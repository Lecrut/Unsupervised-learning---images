#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Dynamic Degradation-Aware Convolution
class DAConv(nn.Module):
    def __init__(self, channels=64, rep_dim=256):
        super().__init__()
        self.kernel_fc = nn.Linear(rep_dim, channels * 9)
        self.modulation_fc = nn.Sequential(
            nn.Linear(rep_dim, channels),
            nn.Sigmoid()
        )
        self.pw = nn.Conv2d(channels, channels, 1)

    def forward(self, x, rep):
        B, C, H, W = x.shape

        kernel = self.kernel_fc(rep).view(B * C, 1, 3, 3)

        x_reshaped = x.view(1, B * C, H, W)
        out = F.conv2d(x_reshaped, kernel, groups=B * C, padding=1)
        out = out.view(B, C, H, W)

        mod = self.modulation_fc(rep).view(B, C, 1, 1)
        out = out * mod

        return self.pw(out)


#%% DASR Block (uses DAConv)
class DASRBlock(nn.Module):
    def __init__(self, channels=64, rep_dim=256):
        super().__init__()
        self.da = DAConv(channels, rep_dim)

    def forward(self, x, rep):
        return x + self.da(x, rep)


#%% DASR Model
class DASR(nn.Module):
    def __init__(self, scale=4, rep_dim=256, channels=64, num_blocks=6):
        super().__init__()

        self.head = nn.Conv2d(3, channels, 3, padding=1)

        self.blocks = nn.ModuleList([
            DASRBlock(channels, rep_dim) for _ in range(num_blocks)
        ])

        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, padding=1)
        )

    def forward(self, img_lr: torch.Tensor, rep: torch.Tensor):
        x = self.head(img_lr)

        for block in self.blocks:
            x = block(x, rep)

        return self.upsample(x)


#%% Super resolution function
def super_resolve_with_rep(image_tensor: torch.Tensor, rep: torch.Tensor, scale=4):
    model = DASR(scale=scale, rep_dim=rep.shape[-1])
    return model(image_tensor, rep)
