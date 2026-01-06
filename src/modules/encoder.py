#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        if in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, 1, 1, 0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h); h = F.silu(h); h = self.conv1(h)
        h = self.norm2(h); h = F.silu(h); h = self.dropout(h); h = self.conv2(h)
        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_); k = self.k(h_); v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h*w).permute(0, 2, 1)
        h_ = torch.bmm(w_, v)
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj_out(h_)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels or in_channels, 3, 2, 0)
    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024, input_channels=4, image_size=256):
        super().__init__()
        self.conv_in = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList([
            nn.Sequential(ResnetBlock(64, 128), Downsample(128, 128)),
            nn.Sequential(ResnetBlock(128, 256), Downsample(256, 256)),
            nn.Sequential(ResnetBlock(256, 512), Downsample(512, 512)),
            nn.Sequential(ResnetBlock(512, 512), Downsample(512, 512)),
            nn.Sequential(ResnetBlock(512, 512), Downsample(512, 512)),
        ])
        self.mid = nn.Sequential(
            ResnetBlock(512, 512),
            AttnBlock(512),
            ResnetBlock(512, 512),
        )
        self.norm_out = nn.GroupNorm(32, 512)
        self.conv_out = nn.Conv2d(512, 512, 3, 1, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
    
    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down: x = block(x)
        x = self.mid(x)
        x = self.norm_out(x); x = F.silu(x); x = self.conv_out(x)
        flat = self.flatten(x)
        latent = self.ln(self.fc(flat))
        return latent, []