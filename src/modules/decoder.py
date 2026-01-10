import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, output_channels=4, image_size=256):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        self.image_size = image_size

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU()
            )

        self.up1 = up_block(512, 512)   # 8 → 16
        self.up2 = up_block(512, 256)   # 16 → 32
        self.up3 = up_block(256, 128)   # 32 → 64
        self.up4 = up_block(128, 64)    # 64 → 128

        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.unflatten(self.fc(z))

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.final(x)
        return x