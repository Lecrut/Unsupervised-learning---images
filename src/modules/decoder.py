import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=2048, output_channels=3, image_size=256):
        super().__init__()
        
        self.reshape_h = 4
        self.reshape_w = 4
        self.reshape_c = 128
        self.reshape_flat = self.reshape_c * self.reshape_h * self.reshape_w 
        
        self.fc = nn.Linear(latent_dim, self.reshape_flat)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        def up_block(in_c, out_c):
            return nn.Sequential(
            nn.Conv2d(in_c, out_c * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2), 
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.net = nn.Sequential(
            # Start: 4x4 -> 8x8
            up_block(self.reshape_c, 256),
            
            # 8 -> 16
            up_block(256, 256),
            
            # 16 -> 32
            up_block(256, 128),
            
            # 32 -> 64
            up_block(128, 64),
            
            # 64 -> 128
            up_block(64, 32),
            
            # 128 -> 256
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.act(x)
        
        x = x.view(-1, self.reshape_c, self.reshape_h, self.reshape_w)
        
        x = self.net(x)
        return x