#%% Imports
import torch
import torch.nn as nn

#%% Encoder Definition - Optimized for WikiArt Classification
class Encoder(nn.Module):
    def __init__(self, latent_dim=768, input_channels=4, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.encoder_block1 = self._make_conv_block(input_channels, 64)
        self.encoder_block2 = self._make_conv_block(64, 128)
        self.encoder_block3 = self._make_conv_block(128, 256)
        self.encoder_block4 = self._make_conv_block(256, 512)
        self.encoder_block5 = self._make_conv_block(512, 512)
        
        flat_size = 512 * 8 * 8
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        x = 2 * x - 1
        
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)
        x5 = self.encoder_block5(x4)
        
        latent = self.to_latent(x5)
        return latent, (x1, x2, x3, x4, x5)

