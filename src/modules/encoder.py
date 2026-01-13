import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=2048, input_channels=4, image_size=256):
        super().__init__()
        
        # 1. Feature Extractor (Schodzimy do 16x16)
        self.features = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(input_channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            # 128 -> 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            # 64 -> 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            # 32 -> 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
        )
        
    
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        
        # Wielkość po spłaszczeniu: 32 kanały * 16 * 16 = 8192 features
        self.flat_size = 32 * 16 * 16
        
        self.fc_head = nn.Sequential(
            nn.Linear(self.flat_size, latent_dim * 2), 
            nn.BatchNorm1d(latent_dim * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(latent_dim * 2, latent_dim)      
        )
        
    def forward(self, x):
        x = self.features(x)       # [B, 256, 16, 16]
        x = self.reduce_conv(x)    # [B, 32, 16, 16] 
        x = torch.flatten(x, 1)    # [B, 8192]
        latent = self.fc_head(x)   # [B, 2048]
        return latent, None