import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder Module - ekstrakcja reprezentacji latentnej"""
    
    def __init__(self, latent_dim=128, input_channels=3, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.conv_output_size = image_size // 8
        
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1)
        )
        
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.conv_output_size * self.conv_output_size, latent_dim),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x):
        x = 2 * x - 1
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        z = self.to_latent(x3)
        return z, (x1, x2, x3)