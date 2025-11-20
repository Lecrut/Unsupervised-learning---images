import torch
import torch.nn as nn

class DEC(nn.Module):
    """Decoder - rekonstrukcja obrazu z przestrzeni latentnej"""
    
    def __init__(self, latent_dim=128, output_channels=3, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.conv_output_size = image_size // 8
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.conv_output_size * self.conv_output_size),
            nn.ReLU(True),
            nn.Unflatten(1, (256, self.conv_output_size, self.conv_output_size))
        )
        
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, encoder_features=None):
        x = self.from_latent(z)
        
        if encoder_features is not None:
            x1, x2, x3 = encoder_features
            x = self.decoder_block1(x + x3)
            x = self.decoder_block2(x + x2)
            x = self.final_block(x + x1)
        else:
            x = self.decoder_block1(x)
            x = self.decoder_block2(x)
            x = self.final_block(x)
        
        x = (x + 1) / 2
        return x