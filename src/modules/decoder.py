import torch
import torch.nn as nn

#%% Decoder Module - VaDE-inspired VAE Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.conv_output_size = image_size // 8
        
        flat_size = 256 * self.conv_output_size * self.conv_output_size
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, flat_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(flat_size)
        )
        
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, latent, skip_connections=None):
        batch_size = latent.size(0)
        x = self.from_latent(latent)
        x = x.view(batch_size, 256, self.conv_output_size, self.conv_output_size)
        
        x = self.decoder_block1(x)
        if skip_connections is not None:
            x = torch.cat([x, skip_connections[2]], dim=1)
        
        x = self.decoder_block2(x)
        if skip_connections is not None:
            x = torch.cat([x, skip_connections[1]], dim=1)
        
        x = self.decoder_block3(x)
        x = self.final_conv(x)
        
        return x