import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=768, output_channels=4, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        flat_size = 512 * 16 * 16
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, flat_size),
            nn.BatchNorm1d(flat_size),
            nn.GELU()
        )
        
        self.decoder_block1 = self._make_deconv_block(512, 256)
        self.decoder_block2 = self._make_deconv_block(256, 128)
        self.decoder_block3 = self._make_deconv_block(128, 64)
        self.decoder_block4 = self._make_deconv_block(64, 32)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _make_deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, latent, skip_connections=None):
        batch_size = latent.size(0)
        x = self.from_latent(latent)
        x = x.view(batch_size, 512, 16, 16)
        
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        
        x = self.final_conv(x)
        return x