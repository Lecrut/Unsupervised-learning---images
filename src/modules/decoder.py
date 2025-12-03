#%% Imports
import torch
import torch.nn as nn

#%% Decoder Module - Optimized for WikiArt
class Decoder(nn.Module):
    def __init__(self, latent_dim=768, output_channels=4, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        flat_size = 512 * 8 * 8
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, flat_size),
            nn.LayerNorm(flat_size),
            nn.GELU()
        )
        
        self.decoder_block1 = self._make_deconv_block(512, 512)
        self.decoder_block2 = self._make_deconv_block(512 + 512, 256)
        self.decoder_block3 = self._make_deconv_block(256 + 256, 128)
        self.decoder_block4 = self._make_deconv_block(128 + 128, 64)
        self.decoder_block5 = self._make_deconv_block(64 + 64, 32)
        
        self.decoder_block2_no_skip = self._make_deconv_block(512, 256)
        self.decoder_block3_no_skip = self._make_deconv_block(256, 128)
        self.decoder_block4_no_skip = self._make_deconv_block(128, 64)
        self.decoder_block5_no_skip = self._make_deconv_block(64, 32)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _make_deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(32, out_channels // 2), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, out_channels // 2), out_channels),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, latent, skip_connections=None):
        batch_size = latent.size(0)
        x = self.from_latent(latent)
        x = x.view(batch_size, 512, 8, 8)
        
        x = self.decoder_block1(x)
        
        if skip_connections is not None and len(skip_connections) >= 5:
            x = torch.cat([x, skip_connections[4]], dim=1)
            x = self.decoder_block2(x)
            x = torch.cat([x, skip_connections[3]], dim=1)
            x = self.decoder_block3(x)
            x = torch.cat([x, skip_connections[2]], dim=1)
            x = self.decoder_block4(x)
            x = torch.cat([x, skip_connections[1]], dim=1)
            x = self.decoder_block5(x)
        else:
            x = self.decoder_block2_no_skip(x)
            x = self.decoder_block3_no_skip(x)
            x = self.decoder_block4_no_skip(x)
            x = self.decoder_block5_no_skip(x)
        
        x = self.final_conv(x)
        return x