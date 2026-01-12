import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=2048, output_channels=3, image_size=256):
        super().__init__()
        
        self.reshape_h = 16
        self.reshape_w = 16
        self.reshape_c = 32  
        self.reshape_flat = self.reshape_c * self.reshape_h * self.reshape_w 
        
        self.fc_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(latent_dim * 2, self.reshape_flat),
            nn.LeakyReLU(0.2, True)
        )

        self.expand_conv = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c * 4, 3, 1, 1), # PixelShuffle dzieli kanały przez 4
                nn.PixelShuffle(2),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, True)
            )

        self.net = nn.Sequential(
            # 16x16 -> 32x32
            up_block(256, 128), 
            # 32x32 -> 64x64
            up_block(128, 64),  
            # 64x64 -> 128x128
            up_block(64, 32),  
            
            # 128x128 -> 256x256 (Ostatni blok - poprawiony)
            nn.Conv2d(32, 16 * 4, 3, 1, 1), 
            nn.PixelShuffle(2),             
            
            nn.Conv2d(16, 16, 3, 1, 1),     # Zmienione z 32 na 16
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(16, output_channels, 3, 1, 1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.fc_head(z)
        x = x.view(-1, self.reshape_c, self.reshape_h, self.reshape_w)
        x = self.expand_conv(x)
        x = self.net(x)
        return x