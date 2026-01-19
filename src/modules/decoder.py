import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_channels=64, output_channels=4, base_channels=32):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(latent_channels, base_channels*8, kernel_size=3, padding=1)
        self.initial_norm = nn.BatchNorm2d(base_channels*8)
        self.initial_act = nn.GELU()
        
        self.up1 = nn.ConvTranspose2d(base_channels*8, base_channels*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels*8)
        self.act1 = nn.GELU()
        
        self.up2 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*4)
        self.act2 = nn.GELU()
        
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*2)
        self.act3 = nn.GELU()
        
        self.up4 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_channels)
        self.act4 = nn.GELU()
        
        self.up5 = nn.ConvTranspose2d(base_channels, output_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z):
        x = self.initial_act(self.initial_norm(self.initial_conv(z)))
        
        x = self.act1(self.bn1(self.up1(x)))
        x = self.act2(self.bn2(self.up2(x)))
        x = self.act3(self.bn3(self.up3(x)))
        x = self.act4(self.bn4(self.up4(x)))
        
        x = self.up5(x)
        return x