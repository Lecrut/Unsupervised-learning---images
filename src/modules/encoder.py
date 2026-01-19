import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_channels=64, input_channels=4, base_channels=32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels) 
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        self.act2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        self.act3 = nn.GELU()
        
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_channels*8)
        self.act4 = nn.GELU()
        
        self.conv5 = nn.Conv2d(base_channels*8, base_channels*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(base_channels*8)
        self.act5 = nn.GELU()
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(base_channels*8, latent_channels, kernel_size=3, padding=1)
        self.bottleneck_norm = nn.GroupNorm(4, latent_channels)

    def forward(self, x):        
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        
        latent = self.bottleneck_norm(self.bottleneck(x))
        return latent

