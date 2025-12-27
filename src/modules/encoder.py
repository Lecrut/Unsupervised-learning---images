#%% Imports
import torch
import torch.nn as nn

#%% Encoder Definition - Simplified and Faster
class Encoder(nn.Module):
    def __init__(self, latent_dim=768, input_channels=4, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()
        
        # 128 -> 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.GELU()
        
        # 64 -> 32
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.GELU()
        
        # 32 -> 16
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.GELU()
        
        # 16 -> 8
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.GELU()
        
        # latent: 512 x 8 x 8 = 32768
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x):        
        x1 = self.act1(self.bn1(self.conv1(x)))
        x2 = self.act2(self.bn2(self.conv2(x1)))
        x3 = self.act3(self.bn3(self.conv3(x2)))
        x4 = self.act4(self.bn4(self.conv4(x3)))
        x5 = self.act5(self.bn5(self.conv5(x4)))
        
        flat = self.flatten(x5)
        latent = self.norm(self.fc(flat))
        
        return latent, [x1, x2, x3, x4]

