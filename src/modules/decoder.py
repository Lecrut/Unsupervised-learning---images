import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=768, output_channels=4):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.act1 = nn.GELU()
        
        self.up2 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.act2 = nn.GELU()
        
        self.up3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.GELU()
        
        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.GELU()
        
        self.up5 = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z, skips):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8) 
        
        s1, s2, s3, s4 = skips
        

        x = self.act1(self.bn1(self.up1(x)))
        x = torch.cat([x, s4], dim=1) 

        x = self.act2(self.bn2(self.up2(x)))
        x = torch.cat([x, s3], dim=1)

        x = self.act3(self.bn3(self.up3(x)))
        x = torch.cat([x, s2], dim=1)

        x = self.act4(self.bn4(self.up4(x)))
        x = torch.cat([x, s1], dim=1)
        
        x = self.up5(x)
        
        return x
