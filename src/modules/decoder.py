import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Decoder(nn.Module):
    def __init__(self, output_channels=3, filter_sizes=[32, 64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        reversed_filters = filter_sizes[::-1]
        
        in_c = reversed_filters[0] * 2
        
        for i in range(len(reversed_filters)):
            out_c = reversed_filters[i]
            
            self.ups.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
            
            self.convs.append(DoubleConv(out_c * 2, out_c))
            
            in_c = out_c
        
        self.final_conv = nn.Conv2d(filter_sizes[0], output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skips):
        for i, (up, conv) in enumerate(zip(self.ups, self.convs)):
            x = up(x)
            skip = skips[i]
            
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
            
        return self.sigmoid(self.final_conv(x))