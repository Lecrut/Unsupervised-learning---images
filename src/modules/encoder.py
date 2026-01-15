import torch
import torch.nn as nn

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

class Encoder(nn.Module):
    def __init__(self, input_channels=4, filter_sizes=[32, 64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        in_c = input_channels
        for out_c in filter_sizes:
            self.downs.append(DoubleConv(in_c, out_c))
            in_c = out_c
            
        self.bottleneck = DoubleConv(filter_sizes[-1], filter_sizes[-1] * 2)
        
    def forward(self, x):
        skips = []
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        return x, skips[::-1]