"""
Dedykowany model do inpaintingu (uzupełniania uszkodzonych fragmentów obrazu).

Implementuje architekturę U-Net z Attention oraz Partial Convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Self-attention block do skupienia się na ważnych fragmentach obrazu.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class PartialConv2d(nn.Module):
    """
    Partial Convolution - traktuje uszkodzone fragmenty inaczej niż zdrowe.
    
    Bazuje na: "Image Inpainting for Irregular Holes Using Partial Convolutions"
    (Liu et al., ECCV 2018)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # Inicjalizuj maskę wagami 1.0
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        
        # Freeze mask conv weights
        for param in self.mask_conv.parameters():
            param.requires_grad = False
            
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            mask: Binary mask [B, 1, H, W], 1=valid, 0=invalid
        """
        if mask is None:
            mask = torch.ones_like(x[:, :1, :, :])
            
        # Expand mask to match input channels
        mask_expanded = mask.expand(-1, x.size(1), -1, -1)
        
        # Apply mask to input
        with torch.no_grad():
            output_mask = self.mask_conv(mask_expanded)
            
        # Normalize
        output = self.conv(x * mask_expanded)
        
        # Renormalize based on mask coverage
        mask_sum = output_mask + 1e-8
        output = output / mask_sum
        
        # Update mask
        new_mask = torch.clamp(output_mask, 0, 1)
        
        return output, new_mask


class UNetInpainting(nn.Module):
    """
    U-Net z attention do inpaintingu obrazów.
    
    Architektura:
    - Encoder: 4 poziomy z downsamplingiem
    - Bottleneck: Attention block
    - Decoder: 4 poziomy z upsamplingiem i skip connections
    """
    
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        in_ch = in_channels
        for feature in features:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = feature
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            AttentionBlock(features[-1] * 2),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        for feature in reversed(features):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx, decoder in enumerate(self.decoder_blocks):
            x = decoder[0](x)  # Upsampling
            skip = skip_connections[idx]
            
            # Concatenate skip connection
            x = torch.cat([skip, x], dim=1)
            
            # Apply remaining convolutions
            x = decoder[1:](x)
        
        # Final output
        x = self.final_conv(x)
        return torch.sigmoid(x)


class PartialConvUNet(nn.Module):
    """
    U-Net używający Partial Convolutions - lepszy dla nieregularnych uszkodzeń.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder with partial convs
        self.enc1 = PartialConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.enc2 = PartialConv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc3 = PartialConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc4 = PartialConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Decoder
        self.dec4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, mask=None):
        # Encoder
        x1, mask1 = self.enc1(x, mask)
        x1 = self.relu(x1)
        
        x2, mask2 = self.enc2(x1, mask1)
        x2 = self.relu(x2)
        
        x3, mask3 = self.enc3(x2, mask2)
        x3 = self.relu(x3)
        
        x4, mask4 = self.enc4(x3, mask3)
        x4 = self.relu(x4)
        
        # Decoder
        x = self.relu(self.dec4(x4))
        x = self.relu(self.dec3(x + x3))  # Skip connection
        x = self.relu(self.dec2(x + x2))
        x = self.dec1(x + x1)
        
        return torch.sigmoid(x)


class SimpleInpainting(nn.Module):
    """
    Prostsza wersja modelu do inpaintingu oparta na autoenkodzie z skip connections.
    
    Dobra jako baseline lub do szybkich eksperymentów.
    """
    
    def __init__(self, latent_dim=512):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded
