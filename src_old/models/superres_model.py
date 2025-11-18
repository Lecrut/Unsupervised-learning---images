"""
Moduł super-resolution do zwiększania rozdzielczości obrazów dzieł sztuki.

Implementuje model oparty na Residual Blocks do uczenia się
zwiększania rozdzielczości z 128x128 do 256x256.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Blok rezydualny z skip connection dla lepszego przepływu gradientów.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual  # skip connection


class PixelShuffleUpsampler(nn.Module):
    """
    Upsampling za pomocą pixel shuffle (sub-pixel convolution).
    Bardziej efektywny niż transposed convolution.
    """
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        out_channels = in_channels * (upscale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class SuperResolutionModel(nn.Module):
    """
    Model do zwiększania rozdzielczości obrazów (super-resolution).
    
    Architektura oparta na SRResNet:
    - Initial feature extraction
    - Multiple residual blocks
    - Upsampling blocks (pixel shuffle)
    - Final reconstruction
    
    Args:
        upscale_factor (int): Współczynnik zwiększenia rozdzielczości (2 = 128->256)
        num_residual_blocks (int): Liczba bloków rezydualnych (domyślnie 16)
        num_channels (int): Liczba kanałów w warstwach ukrytych (domyślnie 64)
    """
    
    def __init__(self, upscale_factor=2, num_residual_blocks=16, num_channels=64):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        residual_layers = []
        for _ in range(num_residual_blocks):
            residual_layers.append(ResidualBlock(num_channels))
        self.residual_blocks = nn.Sequential(*residual_layers)
        
        # Post-residual convolution
        self.post_residual = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        
        # Upsampling layers
        upsampling_layers = []
        for _ in range(upscale_factor // 2):  # Dla x2 = 1 iteracja, dla x4 = 2 iteracje
            upsampling_layers.append(PixelShuffleUpsampler(num_channels, upscale_factor=2))
        self.upsampling = nn.Sequential(*upsampling_layers)
        
        # Final reconstruction
        self.reconstruction = nn.Conv2d(num_channels, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Obraz niskej rozdzielczości [B, 3, H, W]
            
        Returns:
            Obraz wysokiej rozdzielczości [B, 3, H*upscale_factor, W*upscale_factor]
        """
        # Initial features
        initial = self.initial(x)
        
        # Residual learning
        residual = self.residual_blocks(initial)
        residual = self.post_residual(residual)
        
        # Skip connection z initial features
        x = initial + residual
        
        # Upsampling
        x = self.upsampling(x)
        
        # Final reconstruction
        x = self.reconstruction(x)
        
        return torch.sigmoid(x)  # Output w zakresie [0, 1]


class LightweightSuperRes(nn.Module):
    """
    Lżejsza wersja modelu super-resolution dla szybszego trenowania.
    
    Używa mniej bloków rezydualnych i kanałów.
    Dobra do testowania i eksperymentów.
    """
    
    def __init__(self, upscale_factor=2, num_residual_blocks=8, num_channels=32):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=5, padding=2),
            nn.PReLU()
        )
        
        # Residual blocks
        residual_layers = []
        for _ in range(num_residual_blocks):
            residual_layers.append(ResidualBlock(num_channels))
        self.residual_blocks = nn.Sequential(*residual_layers)
        
        # Upsampling
        self.upsampling = PixelShuffleUpsampler(num_channels, upscale_factor=upscale_factor)
        
        # Final reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_channels, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        x = self.reconstruction(x)
        return x


class ESPCNSuperRes(nn.Module):
    """
    ESPCN (Efficient Sub-Pixel Convolutional Neural Network) - bardzo szybki model SR.
    
    Dobry wybór dla dużych datasetów gdzie szybkość jest ważna.
    """
    
    def __init__(self, upscale_factor=2, num_channels=64):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels // 2, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return torch.sigmoid(x)


def create_lowres_highres_pairs(images: torch.Tensor, downscale_factor: int = 2) -> tuple:
    """
    Tworzy pary obrazów (niska rozdzielczość, wysoka rozdzielczość) do trenowania SR.
    
    Args:
        images: Batch obrazów wysokiej rozdzielczości [B, C, H, W]
        downscale_factor: Współczynnik zmniejszenia (2 = 256->128)
        
    Returns:
        Tuple: (obrazy_lowres, obrazy_highres)
    """
    import torch.nn.functional as F
    
    # Wysoką rozdzielczość traktujemy jako target
    highres = images
    
    # Zmniejszamy rozdzielczość używając interpolacji
    lowres = F.interpolate(
        images,
        scale_factor=1/downscale_factor,
        mode='bicubic',
        align_corners=False
    )
    
    return lowres, highres
