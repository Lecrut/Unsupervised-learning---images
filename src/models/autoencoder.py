"""
Implementacja konwolucyjnego autoencodera dla rekonstrukcji obrazów
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Konwolucyjny autoencoder do rekonstrukcji obrazów i inpaintingu.
    
    Architektura:
    - Encoder: 3 warstwy konwolucyjne + flatten + linear
    - Decoder: linear + unflatten + 3 warstwy transponowane konwolucyjne
    
    Args:
        latent_dim (int): Wymiar przestrzeni latentnej (domyślnie 128)
        input_channels (int): Liczba kanałów wejściowych (domyślnie 3 dla RGB)
        image_size (int): Rozmiar obrazu wejściowego (domyślnie 256)
    """
    
    def __init__(self, latent_dim=128, input_channels=3, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.image_size = image_size
        
        # Oblicz rozmiar po konwolucjach
        # 256 -> 128 -> 64 -> 32 (3 warstwy z stride=2)
        self.conv_output_size = image_size // 8  # 32 dla 256x256
        self.flattened_size = 128 * self.conv_output_size * self.conv_output_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            # 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.Flatten(),
            nn.Linear(self.flattened_size, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.ReLU(True),
            nn.Unflatten(1, (128, self.conv_output_size, self.conv_output_size)),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # wynik w [0,1]
        )

    def encode(self, x):
        """Enkoduje obraz do przestrzeni latentnej"""
        return self.encoder(x)
    
    def decode(self, z):
        """Dekoduje wektor latentny do obrazu"""
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass przez autoencoder
        
        Returns:
            tuple: (rekonstruowany_obraz, wektor_latentny)
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z
    
    def get_latent_representation(self, x):
        """Zwraca tylko reprezentację latentną"""
        with torch.no_grad():
            return self.encode(x)
    
    def reconstruct(self, x):
        """Rekonstruuje obraz (bez gradientów)"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            return reconstructed