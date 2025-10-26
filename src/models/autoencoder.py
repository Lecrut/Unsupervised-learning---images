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
        self.encoder_block1 = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block2 = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block3 = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1)
        )
        
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.conv_output_size * self.conv_output_size, latent_dim),
            nn.LeakyReLU(0.2, True)
        )
        
        # Decoder
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.conv_output_size * self.conv_output_size),
            nn.ReLU(True),
            nn.Unflatten(1, (256, self.conv_output_size, self.conv_output_size))
        )
        
        self.decoder_block1 = nn.Sequential(
            # 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.decoder_block2 = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.final_block = nn.Sequential(
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # wynik w [-1,1] - lepszy zakres niż sigmoid
        )

    def encode(self, x):
        """Enkoduje obraz do przestrzeni latentnej"""
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        z = self.to_latent(x3)
        return z, (x1, x2, x3)
    
    def decode(self, z, encoder_features=None):
        """Dekoduje wektor latentny do obrazu"""
        x = self.from_latent(z)
        
        # Skip connections if available
        if encoder_features is not None:
            x1, x2, x3 = encoder_features
            x = self.decoder_block1(x + x3)  # Skip connection
            x = self.decoder_block2(x + x2)  # Skip connection
            x = self.final_block(x + x1)     # Skip connection
        else:
            x = self.decoder_block1(x)
            x = self.decoder_block2(x)
            x = self.final_block(x)
        
        return x

    def forward(self, x):
        """
        Forward pass przez autoencoder z residual connections
        
        Returns:
            tuple: (rekonstruowany_obraz, wektor_latentny)
        """
        # Normalizacja wejścia do [-1,1]
        x = 2 * x - 1
        
        # Encode
        z, encoder_features = self.encode(x)
        
        # Decode with skip connections
        reconstructed = self.decode(z, encoder_features)
        
        # Denormalizacja wyjścia do [0,1]
        reconstructed = (reconstructed + 1) / 2
        
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