import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Konwolucyjny autoencoder z encoder-decoder i skip connections"""
    def __init__(self, latent_dim=128, input_channels=3, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.conv_output_size = image_size // 8
        self.flattened_size = 128 * self.conv_output_size * self.conv_output_size
        
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block3 = nn.Sequential(
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
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.conv_output_size * self.conv_output_size),
            nn.ReLU(True),
            nn.Unflatten(1, (256, self.conv_output_size, self.conv_output_size))
        )
        
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        z = self.to_latent(x3)
        return z, (x1, x2, x3)
    
    def decode(self, z, encoder_features=None):
        x = self.from_latent(z)
        
        if encoder_features is not None:
            x1, x2, x3 = encoder_features
            x = self.decoder_block1(x + x3)
            x = self.decoder_block2(x + x2)
            x = self.final_block(x + x1)
        else:
            x = self.decoder_block1(x)
            x = self.decoder_block2(x)
            x = self.final_block(x)
        
        return x

    def forward(self, x):
        x = 2 * x - 1
        z, encoder_features = self.encode(x)
        reconstructed = self.decode(z, encoder_features)
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