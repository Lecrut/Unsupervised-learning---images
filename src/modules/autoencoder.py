import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List

from .encoder import Encoder
from .decoder import Decoder

#%% Autoencoder Module
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, input_channels=4, learning_rate=1e-4, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': []
        }
    
    def forward(self, x):
        latent, skip_connections = self.encoder(x)
        reconstruction = self.decoder(latent, skip_connections)
        return reconstruction, latent
    
    def compute_loss(self, original, reconstruction) -> torch.Tensor:
        recon_loss = self.mse_loss(reconstruction, original)
        return recon_loss

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0 #reconstruction error

        for batch in tqdm(dataloader, desc="Training Epoch"):
            if isinstance(batch, dict):
                img = batch['image'].to(self.device)
            else:
                img = batch[0].to(self.device)  if isinstance(batch, (list, tuple)) else batch.to(self.device)
            
            self.optimizer.zero_grad()
            reconstruction, _ = self.forward(img)
            loss = self.compute_loss(img, reconstruction)

            loss.backward() # back propagation
            self.optimizer.step() # wage update by Adam

            epoch_loss += loss.item()
            epoch_recon_loss += loss.item()
        
        return {
            'loss': epoch_loss / len(dataloader),
            'recon_loss': epoch_recon_loss / len(dataloader)
        }

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.eval()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0

        # validation is training without back propagation and weight update
        for batch in tqdm(dataloader, desc="Validation Epoch"):
            if isinstance(batch, dict):
                img = batch['image'].to(self.device)
            else:
                img = batch[0].to(self.device)  if isinstance(batch, (list, tuple)) else batch.to(self.device)
            
            reconstruction, _ = self.forward(img)
            loss = self.compute_loss(img, reconstruction)

            epoch_loss += loss.item()
            epoch_recon_loss += loss.item()
        
        return {
            'loss': epoch_loss / len(dataloader),
            'recon_loss': epoch_recon_loss / len(dataloader)
        }

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
            epochs: int = 20,
            save_path: Optional[Path] = None,
            early_stopping: Optional[int] = None):

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoka {epoch + 1}/{epochs}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader) if val_loader else {'loss': 0.0, 'recon_loss': 0.0}

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            
            print(f"Train Loss: {train_metrics['loss']:.6f}")
            print(f"Val Loss: {val_metrics['loss']:.6f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                if save_path:
                    self.save_checkpoint(save_path / 'best_model.pt', epoch, best_val_loss)
                    print(f"Model zapisany (val_loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping:
                    print(f'\nEarly stopping po {epoch+1} epokach')
                    break
        
        return self.history

    def extract_latent(self, dataloader: DataLoader, return_images: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.eval()
        latent_vectors = []
        images = [] if return_images else None

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Latent Vectors"):
                if isinstance(batch, dict):
                    img = batch['image'].to(self.device)
                else:
                    img = batch[0].to(self.device)  if isinstance(batch, (list, tuple)) else batch.to(self.device)
                
                # extraction from image
                latent, _ = self.encoder(img)
                latent_vectors.append(latent.cpu().numpy())
                
                if return_images:
                    images.append(img.cpu().numpy())

        latent_array = np.concatenate(latent_vectors, axis=0)
        images_array = np.concatenate(images, axis=0) if return_images else None

        return latent_array, images_array

    # in future: encode and decode functions:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent, _ = self.encoder(x)
        return latent

    def decode(self, x: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        with torch.no_grad():
            reconstruction = self.decoder(x, skip_connections)
        return reconstruction

    # Functions to save best model
    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, path)

    # function to load best model
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch'], checkpoint['val_loss']
         
        
    