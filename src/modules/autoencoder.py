#%% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List
import torchvision.models as models
from .encoder import Encoder
from .decoder import Decoder

#%% SSIM Loss
def ssim_loss(x, y, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = nn.functional.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
    mu_y = nn.functional.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = nn.functional.avg_pool2d(x**2, window_size, stride=1, padding=window_size//2) - mu_x_sq
    sigma_y_sq = nn.functional.avg_pool2d(y**2, window_size, stride=1, padding=window_size//2) - mu_y_sq
    sigma_xy = nn.functional.avg_pool2d(x*y, window_size, stride=1, padding=window_size//2) - mu_xy
    
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1)*(sigma_x_sq + sigma_y_sq + C2))
    
    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(dim=[1,2,3])

#%% Autoencoder Module - Optimized for WikiArt
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=768, input_channels=4, learning_rate=3e-4, image_size=256,):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)

        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'learning_rates': []
        }
    
    def forward(self, x):
        latent, skip_connections = self.encoder(x)
        # reconstruction = self.decoder(latent, skip_connections)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def compute_loss(self, original, reconstruction) -> torch.Tensor:
        mse = self.mse_loss(reconstruction, original)
        l1 = self.l1_loss(reconstruction, original)
        ssim = ssim_loss(reconstruction, original)
        
        loss_components = {
            'mse': mse,
            'l1': l1,
            'ssim': ssim
        }
        
        total_loss = 0.5 * mse + 0.3 * l1 + 0.2 * ssim
        
        return total_loss

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0

        for batch in tqdm(dataloader, desc="Training Epoch"):
            if isinstance(batch, dict):
                img = batch['image'].to(self.device, non_blocking=True)
            else:
                img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = self.forward(img)
            loss = self.compute_loss(img, reconstruction)

            loss.backward() 
            self.optimizer.step() 

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

        for batch in tqdm(dataloader, desc="Validation Epoch"):
            if isinstance(batch, dict):
                img = batch['image'].to(self.device, non_blocking=True)
            else:
                img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device, non_blocking=True)
            
            reconstruction, _ = self.forward(img)
            loss = self.compute_loss(img, reconstruction)

            epoch_loss += loss.item()
            epoch_recon_loss += loss.item()
        
        return {
            'loss': epoch_loss / len(dataloader),
            'recon_loss': epoch_recon_loss / len(dataloader)
        }

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
            epochs: int = 50,
            save_path: Optional[Path] = None,
            early_stopping_patience: int = 15):

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
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_metrics['loss']:.6f}")
            print(f"Val Loss: {val_metrics['loss']:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_loader:
                self.scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                if save_path:
                    self.save_checkpoint(save_path / 'best_model.pt', epoch, best_val_loss)
                    print(f"Model zapisany (val_loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                
                if early_stopping_patience and patience_counter >= early_stopping_patience:
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
                
                latent, _ = self.encoder(img)
                latent_vectors.append(latent.cpu().numpy())
                
                if return_images:
                    images.append(img.cpu().numpy())

        latent_array = np.concatenate(latent_vectors, axis=0)
        images_array = np.concatenate(images, axis=0) if return_images else None

        return latent_array, images_array

    # in future: encode and decode functions:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            latent, _ = self.encoder(x)
        return latent

    def decode(self, latent, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        self.eval()
        
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).float()
        
        latent = latent.to(self.device)
        
        with torch.no_grad():
            reconstruction = self.decoder(latent, skip_connections)
        
        return reconstruction
    
    def decode_batch(self, latent_vectors: np.ndarray, batch_size: int = 64) -> np.ndarray:
        self.eval()
        reconstructed = []
        
        num_samples = latent_vectors.shape[0]
        for i in range(0, num_samples, batch_size):
            batch = latent_vectors[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(self.device)
            
            with torch.no_grad():
                recon = self.decoder(batch_tensor, None)
                reconstructed.append(recon.cpu().numpy())
        
        return np.concatenate(reconstructed, axis=0)

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

        self.to(self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch'], checkpoint['val_loss']
         
        
    