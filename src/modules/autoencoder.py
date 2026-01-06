#%% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List
import torchvision.models as models
from datetime import datetime
from .encoder import Encoder
from .decoder import Decoder

#%% VGG Loss:
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([vgg[:2], vgg[2:7], vgg[7:12], vgg[12:21], vgg[21:30]])
        for param in self.parameters(): param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # Input to rekonstrukcja (3 kanały), Target to oryginał (3 kanały RGB po obcięciu)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x, y = input, target
        for block in self.blocks:
            x, y = block(x), block(y)
            loss += torch.mean(torch.abs(x - y))
        return loss

#%% Autoencoder Module - Optimized for WikiArt
class Autoencoder(nn.Module):
    def __init__(self, 
                 latent_dim=768, 
                 input_channels=4, 
                 learning_rate=3e-4, 
                 image_size=256, 
                 use_amp=True, 
                 load_best=False
                ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.load_best = load_best
        self.best_model_path = Path('checkpoints/best_autoencoder.pt')
        self.model_loaded = False

        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)

        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, output_channels=3, image_size=image_size)

        self.perceptual_loss_fn = VGGPerceptualLoss().to(self.device)

        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_perceptual_loss': [], 'val_perceptual_loss': [],
            'learning_rates': []
        }

        if load_best:
            if not self.best_model_path.exists():
                print(f"Brak zapisanego modelu w {self.best_model_path}")
                print("Model zostanie wytrenowany od nowa")
                self.model_loaded = False
            else:
                try:
                    self.load_checkpoint()
                    self.model_loaded = True
                    print("Model wczytany pomyslnie")
                except Exception as e:
                    print(f"Blad podczas wczytywania modelu: {e}")
                    print("Model zostanie wytrenowany od nowa")
                    self.model_loaded = False
    
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            latent, _ = self.encoder(x)
            reconstruction = self.decoder(latent) # Zwraca [B, 3, H, W]
        return reconstruction, latent
    
    def compute_loss(self, original, reconstruction):
        original = original.to(self.device)
        reconstruction = reconstruction.to(self.device)
        
        original_rgb = original[:, :3, :, :]
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 1. Pixel Loss (Charbonnier)
            diff = reconstruction - original_rgb
            loss_pix = torch.mean(torch.sqrt(diff * diff + 1e-6))
            
            # 2. Perceptual Loss (VGG)
            loss_percep = self.perceptual_loss_fn(reconstruction, original_rgb) * 0.1
            
            total_loss = loss_pix + loss_percep
            
        return total_loss, loss_pix, loss_percep

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_percep_loss = 0.0

        for batch in tqdm(dataloader, desc="Training Epoch"):
            if isinstance(batch, dict):
                img = batch['image'].to(self.device, non_blocking=True)
            else:
                img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device, non_blocking=True)
            
            if torch.isnan(img).any() or torch.isinf(img).any():
                print("Wykryto NaN/Inf w danych wejściowych! Pomijam batch.")
                continue

            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                recon, _ = self.forward(img)
                loss, l_pix, l_percep = self.compute_loss(img, recon)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Wykryto NaN w Loss: {loss.item()}. Pomijam krok.")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += l_pix.item()
            epoch_percep_loss += l_percep.item()
            
        n = len(dataloader)
        return {
            'loss': epoch_loss / n,
            'recon_loss': epoch_recon_loss / n,
            'perceptual_loss': epoch_percep_loss / n
        }
    
    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.eval()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_percep_loss = 0.0

        for _, batch in enumerate(tqdm(dataloader, desc="Validation Epoch")):
            if isinstance(batch, dict):
                img = batch['image'].to(self.device, non_blocking=True)
            else:
                img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device, non_blocking=True)
            
            recon, _ = self.forward(img)
            loss, l_pix, l_percep = self.compute_loss(img, recon)

            epoch_loss += loss.item()
            epoch_recon_loss += l_pix.item()
            epoch_percep_loss += l_percep.item()
            
        n = len(dataloader)
        return {
            'loss': epoch_loss / n,
            'recon_loss': epoch_recon_loss / n,
            'perceptual_loss': epoch_percep_loss / n
        }

    def fit(self, train_loader, val_loader=None, epochs=50, early_stopping_patience=15):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoka {epoch + 1}/{epochs}")

            train_metrics = self.train_epoch(train_loader)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_perceptual_loss'].append(train_metrics['perceptual_loss'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            print(f"Train Loss: {train_metrics['loss']:.5f} (Pix: {train_metrics['recon_loss']:.5f}, VGG: {train_metrics['perceptual_loss']:.5f})")

            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
                
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recon_loss'].append(val_metrics['recon_loss'])
                self.history['val_perceptual_loss'].append(val_metrics['perceptual_loss'])
                
                print(f"Val Loss: {val_metrics['loss']:.5f} (Pix: {val_metrics['recon_loss']:.5f}, VGG: {val_metrics['perceptual_loss']:.5f})")

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self.save_checkpoint(self.best_model_path, epoch, best_val_loss)
                    print(f"Model zapisany.")
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

    def decode(self, latent) -> torch.Tensor:
        self.eval()
        
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).float()
        
        latent = latent.to(self.device)
        
        with torch.no_grad():
            reconstruction = self.decoder(latent)
        
        return reconstruction
    
    def decode_batch(self, latent_vectors, batch_size=128):
        self.decoder.eval()
        
        if isinstance(latent_vectors, np.ndarray):
            latent_vectors = torch.from_numpy(latent_vectors).float()
        
        device = next(self.decoder.parameters()).device
        
        decoded_images = []
        num_samples = latent_vectors.shape[0]
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size)):
                batch = latent_vectors[i:i+batch_size].to(device)
                
                decoded_batch = self.decoder(batch)
                decoded_images.append(decoded_batch.cpu())
                
                if i % (batch_size * 10) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        decoded_images = torch.cat(decoded_images, dim=0)
        return decoded_images.numpy()

    # Functions to save best model
    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists():
            timestamp = datetime.now().strftime("%H-%M-%S-%d-%m")
            old_path = path.parent / f"old-{timestamp}{path.suffix}"
            path.rename(old_path)
            print(f"Stary model przemianowany na: {old_path.name}")
        
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'latent_dim': self.latent_dim
        }, path)

    # function to load best model
    def load_checkpoint(self, path: Optional[Path] = None):
        if path is None:
            path = self.best_model_path
        
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        print(f"Model wczytany z: {path}")
        print(f"Epoka: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")
        
        return checkpoint['epoch'], checkpoint['val_loss']


