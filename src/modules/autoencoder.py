#%% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from datetime import datetime
from .encoder import Encoder
from .decoder import Decoder
import torchvision.transforms as T
import torch.nn.functional as F

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

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.augmenter = nn.Sequential(
            T.RandomResizedCrop(size=image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
        )

        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
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
            projected = self.projection_head(latent)
            reconstruction = self.decoder(latent)   
        return reconstruction, latent, projected
    
    def compute_loss(self, original, reconstruction, projected_vecs):
        original = original.to(self.device)
        reconstruction = reconstruction.to(self.device)
        projected_vecs = projected_vecs.to(self.device)

        original_rgb = original[:, :3, :, :]
        reconstruction_rgb = reconstruction[:, :3, :, :]
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            diff = reconstruction_rgb - original_rgb
            loss_pix = torch.mean(torch.sqrt(diff * diff + 1e-6))
            
            def get_grads(x):
                dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
                dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
                return dy, dx
                
            orig_dy, orig_dx = get_grads(original_rgb)
            recon_dy, recon_dx = get_grads(reconstruction_rgb)
            loss_grad = torch.mean(torch.abs(orig_dy - recon_dy)) + torch.mean(torch.abs(orig_dx - recon_dx))

            loss_recon = loss_pix + loss_grad

            z_norm = F.normalize(projected_vecs, dim=1)
            similarity = torch.matmul(z_norm, z_norm.T)
            
            mask = torch.eye(z_norm.shape[0], device=self.device).bool()
            similarity.masked_fill_(mask, 0)
            
            loss_separation = torch.mean(similarity ** 2)

            total_loss = loss_recon + (0.1 * loss_separation)

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
            
            if torch.isnan(img).any() or torch.isinf(img).any():
                continue

            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                augmented_img = self.augmenter(img)
                
                reconstruction, _, projected = self.forward(augmented_img)
                
                loss = self.compute_loss(img, reconstruction, projected)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

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

        for _, batch in enumerate(tqdm(dataloader, desc="Validation Epoch")):
            if isinstance(batch, dict):
                img = batch['image'].to(self.device, non_blocking=True)
            else:
                img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device, non_blocking=True)
            
            reconstruction, _, projected = self.forward(img)
            
            loss = self.compute_loss(img, reconstruction, projected)

            epoch_loss += loss.item()
            epoch_recon_loss += loss.item()
        
        return {
            'loss': epoch_loss / len(dataloader),
            'recon_loss': epoch_recon_loss / len(dataloader)
        }

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
            epochs: int = 50,
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

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                self.save_checkpoint(self.best_model_path, epoch, best_val_loss)
                print(f"Model zapisany (val_loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f'\nEarly stopping po {epoch+1} epokach')
                    break
        
        print(f"\nNajlepszy model zapisany w: {self.best_model_path}")
        
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
            try:
                path.rename(old_path)
                print(f"Stary model przemianowany na: {old_path.name}")
            except OSError as e:
                print(f"Nie udało się zmienić nazwy starego pliku: {e}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'latent_dim': self.latent_dim
        }, path)

    def load_checkpoint(self, path: Optional[Path] = None):
        if path is None:
            path = self.best_model_path
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Wykryto stary format checkpointu. Ładowanie częściowe...")
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if 'decoder_state_dict' in checkpoint:
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print("UWAGA: Projection Head został zainicjalizowany losowo (nie był obecny w starym modelu).")

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        print(f"Model wczytany z: {path}")
        print(f"Epoka: {checkpoint.get('epoch', 0)}, Val Loss: {checkpoint.get('val_loss', 0.0):.6f}")
        
        return checkpoint.get('epoch', 0), checkpoint.get('val_loss', 0.0)


