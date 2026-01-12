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
import torch.nn.functional as F
import kornia as K

class Autoencoder(nn.Module):
    def __init__(self, 
                 latent_dim=2048, 
                 input_channels=4, 
                 learning_rate=0.001,  
                 image_size=256, 
                 use_amp=True, 
                 load_best=False
                ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.learning_rate = learning_rate 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.best_model_path = Path('checkpoints/best_autoencoder.pt')
        self.model_loaded = False

        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, output_channels=3, image_size=image_size)

        self.lap = K.filters.Laplacian(kernel_size=5, normalized=True)
        self.l1_loss = nn.L1Loss()

        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
    
        self.scheduler = None
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_recon_loss': [], 'val_recon_loss': [],
            'learning_rates': []
        }

        if load_best and self.best_model_path.exists():
            self.load_checkpoint()
            self.model_loaded = True

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            latent, _ = self.encoder(x)
            reconstruction = self.decoder(latent)
        return reconstruction, latent

    def compute_loss(self, target, reconstruction, latent):
        target_rgb = target[:, :3, :, :]

        loss_l1 = F.l1_loss(reconstruction, target_rgb)

        with torch.cuda.amp.autocast(enabled=False):
            lap_recon = self.lap(reconstruction.float())
            lap_target = self.lap(target_rgb.float()).detach()
            loss_hf = F.l1_loss(lap_recon, lap_target)

        loss_latent = 0.0
        if latent is not None:
            z = latent - latent.mean(dim=0, keepdim=True)
            z = F.normalize(z, dim=1)
            sim = z @ z.T
            sim = sim - torch.eye(z.size(0), device=z.device)
            loss_latent = sim.pow(2).mean()

        loss = (
            1.0 * loss_l1 +
            0.4 * loss_hf +
            0.05 * loss_latent
        )

        return loss, loss_l1


    def train_epoch(self, dataloader):
        self.train()
        epoch_loss = 0.0
        epoch_recon = 0.0

        for batch in tqdm(dataloader, desc="Training"):
            img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                recon, latent = self.forward(img)
            
            loss, recon_loss_val = self.compute_loss(img, recon, latent)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer) 
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected! Skipping batch")
                # continue

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss_val.item()

        return {
            'loss': epoch_loss / len(dataloader),
            'recon_loss': epoch_recon / len(dataloader)
        }

    @torch.no_grad()
    def validate_epoch(self, dataloader):
        self.eval()
        epoch_loss = 0.0
        epoch_recon = 0.0

        for batch in tqdm(dataloader, desc="Validation"):
            img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            
            recon, latent = self.forward(img)
            loss, recon_loss_val = self.compute_loss(img, recon, latent)

            epoch_loss += loss.item()
            epoch_recon += recon_loss_val.item()

        return {
            'loss': epoch_loss / len(dataloader),
            'recon_loss': epoch_recon / len(dataloader)
        }

    def fit(self, train_loader, val_loader=None, epochs=30, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Start treningu na: {self.device} z OneCycleLR (Max LR: {self.learning_rate})")

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,     
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1,                  
            div_factor=10.0,                
            final_div_factor=10000.0,       
            anneal_strategy='cos'
        )

        for epoch in range(epochs):
            print(f"\nEpoka {epoch + 1}/{epochs}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader) if val_loader else {'loss': 0.0, 'recon_loss': 0.0}

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f} | LR: {current_lr:.6f}")
            
            current_val_loss = val_metrics['loss']
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                self.save_checkpoint(self.best_model_path, epoch, best_val_loss)
                print("-> Model zapisany (Best Val Loss)")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping po {epoch+1} epokach.")
                    break
        
        return self.history
    
    def extract_latent(self, dataloader):
        self.eval()
        latents = []
        with torch.no_grad():
            for batch in dataloader:
                img = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                l, _ = self.encoder(img)
                latents.append(l.cpu().numpy())
        return np.concatenate(latents, axis=0), None

    def decode_batch(self, latents, batch_size=128):
        self.decoder.eval()
        device = next(self.parameters()).device
        if isinstance(latents, np.ndarray): latents = torch.from_numpy(latents).float()
        outs = []
        with torch.no_grad():
            for i in range(0, len(latents), batch_size):
                batch = latents[i:i+batch_size].to(device)
                outs.append(self.decoder(batch).cpu())
        return torch.cat(outs).numpy()

    def save_checkpoint(self, path, epoch, val_loss):
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch, 'val_loss': val_loss
        }, path)

    def load_checkpoint(self, path=None):
        if path is None: path = self.best_model_path
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])