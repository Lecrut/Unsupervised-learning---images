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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.best_model_path = Path('checkpoints/best_autoencoder.pt')
        self.model_loaded = False
        self.image_size = image_size

        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_recon_loss': [], 'val_cont_loss': []
        }

        if load_best and self.best_model_path.exists():
            try:
                self.load_checkpoint()
                self.model_loaded = True
            except:
                self.model_loaded = False

    def fast_augment(self, x):
        B, C, H, W = x.shape

        padded = F.pad(x, (16, 16, 16, 16), mode='reflect')
        h_off = np.random.randint(0, 32)
        w_off = np.random.randint(0, 32)
        x_aug = padded[:, :, h_off:h_off+H, w_off:w_off+W]
        
        noise = torch.randn_like(x_aug) * 0.05
        return x_aug + noise
    
    def forward(self, x, return_projection=False):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            latent, _ = self.encoder(x)
            latent = F.normalize(latent, dim=1) 
            reconstruction = self.decoder(latent)
            
            if return_projection:
                proj = self.projection_head(latent)
                return reconstruction, latent, proj
                
        return reconstruction, latent

    def compute_loss(self, original, reconstruction, proj_orig, proj_aug):
        loss_recon = F.l1_loss(reconstruction, original)

        with torch.cuda.amp.autocast(enabled=False):
            p1 = F.normalize(proj_orig.float(), dim=1)
            p2 = F.normalize(proj_aug.float(), dim=1)
            
            z = torch.cat([p1, p2], dim=0)
            sim = torch.matmul(z, z.T) / 0.1
            
            n = sim.shape[0]
            mask = torch.eye(n, device=self.device, dtype=torch.bool)
            sim.masked_fill_(mask, -9e15)
            
            batch_size = p1.shape[0]
            labels = torch.cat([
                torch.arange(batch_size, device=self.device) + batch_size,
                torch.arange(batch_size, device=self.device)
            ])
            loss_cont = F.cross_entropy(sim, labels)

        total_loss = loss_recon + (0.05 * loss_cont)
        
        return total_loss, loss_recon, loss_cont

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.train()
        metrics = {'loss': 0.0, 'recon': 0.0, 'cont': 0.0}
        
        for batch in tqdm(dataloader, desc="Train", leave=False):
            img = batch['image'] if isinstance(batch, dict) else batch
            img = img.to(self.device, non_blocking=True)
            
            if torch.isnan(img).any(): continue

            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                img_aug = self.fast_augment(img)
        
                recon, _, proj_orig = self.forward(img, return_projection=True)
                _, _, proj_aug = self.forward(img_aug, return_projection=True)[0:3]
                
                loss, l_recon, l_cont = self.compute_loss(img, recon, proj_orig, proj_aug)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()

            metrics['loss'] += loss.item()
            metrics['recon'] += l_recon.item()
            metrics['cont'] += l_cont.item()
        
        return {k: v / len(dataloader) for k, v in metrics.items()}
    
    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.eval()
        metrics = {'loss': 0.0, 'recon': 0.0}

        for batch in tqdm(dataloader, desc="Val", leave=False):
            img = batch['image'] if isinstance(batch, dict) else batch
            img = img.to(self.device, non_blocking=True)
            
            recon, _ = self.forward(img)
            loss = F.l1_loss(recon, img)
            
            metrics['loss'] += loss.item()
            metrics['recon'] += loss.item()
        
        return {k: v / len(dataloader) for k, v in metrics.items()}

    def fit(self, train_loader, val_loader=None, epochs=50, early_stopping_patience=15):
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            print(f"Epoka {epoch + 1}/{epochs} | ", end='')
            
            train_m = self.train_epoch(train_loader)
            val_m = self.validate_epoch(val_loader) if val_loader else {'loss': 0.0}
            
            self.history['train_loss'].append(train_m['loss'])
            self.history['val_loss'].append(val_m['loss'])
            
            print(f"Loss: {train_m['loss']:.4f} (Recon: {train_m['recon']:.4f}, Cont: {train_m['cont']:.4f}) | Val: {val_m['loss']:.4f}")

            if val_m['loss'] < best_loss:
                best_loss = val_m['loss']
                patience = 0
                self.save_checkpoint(self.best_model_path, epoch, best_loss)
            else:
                patience += 1
                if early_stopping_patience and patience >= early_stopping_patience:
                    print("Early stopping!")
                    break
        return self.history

    def extract_latent(self, dataloader, return_images=False):
        self.eval()
        latents, imgs = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                img = batch['image'] if isinstance(batch, dict) else batch
                img = img.to(self.device)
                latent, _ = self.encoder(img)
                latent = F.normalize(latent, dim=1)
                latents.append(latent.cpu().numpy())
                if return_images: imgs.append(img.cpu().numpy())
        return np.concatenate(latents), (np.concatenate(imgs) if return_images else None)

    def save_checkpoint(self, path, epoch, val_loss):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'history': self.history
        }, path)

    def load_checkpoint(self, path=None):
        if path is None: path = self.best_model_path
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt['state_dict'], strict=False)
        self.history = ckpt.get('history', self.history)
        print(f"Wczytano epokę {ckpt.get('epoch', 0)}")