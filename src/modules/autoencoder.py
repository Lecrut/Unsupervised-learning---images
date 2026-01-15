import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import kornia as K
from .encoder import Encoder
from .decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(self, 
                 input_channels=4, 
                 learning_rate=0.001,  
                 image_size=256,
                 layers=5,           
                 latent_dim=1024,    
                 use_amp=True, 
                 load_best=False
                ):
        super().__init__()
        self.input_channels = input_channels
        self.model_loaded = False
        self.learning_rate = learning_rate 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.best_model_path = Path('checkpoints/best_unet_autoencoder.pt')
        
        start_filters = latent_dim // (2 ** (layers - 1))
        
        self.filter_sizes = [start_filters * (2 ** i) for i in range(layers)]
        

        self.encoder = Encoder(input_channels, self.filter_sizes)
        self.decoder = Decoder(output_channels=3, filter_sizes=self.filter_sizes)

        self.lap = K.filters.Laplacian(kernel_size=5, normalized=True)
        self.l1_loss = nn.L1Loss()

        self.to(self.device)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scheduler = None

        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}

        if load_best and self.best_model_path.exists():
            self.load_checkpoint()
            self.model_loaded = True

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            latent_spatial, skips = self.encoder(x)
            reconstruction = self.decoder(latent_spatial, skips)
        return reconstruction, latent_spatial

    def compute_loss(self, target, reconstruction):
        target_rgb = target[:, :3, :, :]
        loss_l1 = self.l1_loss(reconstruction, target_rgb)
        
        with torch.cuda.amp.autocast(enabled=False):
            lap_recon = self.lap(reconstruction.float())
            lap_target = self.lap(target_rgb.float()).detach()
            loss_hf = F.l1_loss(lap_recon, lap_target)

        loss = loss_l1 + 0.4 * loss_hf
        return loss, loss_l1

    def train_epoch(self, dataloader):
        self.train()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                recon, _ = self.forward(img)
            
            loss, _ = self.compute_loss(img, recon)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer) 
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler: self.scheduler.step()
            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(dataloader)}

    @torch.no_grad()
    def validate_epoch(self, dataloader):
        self.eval()
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc="Validation"):
            img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            recon, _ = self.forward(img)
            loss, _ = self.compute_loss(img, recon)
            epoch_loss += loss.item()
        return {'loss': epoch_loss / len(dataloader)}

    def fit(self, train_loader, val_loader=None, epochs=30, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.learning_rate, steps_per_epoch=len(train_loader),
            epochs=epochs, pct_start=0.1, div_factor=10.0, final_div_factor=10000.0
        )

        for epoch in range(epochs):
            print(f"\nEpoka {epoch + 1}/{epochs}")
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader) if val_loader else {'loss': 0.0}
            
            print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_checkpoint(self.best_model_path, epoch, best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping.")
                    break
        return self.history

    def extract_latent(self, dataloader):
        self.eval()
        latents = []
        with torch.no_grad():
            for batch in dataloader:
                img = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                bottleneck, _ = self.encoder(img)
                latents.append(torch.mean(bottleneck, dim=(2, 3)).cpu().numpy())
        return np.concatenate(latents, axis=0), None

    def save_checkpoint(self, path, epoch, val_loss):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self, path=None):
        if path is None: path = self.best_model_path
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])