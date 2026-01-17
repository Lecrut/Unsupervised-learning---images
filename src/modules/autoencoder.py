import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import kornia as K
from .encoder import Encoder
from .decoder import Decoder 

class Autoencoder(nn.Module):
    def __init__(self, 
                 latent_channels=32,
                 input_channels=4, 
                 learning_rate=0.001,  
                 image_size=256, 
                 use_amp=True, 
                 load_best=False,
                ):
        super().__init__()

        self.latent_channels = latent_channels
        self.input_channels = input_channels
        self.image_size = image_size
        self.learning_rate = learning_rate 
        self.current_epoch = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.best_model_path = Path('checkpoints/best_autoencoder_conv.pt')
        self.model_loaded = False
        
        self.encoder = Encoder(latent_channels=latent_channels, input_channels=input_channels)
        self.decoder = Decoder(latent_channels=latent_channels, output_channels=input_channels)

        self.lap = K.filters.Laplacian(kernel_size=5, normalized=True)
        self.l1_loss = nn.L1Loss()
        
        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(self.device.__str__(), enabled=self.use_amp)

        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}

        if load_best and self.best_model_path.exists():
            self.load_checkpoint()
            self.model_loaded = True

    def forward(self, x):
        with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
            latent = self.encoder(x)
            reconstruction = self.decoder(latent)
        return reconstruction, latent

    def compute_loss(self, target, reconstruction):
        loss_l1 = self.l1_loss(reconstruction, target)

        with torch.amp.autocast(self.device.__str__(), enabled=False):
            lap_recon = self.lap(reconstruction.float())
            lap_target = self.lap(target.float()).detach()
            loss_hf = F.l1_loss(lap_recon, lap_target)

        loss = 1.0 * loss_l1 + 1.5 * loss_hf 
        return loss, loss_l1

    def train_epoch(self, dataloader):
        self.train()
        epoch_loss = 0.0
        epoch_recon = 0.0

        for batch in tqdm(dataloader, desc="Training"):
            img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                recon, _ = self.forward(img)
                loss, recon_loss_val = self.compute_loss(img, recon)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss_val.item()

        return {'loss': epoch_loss / len(dataloader), 'l1_loss': epoch_recon / len(dataloader)}

    @torch.no_grad()
    def validate_epoch(self, dataloader):
        self.eval()
        epoch_loss = 0.0
        epoch_recon = 0.0

        for batch in tqdm(dataloader, desc="Validation"):
            img = batch[0].to(self.device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            recon, _ = self.forward(img)
            loss, recon_loss_val = self.compute_loss(img, recon)
            epoch_loss += loss.item()
            epoch_recon += recon_loss_val.item()

        return {'loss': epoch_loss / len(dataloader), 'l1_loss': epoch_recon / len(dataloader)}

    def fit(self, train_loader, val_loader=None, epochs=30, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1, 
            div_factor=25.0,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )

        for epoch in range(epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader) if val_loader else {'loss': 0.0}

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])

            print(f"Epoch {epoch+1} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_checkpoint(self.best_model_path, epoch, best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
        return self.history

    def extract_latent(self, dataloader):
        self.eval()
        latents = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting latents"):
                img = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                l = self.encoder(img)
                latents.append(l.cpu().numpy())
                
        return np.concatenate(latents, axis=0), None

    def decode_batch(self, latents, batch_size=64):
        self.decoder.eval()
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float()

        outs = []
        with torch.no_grad():
            for i in range(0, len(latents), batch_size):
                batch_z = latents[i:i+batch_size].to(self.device)
                decoded = self.decoder(batch_z)
                outs.append(decoded.cpu())

        return torch.cat(outs).numpy()

    def save_checkpoint(self, path, epoch, val_loss):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'latent_channels': self.latent_channels 
        }, path)

    def load_checkpoint(self, path=None):
        if path is None: path = self.best_model_path
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")