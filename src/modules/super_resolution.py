#%% Imports
from pathlib import Path
from src.modules.autoencoder import Autoencoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Dict
from tqdm import tqdm


#%% Super Resolution Model
class SuperResolutionModel(Autoencoder): 
    def __init__(self, latent_dim=768, input_channels=3, learning_rate=0.001, image_size=256, use_amp=True, load_best=False):
        super().__init__(latent_dim=latent_dim, input_channels=input_channels, learning_rate=learning_rate, image_size=image_size, use_amp=use_amp, load_best=load_best)
        self.upsampler = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels * 4, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.PixelShuffle(2), 
            nn.Sigmoid()
        ).to(self.device)

        self.best_model_path = Path('checkpoints/super_resolution/best_super_resolution.pt')

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            latent, _ = self.encoder(x)
            feature_256 = self.decoder(latent) 
            output_512 = self.upsampler(feature_256) 
        return output_512, latent

    def train_epoch(self, lr_dataloader: DataLoader, hr_dataloader: DataLoader) -> Dict[str, float]:
        self.train()
        epoch_loss = 0.0
        
        for lr_batch, hr_batch in tqdm(zip(lr_dataloader, hr_dataloader), desc="Training SR"):
            lr_img = lr_batch.to(self.device, non_blocking=True) if isinstance(lr_batch, torch.Tensor) else lr_batch['image'].to(self.device, non_blocking=True)
            hr_img = hr_batch.to(self.device, non_blocking=True) if isinstance(hr_batch, torch.Tensor) else hr_batch['image'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                sr_img, _ = self.forward(lr_img)
                loss = self.compute_loss(hr_img, sr_img)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(lr_dataloader)}

    def compute_loss(self, hr_img, sr_img):
        loss_fn = nn.L1Loss()
        return loss_fn(sr_img, hr_img)

    @torch.no_grad()
    def validate_epoch(self, lr_dataloader: DataLoader, hr_dataloader: DataLoader) -> Dict[str, float]:
        self.eval()
        epoch_loss = 0.0
        
        for lr_batch, hr_batch in tqdm(zip(lr_dataloader, hr_dataloader), desc="Validating SR"):
            lr_img = lr_batch.to(self.device, non_blocking=True) if isinstance(lr_batch, torch.Tensor) else lr_batch['image'].to(self.device, non_blocking=True)
            hr_img = hr_batch.to(self.device, non_blocking=True) if isinstance(hr_batch, torch.Tensor) else hr_batch['image'].to(self.device, non_blocking=True)
            
            sr_img, _ = self.forward(lr_img)
            loss = self.compute_loss(hr_img, sr_img)
            
            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(lr_dataloader)}

    def fit(self, train_small: DataLoader, train_big: DataLoader, val_small: DataLoader, val_big: DataLoader, epochs: int = 50, early_stopping_patience: int = 15):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoka {epoch + 1}/{epochs}")

            train_metrics = self.train_epoch(train_small, train_big)
            val_metrics = self.validate_epoch(val_small, val_big)

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
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