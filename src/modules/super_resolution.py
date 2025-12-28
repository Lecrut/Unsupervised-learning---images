#%% Imports
from pathlib import Path
from src.modules.autoencoder import Autoencoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Dict
from tqdm import tqdm

#%% Helper Classes
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

#%% Super Resolution Model
class SuperResolutionModel(Autoencoder): 
    def __init__(self, latent_dim=768, input_channels=3, learning_rate=0.001, image_size=256, use_amp=True, load_best=False):
        super().__init__(latent_dim=latent_dim, input_channels=input_channels, learning_rate=learning_rate, image_size=image_size, use_amp=use_amp, load_best=load_best)
        
        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            *[ResidualBlock(64) for _ in range(5)],
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, self.input_channels, kernel_size=9, padding=4),
            nn.Sigmoid()
        ).to(self.device)

        self.l1_loss = nn.L1Loss()
        
        self.best_model_path = Path('checkpoints/super_resolution/best_super_resolution.pt')

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model(x)
        return output, None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.train()
        epoch_loss = 0.0
        
        for lr_img, hr_img in tqdm(dataloader, desc="Training SR"):
            lr_img = lr_img.to(self.device, non_blocking=True)
            hr_img = hr_img.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                sr_img, _ = self.forward(lr_img)
                loss = self.compute_loss(hr_img, sr_img)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        return {'loss': avg_loss, 'recon_loss': avg_loss}

    def compute_loss(self, hr_img, sr_img):
        return self.l1_loss(sr_img, hr_img)
        
    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.eval()
        epoch_loss = 0.0
        
        for lr_img, hr_img in tqdm(dataloader, desc="Validating SR"):
            lr_img = lr_img.to(self.device, non_blocking=True)
            hr_img = hr_img.to(self.device, non_blocking=True)
            
            sr_img, _ = self.forward(lr_img)
            loss = self.compute_loss(hr_img, sr_img)
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        return {'loss': avg_loss, 'recon_loss': avg_loss}