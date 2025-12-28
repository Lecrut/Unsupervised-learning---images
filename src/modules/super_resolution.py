#%% Imports
from pathlib import Path
from src.modules.autoencoder import Autoencoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Dict
from tqdm import tqdm

#%% EDSR Block
class EDSRBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + (res * self.res_scale)


#%% Upsampler x2
class Upsampler2x(nn.Sequential):
    def __init__(self, n_feats):
        m = [
            nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*m)

#%% EDSR Model - Super Resolution x2
class SuperResolutionModel(Autoencoder):
    def __init__(self, 
                 input_channels=3, 
                 n_res_blocks=16,
                 n_feats=64,      
                 learning_rate=1e-4, 
                 use_amp=True, 
                 load_best=False,
                 image_size=None):
        
        super().__init__(latent_dim=0, input_channels=input_channels, learning_rate=learning_rate, use_amp=use_amp, load_best=False)
        
        self.best_model_path = Path('checkpoints/super_resolution/best_edsr_x2.pt')
        
        self.head = nn.Conv2d(input_channels, n_feats, kernel_size=3, padding=1)

        self.body = nn.Sequential(*[
            EDSRBlock(n_feats, res_scale=0.1) for _ in range(n_res_blocks)
        ])
        
        self.body_tail = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        self.tail = nn.Sequential(
            Upsampler2x(n_feats),
            nn.Conv2d(n_feats, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_l1 = nn.L1Loss() 

        if load_best:
            self.load_checkpoint()

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            x = self.head(x)
            
            res = self.body(x)
            res = self.body_tail(res)
            
            res += x 
            
            x = self.tail(res)
            
            return x, None 

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.train()
        epoch_loss = 0.0
        
        for lr_img, hr_target in tqdm(dataloader, desc="Training EDSR x2"):
            lr_img = lr_img.to(self.device, non_blocking=True)
            hr_target = hr_target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                sr_img, _ = self.forward(lr_img)
                loss = self.loss_l1(sr_img, hr_target)

            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(dataloader), 'recon_loss': epoch_loss / len(dataloader)}

    @torch.no_grad()
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        self.eval()
        epoch_loss = 0.0
        
        for lr_img, hr_target in tqdm(dataloader, desc="Validating EDSR x2"):
            lr_img = lr_img.to(self.device, non_blocking=True)
            hr_target = hr_target.to(self.device, non_blocking=True)
            
            sr_img, _ = self.forward(lr_img)
            loss = self.criterion(sr_img, hr_target)
            
            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(dataloader), 'recon_loss': epoch_loss / len(dataloader)}

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
         path.parent.mkdir(parents=True, exist_ok=True)
         torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, path)

    def load_checkpoint(self, path: Path = None):
        if path is None: path = self.best_model_path
        if not path.exists(): return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint: self.history = checkpoint['history']
        
        print(f"EDSR (x2) wczytany. Epoka: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")