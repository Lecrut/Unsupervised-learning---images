#%% Imports
from pathlib import Path
from src.modules.autoencoder import Autoencoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm

#%% Super Resolution Model Definition - Fast Anchor SR
class SuperResolutionModel(Autoencoder):
    def __init__(self, 
                 input_channels=3, 
                 n_feats=24,      
                 n_layers=4,      
                 scale=2,
                 learning_rate=1e-3, 
                 use_amp=True, 
                 load_best=False,
                 image_size=None):
        
        super().__init__(input_channels=input_channels, learning_rate=learning_rate, use_amp=use_amp, load_best=False)
        
        self.scale = scale
        self.best_model_path = Path('checkpoints/super_resolution/best_fast_anchor_x2.pt')
        
        layers = [nn.Conv2d(input_channels, n_feats, kernel_size=3, padding=1), nn.PReLU()]
        for _ in range(n_layers):
            layers.extend([nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1), nn.PReLU()])
        self.body = nn.Sequential(*layers)
        
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, input_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
        self._init_weights()
        self.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_l1 = nn.L1Loss()

        if load_best:
            if self.best_model_path.exists():
                try:
                    self.load_checkpoint()
                    self.model_loaded = True
                except Exception as e:
                    print(f"Błąd wczytywania modelu SR: {e}")
                    self.model_loaded = False
            else:
                print(f"Brak zapisanego modelu SR w {self.best_model_path}")
                self.model_loaded = False

    def _init_weights(self):
        for m in self.tail:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            base = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
            res = self.tail(self.body(x))
            return base + res, None

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.train()
        epoch_loss = 0.0
        
        for lr_img, hr_target in tqdm(dataloader, desc="Training FastSR"):
            lr_img = lr_img.to(self.device, non_blocking=True)
            hr_target = hr_target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                sr_img, _ = self.forward(lr_img)
                loss = self.loss_l1(sr_img, hr_target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(dataloader), 'recon_loss': epoch_loss / len(dataloader)}

    @torch.no_grad()
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        self.eval()
        epoch_loss = 0.0
        
        for lr_img, hr_target in tqdm(dataloader, desc="Validating FastSR"):
            lr_img = lr_img.to(self.device, non_blocking=True)
            hr_target = hr_target.to(self.device, non_blocking=True)
            
            sr_img, _ = self.forward(lr_img)
            loss = self.loss_l1(sr_img, hr_target)
            
            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(dataloader), 'recon_loss': epoch_loss / len(dataloader)}

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
         path.parent.mkdir(parents=True, exist_ok=True)
         torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        
        print(f"FastSR (x{self.scale}) wczytany. Epoka: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 0):.6f}")