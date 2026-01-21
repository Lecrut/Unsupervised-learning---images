import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

class SuperResolutionModel(nn.Module):
    def __init__(
        self, 
        input_channels=3, 
        n_feats=64,       
        n_layers=4,      
        scale=2,
        learning_rate=1e-4, 
        max_lr=1e-3,
        use_amp=True, 
        load_best=False,
    ):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_path = Path('checkpoints/best_super_resolution.pt')
        self.scale = scale
        self.history = {'train_loss': [], 'val_loss': []}
        self.model_loaded = False
        
        layers = [nn.Conv2d(input_channels, n_feats, kernel_size=3, padding=1), nn.PReLU()]
        for _ in range(n_layers):
            layers.extend([nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1), nn.PReLU()])
        
        self.body = nn.Sequential(*layers)
        
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, input_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
        self._init_weights()
        
        self.l1_loss = nn.L1Loss()
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler(self.device.__str__(), enabled=self.use_amp)
        self.max_lr = max_lr

        self.to(self.device)

        if load_best and self.save_path.exists():
            self.load_checkpoint()
            self.model_loaded = True

    def _init_weights(self):
        for m in self.tail:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        feat = self.body(x)
        res = self.tail(feat)
        
        return base + res

    def fit(self, train_loader, val_loader=None, epochs=50, patience=10):
        print(f"Start treningu SR (x{self.scale}) | AMP: {self.use_amp}")
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1
        )
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            train_loss_accum = 0.0
            
            for batch in loop:
                if isinstance(batch, (list, tuple)):
                    lr_img, hr_target = batch[0], batch[1]
                else:
                    lr_img, hr_target = batch['lr'], batch['hr']

                lr_img = lr_img.to(self.device, non_blocking=True)
                hr_target = hr_target.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                    sr_img = self(lr_img)
                    loss = self.l1_loss(sr_img, hr_target)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                scheduler.step()
                
                train_loss_accum += loss.item()
                loop.set_postfix(loss=loss.item(), lr=f"{scheduler.get_last_lr()[0]:.2e}")
            
            avg_train_loss = train_loss_accum / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)

            avg_val_loss = 0.0
            if val_loader:
                self.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        if isinstance(batch, (list, tuple)):
                            lr_img, hr_target = batch[0], batch[1]
                        else:
                            lr_img, hr_target = batch['lr'], batch['hr']

                        lr_img = lr_img.to(self.device, non_blocking=True)
                        hr_target = hr_target.to(self.device, non_blocking=True)
                        
                        sr_img = self(lr_img)
                        loss = self.l1_loss(sr_img, hr_target)
                        val_loss_accum += loss.item()
                
                avg_val_loss = val_loss_accum / len(val_loader)
                self.history['val_loss'].append(avg_val_loss)
                
                print(f" -> Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch=epoch, loss=best_val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                self.save_checkpoint(epoch=epoch, loss=avg_train_loss)

        return self.history

    def save_checkpoint(self, path=None, epoch=0, loss=0.0):
        if path is None: path = self.save_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': loss,
            'scale': self.scale,
            'history': self.history
        }, path)
        print(f"-> Zapisano model SR: {path} (Loss: {loss:.5f})")

    def load_checkpoint(self, path=None):
        if path is None: path = self.save_path
        path = Path(path)
        
        if not path.exists():
            print(f"-> Brak checkpointu SR w {path}. Start od zera.")
            return

        print(f"-> Wczytywanie SR z {path}...")
        ckpt = torch.load(path, map_location=self.device)
        
        self.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except:
                pass
                
        if 'history' in ckpt:
            self.history = ckpt['history']
            
        print(f"-> Model SR załadowany (Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('val_loss', '?')})")