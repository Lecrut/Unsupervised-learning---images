import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm

class LatentInpainter(nn.Module):
    def __init__(self, 
                 latent_channels=64,
                 num_clusters=10,
                 hidden_channels=128,
                 learning_rate=0.001,
                 use_amp=True,
                 load_best=False,
                 class_loss_weight=0.1
                ):
        super().__init__()

        self.latent_channels = latent_channels
        self.num_clusters = num_clusters
        self.learning_rate = learning_rate
        self.class_loss_weight = class_loss_weight
        self.current_epoch = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.best_model_path = Path('checkpoints/best_inpainter.pt')
        
        self.encoder_block = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        
        self.middle_block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels*2),
            nn.GELU(),
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        
        self.decoder_block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_channels, 128),
            nn.GELU(),
            nn.Linear(128, num_clusters)
        )

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(self.device.__str__(), enabled=self.use_amp)

        self.history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'class_loss': []}

        if load_best and self.best_model_path.exists():
            self.load_checkpoint()

    def forward(self, x):
        enc = self.encoder_block(x)
        mid = self.middle_block(enc)
        correction = self.decoder_block(mid)
        return x + correction

    def forward_train(self, x):
        repaired = self.forward(x)
        class_logits = self.classifier(repaired)
        return repaired, class_logits

    def compute_loss(self, repaired_latent, target_latent, pred_logits, target_labels):
        loss_recon = self.mse_loss(repaired_latent, target_latent)
        loss_class = self.ce_loss(pred_logits, target_labels)
        total_loss = loss_recon + (self.class_loss_weight * loss_class)
        return total_loss, loss_recon, loss_class

    def _process_batch_data(self, batch_idx, clean_batch, corr_batch, all_labels, encoder_model):
        clean_img = clean_batch[0].to(self.device, non_blocking=True) if isinstance(clean_batch, (list, tuple)) else clean_batch.to(self.device)
        corr_img = corr_batch[0].to(self.device, non_blocking=True) if isinstance(corr_batch, (list, tuple)) else corr_batch.to(self.device)
        
        with torch.no_grad():
            clean_latent = encoder_model(clean_img)
            corr_latent = encoder_model(corr_img)

        batch_size = clean_latent.size(0)
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        current_labels = all_labels[start_idx:end_idx]
        
        if not isinstance(current_labels, torch.Tensor):
            current_labels = torch.tensor(current_labels)
            
        current_labels = current_labels.to(self.device, dtype=torch.long)
        
        return clean_latent, corr_latent, current_labels

    def train_epoch(self, clean_loader, corr_loader, train_labels, encoder_model):
        self.train()
        encoder_model.eval() 
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_class = 0.0
        
        loop = tqdm(zip(clean_loader, corr_loader), total=len(clean_loader), desc="Training")
        
        for batch_idx, (clean_batch, corr_batch) in enumerate(loop):
            
            clean_z, corr_z, labels = self._process_batch_data(
                batch_idx, clean_batch, corr_batch, train_labels, encoder_model
            )

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                repaired_z, logits = self.forward_train(corr_z)
                loss, l_recon, l_class = self.compute_loss(repaired_z, clean_z, logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss += loss.item()
            epoch_recon += l_recon.item()
            epoch_class += l_class.item()
            
            loop.set_postfix(loss=loss.item())

        N = len(clean_loader)
        return {
            'loss': epoch_loss / N, 
            'recon_loss': epoch_recon / N, 
            'class_loss': epoch_class / N
        }

    @torch.no_grad()
    def validate_epoch(self, clean_loader, corr_loader, val_labels, encoder_model):
        self.eval()
        encoder_model.eval()
        
        epoch_loss = 0.0
        
        loop = tqdm(zip(clean_loader, corr_loader), total=len(clean_loader), desc="Validation")
        
        for batch_idx, (clean_batch, corr_batch) in enumerate(loop):
            clean_z, corr_z, labels = self._process_batch_data(
                batch_idx, clean_batch, corr_batch, val_labels, encoder_model
            )

            with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                repaired_z, logits = self.forward_train(corr_z)
                loss, _, _ = self.compute_loss(repaired_z, clean_z, logits, labels)

            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(clean_loader)}

    def fit(self, 
            clean_train_loader, corr_train_loader, train_labels, 
            encoder_model, 
            clean_val_loader=None, corr_val_loader=None, val_labels=None,
            epochs=30, early_stopping_patience=10):
        
        best_val_loss = float('inf')
        patience_counter = 0

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=len(clean_train_loader),
            epochs=epochs,
            pct_start=0.1, 
            div_factor=25.0,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )

        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch(
                clean_train_loader, corr_train_loader, train_labels, encoder_model
            )
            
            val_metrics = {'loss': 0.0}
            if clean_val_loader and corr_val_loader and val_labels is not None:
                val_metrics = self.validate_epoch(
                    clean_val_loader, corr_val_loader, val_labels, encoder_model
                )

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_metrics['loss']:.4f} (Rec:{train_metrics['recon_loss']:.3f}, Cls:{train_metrics['class_loss']:.3f}) | "
                  f"Val: {val_metrics['loss']:.4f}")

            if val_metrics['loss'] < best_val_loss and val_metrics['loss'] != 0.0:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_checkpoint(self.best_model_path, epoch, best_val_loss)
            elif val_metrics['loss'] != 0.0:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
                    
        return self.history

    def save_checkpoint(self, path, epoch, val_loss):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.best_model_path
        if not path.exists():
            print(f"Checkpoint {path} not found.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded inpainter from {path}")