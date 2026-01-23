import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.GELU()
        self.attn = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attn(out)
        return self.relu(out + residual)


class LatentInpainter(nn.Module):
    def __init__(self, 
                 latent_channels=32,
                 num_clusters=12,
                 embedding_dim=32,
                 hidden_channels=128,
                 learning_rate=0.001,
                 use_amp=True,
                 load_best=False,
                 class_loss_weight=0.1
                ):
        super().__init__()

        self.latent_channels = latent_channels
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.class_loss_weight = class_loss_weight
        self.current_epoch = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.best_model_path = Path('checkpoints/best_inpainter.pt')
        
        self.label_embedding = nn.Embedding(num_clusters, embedding_dim)
        
        input_dim = latent_channels + embedding_dim
        
        self.encoder_block = nn.Sequential(
            nn.Conv2d(input_dim, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        
        self.middle_block = nn.Sequential(
            DilatedResidualBlock(hidden_channels, dilation=1),
            DilatedResidualBlock(hidden_channels, dilation=2),
            DilatedResidualBlock(hidden_channels, dilation=4),
            DilatedResidualBlock(hidden_channels, dilation=2),
            DilatedResidualBlock(hidden_channels, dilation=1),
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
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(self.device.__str__(), enabled=self.use_amp)

        self.history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'class_loss': []}

        if load_best and self.best_model_path.exists():
            self.load_checkpoint()

    def forward(self, x, labels):
        # x: [Batch, 32, 8, 8]
        # labels: [Batch] (int)
        
        # Zamiana labela na mapę cech [Batch, 32, 1, 1]
        emb = self.label_embedding(labels)
        
        # To size [Batch, 32, 8, 8]
        emb_map = emb.view(emb.size(0), emb.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        
        # Concat
        x_input = torch.cat([x, emb_map], dim=1)
        
        enc = self.encoder_block(x_input)
        mid = self.middle_block(enc)
        correction = self.decoder_block(mid)
        
        return x + correction

    def forward_train(self, x, labels):
        repaired = self.forward(x, labels)
        class_logits = self.classifier(repaired)
        return repaired, class_logits

    def compute_loss(self, repaired_latent, target_latent, pred_logits, target_labels):
        l_mse = self.mse_loss(repaired_latent, target_latent)
        l_l1 = self.l1_loss(repaired_latent, target_latent)
        loss_recon = 0.5 * l_mse + 0.5 * l_l1
        
        if self.class_loss_weight > 0:
            loss_class = self.ce_loss(pred_logits, target_labels)
        else:
            loss_class = torch.tensor(0.0, device=self.device)
            
        total_loss = loss_recon + (self.class_loss_weight * loss_class)
        return total_loss, loss_recon, loss_class

    def train_epoch(self, train_loader, encoder_model=None):
        self.train()
        if encoder_model: encoder_model.eval() 
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_class = 0.0
        
        loop = tqdm(train_loader, desc="Training")
        
        for batch_idx, (clean_batch, corr_batch, labels) in enumerate(loop):
            clean_data = clean_batch.to(self.device, non_blocking=True)
            corr_data = corr_batch.to(self.device, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long) 

            if encoder_model is not None:
                with torch.no_grad():
                    clean_z = encoder_model(clean_data)
                    corr_z = encoder_model(corr_data)
            else:
                clean_z = clean_data
                corr_z = corr_data

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                repaired_z, logits = self.forward_train(corr_z, labels)
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

        N = len(train_loader)
        return {
            'loss': epoch_loss / N, 
            'recon_loss': epoch_recon / N, 
            'class_loss': epoch_class / N
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader, encoder_model=None):
        self.eval()
        if encoder_model: encoder_model.eval()
        
        epoch_loss = 0.0
        
        for batch_idx, (clean_batch, corr_batch, labels) in enumerate(val_loader):
            clean_data = clean_batch.to(self.device, non_blocking=True)
            corr_data = corr_batch.to(self.device, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long)

            if encoder_model is not None:
                clean_z = encoder_model(clean_data)
                corr_z = encoder_model(corr_data)
            else:
                clean_z = clean_data
                corr_z = corr_data

            with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                repaired_z, logits = self.forward_train(corr_z, labels)
                loss, _, _ = self.compute_loss(repaired_z, clean_z, logits, labels)

            epoch_loss += loss.item()

        return {'loss': epoch_loss / len(val_loader)}

    def fit(self, 
            train_loader, 
            encoder_model, 
            val_loader=None, 
            epochs=30, 
            early_stopping_patience=10):
        
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
        
        print(f"Start treningu (Conditional) na urządzeniu: {self.device}")

        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch(train_loader, encoder_model)
            
            val_metrics = {'loss': 0.0}
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader, encoder_model)

            self.history['train_loss'].append(train_metrics['loss'])
            if val_loader: self.history['val_loss'].append(val_metrics['loss'])

            val_str = f"| Val: {val_metrics['loss']:.4f}" if val_loader else ""
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_metrics['loss']:.4f} (Rec:{train_metrics['recon_loss']:.3f}, Cls:{train_metrics['class_loss']:.3f}) "
                  f"{val_str}")

            current_val_loss = val_metrics['loss'] if val_loader else train_metrics['loss']
            
            if current_val_loss < best_val_loss and current_val_loss != 0.0:
                best_val_loss = current_val_loss
                patience_counter = 0
                self.save_checkpoint(self.best_model_path, epoch, best_val_loss)
            elif current_val_loss != 0.0:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
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