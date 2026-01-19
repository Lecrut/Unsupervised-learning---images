import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia as K
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .encoder import Encoder
from .decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(
        self,
        latent_channels=32,
        input_channels=4,
        num_prototypes=20,
        proto_dim=128,
        learning_rate=1e-4,          
        max_lr=1e-3,      
        use_amp=True,
        load_best=False,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_path = Path('checkpoints/best_clustering_model.pt')
        self.num_prototypes = num_prototypes
        self.lr = learning_rate
        self.max_lr = max_lr
        self.model_loaded = False
        
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        
        self.encoder = Encoder(latent_channels, input_channels)
        self.decoder = Decoder(latent_channels, input_channels)

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels * 2, proto_dim),
            nn.BatchNorm1d(proto_dim),
            nn.GELU(),
            nn.Linear(proto_dim, proto_dim) 
        )

        self.prototypes = nn.Linear(proto_dim, num_prototypes, bias=False)
        self.temperature = 0.1 

        self.l1 = nn.L1Loss()
        self.lap = K.filters.Laplacian(kernel_size=5, normalized=True)
        
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler(self.device.__str__(), enabled=self.use_amp)
        
        self.to(self.device)

        if load_best and self.save_path.exists():
            self.load_checkpoint()
            print('Best model loaded')
            self.model_loaded = True

    def forward(self, x):
        z_spatial = self.encoder(x)

        rec = self.decoder(z_spatial)

        z_mean = torch.mean(z_spatial, dim=[2, 3])
        z_std  = torch.std(z_spatial, dim=[2, 3])
        style_raw = torch.cat([z_mean, z_std], dim=1)

        style_detached = style_raw.detach() 

        style_emb = self.projector(style_detached)
        style_emb = F.normalize(style_emb, dim=1, p=2)

        W = F.normalize(self.prototypes.weight, dim=1, p=2)
        logits = F.linear(style_emb, W) / self.temperature
        
        return rec, logits, style_emb, style_raw

    def compute_cluster_loss(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy_samples = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        avg_probs = torch.mean(probs, dim=0)
        entropy_batch = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        return entropy_samples - entropy_batch

    def compute_rec_loss(self, img, rec):
        loss_l1 = self.l1(rec, img)
        with torch.no_grad():
            lap_img = self.lap(img)
        lap_rec = self.lap(rec)
        return 1.0 * loss_l1 + 2.0 * F.l1_loss(lap_rec, lap_img)

    def save_checkpoint(self, path=None, epoch=0, loss=0.0):
        if path is None: path = self.save_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'num_prototypes': self.num_prototypes,
            'history': self.history
        }, path)
        print(f"-> Zapisano checkpoint: {path} (Loss: {loss:.4f})")

    def load_checkpoint(self, path=None):
        if path is None: path = self.save_path
        path = Path(path)
        
        if not path.exists():
            print(f"-> Brak checkpointu w {path}. Start od zera.")
            return

        print(f"-> Wczytywanie wag z {path}...")
        ckpt = torch.load(path, map_location=self.device)
        
        missing, unexpected = self.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except:
                print("-> Nie udało się wczytać stanu optimizera (zignorowano).")

        if 'history' in ckpt:
            self.history = ckpt['history']

        epoch = ckpt.get('epoch', 0)
        loss = ckpt.get('loss', 'N/A')
        
        print(f"-> Model załadowany! Epoka: {epoch} | Loss: {loss}")
        if missing:
            print(f"-> Zainicjowano losowo nowe warstwy: {missing}")

    def fit(self, train_loader, val_loader=None, epochs=50, early_stopping_patience=10, warmup_epochs=5):
        print(f"Start treningu: {epochs} epok (Warmup: {warmup_epochs}) | AMP: {self.use_amp}")
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1,
            div_factor=10.0,        
            final_div_factor=100.0
        )
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            is_warmup = False if self.model_loaded else epoch < warmup_epochs 
            self.projector.requires_grad_(not is_warmup)
            self.prototypes.requires_grad_(not is_warmup)
            
            train_loss_accum = 0.0
            lrs = []
            
            for batch in loop:
                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                    rec, logits, _, _ = self(img)
                    
                    loss_rec = self.compute_rec_loss(img, rec)
                    
                    if not is_warmup:
                        loss_clust = self.compute_cluster_loss(logits)
                        loss = loss_rec + 0.1 * loss_clust
                    else:
                        loss_clust = torch.tensor(0.0)
                        loss = loss_rec

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                current_lr = scheduler.get_last_lr()[0]
                lrs.append(current_lr)
                scheduler.step()
                
                train_loss_accum += loss.item()
                
                loop.set_postfix(
                    loss=loss.item(),
                    cl=f"{loss_clust.item():.2f}" if not is_warmup else "warm",
                    lr=f"{current_lr:.2e}"
                )
            
            avg_train_loss = train_loss_accum / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            self.history['learning_rates'].extend(lrs) # Dodajemy listę LR z całej epoki

            avg_val_loss = 0.0
            
            if val_loader:
                self.eval()
                val_loss_accum = 0.0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        img = batch[0] if isinstance(batch, (list, tuple)) else batch
                        img = img.to(self.device, non_blocking=True)
                        
                        rec, logits, _, _ = self(img)
                        loss_rec = self.compute_rec_loss(img, rec)
                        
                        if not is_warmup:
                            loss_clust = self.compute_cluster_loss(logits)
                            val_batch_loss = loss_rec + 0.1 * loss_clust
                        else:
                            val_batch_loss = loss_rec
                            
                        val_loss_accum += val_batch_loss.item()
                
                avg_val_loss = val_loss_accum / len(val_loader)
                self.history['val_loss'].append(avg_val_loss)
                
                print(f" -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch=epoch, loss=best_val_loss) 
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                self.save_checkpoint(epoch=epoch, loss=avg_train_loss)
                
        return self.history
    
    def extract_latent(self, dataloader, use_projector=False): 
        self.eval()
        latents = []
        print("Extracting latents...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device)
                
                z_spatial = self.encoder(img)
                
                if use_projector:
                    z_mean = torch.mean(z_spatial, dim=[2, 3])
                    z_std  = torch.std(z_spatial, dim=[2, 3])
                    style_raw = torch.cat([z_mean, z_std], dim=1)

                    style_emb = self.projector(style_raw)
                    style_emb = F.normalize(style_emb, dim=1, p=2)
                    latents.append(style_emb.cpu().numpy())
                else:
                    latents.append(z_spatial.cpu().numpy())
                    
        return np.concatenate(latents, axis=0)