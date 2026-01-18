import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
import kornia.augmentation as K
from .encoder import Encoder

class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_features = x.shape[1]

        repr_loss = F.mse_loss(x, y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow(2).sum().div(num_features) + \
                   self.off_diagonal(cov_y).pow(2).sum().div(num_features)

        loss = (self.sim_coeff * repr_loss + 
                self.std_coeff * std_loss + 
                self.cov_coeff * cov_loss)
        return loss

class ClusteringAutoencoder(nn.Module):
    def __init__(self, 
                 latent_channels=32, 
                 input_channels=4, 
                 learning_rate=3e-4,  
                 num_clusters=20,
                 image_size=64,
                 warmup_epochs=5,       
                 entropy_weight=2.0,
                 use_amp=True,
                 load_best=False    
                ):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_epochs = warmup_epochs
        self.entropy_weight = entropy_weight
        self.best_model_path = Path('checkpoints/best_vicreg_cluster.pt')
        self.model_loaded = False
        
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.encoder = Encoder(latent_channels=latent_channels, input_channels=input_channels)
        
        self.projector = nn.Sequential(
            nn.Linear(latent_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256), 
        )
        
        self.cluster_head = nn.Sequential(
            nn.Linear(latent_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_clusters) 
        )
        
        self.augment = nn.Sequential(
            K.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=15.0, p=0.5),
            K.RandomGaussianNoise(mean=0., std=0.05, p=0.2),
        )
        
        self.vicreg_loss = VICRegLoss()
        
        self.to(self.device)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        self.scaler = torch.amp.GradScaler(self.device.__str__(), enabled=self.use_amp)
        
        self.history = {'loss': [], 'vicreg': [], 'cluster': []}

        if load_best and self.best_model_path.exists():
            self.load_checkpoint()
            self.model_loaded = True

    def clustering_loss(self, logits_1, logits_2):
        p1 = F.softmax(logits_1, dim=1)
        p2 = F.softmax(logits_2, dim=1)
        
        log_p1 = F.log_softmax(logits_1, dim=1)
        log_p2 = F.log_softmax(logits_2, dim=1)
        
        loss_const_1 = -torch.mean(torch.sum(p2.detach() * log_p1, dim=1))
        loss_const_2 = -torch.mean(torch.sum(p1.detach() * log_p2, dim=1))
        consistency_loss = (loss_const_1 + loss_const_2) / 2.0

        p_avg = (p1 + p2) / 2.0
        avg_probs = torch.mean(p_avg, dim=0) 

        entropy_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
        
        return consistency_loss, entropy_loss

    def train_epoch(self, dataloader, epoch_idx):
        self.train()
        total_loss = 0
        loss_v_accum = 0
        loss_c_accum = 0
        
        use_clustering = epoch_idx >= self.warmup_epochs

        loop = tqdm(dataloader, desc=f"Epoch {epoch_idx} | Cluster: {use_clustering}")
        
        for batch in loop:
            img = batch[0] if isinstance(batch, (list, tuple)) else batch
            img = img.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                v1 = self.augment(img)
                v2 = self.augment(img)
                
                h1_spatial = self.encoder(v1)
                h2_spatial = self.encoder(v2)
                
                h1 = torch.mean(h1_spatial, dim=[2, 3])
                h2 = torch.mean(h2_spatial, dim=[2, 3])
                
                z1 = self.projector(h1)
                z2 = self.projector(h2)
                loss_vicreg = self.vicreg_loss(z1, z2)
                
                loss_cluster_total = torch.tensor(0.0, device=self.device)
                
                if use_clustering:
                    c1 = self.cluster_head(h1)
                    c2 = self.cluster_head(h2)
                    
                    const_loss, ent_loss = self.clustering_loss(c1, c2)
                    loss_cluster_total = const_loss + (self.entropy_weight * ent_loss)
                
                final_loss = loss_vicreg + loss_cluster_total

            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += final_loss.item()
            loss_v_accum += loss_vicreg.item()
            loss_c_accum += loss_cluster_total.item()
            
            loop.set_postfix(v_loss=loss_vicreg.item(), c_loss=loss_cluster_total.item())

        return {
            'loss': total_loss / len(dataloader),
            'vicreg': loss_v_accum / len(dataloader),
            'cluster': loss_c_accum / len(dataloader)
        }

    def fit(self, train_loader, epochs=50):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader),
            epochs=epochs, pct_start=0.2
        )
        
        min_loss = float('inf')
        
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, epoch)
            
            self.history['loss'].append(metrics['loss'])
            print(f"E{epoch} | Total: {metrics['loss']:.4f} | VICReg: {metrics['vicreg']:.4f} | Cluster: {metrics['cluster']:.4f}")
            
            if metrics['loss'] < min_loss and epoch > self.warmup_epochs:
                min_loss = metrics['loss']
                self.save_checkpoint(self.best_model_path, epoch, min_loss)
            
            if self.scheduler:
                self.scheduler.step()

    def extract_latent(self, dataloader):
        self.eval()
        latents = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device)
                
                with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                    h = self.encoder(img)
                    h = torch.mean(h, dim=[2, 3])
                    h = F.normalize(h, dim=1) 
                
                latents.append(h.float().cpu().numpy()) 
        return np.concatenate(latents, axis=0)

    def save_checkpoint(self, path, epoch, loss):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(), 
            'loss': loss, 
            'epoch': epoch,
            'scaler': self.scaler.state_dict() 
        }, path)
        print(f"Saved to {path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
            
        print(f"Loaded model from {self.best_model_path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")