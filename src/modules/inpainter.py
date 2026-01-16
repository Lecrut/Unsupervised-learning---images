import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):        
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        quantized = self.embedding(encoding_indices).view_as(inputs)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.squeeze()

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)

class ClusterInpainter(nn.Module):
    def __init__(self, 
                 latent_dim=1024,     
                 hidden_dim=1024,      
                 num_clusters=10,      
                 num_layers=6,         
                 dropout=0.1,
                 load_best=False
                 ):
        super().__init__()
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = Path('checkpoints/inpainter_best.pt')
        self.latent_dim = latent_dim
        self.load_best = load_best
        self.best_loaded = False
        
        self.cluster_embedding = nn.Embedding(num_clusters, hidden_dim)
        
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.body = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(num_layers)],
            nn.LayerNorm(hidden_dim)
        )
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.quantizer = VectorQuantizer(num_clusters, latent_dim, commitment_cost=0.25)

        if self.load_best and self.path.exists():
            self.load_state_dict(torch.load(self.path, map_location=self.device))
            self.best_loaded = True
            print(f"   Wczytano najlepszy model Inpaintera z {self.path}")


    def forward(self, z_damaged, cluster_id):
        c_emb = self.cluster_embedding(cluster_id)
        
        h = self.input_proj(z_damaged) + c_emb
        
        h = self.body(h)
        
        z_predicted = self.output_proj(h) + z_damaged
        
        z_quantized, vq_loss, _ = self.quantizer(z_predicted)
        
        return z_quantized, vq_loss

    def fit(self, 
            damaged_data, 
            clean_data, 
            cluster_ids, 
            epochs=50, 
            lr=1e-4, 
            batch_size=128
            ):
        
        if isinstance(damaged_data, np.ndarray):
            damaged_data = torch.from_numpy(damaged_data).float()
        if isinstance(clean_data, np.ndarray):
            clean_data = torch.from_numpy(clean_data).float()
        if isinstance(cluster_ids, (np.ndarray, list)):
            cluster_ids = torch.tensor(cluster_ids).long()

        dataset = TensorDataset(damaged_data, clean_data, cluster_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.to(self.device)
        self.train()
        
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=epochs
        )
        
        best_loss = float('inf')
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"    Start treningu Inpaintera (Device: {self.device})")
        print(f"   Dane: {len(dataset)} próbek, Klastry: {self.cluster_embedding.num_embeddings}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_vq = 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch in pbar:
                z_dam, z_clean, c_ids = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                
                z_repaired, vq_loss = self.forward(z_dam, c_ids)
                
                recon_loss = F.mse_loss(z_repaired, z_clean)
                
                loss = recon_loss + vq_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                epoch_vq += vq_loss.item()
                
                pbar.set_postfix({'MSE': f"{recon_loss.item():.5f}", 'VQ': f"{vq_loss.item():.5f}"})
            
            avg_loss = epoch_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f} (VQ: {epoch_vq/len(dataloader):.6f})")

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"   Nowy najlepszy model zapisany z loss = {best_loss:.6f}")
                torch.save(self.state_dict(), self.path)
                
        self.load_state_dict(torch.load(self.path))
        return self