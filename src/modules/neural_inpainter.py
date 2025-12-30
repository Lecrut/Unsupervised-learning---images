import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        return self.activation(x + self.block(x))

class NeuralInpainter(nn.Module):
    def __init__(self, latent_dim, n_clusters, hidden_dim=1024, n_blocks=4, dropout=0.1):
        super().__init__()
        
        self.cluster_emb = nn.Embedding(n_clusters, 128)
        
        input_dim = latent_dim + 128
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, latent_damaged, cluster_ids):
        style_vector = self.cluster_emb(cluster_ids)
        
        combined = torch.cat([latent_damaged, style_vector], dim=1)
        
        x = self.input_proj(combined)
        
        for block in self.res_blocks:
            x = block(x)
        
        correction = self.output_proj(x)
        
        repaired_latent = latent_damaged + correction
        
        return repaired_latent

# Funkcja pomocnicza do treningu (pseudo-kod)
def train_step(model, optimizer, latent_damaged, latent_original, cluster_ids):
    model.train()
    optimizer.zero_grad()
    
    repaired = model(latent_damaged, cluster_ids)
    
    # Loss: Chcemy, żeby naprawiony latent był jak najbliżej oryginału
    loss = torch.nn.functional.mse_loss(repaired, latent_original)
    
    loss.backward()
    optimizer.step()
    return loss.item()