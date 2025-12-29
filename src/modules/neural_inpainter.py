import torch
import torch.nn as nn

class NeuralInpainter(nn.Module):
    def __init__(self, latent_dim, n_clusters, hidden_dim=1024):
        super().__init__()
        
        self.cluster_emb = nn.Embedding(n_clusters, 64) 
        
        input_dim = latent_dim + 64
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, latent_damaged, cluster_ids):
        style_vector = self.cluster_emb(cluster_ids) # shape: (N, 64)
        
        combined = torch.cat([latent_damaged, style_vector], dim=1)
        
        # Oblicz "poprawkę" (rezyduum)
        correction = self.net(combined)
        
        # Wynik = Uszkodzony + Poprawka
        # (Dzięki temu sieć musi nauczyć się tylko tego, czego brakuje,
        # zamiast generować cały obraz od zera)
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