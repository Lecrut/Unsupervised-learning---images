import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- BLOKI POMOCNICZE ---
class ResBlockConv(nn.Module):
    """Blok zachowujący ostrość i relacje przestrzenne"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels), # GroupNorm jest lepszy dla małych batchy niż BatchNorm
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return x + self.conv(x)

# --- GŁÓWNA KLASA INPAINTERA ---
class ClusterInpainter(nn.Module):
    def __init__(self, 
                 latent_dim=1024,     # Twój wektor wejściowy (z maina)
                 spatial_dim=8,       # Rozmiar przestrzenny w encoderze (8x8)
                 latent_channels=512, # Ilość kanałów w encoderze (512)
                 num_clusters=10,     # Ilość klas
                 load_best=False,
                 **kwargs             # Ignorujemy inne stare parametry
                 ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = Path('checkpoints/inpainter_best.pt')
        self.best_loaded = False
        
        # Zapamiętujemy wymiary do "rozpakowania" wektora
        self.spatial_dim = spatial_dim
        self.latent_channels = latent_channels
        self.flat_features_size = latent_channels * spatial_dim * spatial_dim # 512*8*8 = 32768
        
        # 1. ROZPAKOWYWANIE: Wektor(1024) -> Mapa Cech(32768)
        self.fc_in = nn.Linear(latent_dim, self.flat_features_size)
        
        # 2. EMBEDDING STYLU
        self.style_embedding = nn.Embedding(num_clusters, 64) # Styl to wektor 64
        
        # 3. SIECI KONWOLUCYJNE (U-Net Like)
        # Wejście: 512 (Latent) + 64 (Styl) = 576 kanałów
        in_ch = latent_channels + 64
        
        self.conv_in = nn.Conv2d(in_ch, 512, 3, 1, 1)
        
        # Głębokie przetwarzanie (zamiast płaskich warstw)
        self.body = nn.Sequential(
            ResBlockConv(512),
            ResBlockConv(512),
            ResBlockConv(512),
            ResBlockConv(512)
        )
        
        # Wyjście konwolucyjne (poprawka)
        self.conv_out = nn.Conv2d(512, latent_channels, 3, 1, 1)
        
        # 4. SPAKOWANIE: Mapa Cech -> Wektor(1024)
        self.fc_out = nn.Linear(self.flat_features_size, latent_dim)

        # Ładowanie wag
        if load_best and self.path.exists():
            try:
                self.load_state_dict(torch.load(self.path, map_location=self.device))
                self.best_loaded = True
                print(f"   Wczytano inpaintera: {self.path}")
            except:
                print("   Nie udało się wczytać starego modelu (może inna architektura?). Trenuję od nowa.")

    def forward(self, z_damaged, cluster_id):
        batch_size = z_damaged.shape[0]
        
        # KROK A: Projekcja w górę (Unflatten)
        # Zamieniamy wektor 1D na obrazek 3D, żeby CNN mogło działać
        x = self.fc_in(z_damaged) 
        x = x.view(batch_size, self.latent_channels, self.spatial_dim, self.spatial_dim) # [B, 512, 8, 8]
        
        # KROK B: Dodanie Stylu
        style = self.style_embedding(cluster_id) # [B, 64]
        # Rozciągamy wektor stylu na całą mapę 8x8
        style_map = style.view(batch_size, 64, 1, 1).expand(-1, -1, self.spatial_dim, self.spatial_dim)
        
        # KROK C: Złączenie (Latent + Styl)
        # Maska nie jest potrzebna, sieć sama nauczy się wykrywać anomalie
        inp = torch.cat([x, style_map], dim=1) # [B, 576, 8, 8]
        
        # KROK D: Naprawa (Konwolucje)
        h = self.conv_in(inp)
        h = self.body(h)
        correction = self.conv_out(h)
        
        # Residual connection (x + poprawka)
        x_fixed = x + correction
        
        # KROK E: Projekcja w dół (Flatten)
        # Wracamy do wektora 1024, żeby pasowało do Twojego Decodera
        z_out = self.fc_out(x_fixed.view(batch_size, -1))
        
        return z_out, torch.tensor(0.0).to(self.device)

    def fit(self, damaged_data, clean_data, cluster_ids, epochs=50, lr=1e-4, batch_size=64):
        # Konwersja na tensory (jeśli są numpy)
        if isinstance(damaged_data, np.ndarray): damaged_data = torch.from_numpy(damaged_data).float()
        if isinstance(clean_data, np.ndarray): clean_data = torch.from_numpy(clean_data).float()
        if isinstance(cluster_ids, (np.ndarray, list)): cluster_ids = torch.tensor(cluster_ids).long()

        dataset = TensorDataset(damaged_data, clean_data, cluster_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.to(self.device)
        self.train()
        
        # Weight decay pomaga uniknąć overfittingu w latencie
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=epochs)
        
        best_loss = float('inf')
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Start treningu Spatial Inpaintera (Bez Maski). Batche: {len(dataloader)}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoka {epoch+1}/{epochs}", leave=False)
            
            for z_dam, z_clean, c_ids in pbar:
                z_dam, z_clean, c_ids = z_dam.to(self.device), z_clean.to(self.device), c_ids.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                z_pred, _ = self.forward(z_dam, c_ids)
                
                # Loss MSE (wymuszamy podobieństwo latentów)
                loss = F.mse_loss(z_pred, z_clean)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) 
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.5f}"})
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoka {epoch+1}: Loss = {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.state_dict(), self.path)
                print("   Zapisano model.")
        
        # Wczytujemy najlepszy model na koniec
        self.load_state_dict(torch.load(self.path))
        return self