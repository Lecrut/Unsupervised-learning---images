import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from .encoder import Encoder

class ClusteringAutoencoder(nn.Module):
    def __init__(self, 
                 latent_channels=32,   
                 input_channels=4, 
                 num_clusters=20,      
                 lr=3e-4,
                 load_best=False
                ):
        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_clusters = num_clusters
        self.save_path = Path('checkpoints/best_autoencoder_clustering.pth')
        self.model_loaded = False
        
        self.encoder = Encoder(latent_channels=latent_channels, input_channels=input_channels)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_clusters) 
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=1024, n_init=3)
        
        self.to(self.device)

        if load_best and Path(self.save_path).exists():
            self.load_model()
            self.model_loaded = True

    def get_features_for_clustering(self, dataloader):
        self.eval()
        features = []
        with torch.no_grad():
            for batch in dataloader:
                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device)
                
                z = self.encoder(img)
                z = torch.mean(z, dim=[2, 3]) 
                z = F.normalize(z, dim=1)    
                features.append(z.cpu().numpy())
        return np.concatenate(features, axis=0)

    def fit(self, dataloader, epochs=15):
        print(f"    Start formowania przestrzeni ({epochs} epok). Cel: {self.num_clusters} odseparowanych wysp.")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        min_loss = float('inf')

        for epoch in range(epochs):
            features = self.get_features_for_clustering(dataloader)
            self.kmeans.fit(features)
            
            pseudo_labels = self.kmeans.labels_
            pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
            
            self.train()
            total_loss = 0
            correct = 0
            total_samples = 0
            
            batch_size = dataloader.batch_size
            num_batches = len(dataloader)
            
            loop = tqdm(dataloader, desc=f"Epoch {epoch}")
            
            for i, batch in enumerate(loop):
                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device)
                
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(pseudo_labels))
                targets = pseudo_labels[start_idx:end_idx]
                
                if len(targets) != img.size(0):
                     targets = targets[:img.size(0)]

                self.optimizer.zero_grad()
                
                z = self.encoder(img)
                z = torch.mean(z, dim=[2, 3])
                z = F.normalize(z, dim=1) 
                logits = self.head(z)
                
                loss = self.criterion(logits, targets)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
                
                loop.set_postfix(loss=loss.item(), acc=correct/total_samples)

            avg_loss = total_loss / num_batches
            
            if avg_loss < min_loss:
                min_loss = avg_loss
                self.save_model()

        print(f"    Koniec. Najlepszy model w: {self.save_path}")

    def get_latents(self, dataloader):
        self.load_model()
        self.eval()
        latents = []
        print("     Generowanie finalnych latentów...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device)
                
                z = self.encoder(img)
                z = torch.mean(z, dim=[2, 3])
                z = F.normalize(z, dim=1) 
                latents.append(z.cpu().numpy())
                
        return np.concatenate(latents, axis=0)

    def save_model(self):
        torch.save(self.encoder.state_dict(), self.save_path)

    def load_model(self):
        if self.save_path.exists():
            self.encoder.load_state_dict(torch.load(self.save_path, map_location=self.device))