import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
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
            self._load_model()
            self.model_loaded = True

    def _fast_silhouette(self, z, labels, frac=0.1, max_samples=10000):
        n = len(z)
        k = max(2, int(n * frac))
        k = min(k, max_samples)

        idx = np.random.choice(n, k, replace=False)
        return silhouette_score(z[idx], labels[idx], metric="cosine")


    def _get_features_for_clustering(self, dataloader):
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

    def fit(self, dataloader, epochs=10, cluster_every=2, subset_frac=0.1):
        print(f"Start formowania przestrzeni ({epochs} epok).")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        best_silhouette = -1.0

        for epoch in range(epochs):

            features = self._get_features_for_clustering(dataloader)

            if epoch % cluster_every == 0:
                n = len(features)
                k = max(2, int(n * subset_frac))
                idx = np.random.choice(n, k, replace=False)
                self.kmeans.fit(features[idx])

            labels = self.kmeans.predict(features)
            labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

            sil = self._fast_silhouette(features, labels, frac=subset_frac)
            cluster_sizes = np.bincount(labels, minlength=self.num_clusters)
            alive = (cluster_sizes > 0).sum()

            print(
                f"[Epoch {epoch+1}/{epochs}] "
                f"Silhouette: {sil:.4f} | "
                f"Alive clusters: {alive}/{self.num_clusters} | "
                f"Std size: {cluster_sizes.std():.1f}"
            )

            if sil > best_silhouette and alive >= self.num_clusters * 0.3:
                best_silhouette = sil
                self._save_model()
                print(f"    Model zapisany (silhouette = {sil:.4f})")

            self.train()
            batch_size = dataloader.batch_size

            loop = tqdm(dataloader, desc=f"Train epoch {epoch+1}/{epochs}")
            for i, batch in enumerate(loop):

                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device)

                start = i * batch_size
                end = min(start + batch_size, len(labels_t))
                targets = labels_t[start:end]

                if targets.size(0) != img.size(0):
                    targets = targets[:img.size(0)]

                self.optimizer.zero_grad()

                z = self.encoder(img)
                z = torch.mean(z, dim=[2, 3])
                z = F.normalize(z, dim=1)

                logits = self.head(z)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step()

                loop.set_postfix(loss=loss.item())

        print(f"Koniec. Najlepszy silhouette = {best_silhouette:.4f}")

    def get_latents(self, dataloader, use_best=True):
        if use_best and not self.model_loaded:
            self._load_model()

        self.encoder.eval()
        latents = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting latents"):
                img = batch[0] if isinstance(batch, (list, tuple)) else batch
                img = img.to(self.device)

                z = self.encoder(img)
                latents.append(z.cpu().numpy())

        return np.concatenate(latents, axis=0)


    def _save_model(self, epoch=None, silhouette=None):
        state = {
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "kmeans": self.kmeans,
            "epoch": epoch,
            "silhouette": silhouette,
        }
        torch.save(state, self.save_path)


    def _load_model(self, load_optimizer=False):
        if not self.save_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.save_path}")

        checkpoint = torch.load(self.save_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.head.load_state_dict(checkpoint["head"])

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "kmeans" in checkpoint:
            self.kmeans = checkpoint["kmeans"]

        self.encoder.eval()
        self.head.eval()

        self.model_loaded = True

        return checkpoint.get("epoch"), checkpoint.get("silhouette")
