import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering


class IMG:
    """Moduł wejściowy - przetwarza obraz do tensora"""
    
    def __init__(self, transform=None):
        self.transform = transform
    
    def __call__(self, img):
        if self.transform:
            img = self.transform(img)
        return img


class DMG(nn.Module):
    """Data Modification Generator - generuje zniekształcone wersje obrazu"""
    
    def __init__(self, damage_type='mixed'):
        super().__init__()
        self.damage_type = damage_type
    
    def forward(self, img):
        from ..data.damages import random_mask, rectangular_mask, noise_mask, line_damage, circular_mask
        
        if self.damage_type == 'simple':
            damage_fn = random_mask if torch.rand(1) > 0.5 else rectangular_mask
        elif self.damage_type == 'irregular':
            damage_fns = [line_damage, circular_mask, noise_mask]
            damage_fn = damage_fns[torch.randint(0, len(damage_fns), (1,)).item()]
        else:
            damage_fns = [random_mask, rectangular_mask, noise_mask, line_damage, circular_mask]
            damage_fn = damage_fns[torch.randint(0, len(damage_fns), (1,)).item()]
        
        img_d = torch.stack([damage_fn(img[i]) for i in range(img.size(0))])
        return img_d


class EMC(nn.Module):
    """Encoder Module - ekstrakcja reprezentacji latentnej"""
    
    def __init__(self, latent_dim=128, input_channels=3, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.conv_output_size = image_size // 8
        
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1)
        )
        
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.conv_output_size * self.conv_output_size, latent_dim),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x):
        x = 2 * x - 1
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        z = self.to_latent(x3)
        return z, (x1, x2, x3)


class PCAModule:
    """Redukcja wymiarowości PCA dla przestrzeni latentnej"""
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False
    
    def fit(self, LaSp):
        self.pca.fit(LaSp)
        self.fitted = True
        return self
    
    def transform(self, LaSp):
        if not self.fitted:
            raise ValueError("PCA nie został wytrenowany. Użyj fit() najpierw.")
        return self.pca.transform(LaSp)
    
    def fit_transform(self, LaSp):
        return self.pca.fit_transform(LaSp)


class ClusA:
    """Cluster Assignment - przypisanie klastrów"""
    
    def __init__(self, n_clusters=10, algorithm='kmeans'):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.clusterer = None
        self.fitted = False
    
    def fit(self, pca_lasp):
        if self.algorithm == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.algorithm == 'gaussian_mixture':
            self.clusterer = GaussianMixture(n_components=self.n_clusters, random_state=42)
        elif self.algorithm == 'spectral':
            self.clusterer = SpectralClustering(n_clusters=self.n_clusters, random_state=42)
        elif self.algorithm == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Nieznany algorytm: {self.algorithm}")
        
        self.clusterer.fit(pca_lasp)
        self.fitted = True
        return self
    
    def predict(self, pca_lasp):
        if not self.fitted:
            raise ValueError("Clusterer nie został wytrenowany. Użyj fit() najpierw.")
        
        if hasattr(self.clusterer, 'predict'):
            return self.clusterer.predict(pca_lasp)
        else:
            return self.clusterer.labels_


class IMP(nn.Module):
    """IMPainter - uzupełnia brakujące fragmenty w przestrzeni latentnej"""
    
    def __init__(self, latent_dim=128, n_clusters=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        self.cluster_embedding = nn.Embedding(n_clusters, 64)
        
        self.fixer = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, LaSp_d, K):
        cluster_features = self.cluster_embedding(K)
        combined = torch.cat([LaSp_d, cluster_features], dim=1)
        LaSp_fixed = self.fixer(combined)
        return LaSp_fixed


class DEC(nn.Module):
    """Decoder - rekonstrukcja obrazu z przestrzeni latentnej"""
    
    def __init__(self, latent_dim=128, output_channels=3, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.conv_output_size = image_size // 8
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.conv_output_size * self.conv_output_size),
            nn.ReLU(True),
            nn.Unflatten(1, (256, self.conv_output_size, self.conv_output_size))
        )
        
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, encoder_features=None):
        x = self.from_latent(z)
        
        if encoder_features is not None:
            x1, x2, x3 = encoder_features
            x = self.decoder_block1(x + x3)
            x = self.decoder_block2(x + x2)
            x = self.final_block(x + x1)
        else:
            x = self.decoder_block1(x)
            x = self.decoder_block2(x)
            x = self.final_block(x)
        
        x = (x + 1) / 2
        return x


class DeepClusterPipeline(nn.Module):
    """Kompletny pipeline DeepCluster dla inpainting"""
    
    def __init__(self, latent_dim=128, n_clusters=10, damage_type='mixed'):
        super().__init__()
        self.dmg = DMG(damage_type=damage_type)
        self.emc = EMC(latent_dim=latent_dim)
        self.pca = PCAModule(n_components=min(50, latent_dim))
        self.clusa = ClusA(n_clusters=n_clusters)
        self.imp = IMP(latent_dim=latent_dim, n_clusters=n_clusters)
        self.dec = DEC(latent_dim=latent_dim)
        
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
    
    def encode_and_cluster(self, img):
        img_d = self.dmg(img)
        
        LaSp, features = self.emc(img)
        LaSp_d, features_d = self.emc(img_d)
        
        return LaSp, LaSp_d, features, features_d, img_d
    
    def cluster_latent(self, LaSp_np, LaSp_d_np):
        pca_lasp = self.pca.fit_transform(LaSp_np)
        pca_lasp_d = self.pca.transform(LaSp_d_np)
        
        self.clusa.fit(pca_lasp)
        K = self.clusa.predict(pca_lasp_d)
        
        return K
    
    def inpaint_and_decode(self, LaSp_d, K, features=None):
        LaSp_fixed = self.imp(LaSp_d, K)
        img_fixed = self.dec(LaSp_fixed, features)
        
        return img_fixed, LaSp_fixed
    
    def forward(self, img, return_all=False):
        img_d = self.dmg(img)
        
        LaSp, features = self.emc(img)
        LaSp_d, features_d = self.emc(img_d)
        
        img_recon = self.dec(LaSp, features)
        img_d_recon = self.dec(LaSp_d, features_d)
        
        if return_all:
            return {
                'img_recon': img_recon,
                'img_d_recon': img_d_recon,
                'img_d': img_d,
                'LaSp': LaSp,
                'LaSp_d': LaSp_d,
                'features': features,
                'features_d': features_d
            }
        
        return img_recon, img_d_recon, LaSp, LaSp_d
