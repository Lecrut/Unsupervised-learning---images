import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
from .deepcluster_modules import EMC, DMG, DEC, IMP, PCAModule, ClusA


class EncoderModel:
    """Klasa opakowująca dla enkodera"""
    
    def __init__(self, latent_dim=128, input_channels=3, image_size=256, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = EMC(latent_dim, input_channels, image_size).to(self.device)
        self.latent_dim = latent_dim
    
    def encode(self, images):
        self.encoder.eval()
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            images = images.to(self.device)
            latent, features = self.encoder(images)
            return latent.cpu().numpy(), features
    
    def extract_from_dataloader(self, dataloader, max_samples=None):
        latent_vectors = []
        samples_processed = 0
        
        for batch in dataloader:
            if max_samples and samples_processed >= max_samples:
                break
            
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            latent, _ = self.encode(images)
            latent_vectors.append(latent)
            samples_processed += images.shape[0]
        
        return np.concatenate(latent_vectors, axis=0)[:max_samples] if max_samples else np.concatenate(latent_vectors, axis=0)
    
    def save(self, path):
        torch.save(self.encoder.state_dict(), path)
    
    def load(self, path):
        self.encoder.load_state_dict(torch.load(path, map_location=self.device))


class ClusteringModel:
    """Klasa opakowująca dla klasteryzacji"""
    
    def __init__(self, n_clusters=10, algorithm='kmeans', use_pca=True, n_components=50):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.use_pca = use_pca
        
        if use_pca:
            self.pca = PCAModule(n_components=n_components)
        else:
            self.pca = None
        
        self.clusterer = ClusA(n_clusters=n_clusters, algorithm=algorithm)
        self.fitted = False
    
    def fit(self, latent_vectors):
        if self.use_pca:
            reduced = self.pca.fit_transform(latent_vectors)
        else:
            reduced = latent_vectors
        
        self.clusterer.fit(reduced)
        self.fitted = True
        return self
    
    def predict(self, latent_vectors):
        if not self.fitted:
            raise ValueError("Model nie został wytrenowany. Użyj fit() najpierw.")
        
        if self.use_pca:
            reduced = self.pca.transform(latent_vectors)
        else:
            reduced = latent_vectors
        
        return self.clusterer.predict(reduced)
    
    def fit_predict(self, latent_vectors):
        self.fit(latent_vectors)
        return self.predict(latent_vectors)
    
    def get_metrics(self, latent_vectors, labels):
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        if len(set(labels)) > 1:
            if self.algorithm == 'dbscan':
                mask = labels != -1
                if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                    metrics['silhouette_score'] = silhouette_score(latent_vectors[mask], labels[mask])
                    metrics['davies_bouldin'] = davies_bouldin_score(latent_vectors[mask], labels[mask])
                    metrics['calinski_harabasz'] = calinski_harabasz_score(latent_vectors[mask], labels[mask])
                metrics['n_noise_points'] = np.sum(labels == -1)
            else:
                metrics['silhouette_score'] = silhouette_score(latent_vectors, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(latent_vectors, labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(latent_vectors, labels)
        
        metrics['n_clusters_found'] = len(set(labels))
        return metrics


class InpaintingModel:
    """Klasa opakowująca dla inpaintingu"""
    
    def __init__(self, latent_dim=128, n_clusters=10, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        self.dmg = DMG(damage_type='mixed').to(self.device)
        self.encoder = EMC(latent_dim).to(self.device)
        self.impainter = IMP(latent_dim, n_clusters).to(self.device)
        self.decoder = DEC(latent_dim).to(self.device)
        
        self.clustering = ClusteringModel(n_clusters=n_clusters)
    
    def train_mode(self):
        self.encoder.train()
        self.impainter.train()
        self.decoder.train()
    
    def eval_mode(self):
        self.encoder.eval()
        self.impainter.eval()
        self.decoder.eval()
    
    def forward(self, images, cluster_labels=None):
        self.eval_mode()
        
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            images = images.to(self.device)
            
            img_damaged = self.dmg(images)
            latent_damaged, features = self.encoder(img_damaged)
            
            if cluster_labels is None:
                cluster_labels = torch.zeros(images.shape[0], dtype=torch.long).to(self.device)
            elif not isinstance(cluster_labels, torch.Tensor):
                cluster_labels = torch.tensor(cluster_labels, dtype=torch.long).to(self.device)
            
            latent_fixed = self.impainter(latent_damaged, cluster_labels)
            img_reconstructed = self.decoder(latent_fixed, features)
            
            return img_reconstructed.cpu(), img_damaged.cpu()
    
    def inpaint(self, images, cluster_labels=None):
        return self.forward(images, cluster_labels)
    
    def save(self, path_prefix):
        torch.save(self.encoder.state_dict(), f"{path_prefix}_encoder.pth")
        torch.save(self.impainter.state_dict(), f"{path_prefix}_impainter.pth")
        torch.save(self.decoder.state_dict(), f"{path_prefix}_decoder.pth")
    
    def load(self, path_prefix):
        self.encoder.load_state_dict(torch.load(f"{path_prefix}_encoder.pth", map_location=self.device))
        self.impainter.load_state_dict(torch.load(f"{path_prefix}_impainter.pth", map_location=self.device))
        self.decoder.load_state_dict(torch.load(f"{path_prefix}_decoder.pth", map_location=self.device))


class SuperResolutionModel:
    """Klasa opakowująca dla super-resolution"""
    
    def __init__(self, upscale_factor=2, device='cuda'):
        from .superres_model import LightweightSuperRes
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = LightweightSuperRes(upscale_factor=upscale_factor).to(self.device)
        self.upscale_factor = upscale_factor
    
    def forward(self, images):
        self.model.eval()
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            images = images.to(self.device)
            upscaled = self.model(images)
            return upscaled.cpu()
    
    def upscale(self, images):
        return self.forward(images)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class CometModel:
    """Klasa do logowania eksperymentów z Comet ML i lokalnie"""
    
    def __init__(self, experiment_name, use_comet=True, use_local=True):
        self.experiment_name = experiment_name
        self.use_comet = use_comet
        self.use_local = use_local
        
        self.comet_experiment = None
        self.local_logger = None
        
        if use_comet:
            try:
                from comet_ml import Experiment
                import os
                from dotenv import load_dotenv
                load_dotenv()
                
                api_key = os.getenv("COMET_API_KEY")
                project_name = os.getenv("COMET_PROJECT_NAME")
                workspace = os.getenv("COMET_WORKSPACE")
                
                if not all([api_key, project_name, workspace]):
                    print("Brak zmiennych środowiskowych dla Comet ML")
                    self.use_comet = False
                else:
                    self.comet_experiment = Experiment(
                        api_key=api_key,
                        project_name=project_name,
                        workspace=workspace
                    )
                    print(f"Comet ML aktywny: {project_name}")
            except Exception as e:
                print(f"Comet ML niedostępny: {e}")
                self.use_comet = False
        
        if use_local:
            from ..utils.local_logger import LocalLogger
            self.local_logger = LocalLogger(experiment_name)
            print(f"Lokalny logger aktywny: {experiment_name}")
    
    def log_parameters(self, params: Dict):
        if self.comet_experiment:
            self.comet_experiment.log_parameters(params)
        if self.local_logger:
            self.local_logger.log_parameters(params)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        if self.comet_experiment:
            self.comet_experiment.log_metrics(metrics, step=step)
        if self.local_logger:
            self.local_logger.log_metrics(metrics, step=step)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        if self.comet_experiment:
            self.comet_experiment.log_metric(name, value, step=step)
        if self.local_logger:
            self.local_logger.log_metrics({name: value}, step=step)
    
    def log_image(self, image, name: str, step: Optional[int] = None):
        if self.comet_experiment:
            self.comet_experiment.log_image(image, name=name, step=step)
    
    def log_figure(self, figure_name: str, figure):
        if self.comet_experiment:
            self.comet_experiment.log_figure(figure_name, figure)
    
    def log_model(self, model_name: str, model_path: str):
        if self.comet_experiment:
            self.comet_experiment.log_model(model_name, model_path)
    
    def end(self):
        if self.comet_experiment:
            self.comet_experiment.end()
        if self.local_logger:
            self.local_logger.end()


class ExperimentLogger(CometModel):
    """Alias dla zachowania kompatybilności"""
    pass
