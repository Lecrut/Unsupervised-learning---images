import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from torch.utils.data import DataLoader
from .encoder import Encoder
from .pca_module import PCAReducer, PCAReducerConfig
from .la_sp import LaSP


class MainPipeline:
    """Główna klasa pipeline'u przetwarzania danych i trenowania modelu"""

    def __init__(self, encoder, decoder, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.optimizer = None
        self.latent_vectors = []
        self.pca_reducer: Optional[PCAReducer] = None
        self.pca_latent_vectors: Optional[np.ndarray] = None
        self.lasp_clean: Optional[LaSP] = None
        self.lasp_damaged: Optional[LaSP] = None
        self.share_lasp: bool = True
        
    def setup_training(self, learning_rate=1e-3):
        """Konfiguracja optymalizatora do treningu"""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        
    def train_epoch(self, dataloader, loss_fn, logger=None):
        """
        Trenuje encoder i decoder przez jedną epokę
        
        Args:
            dataloader: DataLoader z obrazami
            loss_fn: funkcja straty
            logger: opcjonalny logger do zapisywania metryk
        
        Returns:
            średnia strata z epoki
        """
        self.encoder.train()
        self.decoder.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
                
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            latent, features = self.encoder(images)
            reconstructed = self.decoder(latent, features)
            
            loss = loss_fn(reconstructed, images)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if logger and batch_idx % 10 == 0:
                logger.log_metric('batch_loss', loss.item(), step=batch_idx)
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def extract_latent_vectors(self, dataloader, save_path: Optional[str] = None) -> np.ndarray:
        """
        Ekstraktuje latent vectors dla wszystkich obrazów
        
        Args:
            dataloader: DataLoader z obrazami
            save_path: opcjonalna ścieżka do zapisania wektorów
        
        Returns:
            numpy array z latent vectors [N, latent_dim]
        """
        self.encoder.eval()
        all_latents = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                    
                images = images.to(self.device)
                latent, _ = self.encoder(images)
                all_latents.append(latent.cpu().numpy())
        
        latent_vectors = np.concatenate(all_latents, axis=0)
        self.latent_vectors = latent_vectors
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, latent_vectors)
            print(f"Latent vectors zapisane: {save_path}")
        
        return latent_vectors
    
    def reduce_latent_with_pca(
        self,
        latent_vectors: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        pca_config: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Redukcja wymiarowości latentów przy użyciu PCA lub Incremental PCA.
        
        Args:
            latent_vectors: macierz latentów [N, latent_dim]; gdy None używa self.latent_vectors
            save_path: opcjonalna ścieżka do zapisu zredukowanych wektorów
            pca_config: dict z kluczami PCAReducerConfig (np. n_components, incremental, batch_size)
        
        Returns:
            numpy array z wymiarowością zredukowaną do n_components
        """
        if latent_vectors is None:
            if self.latent_vectors is None or len(self.latent_vectors) == 0:
                raise ValueError("Brak latent vectors. Użyj extract_latent_vectors najpierw.")
            latent_vectors = self.latent_vectors
        
        self.pca_reducer = PCAReducer(PCAReducerConfig(**(pca_config or {})))
        self.pca_latent_vectors = self.pca_reducer.fit_transform(latent_vectors)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, self.pca_latent_vectors)
            print(f"PCA latent vectors zapisane: {save_path}")
        
        return self.pca_latent_vectors

    def setup_lasp(self, lasp_config: Optional[Dict[str, Any]] = None, share_weights: bool = True):
        """
        Tworzy moduły LaSP (clean i opcjonalnie damaged).
        """
        cfg = lasp_config or {}
        in_dim = cfg.pop("in_dim", None)
        if in_dim is None:
            # Jeśli nie podano, spróbuj wnioskować z encoder.latent_dim
            in_dim = getattr(self.encoder, "latent_dim", None)
        if in_dim is None:
            raise ValueError("Brak in_dim dla LaSP. Podaj in_dim w lasp_config lub ustaw encoder.latent_dim.")

        self.share_lasp = share_weights
        self.lasp_clean = LaSP(in_dim=in_dim, **cfg).to(self.device)
        if not share_weights:
            self.lasp_damaged = LaSP(in_dim=in_dim, **cfg).to(self.device)
        else:
            self.lasp_damaged = self.lasp_clean

    def compute_lasp(self, latent_vectors: np.ndarray, damaged: bool = False) -> np.ndarray:
        """
        Przepuszcza latent vectors przez LaSP (clean lub damaged).
        """
        if self.lasp_clean is None:
            raise ValueError("LaSP nie jest ustawione. Użyj setup_lasp najpierw.")

        lasp_module = self.lasp_damaged if damaged else self.lasp_clean
        z = torch.from_numpy(latent_vectors).to(self.device)
        with torch.no_grad():
            h = lasp_module(z).cpu().numpy()
        return h

    def compute_lasp_and_pca(
        self,
        latent_clean: np.ndarray,
        latent_damaged: np.ndarray,
        lasp_config: Optional[Dict[str, Any]] = None,
        pca_config_clean: Optional[Dict[str, Any]] = None,
        pca_config_damaged: Optional[Dict[str, Any]] = None,
        save_clean_path: Optional[str] = None,
        save_damaged_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pełny krok: LaSP (clean/damaged) + PCA dla obu gałęzi.
        """
        if self.lasp_clean is None:
            self.setup_lasp(lasp_config, share_weights=False)

        lasp_clean = self.compute_lasp(latent_clean, damaged=False)
        lasp_damaged = self.compute_lasp(latent_damaged, damaged=True)

        pca_clean = PCAReducer(PCAReducerConfig(**(pca_config_clean or {}))).fit_transform(lasp_clean)
        pca_damaged = PCAReducer(PCAReducerConfig(**(pca_config_damaged or pca_config_clean or {}))).fit_transform(lasp_damaged)

        if save_clean_path:
            path = Path(save_clean_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, pca_clean)
            print(f"PCA LaSP clean zapisane: {path}")
        if save_damaged_path:
            path = Path(save_damaged_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, pca_damaged)
            print(f"PCA LaSP damaged zapisane: {path}")

        return pca_clean, pca_damaged
    
    def train_and_extract(self, dataloader, loss_fn, num_epochs=10, 
                         save_latents_path: Optional[str] = None,
                         save_model_path: Optional[str] = None,
                         pca_config: Optional[Dict[str, Any]] = None,
                         save_pca_path: Optional[str] = None,
                         return_pca: bool = False,
                         logger=None) -> Tuple:
        """
        Kompletny pipeline: trenowanie i ekstrakcja latent vectors
        
        Args:
            dataloader: DataLoader z obrazami
            loss_fn: funkcja straty
            num_epochs: liczba epok treningu
            save_latents_path: ścieżka do zapisu latent vectors
            save_model_path: ścieżka do zapisu modelu
            pca_config: opcjonalna konfiguracja PCA (np. {"n_components":50, "incremental":False})
            save_pca_path: ścieżka do zapisu latent vectors po PCA
            return_pca: gdy True zwraca również zredukowane latent vectors
            logger: opcjonalny logger
        
        Returns:
            (lista strat per epoka, latent vectors[, pca_latent_vectors])
        """
        if self.optimizer is None:
            self.setup_training()
        
        losses = []
        
        print(f"Rozpoczynam trening na {num_epochs} epok...")
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, loss_fn, logger)
            losses.append(avg_loss)
            
            print(f"Epoka {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
            
            if logger:
                logger.log_metric('epoch_loss', avg_loss, step=epoch)
        
        print("Trening zakończony. Ekstrakcja latent vectors...")
        latent_vectors = self.extract_latent_vectors(dataloader, save_latents_path)
        
        if save_model_path:
            self.save_models(save_model_path)
        
        pca_latent_vectors = None
        if pca_config:
            pca_latent_vectors = self.reduce_latent_with_pca(
                latent_vectors=latent_vectors,
                save_path=save_pca_path,
                pca_config=pca_config,
            )
        
        if return_pca and pca_latent_vectors is not None:
            return losses, latent_vectors, pca_latent_vectors
        
        return losses, latent_vectors
    
    def save_models(self, path_prefix: str):
        """Zapisuje wagi enkodera i dekodera"""
        path = Path(path_prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.encoder.state_dict(), f"{path_prefix}_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{path_prefix}_decoder.pth")
        print(f"Modele zapisane: {path_prefix}_encoder.pth, {path_prefix}_decoder.pth")
    
    def load_models(self, path_prefix: str):
        """Wczytuje wagi enkodera i dekodera"""
        self.encoder.load_state_dict(torch.load(f"{path_prefix}_encoder.pth", map_location=self.device))
        self.decoder.load_state_dict(torch.load(f"{path_prefix}_decoder.pth", map_location=self.device))
        print(f"Modele wczytane z: {path_prefix}")
