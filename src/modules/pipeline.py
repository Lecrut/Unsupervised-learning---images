import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from torch.utils.data import DataLoader
from .encoder import Encoder


class MainPipeline:
    """Główna klasa pipeline'u przetwarzania danych i trenowania modelu"""

    def __init__(self, encoder, decoder, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.optimizer = None
        self.latent_vectors = []
        
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
    
    def train_and_extract(self, dataloader, loss_fn, num_epochs=10, 
                         save_latents_path: Optional[str] = None,
                         save_model_path: Optional[str] = None,
                         logger=None) -> Tuple[List[float], np.ndarray]:
        """
        Kompletny pipeline: trenowanie i ekstrakcja latent vectors
        
        Args:
            dataloader: DataLoader z obrazami
            loss_fn: funkcja straty
            num_epochs: liczba epok treningu
            save_latents_path: ścieżka do zapisu latent vectors
            save_model_path: ścieżka do zapisu modelu
            logger: opcjonalny logger
        
        Returns:
            (lista strat per epoka, latent vectors)
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