import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from .la_sp import LaSP


@dataclass
class PCAReducerConfig:
    """Konfiguracja PCA"""

    n_components: int = 50          #ile głównych składowych zachowujemy
    whiten: bool = True             #normalizacja wariancji
    random_state: int = 42          #ziarno losowe
    svd_solver: str = "randomized"  #randomized PCA per Halko et al. 2011


class PCAReducer:
    """
    Redukcja wymiarowości latentów z użyciem PCA.

    Domyślnie korzystamy z randomized PCA (Halko et al., 2011 --> implementacja w scikit-learn).
    """

    def __init__(self, config: Optional[PCAReducerConfig] = None, **overrides):
        #jeśli nie podamy configu, korzystamy z wartości domyślnych
        cfg = config or PCAReducerConfig()
        for klucz, wartosc in overrides.items():
            if hasattr(cfg, klucz):
                setattr(cfg, klucz, wartosc)

        self.config = cfg
        self.model = None
        self.fitted = False

    def _build_model(self):
        """Tworzy obiekt PCA na podstawie configu (randomized SVD wg Halko et al., 2011)."""
        return PCA(
            n_components=self.config.n_components,
            whiten=self.config.whiten,
            svd_solver=self.config.svd_solver,
            random_state=self.config.random_state,
        )

    def fit(self, latent_vectors: np.ndarray):
        """
        Dopasowuje PCA do wektorów latentnych (X o wymiarach [N, D]).
        """
        model = self._build_model()
        model.fit(latent_vectors)
        self.model = model
        self.fitted = True
        return self

    def transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise ValueError("PCA nie zostało wytrenowane. Użyj fit() lub fit_transform() najpierw.")
        return self.model.transform(latent_vectors)

    def fit_transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Wygodna wersja: najpierw dopasowanie PCA, potem transformacja tych samych danych.
        """
        model = self._build_model()
        przeksztalcone = model.fit_transform(latent_vectors)
        self.model = model
        self.fitted = True
        return przeksztalcone

    @property
    def explained_variance_ratio_(self):
        if self.model is None or not hasattr(self.model, "explained_variance_ratio_"):
            return None
        return getattr(self.model, "explained_variance_ratio_", None)

#LaSP + PCA dla clean / damaged
def run_lasp_pca(
    latent_clean: np.ndarray,
    latent_damaged: np.ndarray,
    proj_dim: Optional[int] = None,
    n_components: int = 50,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zastosowanie LaSP (osobne projekcje dla czystych i uszkodzonych)
    oraz PCA (randomized, wg Halko et al., 2011) do redukcji wymiaru.

    Kroki:
    1) Bierzemy wektory latentne z autoenkodera:
       - latent_clean: latenty dla obrazów oryginalnych,
       - latent_damaged: latenty dla obrazów uszkodzonych.
    2) Każdą gałąź przepuszczamy przez osobny moduł LaSP
       (LaSP wprowadza nieliniową/liniową projekcję na wymiar proj_dim).
    3) Na wynikowych reprezentacjach h_clean, h_dmg
       uruchamiamy PCA z randomized SVD.
    4) Zwracamy dwie macierze:
       - pca_clean: [N_clean, n_components]
       - pca_dmg:   [N_dmg, n_components]
    """
    urzadzenie = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim_clean = latent_clean.shape[1]
    latent_dim_dmg = latent_damaged.shape[1]

    # obie reprezentacje muszą mieć ten sam wymiar
    assert latent_dim_clean == latent_dim_dmg, "Latenty clean i damaged muszą mieć ten sam wymiar"
    latent_dim = latent_dim_clean

    # jeśli proj_dim niepodany lub równy wymiarowi latentów -> używamy tożsamości (bez losowej projekcji)
    if proj_dim is None or proj_dim == latent_dim:
        lasp_clean = nn.Identity().to(urzadzenie)
        lasp_dmg = nn.Identity().to(urzadzenie)
    else:
        lasp_clean = LaSP(in_dim=latent_dim, proj_dim=proj_dim).to(urzadzenie)
        lasp_dmg = LaSP(in_dim=latent_dim, proj_dim=proj_dim).to(urzadzenie)
        # jeżeli chcesz, by LaSP nie uczył się w tym kroku
        for p in lasp_clean.parameters():
            p.requires_grad = False
        for p in lasp_dmg.parameters():
            p.requires_grad = False
        lasp_clean.eval()
        lasp_dmg.eval()

    with torch.no_grad():
        #Konwersja z numpy do torch, przeniesienie na urządzenie i rzutowanie na float
        latent_clean_t = torch.from_numpy(latent_clean).to(urzadzenie).float()
        latent_dmg_t = torch.from_numpy(latent_damaged).to(urzadzenie).float()

        #Przejście przez LaSP – otrzymujemy zredukowane reprezentacje h_clean i h_dmg
        h_clean = lasp_clean(latent_clean_t).cpu().numpy()
        h_dmg = lasp_dmg(latent_dmg_t).cpu().numpy()

    pca_clean = PCAReducer(n_components=n_components).fit_transform(h_clean)
    pca_dmg = PCAReducer(n_components=n_components).fit_transform(h_dmg)

    return pca_clean, pca_dmg
