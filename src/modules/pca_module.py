import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from sklearn.decomposition import PCA, KernelPCA
from .la_sp import LaSP


def run_lasp_pca(
    latent_clean: np.ndarray,
    latent_damaged: np.ndarray,
    proj_dim: Optional[int] = None,
    n_components: int = 50,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LaSP (osobne dla clean/dmg) + liniowy PCA (randomized SVD) osobno dla clean i dmg.
    Zwraca dwie macierze:
      - pca_clean: [N_clean, n_components]
      - pca_dmg:   [N_dmg, n_components]
    """
    urzadzenie = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim_clean = latent_clean.shape[1]
    latent_dim_dmg = latent_damaged.shape[1]
    assert latent_dim_clean == latent_dim_dmg, "Latenty clean i damaged muszą mieć ten sam wymiar"
    latent_dim = latent_dim_clean

    if proj_dim is None or proj_dim == latent_dim:
        lasp_clean = nn.Identity().to(urzadzenie)
        lasp_dmg = nn.Identity().to(urzadzenie)
    else:
        lasp_clean = LaSP(in_dim=latent_dim, proj_dim=proj_dim).to(urzadzenie)
        lasp_dmg = LaSP(in_dim=latent_dim, proj_dim=proj_dim).to(urzadzenie)
        for p in lasp_clean.parameters():
            p.requires_grad = False
        for p in lasp_dmg.parameters():
            p.requires_grad = False
        lasp_clean.eval()
        lasp_dmg.eval()

    with torch.no_grad():
        clean_t = torch.from_numpy(latent_clean).to(urzadzenie).float()
        dmg_t = torch.from_numpy(latent_damaged).to(urzadzenie).float()
        h_clean = lasp_clean(clean_t).cpu().numpy()
        h_dmg = lasp_dmg(dmg_t).cpu().numpy()

    pca_kwargs = dict(n_components=n_components, whiten=True, svd_solver="randomized", random_state=42)
    pca_clean_model = PCA(**pca_kwargs)
    pca_dmg_model = PCA(**pca_kwargs)
    pca_clean = pca_clean_model.fit_transform(h_clean)
    pca_dmg = pca_dmg_model.fit_transform(h_dmg)

    return pca_clean, pca_dmg


# =========================
# Shared LaSP + PCA helpers
# =========================

def train_lasp(
    latent_clean: np.ndarray,
    latent_damaged: np.ndarray,
    proj_dim: int = 64,
    device: Optional[torch.device] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> LaSP:
    """
    Uczy wspólny LaSP na parach (clean, damaged) tą samą sceną.
    Loss: CosineEmbeddingLoss na parach pozytywnych + negatywnych z permutacji.
    """
    urzadzenie = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lasp = LaSP(in_dim=latent_clean.shape[1], proj_dim=proj_dim).to(urzadzenie)
    ds = TensorDataset(
        torch.from_numpy(latent_clean).float(),
        torch.from_numpy(latent_damaged).float(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = torch.nn.CosineEmbeddingLoss()
    optim = torch.optim.Adam(lasp.parameters(), lr=lr)

    for _ in range(epochs):
        for clean, dmg in loader:
            clean = clean.to(urzadzenie)
            dmg = dmg.to(urzadzenie)

            proj_clean = lasp(clean)
            proj_dmg = lasp(dmg)

            pos_labels = torch.ones(clean.size(0), device=urzadzenie)
            pos_loss = criterion(proj_clean, proj_dmg, pos_labels)

            perm = torch.randperm(clean.size(0), device=urzadzenie)
            neg_labels = -torch.ones(clean.size(0), device=urzadzenie)
            neg_loss = criterion(proj_clean, proj_dmg[perm], neg_labels)

            loss = pos_loss + 0.5 * neg_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

    for p in lasp.parameters():
        p.requires_grad = False
    lasp.eval()
    return lasp


def fit_shared_pca(
    rep_clean: np.ndarray,
    rep_dmg: np.ndarray,
    n_components: int = 32,
    whiten: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Dopasowuje jedno PCA na połączonych reprezentacjach i zwraca transformacje dla clean/dmg.
    """
    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="randomized",
        random_state=random_state,
    )
    stacked = np.concatenate([rep_clean, rep_dmg], axis=0)
    pca.fit(stacked)
    return pca.transform(rep_clean), pca.transform(rep_dmg), pca


def run_shared_lasp_pca(
    latent_clean: np.ndarray,
    latent_damaged: np.ndarray,
    proj_dim: Optional[int] = None,
    n_components: int = 32,
    device: Optional[torch.device] = None,
    lasp_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, nn.Module, PCA]:
    """
    Wspólny tor: (opcjonalnie) uczony LaSP -> jedno PCA dla clean + damaged.
    Zwraca (pca_clean, pca_dmg, lasp_model, pca_model).
    """
    urzadzenie = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim_clean = latent_clean.shape[1]
    latent_dim_dmg = latent_damaged.shape[1]
    assert latent_dim_clean == latent_dim_dmg, "Latenty clean i damaged muszą mieć ten sam wymiar"

    if proj_dim is None or proj_dim == latent_dim_clean:
        lasp = nn.Identity().to(urzadzenie)
    else:
        lasp = train_lasp(
            latent_clean=latent_clean,
            latent_damaged=latent_damaged,
            proj_dim=proj_dim,
            device=urzadzenie,
            epochs=lasp_epochs,
            batch_size=batch_size,
            lr=lr,
        )

    with torch.no_grad():
        clean_t = torch.from_numpy(latent_clean).to(urzadzenie).float()
        dmg_t = torch.from_numpy(latent_damaged).to(urzadzenie).float()
        rep_clean = lasp(clean_t).cpu().numpy()
        rep_dmg = lasp(dmg_t).cpu().numpy()

    pca_clean, pca_dmg, pca_model = fit_shared_pca(
        rep_clean=rep_clean,
        rep_dmg=rep_dmg,
        n_components=n_components,
    )
    return pca_clean, pca_dmg, lasp, pca_model


def run_shared_lasp_kpca(
    latent_clean: np.ndarray,
    latent_damaged: np.ndarray,
    proj_dim: Optional[int] = None,
    n_components: int = 32,
    device: Optional[torch.device] = None,
    lasp_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, nn.Module, KernelPCA]:
    """
    Wspólny tor: (opcjonalnie) uczony LaSP -> jedno KernelPCA dla clean + damaged.
    Zwraca (kpca_clean, kpca_dmg, lasp_model, kpca_model).
    """
    urzadzenie = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim_clean = latent_clean.shape[1]
    latent_dim_dmg = latent_damaged.shape[1]
    assert latent_dim_clean == latent_dim_dmg, "Latenty clean i damaged muszą mieć ten sam wymiar"

    if proj_dim is None or proj_dim == latent_dim_clean:
        lasp = nn.Identity().to(urzadzenie)
    else:
        lasp = train_lasp(
            latent_clean=latent_clean,
            latent_damaged=latent_damaged,
            proj_dim=proj_dim,
            device=urzadzenie,
            epochs=lasp_epochs,
            batch_size=batch_size,
            lr=lr,
        )

    with torch.no_grad():
        clean_t = torch.from_numpy(latent_clean).to(urzadzenie).float()
        dmg_t = torch.from_numpy(latent_damaged).to(urzadzenie).float()
        rep_clean = lasp(clean_t).cpu().numpy()
        rep_dmg = lasp(dmg_t).cpu().numpy()

    kpca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=False,
        random_state=42,
    )
    rep_all = np.concatenate([rep_clean, rep_dmg], axis=0)
    kpca.fit(rep_all)
    kpca_clean = kpca.transform(rep_clean)
    kpca_dmg = kpca.transform(rep_dmg)

    return kpca_clean, kpca_dmg, lasp, kpca
