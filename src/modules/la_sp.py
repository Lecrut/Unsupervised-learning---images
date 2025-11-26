import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LaSP(nn.Module):
    """
    Latent Space Projection (LaSP)
    - prosta głowa projekcyjna na latentach + normalizacja L2.
    - domyślnie: Linear -> ReLU -> BatchNorm1d -> Linear -> L2 norm.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 64,
        hidden_dim: Optional[int] = None,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        hidden = hidden_dim or in_dim

        layers = [
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.Linear(hidden, proj_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)
        # L2 normalizacja, by porównania/klasteryzacja były stabilne
        h = F.normalize(h, p=2, dim=1)
        return h

