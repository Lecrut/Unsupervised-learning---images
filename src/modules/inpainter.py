#%% Imports
import torch

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Inpainter Model
def in_painter_model(
        latent_damaged: torch.Tensor,
        clusters,
        lambda_center: float = 0.5,
        lambda_damaged: float = 0.5,
        steps: int = 20,
        lr: float = 0.05
    ):

    if isinstance(clusters.cluster_centers_, torch.Tensor):
        centers = clusters.cluster_centers_
    else:
        centers = torch.tensor(clusters.cluster_centers_, dtype=latent_damaged.dtype)

    distances = torch.norm(centers - latent_damaged.unsqueeze(0), dim=1)
    cluster_label = torch.argmin(distances).item()
    center = centers[cluster_label]

    offset = torch.zeros_like(latent_damaged, requires_grad=True)
    optimizer = torch.optim.Adam([offset], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        candidate = center + offset

        loss_center = lambda_center * torch.norm(offset)
        loss_damaged = lambda_damaged * torch.norm(candidate - latent_damaged)

        loss = loss_center + loss_damaged
        loss.backward()
        optimizer.step()

    repaired = center + offset.detach()
    return repaired


