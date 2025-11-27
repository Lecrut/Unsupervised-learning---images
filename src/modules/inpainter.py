#%% Imports
import torch

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Inpainter Model
def in_painter_model(
        latent_damaged,
        clusters,
        lambda_center: float = 0.5,
        lambda_damaged: float = 0.5,
        steps: int = 20,
        lr: float = 0.05,
        device_arg = None,
    ):
    dev = device_arg if device_arg is not None else device

    if not torch.is_tensor(latent_damaged):
        latent_damaged = torch.as_tensor(latent_damaged, dtype=torch.float32, device=dev)
    else:
        latent_damaged = latent_damaged.to(device=dev, dtype=torch.float32)

    if not torch.is_tensor(clusters):
        clusters = torch.as_tensor(clusters, dtype=torch.long, device=dev)
    else:
        clusters = clusters.to(device=dev, dtype=torch.long)

    if latent_damaged.dim() == 1:
        latent_damaged = latent_damaged.unsqueeze(0)

    N, D = latent_damaged.shape

    if clusters.numel() != N:
        raise ValueError(f"clusters length ({clusters.numel()}) != number of latent vectors ({N})")

    n_clusters = int(clusters.max().item()) + 1

    centers = torch.zeros((n_clusters, D), device=dev, dtype=latent_damaged.dtype)
    for k in range(n_clusters):
        mask = (clusters == k)
        if mask.any():
            centers[k] = latent_damaged[mask].mean(dim=0)
        else:
            centers[k] = torch.zeros(D, device=dev, dtype=latent_damaged.dtype)

    centers_per_sample = centers[clusters]  # (N, D)

    denom = (lambda_center + lambda_damaged) if (lambda_center + lambda_damaged) != 0 else 1.0
    alpha = float(lambda_damaged) / float(denom)

    repaired = centers_per_sample * (1.0 - alpha) + latent_damaged * alpha

    return repaired
