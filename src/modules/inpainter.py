#%% Imports
import torch

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Inpainter Model
def in_painter_model(
        latent_damaged,
        clusters_damaged,
        latent_original,
        clusters_original,
        lambda_center: float = 0.5,
        lambda_damaged: float = 0.5,
    ):

    if not torch.is_tensor(latent_damaged):
        latent_damaged = torch.as_tensor(latent_damaged, dtype=torch.float32, device=device)
    else:
        latent_damaged = latent_damaged.to(device=device, dtype=torch.float32)
    
    if not torch.is_tensor(latent_original):
        latent_original = torch.as_tensor(latent_original, dtype=torch.float32, device=device)
    else:
        latent_original = latent_original.to(device=device, dtype=torch.float32)
    
    if not torch.is_tensor(clusters_damaged):
        clusters_damaged = torch.as_tensor(clusters_damaged, dtype=torch.long, device=device)
    else:
        clusters_damaged = clusters_damaged.to(device=device, dtype=torch.long)
    
    if not torch.is_tensor(clusters_original):
        clusters_original = torch.as_tensor(clusters_original, dtype=torch.long, device=device)
    else:
        clusters_original = clusters_original.to(device=device, dtype=torch.long)

    if latent_damaged.dim() == 1:
        latent_damaged = latent_damaged.unsqueeze(0)
    if latent_original.dim() == 1:
        latent_original = latent_original.unsqueeze(0)

    N_damaged, D = latent_damaged.shape
    N_original = latent_original.shape[0]

    if clusters_damaged.numel() != N_damaged:
        raise ValueError(f"clusters_damaged length ({clusters_damaged.numel()}) != number of damaged latent vectors ({N_damaged})")
    if clusters_original.numel() != N_original:
        raise ValueError(f"clusters_original length ({clusters_original.numel()}) != number of original latent vectors ({N_original})")

    n_clusters = int(max(clusters_damaged.max().item(), clusters_original.max().item())) + 1

    centers = torch.zeros((n_clusters, D), device=device, dtype=latent_original.dtype)
    for k in range(n_clusters):
        mask = (clusters_original == k)
        if mask.any():
            centers[k] = latent_original[mask].mean(dim=0)
        else:
            centers[k] = torch.zeros(D, device=device, dtype=latent_original.dtype)

    centers_per_sample = centers[clusters_damaged]

    denom = (lambda_center + lambda_damaged) if (lambda_center + lambda_damaged) != 0 else 1.0
    alpha = float(lambda_damaged) / float(denom)

    repaired = centers_per_sample * (1.0 - alpha) + latent_damaged * alpha

    return repaired
