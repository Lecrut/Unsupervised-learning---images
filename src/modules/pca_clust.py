#Imports 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import umap
import matplotlib.pyplot as plt

#%$ PCA function
def our_pca(latent_damaged, latent_original, n_components=300):
    X = np.asarray(latent_damaged)

    if X.ndim != 2:
        raise ValueError("latent_damaged must be a 2D array of shape (n_samples, n_features)")

    n_samples, n_features = X.shape
    max_components = max(1, min(n_samples, n_features))
    n_components = int(min(n_components, max_components))

    pca = PCA(n_components=n_components)

    smaller_latent_damaged = pca.fit_transform(latent_damaged)
    smaller_latent_original = pca.transform(latent_original)

    return smaller_latent_damaged, smaller_latent_original

#%% Clustering function
def clustering(latent_damaged, latent_original, n_clusters=10, n_iters=100, device=None):
    if isinstance(latent_damaged, np.ndarray):
        latent_damaged = torch.from_numpy(latent_damaged)
    if isinstance(latent_original, np.ndarray):
        latent_original = torch.from_numpy(latent_original)

    if not torch.is_tensor(latent_damaged) or not torch.is_tensor(latent_original):
        raise TypeError("latent_damaged and latent_original must be numpy arrays or torch tensors")

    latent_damaged = latent_damaged.float()
    latent_original = latent_original.float()
    device = device or torch.device('cpu')
    latent_damaged = latent_damaged.to(device)
    latent_original = latent_original.to(device)

    N_d = latent_damaged.shape[0]
    all_data = torch.cat((latent_damaged, latent_original), dim=0)
    N, D = all_data.shape

    n_clusters = int(min(n_clusters, N))

    perm = torch.randperm(N, device=device)[:n_clusters]
    centroids = all_data[perm].clone()

    for _ in range(int(n_iters)):
        distances = torch.cdist(all_data, centroids)  
        labels = torch.argmin(distances, dim=1)

        converged = True
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centroid = all_data[mask].mean(dim=0)
                if not torch.allclose(centroids[k], new_centroid, atol=1e-6):
                    centroids[k] = new_centroid
                    converged = False
        if converged:
            break

    damaged_labels = labels[:N_d]
    original_labels = labels[N_d:]

    return damaged_labels, original_labels

#%% UMAP visualization function
def display_umap(latent, labels=None, n_clusters=None):
    if torch.is_tensor(latent):
        X = latent.cpu().numpy()
    else:
        X = np.asarray(latent)

    if isinstance(n_clusters, (tuple, list)) and len(n_clusters) == 2:
        _, maybe_original_labels = n_clusters
        labels = maybe_original_labels
        n_clusters = None

    if labels is None:
        labels_np = None
    else:
        if torch.is_tensor(labels):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.asarray(labels)

    if labels_np is not None and labels_np.ndim == 2:
        if labels_np.shape[0] == X.shape[0]:
            labels_np = labels_np.argmax(axis=1)
        else:
            raise ValueError(
                "Provided labels must be 1D of length equal to number of samples. "
                "If you passed the clustering result, unpack it as "
                "damaged_labels, original_labels = clustering(...); then call "
                "display_umap(smaller_latent_original_vectors, original_labels)."
            )

    if labels_np is not None and labels_np.shape[0] != X.shape[0]:
        raise ValueError(
            f"Labels length ({labels_np.shape[0]}) does not match number of samples ({X.shape[0]}). "
            "Ensure you pass the latent vectors for the same set you clustered and the correct labels array."
        )

    if labels_np is not None and n_clusters is None:
        try:
            n_clusters = int(labels_np.max()) + 1
        except Exception:
            n_clusters = 10
    n_clusters = int(n_clusters) if n_clusters is not None else 10

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    if labels_np is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap='Spectral', s=5)
        plt.colorbar(boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))
    plt.title('UMAP projection of the latent space', fontsize=15)
    plt.show()
