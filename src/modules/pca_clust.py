#%% Imports
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

#%% Helper Function - Ensure Numpy Array
def _ensure_numpy(data):
    if torch.is_tensor(data):
        return data.cpu().detach().numpy()
    return np.asarray(data)

#%% UMAP - PCA Reduction
def our_pca(latent_damaged, latent_original, n_components=300):
    X = _ensure_numpy(latent_damaged)
    Y = _ensure_numpy(latent_original)

    combined = np.vstack([X, Y])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    print(f"Redukcja UMAP do {n_components} wymiarów...")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    
    transformed = reducer.fit_transform(combined_scaled)

    n_damaged = len(X)
    return transformed[:n_damaged], transformed[n_damaged:]

#%% HDBSCAN - The "Fresh" One
def clustering(latent_damaged, latent_original, eps=None, min_samples=15):
    X = _ensure_numpy(latent_damaged)
    Y = _ensure_numpy(latent_original)

    print(f"Uruchamianie HDBSCAN (min_cluster_size={min_samples})...")
    
    clusterer = HDBSCAN(
        min_cluster_size=min_samples, 
        min_samples=min_samples, 
        n_jobs=-1)

    damaged_labels = clusterer.fit_predict(X)
    
    clusterer_orig = HDBSCAN(
        min_cluster_size=min_samples, 
        min_samples=min_samples, 
        n_jobs=-1
        )
    
    original_labels = clusterer_orig.fit_predict(Y)

    def print_stats(name, labels):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        pct_noise = (n_noise / len(labels)) * 100 if len(labels) > 0 else 0
        print(f"   {name}: {n_clusters} klastrów, {n_noise} szum ({pct_noise:.1f}%)")

    print_stats("Damaged", damaged_labels)
    print_stats("Original", original_labels)

    return damaged_labels, original_labels

#%% UMAP Visualization
def display_umap(latent, labels=None, n_clusters=None):
    X = _ensure_numpy(latent)
    
    if labels is not None:
        labels = _ensure_numpy(labels)
        if labels.ndim == 2:
            labels = labels.argmax(axis=1)

    print("Generowanie wykresu...")
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine')
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
    else:
        # Szum na szaro
        noise_mask = (labels == -1)
        plt.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], c='#cccccc', s=5, label='Szum (-1)')
        
        # Klastry na kolorowo
        cluster_mask = ~noise_mask
        plt.scatter(embedding[cluster_mask, 0], embedding[cluster_mask, 1], 
                    c=labels[cluster_mask], cmap='Spectral', s=8)
        
    plt.title('UMAP Projection + HDBSCAN', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()