#%% Imports
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier

#%% Helper Function
def _ensure_numpy(data):
    if torch.is_tensor(data):
        return data.cpu().detach().numpy()
    return np.asarray(data)

#%% UMAP - PCA Reduction
def our_pca(latent_damaged, latent_original, n_components=10):
    X = _ensure_numpy(latent_damaged)
    Y = _ensure_numpy(latent_original)
    combined = np.vstack([X, Y])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    print(f"Redukcja UMAP (Manhattan) do {n_components} wymiarów...")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,        
        min_dist=0.0,
        metric='manhattan',
        random_state=42
    )
    
    transformed = reducer.fit_transform(combined_scaled)
    n_damaged = len(X)
    return transformed[:n_damaged], transformed[n_damaged:]

#%% HDBSCAN - Balanced Snake Cutter
def clustering(latent_damaged, latent_original, min_samples=80): 
    X = _ensure_numpy(latent_damaged)
    Y = _ensure_numpy(latent_original)

    def force_assign_noise(data, labels):
        if -1 not in labels:
            return labels
        
        noise_mask = (labels == -1)
        data_clean = data[~noise_mask]
        labels_clean = labels[~noise_mask]
        data_noise = data[noise_mask]

        if len(labels_clean) == 0:
            return labels

        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric='manhattan')
        knn.fit(data_clean, labels_clean)
        predicted_labels = knn.predict(data_noise)
        
        new_labels = labels.copy()
        new_labels[noise_mask] = predicted_labels
        return new_labels

    print(f"Uruchamianie HDBSCAN (Leaf, Manhattan, min_cluster_size={min_samples})...")
    
    clusterer = HDBSCAN(
        min_cluster_size=min_samples,  
        min_samples=1,                  
        cluster_selection_method='leaf',
        metric='manhattan',             
        allow_single_cluster=False,
        n_jobs=-1
    )
    damaged_labels = clusterer.fit_predict(X)
    damaged_labels = force_assign_noise(X, damaged_labels)
    
    clusterer_orig = HDBSCAN(
        min_cluster_size=min_samples,
        min_samples=1,
        cluster_selection_method='leaf',
        metric='manhattan',
        allow_single_cluster=False,
        n_jobs=-1
    )
    original_labels = clusterer_orig.fit_predict(Y)
    original_labels = force_assign_noise(Y, original_labels)

    def print_stats(name, labels):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"   {name}: {n_clusters} klastrów (cel: 10-30), {n_noise} szum.")

    print_stats("Damaged", damaged_labels)
    print_stats("Original", original_labels)

    return damaged_labels, original_labels

#%% UMAP Visualization
def display_umap(latent, labels=None, n_clusters=None):
    X = _ensure_numpy(latent)
    if labels is not None:
        labels = _ensure_numpy(labels)
        if labels.ndim == 2: labels = labels.argmax(axis=1)

    print("Rysowanie...")
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='manhattan')
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(12, 8))
    
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c='gray')
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))

        for i, lbl in enumerate(unique_labels):
            mask = (labels == lbl)
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                        c=[colors[i]], s=8, label=f'{lbl}', alpha=1.0)
        
    plt.title(f'Wynik HDBSCAN (min_size=80, klastrów: {len(unique_labels)})', fontsize=14)
    if len(unique_labels) < 25:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()