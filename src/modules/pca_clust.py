#%% Imports
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

#%% Helper Function
def _ensure_numpy(data):
    if torch.is_tensor(data):
        return data.cpu().detach().numpy()
    return np.asarray(data)

#%% UMAP - Visualization Optimized
def our_pca(latent_damaged, latent_original, n_components=2):
    X = _ensure_numpy(latent_damaged)
    Y = _ensure_numpy(latent_original)
    combined = np.vstack([X, Y])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    print(f"Redukcja UMAP do {n_components} wymiarów (tryb wizualny)...")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,    
        min_dist=0.1,        
        metric='cosine',     
        init='spectral',     
        n_jobs=-1,
        random_state=42
    )
    
    transformed = reducer.fit_transform(combined_scaled)

    min_max = MinMaxScaler()
    transformed = min_max.fit_transform(transformed)

    n_damaged = len(X)
    return transformed[:n_damaged], transformed[n_damaged:]

#%% Clustering - The Visual Cutter
def clustering(latent_damaged, latent_original, n_clusters=20): 
    X = _ensure_numpy(latent_damaged)
    Y = _ensure_numpy(latent_original)

    print(f"Błyskawiczne cięcie węża 2D na {n_clusters} segmentów...")
    
    combined_2d = np.vstack([X, Y])
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        n_init=10,
        random_state=42
    )
    
    kmeans.fit(combined_2d)
    
    damaged_labels = kmeans.predict(X)
    original_labels = kmeans.predict(Y)

    def print_stats(name, labels):
        unique = len(np.unique(labels))
        print(f"   {name}: Pocięto na {unique} segmentów.")

    print_stats("Damaged", damaged_labels)
    print_stats("Original", original_labels)

    return damaged_labels, original_labels

#%% Visualization
def display_umap(latent, labels=None):
    X = _ensure_numpy(latent)
    if labels is not None:
        labels = _ensure_numpy(labels)
        if labels.ndim == 2: labels = labels.argmax(axis=1)

    print("Rysowanie...")
    
    plt.figure(figsize=(12, 8))
    
    if labels is None:
        plt.scatter(X[:, 0], X[:, 1], s=5, c='gray')
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

        for i, lbl in enumerate(unique_labels):
            mask = (labels == lbl)
            plt.scatter(X[mask, 0], X[mask, 1], 
                        c=[colors[i]], s=10, label=f'{lbl}', alpha=1.0)
            
            center = np.mean(X[mask], axis=0)
            plt.text(center[0], center[1], str(lbl), fontsize=9, fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        
    plt.title(f'Wynik: Pocięty Wąż ({len(unique_labels)} segmentów)', fontsize=14)
    
    if len(unique_labels) <= 10:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()