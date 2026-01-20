import numpy as np
import matplotlib.pyplot as plt
import umap
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm


def preprocess_spatial_latents(latents):
    if isinstance(latents, np.ndarray):
        features = latents
    else:
        latents = latents.float().cpu()
        features = latents.numpy()
    
    if features.ndim != 2:
        raise ValueError(f"Oczekiwano latentów 2D [N, D], otrzymano: {list(features.shape)}")
        
    print(f"[Preprocessing] Otrzymano latenty: {list(features.shape)}")
    print("[Preprocessing] Normalizacja L2...")
    
    features = normalize(features, norm='l2', axis=1)
    
    return features

def run_auto_pca(features, variance_threshold=0.95):
    print(f"[PCA] Redukcja wymiarów (cel: zachowanie {variance_threshold*100}% wariancji)...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=variance_threshold)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f" -> Zredukowano z {features.shape[1]} do {features_pca.shape[1]} wymiarów.")
    return features_pca, pca, scaler


def find_optimal_clusters(data, min_k=3, max_k=15):
    print(f"[GMM] Szukanie najlepszej liczby klastrów (zakres {min_k}-{max_k})...")
    
    best_bic = np.inf
    best_model = None
    best_k = min_k
    
    k_range = range(min_k, max_k + 1)
    if len(k_range) > 1:
        k_range = tqdm(k_range, desc="Optymalizacja GMM")

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(data)
        bic = gmm.bic(data)
        
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_k = k
            
    print(f" -> Wybrano {best_k} klastrów (BIC: {best_bic:.0f})")
    
    labels = best_model.predict(data)
    return labels, best_model, best_k

def plot_umap(features, labels, n_clusters):
    print("[UMAP] Generowanie wizualizacji...")
    
    reducer = umap.UMAP(
        n_neighbors=30,    
        min_dist=0.1,      
        n_components=2,
        metric='cosine',   
        random_state=42
    )
    embedding = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=15, alpha=0.8)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'UMAP Projection ({n_clusters} clusters found)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()
    
    return reducer, embedding

def run_clustering_pipeline(latents, min_clusters=3, max_clusters=15):
  
    
    clean_features = preprocess_spatial_latents(latents)
    
    pca_features, pca_model, scaler = run_auto_pca(clean_features, variance_threshold=0.95)
    
    labels, gmm_model, optimal_k = find_optimal_clusters(pca_features, 
                                                         min_k=min_clusters, 
                                                         max_k=max_clusters)
    
    umap_model, embedding = plot_umap(pca_features, labels, optimal_k)
    
    return {
        "labels": labels,          
        "n_clusters": optimal_k,    
        "features_pca": pca_features,
        "embedding": embedding,
        "models": {
            "pca": pca_model,
            "scaler": scaler,
            "gmm": gmm_model,
            "umap": umap_model
        }
    }

def plot_clustered_images(images_loader, labels, samples_per_cluster=5):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    target_indices = {}
    
    print("Losowanie próbek...")
    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        
        count = min(samples_per_cluster, len(cluster_indices))
        if count > 0:
            selected = np.random.choice(cluster_indices, size=count, replace=False)
            for col_idx, global_idx in enumerate(selected):
                target_indices[global_idx] = (cluster_id, col_idx)

    collected_images = {} 

    print("Pobieranie obrazów z Dataloadera (One-Pass)...")
    
    global_idx = 0
    max_target = max(target_indices.keys()) if target_indices else 0
    
    with torch.no_grad():
        for batch in images_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            batch_size = imgs.shape[0]
            batch_end = global_idx + batch_size
            
            relevant_targets = [t for t in target_indices.keys() if global_idx <= t < batch_end]
            
            for t_idx in relevant_targets:
                local_idx = t_idx - global_idx 
                
                img_tensor = imgs[local_idx]
                
                img = img_tensor.permute(1, 2, 0).cpu().numpy()
                
                img = (img - img.min()) / (img.max() - img.min())
                
                cluster_id, col_idx = target_indices[t_idx]
                if cluster_id not in collected_images:
                    collected_images[cluster_id] = {}
                collected_images[cluster_id][col_idx] = img
            
            global_idx += batch_size
            
            if global_idx > max_target:
                break

    print("Generowanie wykresu...")
    plt.figure(figsize=(samples_per_cluster * 2.5, n_clusters * 2.5))
    
    for c_id in unique_labels:
        if c_id not in collected_images: continue
        
        for col in range(samples_per_cluster):
            idx = c_id * samples_per_cluster + col + 1
            plt.subplot(n_clusters, samples_per_cluster, idx)
            
            if col in collected_images[c_id]:
                plt.imshow(collected_images[c_id][col])
            else:
                plt.text(0.5, 0.5, "Brak", ha='center')
                
            plt.axis('off')
            
            if col == 0:
                plt.ylabel(f'Cluster {c_id}', fontsize=12, fontweight='bold')

    plt.suptitle('Sample Images from Each Cluster', fontsize=16)
    plt.tight_layout()
    plt.show()