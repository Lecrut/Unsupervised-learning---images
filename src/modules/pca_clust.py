#%% Imports
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
# from kneed import KneeLocator
from sklearn.preprocessing import normalize

#%% Helper Function
def _ensure_numpy(data):
    if torch.is_tensor(data):
        return data.cpu().detach().numpy()
    return np.asarray(data)

# def our_pca(latent_damaged, latent_original, n_components=2):
#     X = _ensure_numpy(latent_damaged)
#     Y = _ensure_numpy(latent_original)
#     combined = np.vstack([X, Y])

#     scaler = StandardScaler()
#     combined_scaled = scaler.fit_transform(combined)

#     print(f"Redukcja UMAP do {n_components} wymiarów (tryb wizualny)...")
    
#     reducer = umap.UMAP(
#         n_components=n_components,
#         n_neighbors=15,    
#         min_dist=0.1,        
#         metric='cosine',     
#         init='spectral',     
#         n_jobs=-1,
#         random_state=42
#     )
    
#     transformed = reducer.fit_transform(combined_scaled)

#     min_max = MinMaxScaler()
#     transformed = min_max.fit_transform(transformed)

#     n_damaged = len(X)
#     return transformed[:n_damaged], transformed[n_damaged:]

#%% Normalize function
def normalize_l2(latents):
    if torch.is_tensor(latents):
        latents = latents.cpu().numpy()
    
    X_norm = normalize(latents, norm='l2', axis=1)

    return X_norm

#%% umap reduction
def umap_reduction(latents, n_components=15, n_neighbors=30):
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    reduced_data = reducer.fit_transform(latents)
    return reducer, reduced_data


#%% Find optimal number of clusters
def find_optimal_k(X, max_k=30, max_probs=10000):
    print("Redukcja wymiarowości PCA przed klasteryzacją...")
    pca = PCA(n_components=0.95) 
    X_pca = pca.fit_transform(X)
    print(f"Zredukowano wymiary z {X.shape[1]} do {X_pca.shape[1]}")

    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)

    if X_pca.shape[0] > max_probs:
        idx = np.random.choice(X_pca.shape[0], 10000, replace=False)
        X_calc = X_pca[idx]
    else:
        X_calc = X_pca

    print(f"Analiza optymalnej liczby klastrów (2 do {max_k})...")
    for k in tqdm(K_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(X_calc)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_calc, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Liczba klastrów (k)')
    ax1.set_ylabel('Inertia (Suma odległości)')
    ax1.set_title('Metoda Łokcia (Im mniej tym lepiej)')
    ax1.grid(True)
    
    ax2.plot(K_range, silhouettes, 'ro-')
    ax2.set_xlabel('Liczba klastrów (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Jakość separacji (Im więcej tym lepiej)')
    ax2.grid(True)
    
    plt.show()
    # return X_pca

def find_optimal_clusters_gmm(latents, max_k=20):
    if torch.is_tensor(latents):
        latents = latents.cpu().numpy()
    X_norm = normalize(latents, norm='l2', axis=1)
    
    reducer = UMAP(n_components=15, n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
    X_reduced = reducer.fit_transform(X_norm)
    
    if X_reduced.shape[0] > 10000:
        idx = np.random.choice(X_reduced.shape[0], 10000, replace=False)
        data_for_metrics = X_reduced[idx]
    else:
        data_for_metrics = X_reduced

    bics = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in tqdm(K_range):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=3)
        gmm.fit(data_for_metrics)
        
        bics.append(gmm.bic(data_for_metrics))
        labels = gmm.predict(data_for_metrics)
        silhouettes.append(silhouette_score(data_for_metrics, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(K_range, bics, 'bo-')
    ax1.set_xlabel('Liczba klastrów (k)')
    ax1.set_ylabel('BIC Score (im mniej tym lepiej)')
    ax1.set_title('BIC Score (Szukaj minimum lub "łokcia")')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(K_range, silhouettes, 'ro-')
    ax2.set_xlabel('Liczba klastrów (k)')
    ax2.set_ylabel('Silhouette Score (im więcej tym lepiej)')
    ax2.set_title('Silhouette Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


#%% PCA + KMeans
def pca_kmeans_reduction(latents, num_clusters):
    print(f"Redukcja PCA i trening K-Means na {num_clusters} klastrów...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(latents)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca) 

    print("Obliczanie projekcji UMAP...")
    umap_reducer = UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    embedding = umap_reducer.fit_transform(X_pca)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab20', s=10, alpha=0.6)

    for i in range(num_clusters):
        cluster_points = embedding[cluster_labels == i]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            plt.text(center[0], center[1], str(i), fontsize=12, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.colorbar(scatter, ticks=range(num_clusters), label='ID Klastra')
    plt.title(f"Mapa Przestrzeni Latent ({num_clusters} wymuszonych stylów)", fontsize=16)
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.show()

    return pca, kmeans, cluster_labels


#%% Gaussian Mixture Model:
def fit_gmm_clustering(latents_reduced, num_clusters):
    gmm = GaussianMixture(
        n_components=num_clusters,
        covariance_type='full',
        random_state=42,
        n_init=3
    )

    labels = gmm.fit_predict(latents_reduced)
    return gmm, labels

#%% Helper: Visualize Clusters on 2D
def plot_clusters_2d(latents_reduced, labels, num_clusters):
    print("-> Generowanie wykresu 2D...")
    
    if latents_reduced.shape[1] > 2:
        vis_reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
        embedding = vis_reducer.fit_transform(latents_reduced)
    else:
        embedding = latents_reduced

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=8, alpha=0.6)

    # Rysowanie numerków klastrów
    for i in range(num_clusters):
        points = embedding[labels == i]
        if len(points) > 0:
            center = points.mean(axis=0)
            plt.text(center[0], center[1], str(i), fontsize=12, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    plt.colorbar(scatter, ticks=range(num_clusters), label='ID Klastra')
    plt.title(f"Wynik klasteryzacji ({num_clusters} grup)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()


#%% Visualization of images in classes
def visualize_images_kmeans(model, dataloader, kmeans_model, pca_model, device, cluster_labels, num_clusters, samples=5):
    model.eval()
    cluster_gallery = {i: [] for i in range(num_clusters)}
    
    print("Przeszukiwanie obrazów do galerii...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            
            # Ekstrakcja -> PCA -> Predykcja klastra
            latents, _ = model.encoder(imgs)
            latents_np = latents.cpu().numpy()
            latents_pca = pca_model.transform(latents_np)
            labels = kmeans_model.predict(latents_pca)
            
            # Zbieranie próbek
            for i, label in enumerate(labels):
                if len(cluster_gallery[label]) < samples:
                    img_tensor = imgs[i].cpu()
                    # Denormalizacja do [0, 1]
                    img_vis = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
                    cluster_gallery[label].append(img_vis.permute(1, 2, 0).numpy())
            
            # Warunek stopu (gdy mamy komplet)
            if all(len(v) == samples for v in cluster_gallery.values()):
                break
    
    for cluster_id in range(num_clusters):
        images = cluster_gallery[cluster_id]
        if not images: continue
            
        plt.figure(figsize=(15, 3))
        count = np.sum(cluster_labels == cluster_id)
        percent = (count / len(cluster_labels)) * 100
        plt.suptitle(f"Styl #{cluster_id} ({count} obrazów, {percent:.1f}%)", fontsize=14, x=0.1)
        
        for i, img in enumerate(images):
            plt.subplot(1, samples, i+1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

#%% Visualization of GMM process:
def visualize_images_gmm(model, dataloader, umap_model, gmm_model, device, num_clusters, samples=5):
    model.eval()
    cluster_gallery = {i: [] for i in range(num_clusters)}

    with torch.no_grad():
        for batch in dataloader:
            # Obsługa batcha
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            
            # 1. Ekstrakcja z Autoencodera
            latents, _ = model.encoder(imgs)
            
            # 2. Normalizacja L2 (Musi być na CPU numpy)
            latents_np = latents.cpu().numpy()
            latents_norm = normalize(latents_np, norm='l2', axis=1)
            
            # 3. Transformacja UMAP (używamy wytrenowanego modelu!)
            latents_reduced = umap_model.transform(latents_norm)
            
            # 4. Predykcja GMM
            labels = gmm_model.predict(latents_reduced)
            
            # 5. Zbieranie obrazków
            for i, label in enumerate(labels):
                if len(cluster_gallery[label]) < samples:
                    img_tensor = imgs[i].cpu()
                    # Denormalizacja do [0,1] dla wyświetlania
                    img_vis = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
                    cluster_gallery[label].append(img_vis.permute(1, 2, 0).numpy())
            
            # Warunek stopu
            if all(len(v) == samples for v in cluster_gallery.values()):
                break
    
    for cluster_id in range(num_clusters):
        images = cluster_gallery[cluster_id]
        if not images: continue
            
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Styl #{cluster_id}", fontsize=14, x=0.1)
        for i, img in enumerate(images):
            plt.subplot(1, samples, i+1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()