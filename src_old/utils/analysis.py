"""
Funkcje do analizy przestrzeni latentnej i klasteryzacji.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from torch.utils.data import DataLoader
import umap
from typing import Tuple, Optional


def extract_latent_vectors(model: torch.nn.Module,
                          dataloader: DataLoader,
                          device: torch.device,
                          max_samples: Optional[int] = None) -> np.ndarray:
    """
    Ekstraktuje wektory latentne z trenowanego autoencodera.
    
    Args:
        model: trenowany autoencoder
        dataloader: DataLoader z danymi
        device: urządzenie (cuda/cpu)
        max_samples: maksymalna liczba próbek (None = wszystkie)
        
    Returns:
        numpy array z wektorami latentnymi [n_samples, latent_dim]
    """
    model.eval()
    latent_vectors = []
    samples_processed = 0
    
    print(f"Ekstraktowanie wektorów latentnych...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and samples_processed >= max_samples:
                break
            
            # Obsługa różnych formatów danych
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imgs = batch[0]  # Pierwszy element to obrazy
            else:
                imgs = batch
                
            imgs = imgs.to(device)
            
            # Ekstraktuj reprezentację latentną
            if hasattr(model, 'get_latent_representation'):
                z, _ = model.get_latent_representation(imgs)
            elif hasattr(model, 'encode'):
                z, _ = model.encode(imgs)  # ignore encoder features
            else:
                output_tuple = model(imgs)  # Zakładamy że model zwraca (output, latent)
                if isinstance(output_tuple, tuple):
                    _, z = output_tuple
                else:
                    z = output_tuple
            
            latent_vectors.append(z.cpu().numpy())
            samples_processed += imgs.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Przetworzono {samples_processed} próbek...")

    latent_array = np.concatenate(latent_vectors, axis=0)
    
    if max_samples:
        latent_array = latent_array[:max_samples]
    
    print(f"Wyekstraktowano {latent_array.shape[0]} wektorów latentnych "
          f"o wymiarze {latent_array.shape[1]}")
    
    return latent_array


def cluster_latent_space(latent_vectors: np.ndarray,
                        n_clusters: int = 10,
                        algorithm: str = 'kmeans',
                        random_state: int = 42,
                        **kwargs) -> Tuple[np.ndarray, dict]:
    """
    Klasteryzuje wektory latentne.
    
    Args:
        latent_vectors: array z wektorami latentnymi
        n_clusters: liczba klastrów
        algorithm: algorytm klasteryzacji ('kmeans', 'spectral', 'dbscan', 'gaussian_mixture')
        random_state: seed dla reprodukowalności
        **kwargs: dodatkowe argumenty dla algorytmu
        
    Returns:
        Tuple: (etykiety_klastrów, słownik_z_metrykami)
    """
    print(f"Klasteryzacja {algorithm.upper()} na {n_clusters} klastrów...")
    
    if algorithm == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state, 
            n_init=kwargs.get('n_init', 10)
        )
    elif algorithm == 'spectral':
        from sklearn.cluster import SpectralClustering
        clusterer = SpectralClustering(
            n_clusters=n_clusters, 
            random_state=random_state,
            affinity=kwargs.get('affinity', 'rbf')
        )
    elif algorithm == 'dbscan':
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(
            eps=kwargs.get('eps', 0.5), 
            min_samples=kwargs.get('min_samples', 5)
        )
    elif algorithm == 'gaussian_mixture':
        from sklearn.mixture import GaussianMixture
        clusterer = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            covariance_type=kwargs.get('covariance_type', 'full')
        )
    else:
        raise ValueError(f"Nieznany algorytm: {algorithm}")
    
    labels = clusterer.fit_predict(latent_vectors)
    
    # Oblicz metryki
    metrics = {}
    
    if len(set(labels)) > 1:  # Sprawdź czy są różne klastry
        # Silhouette score (tylko dla non-outlier punktów w DBSCAN)
        if algorithm == 'dbscan':
            mask = labels != -1
            if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                metrics['silhouette_score'] = silhouette_score(latent_vectors[mask], labels[mask])
            metrics['n_noise_points'] = np.sum(labels == -1)
        else:
            metrics['silhouette_score'] = silhouette_score(latent_vectors, labels)
        
        metrics['n_clusters_found'] = len(set(labels))
        
        # Davies-Bouldin Index (niższa = lepsza)
        if algorithm != 'dbscan' or np.sum(labels != -1) > 0:
            from sklearn.metrics import davies_bouldin_score
            try:
                if algorithm == 'dbscan':
                    mask = labels != -1
                    metrics['davies_bouldin'] = davies_bouldin_score(latent_vectors[mask], labels[mask])
                else:
                    metrics['davies_bouldin'] = davies_bouldin_score(latent_vectors, labels)
            except:
                pass
        
        # Calinski-Harabasz Index (wyższa = lepsza)
        from sklearn.metrics import calinski_harabasz_score
        try:
            if algorithm == 'dbscan':
                mask = labels != -1
                if len(set(labels[mask])) > 1:
                    metrics['calinski_harabasz'] = calinski_harabasz_score(latent_vectors[mask], labels[mask])
            else:
                metrics['calinski_harabasz'] = calinski_harabasz_score(latent_vectors, labels)
        except:
            pass
    
    print(f"Znaleziono {len(set(labels))} klastrów")
    if 'silhouette_score' in metrics:
        print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
    if 'davies_bouldin' in metrics:
        print(f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
    if 'calinski_harabasz' in metrics:
        print(f"Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
    
    return labels, metrics


def reduce_dimensionality(latent_vectors: np.ndarray,
                         method: str = 'umap',
                         n_components: int = 2,
                         **kwargs) -> np.ndarray:
    """
    Redukuje wymiarowość wektorów latentnych do wizualizacji.
    
    Args:
        latent_vectors: wektory latentne
        method: metoda redukcji ('umap', 'tsne', 'pca')
        n_components: docelowa liczba wymiarów
        **kwargs: dodatkowe argumenty dla algorytmu
        
    Returns:
        Array z zredukowanymi wymiarami
    """
    print(f"Redukcja wymiarowości metodą {method.upper()} do {n_components}D...")
    
    if method == 'umap':
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=kwargs.get('perplexity', 30),
            random_state=kwargs.get('random_state', 42)
        )
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Nieznana metoda: {method}")
    
    embedding = reducer.fit_transform(latent_vectors)
    print(f"Zredukowano z {latent_vectors.shape[1]}D do {embedding.shape[1]}D")
    
    return embedding


def cluster_and_visualize(latent_vectors: np.ndarray,
                         n_clusters: int = 10,
                         n_samples_viz: int = 2000,
                         experiment = None,
                         method: str = 'umap',
                         figsize: Tuple[int, int] = (12, 8)) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Kompletna analiza: klasteryzacja + redukcja wymiarowości + wizualizacja.
    
    Args:
        latent_vectors: wektory latentne
        n_clusters: liczba klastrów dla KMeans
        n_samples_viz: liczba próbek do wizualizacji
        experiment: obiekt CometML do logowania
        method: metoda redukcji wymiarowości
        figsize: rozmiar wykresu
        
    Returns:
        Tuple: (etykiety_klastrów, embedding_2d, słownik_metryk)
    """
    # Ograniczenie liczby próbek dla wizualizacji
    viz_vectors = latent_vectors[:n_samples_viz] if len(latent_vectors) > n_samples_viz else latent_vectors
    
    # Klasteryzacja na wszystkich danych
    cluster_labels, cluster_metrics = cluster_latent_space(latent_vectors, n_clusters)
    
    # Redukcja wymiarowości dla wizualizacji
    embedding = reduce_dimensionality(viz_vectors, method=method)
    
    # Przygotuj dane do wizualizacji
    viz_labels = cluster_labels[:len(viz_vectors)]
    
    # Tworzenie wykresu
    plt.figure(figsize=figsize)
    
    # Scatter plot z kolorami klastrów
    scatter = plt.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=viz_labels, 
        cmap='Spectral', 
        alpha=0.7,
        s=20
    )
    
    plt.colorbar(scatter, label='Numer klastra')
    plt.title(f'Klasteryzacja przestrzeni latentnej\n'
              f'{method.upper()} + KMeans (k={n_clusters})')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    
    # Dodaj statystyki na wykres
    if 'silhouette_score' in cluster_metrics:
        plt.text(0.02, 0.98, f"Silhouette: {cluster_metrics['silhouette_score']:.3f}", 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Logowanie do Comet ML
    if experiment:
        experiment.log_figure("latent_space_clustering", plt)
        experiment.log_metrics(cluster_metrics)
    
    plt.show()
    
    # Analiza klastrów
    print("\nAnaliza klastrów:")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = count / len(cluster_labels) * 100
        print(f"  Klaster {label}: {count} próbek ({percentage:.1f}%)")
    
    return cluster_labels, embedding, cluster_metrics


def analyze_cluster_characteristics(latent_vectors: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  feature_names: Optional[list] = None) -> dict:
    """
    Analizuje charakterystyki każdego klastra w przestrzeni latentnej.
    
    Args:
        latent_vectors: wektory latentne
        cluster_labels: etykiety klastrów
        feature_names: nazwy cech (opcjonalne)
        
    Returns:
        Słownik z charakterystykami klastrów
    """
    analysis = {}
    unique_labels = np.unique(cluster_labels)
    
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_vectors = latent_vectors[mask]
        
        analysis[f'cluster_{label}'] = {
            'size': np.sum(mask),
            'percentage': np.sum(mask) / len(cluster_labels) * 100,
            'mean': np.mean(cluster_vectors, axis=0),
            'std': np.std(cluster_vectors, axis=0),
            'centroid': np.mean(cluster_vectors, axis=0)
        }
    
    return analysis