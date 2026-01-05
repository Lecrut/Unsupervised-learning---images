#%% Imports
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from kneed import KneeLocator

#%% Helper Function
def _ensure_numpy(data):
    if torch.is_tensor(data):
        return data.cpu().detach().numpy()
    return np.asarray(data)

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

#%% Visualization UMAP
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


#%% Funkcja klasteryzująca ale z PCA przed (tez będzie do wywalenia)
def clustering_with_pca(latent_damaged, latent_original, n_clusters=20, n_components=100):
    X = _ensure_numpy(latent_damaged)
    Y = _ensure_numpy(latent_original)
    combined = np.vstack([X, Y])
    
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    
    print(f"Redukcja PCA: {X.shape[1]}D -> {n_components}D")
    pca = PCA(n_components=n_components, random_state=42)
    combined_pca = pca.fit_transform(combined_scaled)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        n_init=50,
        max_iter=500,
        random_state=42
    )
    
    labels_all = kmeans.fit_predict(combined_pca)
    
    damaged_labels = labels_all[:len(X)]
    original_labels = labels_all[len(X):]
    
    cluster_info = {
        'centroids': kmeans.cluster_centers_,
        'scaler': scaler,
        'pca': pca,
        'inertia': kmeans.inertia_,
        'n_features': n_components
    }
    
    return damaged_labels, original_labels, cluster_info

#%% szuka optymalnego K dla zbioru (wykorzystywana w find_optimal_clusters_with_pca)
def find_optimal_clusters(latent_vectors, min_clusters=3, max_clusters=15, sample_size=10000):
    """
    Automatyczne znajdowanie optymalnej liczby klastrów.
    Używa Elbow Method (inertia) i Silhouette Score.
    
    Args:
        latent_vectors: wektory latentne do klasteryzacji
        min_clusters: minimalna liczba klastrów do przetestowania
        max_clusters: maksymalna liczba klastrów do przetestowania
        sample_size: ile próbek użyć do obliczenia Silhouette Score (dla szybkości)
    
    Returns:
        int: rekomendowana liczba klastrów
    """
    X = _ensure_numpy(latent_vectors)
    
    print(f"Szukanie optymalnej liczby klastrów ({min_clusters}-{max_clusters})...")
    print(f"Próbek: {X.shape[0]}, wymiarów: {X.shape[1]}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouettes = []
    K_range = range(min_clusters, max_clusters + 1)
    
    for k in K_range:
        print(f"  Testowanie k={k}...", end=" ")
        
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=4096,
            n_init=20,
            max_iter=300,
            random_state=42
        )
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        
        sample_indices = np.random.choice(X_scaled.shape[0], min(sample_size, X_scaled.shape[0]), replace=False)
        sil_score = silhouette_score(X_scaled[sample_indices], labels[sample_indices], metric='euclidean')
        silhouettes.append(sil_score)
        
        print(f"Inertia: {kmeans.inertia_:.0f}, Silhouette: {sil_score:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Liczba klastrów', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method - im niższa tym lepiej', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(K_range)
    
    best_k_silhouette = K_range[np.argmax(silhouettes)]
    axes[1].plot(K_range, silhouettes, marker='o', color='orange', linewidth=2, markersize=8)
    axes[1].axvline(best_k_silhouette, color='red', linestyle='--', linewidth=2, label=f'Najlepsze: k={best_k_silhouette}')
    axes[1].set_xlabel('Liczba klastrów', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score - im wyższy tym lepiej', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(K_range)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    inertia_diffs = np.diff(inertias)
    elbow_scores = np.abs(np.diff(inertia_diffs))
    best_k_elbow = K_range[np.argmax(elbow_scores) + 1] if len(elbow_scores) > 0 else best_k_silhouette
    
    print("\n" + "="*60)
    print("REKOMENDACJE:")
    print("="*60)
    print(f"Najlepsze k wg Silhouette Score: {best_k_silhouette} (score: {silhouettes[best_k_silhouette - min_clusters]:.4f})")
    print(f"Najlepsze k wg Elbow Method: {best_k_elbow}")
    
    recommended_k = best_k_silhouette
    print(f"\nREKOMENDOWANA LICZBA KLASTRÓW: {recommended_k}")
    print("="*60 + "\n")
    
    return recommended_k


#%% szuka optymalnego K, tylko że PCA przed
def find_optimal_clusters_with_pca(latent_damaged_vectors, latent_original_vectors, 
                                    n_components=100, min_clusters=5, max_clusters=30, sample_size=10000):
    """
    Najpierw redukuje wymiary przez PCA, potem znajduje optymalne klastry.
    
    Args:
        latent_damaged_vectors: wektory uszkodzonych obrazów
        latent_original_vectors: wektory oryginalnych obrazów
        n_components: liczba komponentów PCA
        min_clusters: minimalna liczba klastrów
        max_clusters: maksymalna liczba klastrów
        sample_size: liczba próbek do Silhouette Score
    
    Returns:
        tuple: (optimal_k, pca_damaged, pca_original, pca_model, scaler)
    """
    X = _ensure_numpy(latent_damaged_vectors)
    Y = _ensure_numpy(latent_original_vectors)
    combined = np.vstack([X, Y])
    
    print(f"Redukcja PCA: {X.shape[1]}D -> {n_components}D przed analizą klastrów...")
    
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    
    pca = PCA(n_components=n_components, random_state=42)
    combined_pca = pca.fit_transform(combined_scaled)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    pca_damaged = combined_pca[:len(X)]
    pca_original = combined_pca[len(X):]
    
    optimal_k = find_optimal_clusters(
        pca_damaged,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        sample_size=sample_size
    )
    
    return optimal_k, pca_damaged, pca_original, pca, scaler

#%% skleja dwa zbiory
def vstack_data(latent_damaged_vectors, latent_original_vectors):
    X = _ensure_numpy(latent_damaged_vectors)
    Y = _ensure_numpy(latent_original_vectors)

    latents = np.vstack([X, Y])
    return latents

#%% PCA:
def use_pca(latents, n_components=100):
    # latents = vstack_data(latent_damaged_vectors, latent_original_vectors)

    #skalujemy dane
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    pca = PCA(n_components=n_components, random_state=42)
    latents_pca = pca.fit_transform(latents_scaled)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # pca_damaged = latent_pca[:len(X)]
    # pca_original = latent_pca[len(X):]

    return latents_pca


def find_optimal_k(latent_vectors, min_clusters=3, max_clusters=20, sample_size=10000):
    """
    Szuka optymalnego k, zwraca wartość dla metody łokciowej oraz silhouette_score.
    """
    X = _ensure_numpy(latent_vectors)
    
    print(f"Szukanie optymalnej liczby klastrów ({min_clusters}-{max_clusters})...")
    print(f"Próbek: {X.shape[0]}, wymiarów: {X.shape[1]}")
    
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouettes = []
    K_range = range(min_clusters, max_clusters + 1)
    
    for k in K_range:
        print(f"  Testowanie k={k}...", end=" ")
        
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=4096,
            n_init=20,
            max_iter=300,
            random_state=42
        )
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        
        sample_indices = np.random.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
        sil_score = silhouette_score(X[sample_indices], labels[sample_indices], metric='euclidean')
        silhouettes.append(sil_score)
        
        print(f"Inertia: {kmeans.inertia_:.0f}, Silhouette: {sil_score:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Liczba klastrów', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method - im niższa tym lepiej', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(K_range)
    
    best_k_silhouette = K_range[np.argmax(silhouettes)]
    axes[1].plot(K_range, silhouettes, marker='o', color='orange', linewidth=2, markersize=8)
    axes[1].axvline(best_k_silhouette, color='red', linestyle='--', linewidth=2, label=f'Najlepsze: k={best_k_silhouette}')
    axes[1].set_xlabel('Liczba klastrów', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score - im wyższy tym lepiej', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(K_range)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    inertia_diffs = np.diff(inertias)
    elbow_scores = np.abs(np.diff(inertia_diffs))
    best_k_elbow = K_range[np.argmax(elbow_scores) + 1] if len(elbow_scores) > 0 else best_k_silhouette
    
    print("REKOMENDACJE:")
    print(f"Najlepsze k wg Silhouette Score: {best_k_silhouette} (score: {silhouettes[best_k_silhouette - min_clusters]:.4f})")
    print(f"Najlepsze k wg Elbow Method: {best_k_elbow}")
    
    # recommended_k = best_k_silhouette
    # print(f"\nREKOMENDOWANA LICZBA KLASTRÓW: {recommended_k}")
    
    return best_k_elbow, best_k_silhouette

#%% finalna funkcja klasteryzacji wykonująca kMeans
def cluster_kmeans(latent_vectors, optimal_k):
    """
    Klasteryzacja zbioru przy pomocy kMeans. Zwraca etykiety dla każdej z próbek oraz model kMeans.
    """
    data = _ensure_numpy(latent_vectors)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(data)
    
    print(f"Klasteryzacja zakończona:")
    print(f"  Liczba próbek: {len(data)}")
    print(f"  Liczba klastrów: {optimal_k}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nRozkład klastrów:")
    for label, count in zip(unique, counts):
        print(f"  Klaster {label}: {count} próbek")
    
    cluster_info = {
        'centroids': kmeans.cluster_centers_,
        # 'scaler': scaler,
        # 'pca': pca,
        'inertia': kmeans.inertia_,
        'n_features': optimal_k
    }
    
    return labels, kmeans, cluster_info

#%% Helper do analizy klastrów (zostawić)
def analyze_clusters(cluster_info, labels):
    """
    Analiza jakości klasteryzacji.
    
    Args:
        cluster_info: dict z informacjami o klastrach (z funkcji clustering)
        labels: etykiety klastrów dla próbek
    """
    labels = _ensure_numpy(labels)
    unique_labels = np.unique(labels)
    
    print("\n" + "="*60)
    print("ANALIZA KLASTRÓW")
    print("="*60)
    print(f"Liczba klastrów: {len(unique_labels)}")
    print(f"Wymiarowość: {cluster_info['n_features']}D")
    print(f"Inertia (niższa = lepiej): {cluster_info['inertia']:.2f}")
    
    print("\nRozkład próbek w klastrach:")
    for label in unique_labels:
        count = np.sum(labels == label)
        percentage = (count / len(labels)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  Klaster {label:2d}: {count:5d} próbek ({percentage:5.2f}%) {bar}")
    
    counts = [np.sum(labels == label) for label in unique_labels]
    print(f"\nStatystyki:")
    print(f"  Średnia: {np.mean(counts):.1f} próbek/klaster")
    print(f"  Mediana: {np.median(counts):.1f} próbek/klaster")
    print(f"  Min: {np.min(counts)} | Max: {np.max(counts)}")
    print(f"  Odchylenie std: {np.std(counts):.1f}")
    print("="*60 + "\n")



#%% BOTOWA FUNKCJA - narazie zostawić ale ogólnie do usunięcia będzie
def pca_clustering_pipeline(latent_damaged_vectors, latent_original_vectors, 
                             n_components=100, min_clusters=5, max_clusters=30):
    """
    Pipeline: PCA -> znajdz optymalne K -> klasteryzacja.
    
    Args:
        latent_damaged_vectors: wektory uszkodzonych obrazów (N, 768)
        latent_original_vectors: wektory oryginalnych obrazów (N, 768)
        n_components: liczba komponentów PCA
        min_clusters: minimalna liczba klastrów do testowania
        max_clusters: maksymalna liczba klastrów do testowania
        
    Returns:
        tuple: (clusters_damaged, clusters_original, optimal_k, pca_damaged, pca_original)
    """
    
    X = _ensure_numpy(latent_damaged_vectors)
    Y = _ensure_numpy(latent_original_vectors)
    combined = np.vstack([X, Y])
    
    print(f"[1/3] PCA: {X.shape[1]}D -> {n_components}D")
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    
    pca = PCA(n_components=n_components, random_state=42)
    combined_pca = pca.fit_transform(combined_scaled)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    pca_damaged = combined_pca[:len(X)]
    pca_original = combined_pca[len(X):]
    
    print(f"\n[2/3] Szukanie optymalnego K (metoda łokcia)...")
    inertias = []
    K_range = list(range(min_clusters, max_clusters + 1))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(pca_damaged)
        inertias.append(kmeans.inertia_)
        if k % 5 == 0:
            print(f"  K={k}: inertia={kmeans.inertia_:.2f}")
    
    diffs = np.diff(inertias)
    diffs_ratio = np.abs(diffs[1:] / diffs[:-1])
    optimal_idx = np.argmax(diffs_ratio) + 2
    optimal_k = K_range[optimal_idx]
    
    print(f"\nOptymalne K: {optimal_k}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Inertia')
    plt.title('Metoda łokcia')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\n[3/3] Klasteryzacja z K={optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    
    combined_pca_full = np.vstack([pca_damaged, pca_original])
    labels = kmeans_final.fit_predict(combined_pca_full)
    
    clusters_damaged = labels[:len(pca_damaged)]
    clusters_original = labels[len(pca_damaged):]
    
    print(f"Klasteryzacja zakończona:")
    print(f"  Liczba klastrów: {optimal_k}")
    print(f"  Inertia: {kmeans_final.inertia_:.2f}")
    
    unique, counts = np.unique(clusters_damaged, return_counts=True)
    print(f"\nRozkład klastrów:")
    for label, count in zip(unique, counts):
        print(f"  Klaster {label}: {count} próbek")
    
    return clusters_damaged, clusters_original, optimal_k, pca_damaged, pca_original

