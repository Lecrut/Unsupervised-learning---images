"""
DeepCluster Architecture - Utility Functions
Funkcje pomocnicze dedykowane dla architektury DeepCluster
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap


def evaluate_deepcluster_pipeline(model, test_loader, device, max_batches=20):
    """
    Ewaluacja kompletnego DeepCluster Pipeline
    
    Args:
        model: DeepClusterPipeline model
        test_loader: DataLoader z danymi testowymi
        device: urządzenie (cuda/cpu)
        max_batches: maksymalna liczba batchy do ewaluacji
    
    Returns:
        Dict z metrykami ewaluacji
    """
    from .metrics import calculate_psnr, calculate_ssim
    
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
                
            original_imgs = batch[0].to(device)
            
            # Użyj pełnego DeepCluster pipeline
            outputs = model(original_imgs, return_all=True)
            
            # Ewaluuj rekonstrukcję oryginału
            img_recon = outputs['img_recon']
            
            # Oblicz metryki dla batcha
            for j in range(original_imgs.size(0)):
                orig = original_imgs[j].cpu()
                recon = img_recon[j].cpu()
                
                psnr_val = calculate_psnr(orig, recon)
                ssim_val = calculate_ssim(orig, recon)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_samples += 1
    
    return {
        'psnr': total_psnr / total_samples,
        'ssim': total_ssim / total_samples,
        'samples_evaluated': total_samples,
        'architecture': 'DeepCluster',
        'stages_evaluated': ['IMG→DMG→EMC', 'PCA→ClusA', 'IMP→DEC']
    }


def extract_deepcluster_representations(model, dataloader, device, max_samples=2000):
    """
    Ekstraktuje reprezentacje latentne z EMC (Encoder Module Component)
    
    Args:
        model: DeepClusterPipeline model
        dataloader: DataLoader z danymi
        device: urządzenie (cuda/cpu)
        max_samples: maksymalna liczba próbek
    
    Returns:
        numpy array z reprezentacjami latentnymi [n_samples, latent_dim]
    """
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= max_samples:
                break
                
            images = batch[0].to(device)
            
            # Użyj EMC (Encoder Module Component) z DeepCluster
            LaSp, features = model.emc(images)
            
            latent_vectors.append(LaSp.cpu().numpy())
            samples_processed += images.size(0)
    
    return np.vstack(latent_vectors)[:max_samples]


def deepcluster_clustering_analysis(latent_vectors, algorithms=['kmeans', 'gaussian_mixture', 'spectral']):
    """
    Analiza klasteryzacji używając komponentów DeepCluster (PCA + ClusA)
    
    Args:
        latent_vectors: reprezentacje latentne z EMC
        algorithms: lista algorytmów do przetestowania
    
    Returns:
        Dict z wynikami klasteryzacji dla każdego algorytmu
    """
    from ..models import PCAModule, ClusA
    
    results = {}
    
    for algo in algorithms:
        print(f"🔍 Testowanie DeepCluster ClusA: {algo.upper()}")
        
        try:
            # PCAModule z DeepCluster
            pca_module = PCAModule(n_components=50)
            pca_vectors = pca_module.fit_transform(latent_vectors)
            
            # ClusA z DeepCluster  
            clusa = ClusA(n_clusters=10, algorithm=algo)
            clusa.fit(pca_vectors)
            labels = clusa.predict(pca_vectors)
            
            # Metryki
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(pca_vectors, labels)
                n_clusters_found = len(np.unique(labels))
            else:
                sil_score = -1.0
                n_clusters_found = 1
            
            results[algo] = {
                'labels': labels,
                'pca_vectors': pca_vectors,
                'silhouette_score': sil_score,
                'n_clusters_found': n_clusters_found,
                'pca_components': pca_module.n_components,
                'original_dim': latent_vectors.shape[1],
                'reduced_dim': pca_vectors.shape[1]
            }
            
            print(f"   ✓ PCA: {latent_vectors.shape[1]} → {pca_vectors.shape[1]} wymiarów")
            print(f"   ✓ ClusA ({algo}): {n_clusters_found} klastrów, Silhouette: {sil_score:.4f}")
            
        except Exception as e:
            print(f"   ✗ Błąd w {algo}: {e}")
            results[algo] = None
    
    return results


def visualize_deepcluster_clustering(clustering_results, title_prefix="DeepCluster"):
    """
    Wizualizacja wyników klasteryzacji DeepCluster
    
    Args:
        clustering_results: wyniki z deepcluster_clustering_analysis
        title_prefix: prefiks dla tytułów wykresów
    """
    valid_results = {k: v for k, v in clustering_results.items() if v is not None}
    
    if not valid_results:
        print("⚠️ Brak wyników do wizualizacji")
        return
    
    # Znajdź najlepszy algorytm
    best_algo_name = max(valid_results.keys(), 
                        key=lambda x: valid_results[x]['silhouette_score'])
    best_result = valid_results[best_algo_name]
    
    print(f"\n🏆 Najlepszy algorytm: {best_algo_name.upper()}")
    print(f"   Silhouette Score: {best_result['silhouette_score']:.4f}")
    
    # Wizualizacja UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(best_result['pca_vectors'])
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                        c=best_result['labels'], cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    
    plt.title(f'{title_prefix} UMAP - {best_algo_name.upper()}\n'
             f'PCA: {best_result["original_dim"]} → {best_result["reduced_dim"]} dim, '
             f'Silhouette: {best_result["silhouette_score"]:.4f}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_algo_name, best_result


def deepcluster_stage_analysis(model, sample_batch, device):
    """
    Analiza wszystkich 3 etapów DeepCluster na przykładowym batchu
    
    Args:
        model: DeepClusterPipeline
        sample_batch: batch obrazów do analizy
        device: urządzenie
    
    Returns:
        Dict z wynikami wszystkich etapów
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        original_imgs = sample_batch[0][:4].to(device)
        
        # ETAP I: IMG → DMG → EMC
        LaSp, LaSp_d, features, features_d, img_d = model.encode_and_cluster(original_imgs)
        
        results['stage_1'] = {
            'name': 'IMG → DMG → EMC',
            'LaSp_shape': LaSp.shape,
            'LaSp_d_shape': LaSp_d.shape,
            'damaged_img_shape': img_d.shape,
            'features_count': len(features)
        }
        
        # ETAP II: PCA → ClusA  
        LaSp_np = LaSp.cpu().numpy()
        LaSp_d_np = LaSp_d.cpu().numpy()
        
        K = model.cluster_latent(LaSp_np, LaSp_d_np)
        K_tensor = torch.tensor(K, device=device).long()
        
        results['stage_2'] = {
            'name': 'PCA → ClusA',
            'pca_components': model.pca.n_components,
            'cluster_assignments_shape': K_tensor.shape,
            'unique_clusters': len(torch.unique(K_tensor)),
            'clusters_found': K.tolist()
        }
        
        # ETAP III: IMP → DEC
        img_fixed, LaSp_fixed = model.inpaint_and_decode(LaSp_d, K_tensor, features_d)
        img_reconstructed = model.dec(LaSp, features)
        
        results['stage_3'] = {
            'name': 'IMP → DEC',
            'LaSp_fixed_shape': LaSp_fixed.shape,
            'img_fixed_shape': img_fixed.shape,
            'img_reconstructed_shape': img_reconstructed.shape
        }
    
    # Podsumowanie
    results['summary'] = {
        'architecture': 'DeepCluster',
        'total_stages': 3,
        'input_shape': original_imgs.shape,
        'latent_dimension': LaSp.shape[1],
        'n_clusters': len(torch.unique(K_tensor)),
        'all_stages_completed': True
    }
    
    return results


def print_deepcluster_summary(results):
    """
    Wyświetla podsumowanie analizy DeepCluster
    """
    print("\n" + "="*70)
    print("🏗️  DEEPCLUSTER ARCHITECTURE SUMMARY")
    print("="*70)
    
    print(f"📥 Input: {results['summary']['input_shape']}")
    print(f"🧠 Latent Dimension: {results['summary']['latent_dimension']}")
    print(f"🎯 Number of Clusters: {results['summary']['n_clusters']}")
    print()
    
    for stage_key in ['stage_1', 'stage_2', 'stage_3']:
        stage = results[stage_key]
        print(f"📋 {stage['name']}")
        
        if stage_key == 'stage_1':
            print(f"   • Original representation: {stage['LaSp_shape']}")
            print(f"   • Damaged representation: {stage['LaSp_d_shape']}")
            print(f"   • Damaged image: {stage['damaged_img_shape']}")
            
        elif stage_key == 'stage_2':
            print(f"   • PCA components: {stage['pca_components']}")
            print(f"   • Cluster assignments: {stage['cluster_assignments_shape']}")
            print(f"   • Unique clusters: {stage['unique_clusters']}")
            
        elif stage_key == 'stage_3':
            print(f"   • Fixed representation: {stage['LaSp_fixed_shape']}")
            print(f"   • Fixed image: {stage['img_fixed_shape']}")
            print(f"   • Reconstructed image: {stage['img_reconstructed_shape']}")
        
        print()
    
    print("✅ DeepCluster Architecture - All Stages Completed!")
    print("🎯 Compliant with Project Requirements")
    print("="*70)