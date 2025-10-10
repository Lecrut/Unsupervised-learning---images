"""
Funkcje do wizualizacji wyników autoencodera i analizy.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List
import seaborn as sns


def visualize_reconstructions(model: torch.nn.Module,
                            dataloader: DataLoader,
                            device: torch.device,
                            n_samples: int = 8,
                            figsize: Tuple[int, int] = (16, 6),
                            save_path: Optional[str] = None) -> None:
    """
    Wizualizuje przykładowe rekonstrukcje autoencodera.
    
    Args:
        model: trenowany autoencoder
        dataloader: DataLoader z danymi (masked, target)
        device: urządzenie
        n_samples: liczba przykładów do pokazania
        figsize: rozmiar wykresu
        save_path: ścieżka do zapisania wykresu
    """
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                masked_imgs, target_imgs = batch[0], batch[1]
            else:
                # Jeśli tylko obrazy bez masek, zastosuj prostą maskę
                imgs = batch
                masked_imgs = imgs.clone()
                # Dodaj prostą maskę 32x32 w centrum
                _, _, h, w = imgs.shape
                mask_size = 32
                start_h, start_w = (h - mask_size) // 2, (w - mask_size) // 2
                masked_imgs[:, :, start_h:start_h+mask_size, start_w:start_w+mask_size] = 1.0
                target_imgs = imgs
            
            masked_imgs, target_imgs = masked_imgs.to(device), target_imgs.to(device)
            outputs, _ = model(masked_imgs)
            
            # Ograniczenie do n_samples
            n_show = min(n_samples, masked_imgs.size(0))
            
            # Konwersja do numpy i normalizacja
            masked = masked_imgs[:n_show].cpu().permute(0, 2, 3, 1).numpy()
            targets = target_imgs[:n_show].cpu().permute(0, 2, 3, 1).numpy()
            reconstructed = outputs[:n_show].detach().cpu().permute(0, 2, 3, 1).numpy()
            
            # Klampowanie wartości do [0, 1]
            masked = np.clip(masked, 0, 1)
            targets = np.clip(targets, 0, 1)
            reconstructed = np.clip(reconstructed, 0, 1)
            
            # Tworzenie wykresu
            fig, axes = plt.subplots(3, n_show, figsize=figsize)
            if n_show == 1:
                axes = axes.reshape(3, 1)
            
            for i in range(n_show):
                # Uszkodzony obraz
                axes[0, i].imshow(masked[i])
                axes[0, i].set_title("Uszkodzony", fontsize=10)
                axes[0, i].axis('off')
                
                # Oryginalny obraz
                axes[1, i].imshow(targets[i])
                axes[1, i].set_title("Oryginalny", fontsize=10)
                axes[1, i].axis('off')
                
                # Rekonstrukcja
                axes[2, i].imshow(reconstructed[i])
                axes[2, i].set_title("Rekonstrukcja", fontsize=10)
                axes[2, i].axis('off')
                
                # Oblicz i pokaż błąd
                mse = np.mean((targets[i] - reconstructed[i]) ** 2)
                axes[2, i].text(0.5, -0.1, f'MSE: {mse:.4f}', 
                              transform=axes[2, i].transAxes, 
                              ha='center', fontsize=8)
            
            plt.suptitle("Porównanie: Uszkodzony → Oryginalny → Rekonstrukcja", fontsize=14)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 Zapisano wizualizację: {save_path}")
            
            plt.show()
            break  # Tylko pierwszy batch


def plot_training_history(losses: List[float],
                         val_losses: Optional[List[float]] = None,
                         title: str = "Historia trenowania",
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Wizualizuje historię strat podczas trenowania.
    
    Args:
        losses: lista strat treningowych
        val_losses: lista strat walidacyjnych (opcjonalna)
        title: tytuł wykresu
        figsize: rozmiar wykresu
        save_path: ścieżka do zapisania
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'b-', label='Strata treningowa', linewidth=2)
    
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Strata walidacyjna', linewidth=2)
    
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dodaj adnotacje
    min_loss_epoch = np.argmin(losses) + 1
    min_loss_value = min(losses)
    plt.annotate(f'Min: {min_loss_value:.4f}', 
                xy=(min_loss_epoch, min_loss_value),
                xytext=(min_loss_epoch + len(losses)*0.1, min_loss_value + max(losses)*0.1),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Zapisano wykres strat: {save_path}")
    
    plt.show()


def visualize_latent_space_2d(embedding: np.ndarray,
                             labels: np.ndarray,
                             title: str = "Przestrzeń latentna 2D",
                             figsize: Tuple[int, int] = (12, 8),
                             save_path: Optional[str] = None) -> None:
    """
    Wizualizuje przestrzeń latentną w 2D z kolorami klastrów.
    
    Args:
        embedding: embedding 2D (UMAP/t-SNE/PCA)
        labels: etykiety klastrów
        title: tytuł wykresu
        figsize: rozmiar wykresu
        save_path: ścieżka do zapisania
    """
    plt.figure(figsize=figsize)
    
    # Unikalne klastry
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                   c=[color], label=f'Klaster {label}', 
                   alpha=0.7, s=20)
    
    plt.xlabel('Wymiar 1')
    plt.ylabel('Wymiar 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Zapisano wizualizację 2D: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_cluster_analysis(latent_vectors: np.ndarray,
                         labels: np.ndarray,
                         max_clusters_show: int = 8,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Szczegółowa analiza klastrów z wykresami.
    
    Args:
        latent_vectors: wektory latentne
        labels: etykiety klastrów
        max_clusters_show: maksymalna liczba klastrów do pokazania szczegółowo
        figsize: rozmiar wykresu
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Rozkład wielkości klastrów
    axes[0, 0].bar(unique_labels, counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel('Numer klastra')
    axes[0, 0].set_ylabel('Liczba próbek')
    axes[0, 0].set_title('Rozkład wielkości klastrów')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dodaj procenty
    total_samples = len(labels)
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        percentage = count / total_samples * 100
        axes[0, 0].text(label, count + max(counts)*0.01, f'{percentage:.1f}%', 
                       ha='center', fontsize=9)
    
    # 2. Średnie wartości cech dla każdego klastra
    cluster_means = []
    for label in unique_labels[:max_clusters_show]:
        mask = labels == label
        cluster_mean = np.mean(latent_vectors[mask], axis=0)
        cluster_means.append(cluster_mean)
    
    if cluster_means:
        cluster_means = np.array(cluster_means)
        im = axes[0, 1].imshow(cluster_means, aspect='auto', cmap='viridis')
        axes[0, 1].set_xlabel('Wymiar latentny')
        axes[0, 1].set_ylabel('Klaster')
        axes[0, 1].set_title('Średnie wartości cech w klastrach')
        axes[0, 1].set_yticks(range(len(cluster_means)))
        axes[0, 1].set_yticklabels([f'K{i}' for i in unique_labels[:max_clusters_show]])
        plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Histogram odległości od centroidów
    distances = []
    for label in unique_labels:
        mask = labels == label
        cluster_vectors = latent_vectors[mask]
        centroid = np.mean(cluster_vectors, axis=0)
        cluster_distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        distances.extend(cluster_distances)
    
    axes[1, 0].hist(distances, bins=30, alpha=0.7, color='lightcoral')
    axes[1, 0].set_xlabel('Odległość od centroidu')
    axes[1, 0].set_ylabel('Liczba próbek')
    axes[1, 0].set_title('Rozkład odległości od centroidów klastrów')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Macierz korelacji między klastrami (na podstawie centroidów)
    if len(unique_labels) > 1:
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(latent_vectors[mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        correlation_matrix = np.corrcoef(centroids)
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_xlabel('Klaster')
        axes[1, 1].set_ylabel('Klaster')
        axes[1, 1].set_title('Korelacja między centroidami klastrów')
        axes[1, 1].set_xticks(range(len(unique_labels)))
        axes[1, 1].set_yticks(range(len(unique_labels)))
        axes[1, 1].set_xticklabels([f'K{i}' for i in unique_labels])
        axes[1, 1].set_yticklabels([f'K{i}' for i in unique_labels])
        plt.colorbar(im, ax=axes[1, 1])
        
        # Dodaj wartości do macierzy
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def compare_damage_types(model: torch.nn.Module,
                        sample_image: torch.Tensor,
                        damage_functions: list,
                        device: torch.device,
                        figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Porównuje rekonstrukcje dla różnych typów uszkodzeń.
    
    Args:
        model: trenowany autoencoder
        sample_image: przykładowy obraz [C, H, W]
        damage_functions: lista funkcji uszkadzających
        device: urządzenie
        figsize: rozmiar wykresu
    """
    model.eval()
    
    n_damages = len(damage_functions)
    fig, axes = plt.subplots(3, n_damages + 1, figsize=figsize)
    
    # Oryginalny obraz
    original_np = sample_image.permute(1, 2, 0).numpy()
    original_np = np.clip(original_np, 0, 1)
    
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title("Oryginalny")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    
    with torch.no_grad():
        for i, damage_func in enumerate(damage_functions):
            # Zastosuj uszkodzenie
            damaged = damage_func(sample_image)
            damaged_input = damaged.unsqueeze(0).to(device)
            
            # Rekonstrukcja
            reconstructed, _ = model(damaged_input)
            reconstructed = reconstructed.squeeze(0).cpu()
            
            # Konwersja do numpy
            damaged_np = damaged.permute(1, 2, 0).numpy()
            reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
            damaged_np = np.clip(damaged_np, 0, 1)
            reconstructed_np = np.clip(reconstructed_np, 0, 1)
            
            # Oblicz błąd
            mse = np.mean((original_np - reconstructed_np) ** 2)
            
            # Uszkodzony
            axes[0, i + 1].imshow(damaged_np)
            axes[0, i + 1].set_title(f"{damage_func.__name__}")
            axes[0, i + 1].axis('off')
            
            # Rekonstrukcja
            axes[1, i + 1].imshow(reconstructed_np)
            axes[1, i + 1].set_title(f"Rekonstrukcja")
            axes[1, i + 1].axis('off')
            
            # Różnica
            diff = np.abs(original_np - reconstructed_np)
            axes[2, i + 1].imshow(diff, cmap='hot')
            axes[2, i + 1].set_title(f"Różnica (MSE: {mse:.4f})")
            axes[2, i + 1].axis('off')
    
    plt.suptitle("Porównanie rekonstrukcji dla różnych typów uszkodzeń", fontsize=16)
    plt.tight_layout()
    plt.show()