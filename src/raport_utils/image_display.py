from typing import Counter
import torch
import gc
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.modules.pca_clust import preprocess_spatial_latents

from src.data.damage import square_damage, multiple_squares_damage, line_damage


#%% Visualize damaged images
def visualize_damaged_images(data_loader, num_samples=3, figsize=(15, 5)):
    """
    Pobiera zadaną liczbę próbek z data_loadera i wizualizuje ich kanały.
    Dzieli obraz na: Pełny podgląd, RGB, oraz 4. kanał (jeśli istnieje).
    
    Args:
        data_loader: PyTorch DataLoader zwracający batche obrazów.
        num_samples (int): Liczba obrazów do wyświetlenia (domyślnie 3).
        figsize (tuple): Rozmiar pojedynczego wiersza wykresu (domyślnie 15x5).
    """
    it = iter(data_loader)
    collected = []
    
    while len(collected) < num_samples:
        try:
            batch = next(it)
        except StopIteration:
            break # Koniec danych w loaderze
            
        # Obsługa sytuacji, gdy loader zwraca (img, label) lub samo img
        imgs_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        for i in range(imgs_batch.shape[0]):
            collected.append(imgs_batch[i].detach().cpu())
            if len(collected) == num_samples:
                break

    for idx, img in enumerate(collected):
        # img: tensor C x H x W
        img_np = img.numpy()
        C, H, W = img_np.shape
        full = np.transpose(img_np, (1, 2, 0))           # H x W x C
        
        # Rozdzielenie kanałów
        rgb = full[..., :3] if C >= 3 else np.repeat(full[..., :1], 3, axis=2)
        fourth = full[..., 3] if C >= 4 else None

        # Clip to [0,1] dla bezpiecznego wyświetlania (float vs int display)
        full = np.clip(full, 0.0, 1.0)
        rgb = np.clip(rgb, 0.0, 1.0)
        if fourth is not None:
            fourth = np.clip(fourth, 0.0, 1.0)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Pełne (RGBA lub RGB)
        if C >= 4:
            axes[0].imshow(full)
            axes[0].set_title("Pełne (RGBA)")
        else:
            axes[0].imshow(rgb)
            axes[0].set_title("Pełne (RGB)")
        axes[0].axis("off")

        # 2. Tylko pierwsze 3 kanały (RGB)
        axes[1].imshow(rgb)
        axes[1].set_title("Tylko pierwsze 3 kanały (RGB)")
        axes[1].axis("off")

        # 3. Tylko czwarty kanał jako skala szarości
        if fourth is not None:
            axes[2].imshow(fourth, cmap="gray")
            axes[2].set_title("Tylko 4. kanał (grayscale)")
        else:
            axes[2].text(0.5, 0.5, "Brak 4. kanału", ha="center", va="center")
            axes[2].set_title("Tylko 4. kanał")
        axes[2].axis("off")

        plt.suptitle(f"Przykład {idx+1}")
        plt.tight_layout()
        plt.show()

#%% Visualization of Autoencoder reconstruction
def visualize_reconstruction(
    model, 
    test_loader, 
    damaged_test_loader, 
    device=None, 
    num_display=3
):
    """
    Przeprowadza inferencję na losowych próbkach z zestawu testowego,
    wizualizuje porównanie (Oryginał vs Rekonstrukcja vs Różnica)
    i oblicza metryki MSE.

    Args:
        model: Model autoenkodera (musi zwracać krotkę, gdzie 1. element to rekonstrukcja).
        test_loader: DataLoader z oryginalnymi obrazami.
        damaged_test_loader: DataLoader z uszkodzonymi obrazami (musi odpowiadać indeksami test_loader).
        device (torch.device, optional): Urządzenie (CPU/CUDA). Jeśli None, wykrywa automatycznie.
        num_display (int): Liczba losowych próbek do wizualizacji.
        
    Returns:
        dict: Słownik z obliczonymi metrykami MSE ('mse_orig', 'mse_dmg').
    """
    
    # 1. Zarządzanie pamięcią i urządzeniem
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)

    # 2. Pobieranie próbek
    total_samples = len(test_loader.dataset)
    actual_num_display = min(num_display, total_samples)
    
    # Losowanie indeksów (zakładamy, że dataset w obu loaderach jest wyrównany indeksami)
    random_indices = np.random.choice(total_samples, actual_num_display, replace=False)

    displayed_originals = []
    displayed_damaged = []

    for idx in random_indices:
        original_sample = test_loader.dataset[idx]
        damaged_sample = damaged_test_loader.dataset[idx]
        
        # Obsługa krotek (img, label) vs sam tensor
        original_img = original_sample[0] if isinstance(original_sample, (tuple, list)) else original_sample
        damaged_img = damaged_sample[0] if isinstance(damaged_sample, (tuple, list)) else damaged_sample
        
        displayed_originals.append(original_img)
        displayed_damaged.append(damaged_img)

    # Stackowanie do batcha
    original_imgs = torch.stack(displayed_originals).to(device)
    damaged_imgs = torch.stack(displayed_damaged).to(device)

    # 3. Inferencja
    with torch.no_grad():
        # Uwaga: Zakładamy, że model zwraca: recon, _, _, latent
        recon_original, _, _, latent_orig = model(original_imgs)
        recon_damaged, _, _, latent_dmg = model(damaged_imgs)

    # 4. Konwersja do Numpy
    original_np = original_imgs.cpu().float().numpy()
    damaged_np = damaged_imgs.cpu().float().numpy()
    recon_original_np = recon_original.cpu().float().numpy()
    recon_damaged_np = recon_damaged.cpu().float().numpy()

    print(f"Prezentacja rekonstrukcji (Model: {model.__class__.__name__}, Latent Channels: {latent_orig.shape[1]}):")

    # 5. Wizualizacja
    for i in range(actual_num_display):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # --- Wiersz 1: Oryginał ---
        
        # Input
        img_orig_rgb = np.transpose(original_np[i, :3], (1, 2, 0))
        axes[0, 0].imshow(np.clip(img_orig_rgb, 0, 1))
        axes[0, 0].set_title("Original Input")
        axes[0, 0].axis("off")
        
        # Rekonstrukcja
        recon_orig_rgb = np.transpose(recon_original_np[i, :3], (1, 2, 0))
        axes[0, 1].imshow(np.clip(recon_orig_rgb, 0, 1))
        axes[0, 1].set_title("Reconstruction (Orig)")
        axes[0, 1].axis("off")
        
        # Różnica
        diff_orig = np.mean(np.abs(img_orig_rgb - recon_orig_rgb), axis=2) 
        im_diff1 = axes[0, 2].imshow(diff_orig, cmap='hot')
        axes[0, 2].set_title("Difference Intensity")
        axes[0, 2].axis("off")
        plt.colorbar(im_diff1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # --- Wiersz 2: Uszkodzony ---
        
        # Input
        img_dmg_rgb = np.transpose(damaged_np[i, :3], (1, 2, 0))
        axes[1, 0].imshow(np.clip(img_dmg_rgb, 0, 1))
        axes[1, 0].set_title("Damaged Input")
        axes[1, 0].axis("off")
        
        # Rekonstrukcja
        recon_dmg_rgb = np.transpose(recon_damaged_np[i, :3], (1, 2, 0))
        axes[1, 1].imshow(np.clip(recon_dmg_rgb, 0, 1))
        axes[1, 1].set_title("Reconstruction (Dmg)")
        axes[1, 1].axis("off")
        
        # Różnica
        diff_dmg = np.mean(np.abs(img_dmg_rgb - recon_dmg_rgb), axis=2)
        im_diff2 = axes[1, 2].imshow(diff_dmg, cmap='hot')
        axes[1, 2].set_title("Difference Intensity")
        axes[1, 2].axis("off")
        plt.colorbar(im_diff2, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"Sample Index: {random_indices[i]}", fontsize=16)
        plt.tight_layout()
        plt.show()

    # 6. Metryki
    mse_orig = np.mean((original_np - recon_original_np)**2)
    mse_dmg = np.mean((damaged_np - recon_damaged_np)**2)
    
    print("\nReconstruction metrics:")
    print(f"MSE Original (All channels): {mse_orig:.6f}")
    print(f"MSE Damaged (All channels): {mse_dmg:.6f}")
    
    return {
        "mse_orig": mse_orig, 
        "mse_dmg": mse_dmg
    }

#%% Test cluster assignment
def visualize_cluster_assignment(autoencoder, dataloader, clustering_models, device, batch_size=128):
    """
    Testuje przypisywanie klas dla całego datasetu
    """
    autoencoder.eval()
    
    pca = clustering_models['pca']
    scaler = clustering_models['scaler']
    gmm = clustering_models['gmm']
    
    all_predictions = []
    all_confidences = []
    
    print(f"Testowanie przypisywania klas ({len(dataloader.dataset)} obrazów)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)
            
            temp_dataset = torch.utils.data.TensorDataset(imgs.cpu())
            temp_loader = DataLoader(temp_dataset, batch_size=len(imgs), shuffle=False)
            
            z_1d = autoencoder.extract_latent(temp_loader, use_projector=True, verbose=False)
            
            features = preprocess_spatial_latents(z_1d)
            features_scaled = scaler.transform(features)
            latents_pca = pca.transform(features_scaled)
            
            predictions = gmm.predict(latents_pca)
            proba = gmm.predict_proba(latents_pca)
            confidences = np.max(proba, axis=1)
            
            all_predictions.extend(predictions)
            all_confidences.extend(confidences)

    
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    
    cluster_counts = Counter(all_predictions)
    n_clusters = len(cluster_counts)
    
    print(f"\nRozkład klas w datasecie:")
    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        percentage = (count / len(all_predictions)) * 100
        print(f"  Klasa {cluster_id}: {count} obrazów ({percentage:.2f}%)")
    
    print(f"\nŚrednia pewność przypisania: {np.mean(all_confidences):.4f}")
    print(f"Min pewność: {np.min(all_confidences):.4f}")
    print(f"Max pewność: {np.max(all_confidences):.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    cluster_ids = sorted(cluster_counts.keys())
    counts = [cluster_counts[cid] for cid in cluster_ids]
    
    axes[0].bar(cluster_ids, counts, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Klaster ID')
    axes[0].set_ylabel('Liczba obrazów')
    axes[0].set_title(f'Rozkład klas ({n_clusters} klastrów)')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (cid, count) in enumerate(zip(cluster_ids, counts)):
        percentage = (count / len(all_predictions)) * 100
        axes[0].text(cid, count + max(counts)*0.01, f'{percentage:.1f}%', 
                     ha='center', va='bottom', fontsize=9)
    
    axes[1].hist(all_confidences, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(all_confidences), color='red', linestyle='--', 
                    linewidth=2, label=f'Średnia: {np.mean(all_confidences):.3f}')
    axes[1].set_xlabel('Pewność przypisania')
    axes[1].set_ylabel('Liczba próbek')
    axes[1].set_title('Rozkład pewności przypisania klas')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'predictions': all_predictions,
        'confidences': all_confidences,
        'cluster_counts': cluster_counts,
        'n_clusters': n_clusters
    }


#%% Visualization of inpainter results
def visualize_images_inpainter(inpainter, autoencoder, clean_dataloader, clustering_models, device, num_samples=3):
    inpainter.eval()
    autoencoder.eval()
    
    DAMAGE_FUNCTIONS = [square_damage, multiple_squares_damage, line_damage]
    
    dataset = clean_dataloader.dataset
    total_len = len(dataset)
    
    random_indices = np.random.choice(total_len, num_samples, replace=False)
    print(f"Wylosowane indeksy: {random_indices}")

    sampled_clean_images = []
    
    for idx in random_indices:
        item = dataset[idx]
        img = item[0] if isinstance(item, (list, tuple)) else item
        sampled_clean_images.append(img)
    
    clean_images = torch.stack(sampled_clean_images).to(device)
    
    print(f"[DEBUG] Input Clean Images - Min: {clean_images.min():.4f}, Max: {clean_images.max():.4f}, Mean: {clean_images.mean():.4f}")

    pca = clustering_models['pca']
    scaler = clustering_models['scaler']
    gmm = clustering_models['gmm']

    with torch.no_grad():
        temp_dataset = TensorDataset(clean_images.cpu())
        temp_loader = DataLoader(temp_dataset, batch_size=num_samples, shuffle=False)
        
        z_1d = autoencoder.extract_latent(temp_loader, use_projector=True)
        
        features = preprocess_spatial_latents(z_1d) 
        features_scaled = scaler.transform(features)
        latents_pca = pca.transform(features_scaled)
        
        predicted_labels_numpy = gmm.predict(latents_pca)
        print(f"Przewidziane klasy (GMM): {predicted_labels_numpy}")

        damaged_images_list = []
        for i in range(num_samples):
            img_to_damage = clean_images[i].clone()
            damage_fn = np.random.choice(DAMAGE_FUNCTIONS)
            damaged_img = damage_fn(img_to_damage)
            damaged_images_list.append(damaged_img)
            
        damaged_images = torch.stack(damaged_images_list).to(device)
        
        z_damaged = autoencoder.encoder(damaged_images)
        
        z_repaired = inpainter(z_damaged)

        img_damaged_rec = autoencoder.decoder(z_damaged)
        img_repaired_rec = autoencoder.decoder(z_repaired)
        
        print(f"[DEBUG] Repaired Output - Min: {img_repaired_rec.min():.4f}, Max: {img_repaired_rec.max():.4f}, Mean: {img_repaired_rec.mean():.4f}")

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    plt.suptitle("Inpainting View (Auto-Detect Logic)", fontsize=16, y=1.02)
    
    cols = ["Oryginał", "Generated Damage", "Encoder Rec.", "Wynik Naprawy"]
    if num_samples == 1: axes = np.expand_dims(axes, axis=0)
    for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=12, fontweight='bold')

    def standard_process_img(tensor):
        img = tensor.cpu().permute(1, 2, 0).numpy()
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return np.clip(img, 0, 1)

    for i in range(num_samples):
        axes[i, 0].imshow(standard_process_img(clean_images[i]))
        
        axes[i, 1].imshow(standard_process_img(damaged_images[i]))
        
        axes[i, 2].imshow(standard_process_img(img_damaged_rec[i]))
        
        axes[i, 3].imshow(standard_process_img(img_repaired_rec[i]))
        axes[i, 3].set_title(f"GMM Label: {predicted_labels_numpy[i]}", color='green')
        
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.show()

#%% Visualize SR results
def visualize_sr_results(model, dataloader, num_samples=3):
    model.eval()
    device = model.device
    
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        lr_imgs = batch[0]
    else:
        lr_imgs = batch
    
    indices = torch.randperm(len(lr_imgs))[:num_samples]
    
    _, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_256 = lr_imgs[idx].unsqueeze(0).to(device)
            
            if img_256.shape[1] == 4 and getattr(model, 'input_channels', 3) == 3:
                img_256 = img_256[:, :3, :, :]
            
            img_512 = model(img_256)
            img_1024 = model(img_512)
            imgs = [img_256, img_512, img_1024]
            titles = [f"Input ({img_256.shape[-2]}x{img_256.shape[-1]})", 
                      f"Upscaled x2 ({img_512.shape[-2]}x{img_512.shape[-1]})", 
                      f"Upscaled x4 ({img_1024.shape[-2]}x{img_1024.shape[-1]})"]
            
            for j, img_tensor in enumerate(imgs):
                img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                
                ax = axes[i, j]
                ax.imshow(img_np)
                ax.set_title(titles[j])
                ax.axis('off')

    plt.tight_layout()
    plt.show()