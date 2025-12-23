import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from src.data.load_dataset import load_data
from src.modules.autoencoder import Autoencoder

def test_model_save_load():
    print("=" * 60)
    print("TEST ZAPISYWANIA I WCZYTYWANIA MODELU")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    train_loader, _, _ = load_data(add_fourth_channel=True)
    
    it = iter(train_loader)
    batch = next(it)
    single_image = batch[0] if isinstance(batch, (tuple, list)) else batch
    single_image = single_image[:1]
    
    print(f"\nObraz testowy shape: {single_image.shape}")
    
    single_dataset = TensorDataset(single_image)
    single_loader = DataLoader(single_dataset, batch_size=1, shuffle=False)
    
    print("\n" + "=" * 60)
    print("ETAP 1: TRENING I ZAPISANIE MODELU (load_best=False)")
    print("=" * 60)
    
    LOAD_BEST = False
    
    autoencoder = Autoencoder(
        latent_dim=768,
        input_channels=4,
        image_size=256,
        learning_rate=0.001,
        load_best=LOAD_BEST
    )
    
    if not autoencoder.model_loaded:
        print("\nModel nie został wczytany - rozpoczynam trening...")
        history = autoencoder.fit(
            train_loader=single_loader,
            val_loader=single_loader,
            epochs=1,
            early_stopping_patience=5
        )
        
        print(f"\nTrain Loss: {history['train_loss'][0]:.6f}")
        print(f"Val Loss: {history['val_loss'][0]:.6f}")
    else:
        print("\nModel został wczytany - pomijam trening")
    
    latent_vectors_train, _ = autoencoder.extract_latent(single_loader)
    reconstructed_train = autoencoder.decode_batch(latent_vectors_train)
    
    print(f"\nLatent vector (po treningu) - shape: {latent_vectors_train.shape}")
    print(f"Latent vector (po treningu) - mean: {np.mean(latent_vectors_train):.6f}")
    print(f"Latent vector (po treningu) - std: {np.std(latent_vectors_train):.6f}")
    
    del autoencoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n" + "=" * 60)
    print("ETAP 2: WCZYTANIE MODELU (load_best=True)")
    print("=" * 60)
    
    LOAD_BEST = True
    
    autoencoder = Autoencoder(
        latent_dim=768,
        input_channels=4,
        image_size=256,
        learning_rate=0.001,
        load_best=LOAD_BEST
    )
    
    if not autoencoder.model_loaded:
        print("\nModel nie został wczytany - rozpoczynam trening...")
        history = autoencoder.fit(
            train_loader=single_loader,
            val_loader=single_loader,
            epochs=1,
            early_stopping_patience=5
        )
    else:
        print("\nModel został pomyslnie wczytany - pomijam trening")
    
    latent_vectors_load, _ = autoencoder.extract_latent(single_loader)
    reconstructed_load = autoencoder.decode_batch(latent_vectors_load)
    
    print(f"\nLatent vector (po wczytaniu) - shape: {latent_vectors_load.shape}")
    print(f"Latent vector (po wczytaniu) - mean: {np.mean(latent_vectors_load):.6f}")
    print(f"Latent vector (po wczytaniu) - std: {np.std(latent_vectors_load):.6f}")
    
    print("\n" + "=" * 60)
    print("ETAP 3: PORÓWNANIE WYNIKÓW")
    print("=" * 60)
    
    latent_diff = np.abs(latent_vectors_train - latent_vectors_load)
    recon_diff = np.abs(reconstructed_train - reconstructed_load)
    
    print(f"\nRóżnica w latent space:")
    print(f"  Max diff: {np.max(latent_diff):.10f}")
    print(f"  Mean diff: {np.mean(latent_diff):.10f}")
    print(f"  Sum diff: {np.sum(latent_diff):.10f}")
    
    print(f"\nRóżnica w rekonstrukcji:")
    print(f"  Max diff: {np.max(recon_diff):.10f}")
    print(f"  Mean diff: {np.mean(recon_diff):.10f}")
    print(f"  Sum diff: {np.sum(recon_diff):.10f}")
    
    if np.allclose(latent_vectors_train, latent_vectors_load, atol=1e-6):
        print("\nSUKCES: Latent vectors są identyczne!")
    else:
        print("\nUWAGA: Latent vectors różnią się!")
    
    if np.allclose(reconstructed_train, reconstructed_load, atol=1e-6):
        print("SUKCES: Rekonstrukcje są identyczne!")
    else:
        print("UWAGA: Rekonstrukcje różnią się!")
    
    print("\n" + "=" * 60)
    print("ETAP 4: WIZUALIZACJA")
    print("=" * 60)
    
    original_np = single_image[0].cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    original_rgb = np.transpose(original_np[:3], (1, 2, 0))
    original_rgb = np.clip(original_rgb, 0, 1)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Oryginalny obraz (RGB)")
    axes[0, 0].axis("off")
    
    recon_train_rgb = np.transpose(reconstructed_train[0, :3], (1, 2, 0))
    recon_train_rgb = np.clip(recon_train_rgb, 0, 1)
    axes[0, 1].imshow(recon_train_rgb)
    axes[0, 1].set_title("Rekonstrukcja (po treningu)")
    axes[0, 1].axis("off")
    
    recon_load_rgb = np.transpose(reconstructed_load[0, :3], (1, 2, 0))
    recon_load_rgb = np.clip(recon_load_rgb, 0, 1)
    axes[0, 2].imshow(recon_load_rgb)
    axes[0, 2].set_title("Rekonstrukcja (po wczytaniu)")
    axes[0, 2].axis("off")
    
    diff_train = np.abs(original_rgb - recon_train_rgb)
    axes[1, 0].imshow(diff_train)
    axes[1, 0].set_title(f"Diff po treningu (MSE: {np.mean(diff_train**2):.6f})")
    axes[1, 0].axis("off")
    
    diff_load = np.abs(original_rgb - recon_load_rgb)
    axes[1, 1].imshow(diff_load)
    axes[1, 1].set_title(f"Diff po wczytaniu (MSE: {np.mean(diff_load**2):.6f})")
    axes[1, 1].axis("off")
    
    diff_reconstructions = np.abs(recon_train_rgb - recon_load_rgb)
    axes[1, 2].imshow(diff_reconstructions * 100)
    axes[1, 2].set_title(f"Diff rekonstrukcji (x100)")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    
    save_dir = Path('test_checkpoints')
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nWizualizacja zapisana do: {save_dir / 'comparison.png'}")
    plt.show()
    
    print("\n" + "=" * 60)
    print("TEST ZAKOŃCZONY")
    print("=" * 60)
    
    return {
        'latent_diff_max': np.max(latent_diff),
        'latent_diff_mean': np.mean(latent_diff),
        'recon_diff_max': np.max(recon_diff),
        'recon_diff_mean': np.mean(recon_diff),
        'latent_identical': np.allclose(latent_vectors_train, latent_vectors_load, atol=1e-6),
        'recon_identical': np.allclose(reconstructed_train, reconstructed_load, atol=1e-6)
    }

if __name__ == "__main__":
    results = test_model_save_load()
    
    print("\n" + "=" * 60)
    print("WYNIKI KOŃCOWE")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value}")
