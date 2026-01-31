import torch
import tqdm

#%% Cache latent data from dataloader
def cache_latent_data(loader, encoder, device):
    print(f"Rozpoczynam caching danych ({len(loader)} batchy)... To potrwa kilka minut.")
    
    clean_latents = []
    damaged_latents = []
    all_labels = []
    
    encoder.eval()
    encoder.to(device)
    
    with torch.no_grad():
        for clean_batch, damaged_batch, labels in tqdm.tqdm(loader, desc="Caching Latents"):

            clean_img = clean_batch.to(device)
            damaged_img = damaged_batch.to(device)
            
            z_clean = encoder(clean_img)
            z_damaged = encoder(damaged_img)
            
            clean_latents.append(z_clean.cpu())
            damaged_latents.append(z_damaged.cpu())
            all_labels.append(labels.cpu()) 
            
    full_clean_z = torch.cat(clean_latents, dim=0)
    full_damaged_z = torch.cat(damaged_latents, dim=0)
    full_labels = torch.cat(all_labels, dim=0)
    
    print(f"Caching zakończony!")
    print(f"Wymiary danych w RAM: {full_clean_z.shape} (ok. {full_clean_z.element_size() * full_clean_z.numel() / 1024**2:.1f} MB)")
    
    return full_clean_z, full_damaged_z, full_labels