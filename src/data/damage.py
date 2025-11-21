import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#%% Define max size 
MAX_MASK_SIZE = 0.0625

#%% Damage Function - Square Mask - 4th channel output
def square_damage(image: torch.Tensor) -> torch.Tensor:
    C, H, W = image.shape

    max_mask_size = int(min(H, W) * MAX_MASK_SIZE)
    mask_size = np.random.randint(1, max_mask_size + 1)

    y = np.random.randint(0, H - mask_size + 1)
    x = np.random.randint(0, W - mask_size + 1)

    mask = torch.zeros(1, H, W, dtype=image.dtype, device=image.device)
    mask[:, y:y+mask_size, x:x+mask_size] = 1.0

    return torch.cat([image, mask], dim=0)
    
#%% Damage Function - Noise Mask - 4th channel output
def noise_damage(image: torch.Tensor) -> torch.Tensor:
    _, H, W = image.shape

    num_noisy_pixels = int(H * W * MAX_MASK_SIZE)

    ys = np.random.randint(0, H, size=num_noisy_pixels)
    xs = np.random.randint(0, W, size=num_noisy_pixels)

    mask = torch.zeros(1, H, W, dtype=image.dtype, device=image.device)

    for y, x in zip(ys, xs):
        mask[:, y, x] = 1.0

    return torch.cat([image, mask], dim=0)

#%% Damage Function - Line Mask - 4th channel output
def line_damage(image: torch.Tensor) -> torch.Tensor:
    _, h, w = image.shape

    mask = torch.zeros(1, h, w, dtype=image.dtype, device=image.device)
    num_lines = np.random.randint(1, 5)
    line_width = max(1, int(min(h, w) * MAX_MASK_SIZE / 10))
    
    for _ in range(num_lines):
        y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
        y2, x2 = np.random.randint(0, h), np.random.randint(0, w)
        
        steps = max(abs(y2-y1), abs(x2-x1))
        if steps > 0:
            for i in range(steps):
                y = int(y1 + (y2-y1) * i / steps)
                x = int(x1 + (x2-x1) * i / steps)
                
                for dy in range(-line_width//2, line_width//2 + 1):
                    for dx in range(-line_width//2, line_width//2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            mask[:, ny, nx] = 1.0
    
    return torch.cat([image, mask], dim=0)

#%% Damage Function - Transparent Mask - 4th channel output
def transparent_damage(image: torch.Tensor) -> torch.Tensor:
    C, H, W = image.shape
    mask = torch.zeros(1, H, W, dtype=image.dtype, device=image.device)
    return torch.cat([image, mask], dim=0)


#%% Make Damage DataLoader
def make_damage_loader(dataloader, batch_size=None):
    damage_functions = [square_damage, noise_damage, line_damage]
    
    def collate_fn(batch):
        damaged_images = []
        original_images = []
        
        for item in batch:
            image = item['image'] if isinstance(item, dict) else item
            damage_fn = np.random.choice(damage_functions)
            damaged_image = damage_fn(image)
            original_with_mask = transparent_damage(image)
            
            damaged_images.append(damaged_image)
            original_images.append(original_with_mask)
        
        return torch.stack(damaged_images), torch.stack(original_images)
    
    return DataLoader(
        dataloader.dataset,
        batch_size=batch_size or dataloader.batch_size,
        shuffle=isinstance(dataloader.sampler, type(None)),
        num_workers=dataloader.num_workers,
        collate_fn=collate_fn
    )