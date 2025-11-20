import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#%% Define max size 
MAX_MASK_SIZE = 0.0625

#%% Damage Function - Square Mask - 4th channel output
def square_damage(image: torch.Tensor) -> torch.Tensor:
    """Dodaje kwadratowe uszkodzenie do obrazu"""
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
    """Dodaje losowe piksele szumu jako uszkodzenie"""
    C, H, W = image.shape

    num_noisy_pixels = int(H * W * MAX_MASK_SIZE)

    ys = np.random.randint(0, H, size=num_noisy_pixels)
    xs = np.random.randint(0, W, size=num_noisy_pixels)

    mask = torch.zeros(1, H, W, dtype=image.dtype, device=image.device)

    for y, x in zip(ys, xs):
        mask[:, y, x] = 1.0

    return torch.cat([image, mask], dim=0)

#%% Damage Function - Line Mask - 4th channel output
def line_damage(image: torch.Tensor) -> torch.Tensor:
    """Dodaje losowe linie jako uszkodzenie"""
    C, h, w = image.shape

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

#%% Damage Pipeline
def make_damage(dataset: Dataset) -> Dataset:
    damaged_images = []
    for img in dataset:
        damage_type = np.random.choice(['square', 'noise', 'line'])
        if damage_type == 'square':
            damaged_img = square_damage(img)
        elif damage_type == 'noise':
            damaged_img = noise_damage(img)
        else:
            damaged_img = line_damage(img)
        damaged_images.append(damaged_img)
    return Dataset(torch.stack(damaged_images))


#%% Make Damage DataLoader
def make_damage_loader(dataloader, batch_size=None):
    base_dataset = dataloader.dataset
    damaged_dataset = make_damage(base_dataset)
    
    if batch_size is None:
        batch_size = dataloader.batch_size
    
    return DataLoader(
        damaged_dataset, 
        batch_size=batch_size,
        shuffle=dataloader.sampler is None,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory
    )