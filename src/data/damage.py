import torch
import numpy as np
from torch.utils.data import DataLoader

#%% Define max size 
MAX_MASK_SIZE = 0.0625

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Damage Function - Square Mask - 4th channel output
def square_damage(image: torch.Tensor) -> torch.Tensor:
    _, H, W = image.shape
    rgb = image[:3]

    total_area = H * W
    target_mask_area = int(total_area * MAX_MASK_SIZE)
    mask_size = int(np.sqrt(target_mask_area))

    y = np.random.randint(0, H - mask_size + 1)
    x = np.random.randint(0, W - mask_size + 1)

    mask = torch.zeros(1, H, W, dtype=image.dtype, device=image.device)
    mask[:, y:y+mask_size, x:x+mask_size] = 1.0

    rgb[:, y:y+mask_size, x:x+mask_size] = 0.0

    return torch.cat([rgb, mask], dim=0)
    
#%% Damage Function - Multiple Squares Mask - 4th channel output
def multiple_squares_damage(image: torch.Tensor) -> torch.Tensor:
    _, H, W = image.shape
    rgb = image[:3]

    mask = torch.zeros(1, H, W, dtype=image.dtype, device=image.device)
    
    total_mask_area = int(H * W * MAX_MASK_SIZE)
    num_squares = np.random.randint(2, 4)
    square_area = total_mask_area // num_squares
    square_size = int(np.sqrt(square_area))
    
    for _ in range(num_squares):
        y = np.random.randint(0, H - square_size + 1)
        x = np.random.randint(0, W - square_size + 1)
        mask[:, y:y+square_size, x:x+square_size] = 1.0
        rgb[:, y:y+square_size, x:x+square_size] = 0.0

    return torch.cat([rgb, mask], dim=0)

#%% Damage Function - Line Mask - 4th channel output
def line_damage(image: torch.Tensor) -> torch.Tensor:
    _, h, w = image.shape
    rgb = image[:3]

    mask = torch.zeros(1, h, w, dtype=image.dtype, device=image.device)
    
    total_area = h * w
    target_mask_area = int(total_area * MAX_MASK_SIZE)
    
    num_lines = np.random.randint(3, 6)
    line_length = int(np.sqrt(target_mask_area / num_lines))
    line_width = max(2, line_length // 10)
    
    current_area = 0
    attempts = 0
    max_attempts = 100
    
    while current_area < target_mask_area * 0.9 and attempts < max_attempts:
        y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
        angle = np.random.uniform(0, 2 * np.pi)
        y2 = int(y1 + line_length * np.sin(angle))
        x2 = int(x1 + line_length * np.cos(angle))
        
        y2 = np.clip(y2, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        
        steps = max(abs(y2-y1), abs(x2-x1), 1)
        for i in range(steps):
            y = int(y1 + (y2-y1) * i / steps)
            x = int(x1 + (x2-x1) * i / steps)
            
            for dy in range(-line_width//2, line_width//2 + 1):
                for dx in range(-line_width//2, line_width//2 + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[:, ny, nx] == 0:
                            mask[:, ny, nx] = 1.0
                            current_area += 1
                        rgb[:, ny, nx] = 0.0
        
        attempts += 1
    
    return torch.cat([rgb, mask], dim=0)

#%% Damage Function - Transparent Mask - 4th channel output
def transparent_damage(image: torch.Tensor):
    _, H, W = image.shape
    rgb = image[:3]
    mask = torch.zeros(1, H, W, dtype=image.dtype, device=image.device)
    return torch.cat([rgb, mask], dim=0)


#%% Global list of damage functions for collate_fn
DAMAGE_FUNCTIONS = [square_damage, multiple_squares_damage, line_damage]

#%% Collate function for DataLoader (must be at module level for pickle)
def damage_collate_fn(batch):
    damaged_images = []
    original_images = []
    
    for item in batch:
        image = item['image'] if isinstance(item, dict) else item
        image = image.to(device)
        damage_fn = np.random.choice(DAMAGE_FUNCTIONS)
        damaged_image = damage_fn(image)
        original_with_mask = transparent_damage(image)
        
        damaged_images.append(damaged_image)
        original_images.append(original_with_mask)
    
    return torch.stack(damaged_images), torch.stack(original_images)

#%% Make Damage DataLoader
def make_damage_loader(dataloader, batch_size=None):
    return DataLoader(
        dataloader.dataset,
        batch_size=batch_size or dataloader.batch_size,
        shuffle=isinstance(dataloader.sampler, type(None)),
        num_workers=0,
        collate_fn=damage_collate_fn
    )


#%% Test damage functions
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms

    origin_image = np.random.rand(256, 256, 4)

    origin_image[:, :, 3] = 0 

    img_tensor = transforms.ToTensor()(origin_image).to(device)

    damaged_img, original_with_mask = damage_collate_fn([img_tensor])

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    
    damaged = damaged_img[0].cpu()
    original = original_with_mask[0].cpu()
    channel_names = ['Red', 'Green', 'Blue', 'Mask']
    
    for i in range(4):
        axes[i].imshow(damaged[i], cmap='gray')
        axes[i].set_title(f"Damaged {channel_names[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()