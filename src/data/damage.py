import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.draw import line
from torch.utils.data import DataLoader, Dataset
from skimage.morphology import dilation, square

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
    
    num_lines = np.random.randint(2, 4)
    line_length = int(np.sqrt(target_mask_area / num_lines))
    line_width = max(5, line_length // 10)
    
    mask_temp = np.zeros((h, w), dtype=bool)
    current_area = 0
    attempts = 0
    max_attempts = 20
    
    while current_area < target_mask_area * 0.8 and attempts < max_attempts:
        for _ in range(num_lines):
            y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
            angle = np.random.uniform(0, 2 * np.pi)
            y2 = int(y1 + line_length * np.sin(angle))
            x2 = int(x1 + line_length * np.cos(angle))
            
            y2 = np.clip(y2, 0, h - 1)
            x2 = np.clip(x2, 0, w - 1)
            
            rr, cc = line(y1, x1, y2, x2)
            mask_temp[rr, cc] = True
        
        mask_temp = dilation(mask_temp, square(line_width))
        
        current_area = np.sum(mask_temp)
        attempts += 1
    
    mask_np = mask_temp.astype(np.float32)
    mask[0] = torch.from_numpy(mask_np).to(device)
    rgb[:, mask_temp] = 0.0
    
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
def make_damage_loader(dataloader, batch_size=None, shuffle=None):
    return DataLoader(
        dataloader.dataset,
        batch_size=batch_size or dataloader.batch_size,
        shuffle=shuffle if shuffle is not None else dataloader.shuffle,
        num_workers=dataloader.num_workers,
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

#%% Deterministic Damaged Dataset Wrapper
class DeterministicDamagedDataset(Dataset):
    def __init__(self, original_dataset, damage_fn_list=None, seed=42):
        self.original_dataset = original_dataset
        self.damage_fn_list = damage_fn_list or DAMAGE_FUNCTIONS
        self.seed = seed
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)
        
        item = self.original_dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item
        
        damage_fn = rng.choice(self.damage_fn_list)
        
        torch.manual_seed(self.seed + idx)
        np.random.seed(self.seed + idx)
        
        damaged_image = damage_fn(image.to(device))
        
        return damaged_image.cpu()


def create_paired_damaged_loaders(train_loader, test_loader, val_loader, 
                                   damage_fn_list=None, seed=42):
    train_damaged_dataset = DeterministicDamagedDataset(
        train_loader.dataset, damage_fn_list, seed
    )
    test_damaged_dataset = DeterministicDamagedDataset(
        test_loader.dataset, damage_fn_list, seed
    )
    val_damaged_dataset = DeterministicDamagedDataset(
        val_loader.dataset, damage_fn_list, seed
    )
    
    train_damaged_loader = DataLoader(
        train_damaged_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=True
    )
    
    test_damaged_loader = DataLoader(
        test_damaged_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=True
    )
    
    val_damaged_loader = DataLoader(
        val_damaged_dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=True
    )
    
    return train_damaged_loader, test_damaged_loader, val_damaged_loader