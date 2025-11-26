#%% Imports
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch
from typing import Any, Dict

#%% Constants definitions
DATASET_NAME = "huggan/wikiart"
IMAGE_SIZE = 256

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Preprocessing function
def preprocess(batch):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    batch['image'] = [transform(img.convert('RGB')).to(device) for img in batch['image']]
    return batch

#%% Load dataset and create DataLoaders
def load_data(train_split=0.7, test_split=0.15, batch_size=32, num_workers=0):
    dataset = load_dataset(DATASET_NAME, split='train')
    dataset = dataset.with_transform(preprocess)
    
    train_size = int(len(dataset) * train_split)
    test_size = int(len(dataset) * test_split)
    
    splits = dataset.train_test_split(train_size=train_size, seed=42)
    train_ds = splits['train']
    temp_ds = splits['test']
    
    test_val_splits = temp_ds.train_test_split(train_size=test_size, seed=42)
    test_ds = test_val_splits['train']
    val_ds = test_val_splits['test']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, val_loader

def safe_collate(batch: Any) -> Dict[str, torch.Tensor]:
    """
    Bezpieczne collate:
    - obsługuje dict {'image': tensor}, tuple/list
    - dba, by wszystkie elementy miały 4 kanały (RGB + maska). Jeśli obraz ma 3 kanały, dokleja zerową maskę.
    """
    images = []
    for item in batch:
        if isinstance(item, dict) and 'image' in item:
            img = item['image']
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            img = item[0]
        else:
            img = None
        if img is None:
            continue
        if img.dim() == 4:
            # już batch; rozbij na pojedyncze
            for j in range(img.size(0)):
                images.append(img[j])
            continue
        if img.shape[0] == 3:
            _, H, W = img.shape
            mask = torch.zeros((1, H, W), device=img.device, dtype=img.dtype)
            img = torch.cat([img, mask], dim=0)
        images.append(img)
    if len(images) == 0:
        return {'image': torch.empty(0)}
    # ujednolić urządzenie, by uniknąć miksu CPU/GPU
    target_device = images[0].device
    images = [img.to(target_device) for img in images]
    return {'image': torch.stack(images, dim=0)}

#%% Make small subset of dataset for quick testing
def make_small_subset(dataloader, subset_fraction=0.125):    
    dataset_size = len(dataloader.dataset)
    subset_size = int(dataset_size * subset_fraction)
    indices = list(range(subset_size))

    small_dataset = Subset(dataloader.dataset, indices)

    small_loader = DataLoader(
        small_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        pin_memory=False,  # dane już są na GPU po preprocess, więc pin_memory wyłączone
        persistent_workers=False,
        collate_fn=safe_collate
    )

    print(f"Oryginalny zbiór: {dataset_size} obrazów")
    print(f"Mniejszy zbiór ({subset_fraction*100:.1f}%): {subset_size} obrazów")
    print(f"Liczba batchy: {len(dataloader)} -> {len(small_loader)}")
    
    return small_loader

#%% Function to shuffle data (correct images and damaged)
def shuffle_data(correct_dataloader, damaged_dataloader, damaged_percent=0.5):
    correct_size = len(correct_dataloader.dataset)
    damaged_size = len(damaged_dataloader.dataset)
    
    num_damaged = int(min(correct_size, damaged_size) * damaged_percent)
    num_correct = int(num_damaged * (1 - damaged_percent) / damaged_percent)
    
    correct_indices = torch.randperm(correct_size)[:num_correct].tolist()
    damaged_indices = torch.randperm(damaged_size)[:num_damaged].tolist()
    
    correct_subset = Subset(correct_dataloader.dataset, correct_indices)
    damaged_subset = Subset(damaged_dataloader.dataset, damaged_indices)
    
    combined_dataset = torch.utils.data.ConcatDataset([correct_subset, damaged_subset])
    
    shuffled_loader = DataLoader(
        combined_dataset,
        batch_size=correct_dataloader.batch_size,
        shuffle=True,
        num_workers=correct_dataloader.num_workers,
        pin_memory=False,  # dane już na GPU po preprocess
        persistent_workers=False,
        collate_fn=safe_collate
    )
    
    print(f"Utworzono dataset: {num_correct} poprawnych + {num_damaged} uszkodzonych = {len(combined_dataset)} obrazów")
    print(f"Stosunek uszkodzeń: {damaged_percent*100:.1f}%")
    
    return shuffled_loader
