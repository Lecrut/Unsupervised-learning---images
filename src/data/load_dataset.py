#%% Imports
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch

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
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Oryginalny zbiór: {dataset_size} obrazów")
    print(f"Mniejszy zbiór ({subset_fraction*100:.1f}%): {subset_size} obrazów")
    print(f"Liczba batchy: {len(dataloader)} -> {len(small_loader)}")
    
    return small_loader
