#%% Imports
import os
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch
from functools import partial

#%% Constants definitions
DATASET_NAME = "huggan/wikiart"
IMAGE_SIZE = 256
IMAGE_SIZE_BIGGER = 512
NUM_WORKERS = max(1, (os.cpu_count() - 2) // 2)

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Preprocessing function
def preprocess(batch, image_size=IMAGE_SIZE):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    batch['image'] = [transform(img.convert('RGB')) for img in batch['image']]
    return batch

#%% Preprocessing function with alpha channel
def preprocess_with_alpha(batch, image_size=IMAGE_SIZE):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    batch['image'] = [transform(img.convert('RGBA')) for img in batch['image']]
    return batch

#%% Preprocessing function for paired images (small, big)
def preprocess_paired(batch, size_small=IMAGE_SIZE, size_big=IMAGE_SIZE_BIGGER):
    transform_small = transforms.Compose([
        transforms.Resize((size_small, size_small)),
        transforms.ToTensor(),
    ])
    transform_big = transforms.Compose([
        transforms.Resize((size_big, size_big)),
        transforms.ToTensor(),
    ])

    images = [img.convert('RGB') for img in batch['image']]
    batch['small'] = [transform_small(img) for img in images]
    batch['big'] = [transform_big(img) for img in images]
    return batch

#%% Image Dataset Wrapper
class ImageDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item.get('image', None)
        if isinstance(img, list) and len(img) > 0:
            img = img[0]
        return img

#%% Paired Image Dataset Wrapper
class PairedImageDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, patch_size=48, scale_factor=2):
        self.ds = hf_dataset
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
        self.crop = transforms.RandomCrop(patch_size, pad_if_needed=True, padding_mode='reflect')
        self.resize = transforms.Resize((patch_size // scale_factor, patch_size // scale_factor), 
                                      interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['image']
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        hr_patch = self.crop(img)
        
        lr_patch = self.resize(hr_patch)
        
        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

#%% Load dataset and create DataLoaders
def load_data(train_split=0.7, test_split=0.15, batch_size=128, num_workers=NUM_WORKERS, add_fourth_channel=False, use_bigger_image=False):
    image_size = IMAGE_SIZE_BIGGER if use_bigger_image else IMAGE_SIZE
    
    dataset = load_dataset(DATASET_NAME, split='train')
    if add_fourth_channel:
        dataset = dataset.with_transform(partial(preprocess_with_alpha, image_size=image_size))
    else:
        dataset = dataset.with_transform(partial(preprocess, image_size=image_size))
    
    train_size = int(len(dataset) * train_split)
    test_size = int(len(dataset) * test_split)
    
    splits = dataset.train_test_split(train_size=train_size, seed=42)
    train_ds = splits['train']
    temp_ds = splits['test']
    
    test_val_splits = temp_ds.train_test_split(train_size=test_size, seed=42)
    test_ds = test_val_splits['train']
    val_ds = test_val_splits['test']

    train_ds = ImageDatasetWrapper(train_ds)
    test_ds = ImageDatasetWrapper(test_ds)
    val_ds = ImageDatasetWrapper(val_ds)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False if use_bigger_image else True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, val_loader

#%% Load paired dataset and create DataLoaders
def load_paired_data(train_split=0.7, test_split=0.15, batch_size=64, num_workers=NUM_WORKERS, patch_size=48, scale_factor=2):
    dataset = load_dataset(DATASET_NAME, split='train')
    
    train_size = int(len(dataset) * train_split)
    test_size = int(len(dataset) * test_split)
    
    splits = dataset.train_test_split(train_size=train_size, seed=42)
    train_ds = splits['train']
    temp_ds = splits['test']
    
    test_val_splits = temp_ds.train_test_split(train_size=test_size, seed=42)
    test_ds = test_val_splits['train']
    val_ds = test_val_splits['test']

    train_ds = PairedImageDatasetWrapper(train_ds, patch_size=patch_size, scale_factor=scale_factor)
    test_ds = PairedImageDatasetWrapper(test_ds, patch_size=patch_size, scale_factor=scale_factor)
    val_ds = PairedImageDatasetWrapper(val_ds, patch_size=patch_size, scale_factor=scale_factor)
    
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
        pin_memory=False,
        persistent_workers=True
    )

    print(f"Oryginalny zbiór: {dataset_size} obrazów")
    print(f"Mniejszy zbiór ({subset_fraction*100:.1f}%): {subset_size} obrazów")
    print(f"Liczba batchy: {len(dataloader)} -> {len(small_loader)}")
    
    return small_loader

#%% Custom collate function to concatenate images from different datasets
def concatenate_fn(batch):
    if not batch:
        return torch.tensor([])

    if not all(isinstance(x, torch.Tensor) for x in batch):
        raise TypeError("concatenate_fn expected all batch elements to be Tensor")

    batch_cpu = [x.cpu() if x.is_cuda else x for x in batch]
    return torch.stack(batch_cpu, dim=0)

#%% Function to shuffle data (correct images and damaged)
def shuffle_data(correct_dataloader, damaged_dataloader, correct_percent=0.5, damaged_percent=0.5, shuffle=True):
    correct_size = len(correct_dataloader.dataset)
    damaged_size = len(damaged_dataloader.dataset)
    
    num_damaged = int(damaged_size * damaged_percent)
    num_correct = int(correct_size * correct_percent)
    
    correct_indices = torch.randperm(correct_size)[:num_correct].tolist()
    damaged_indices = torch.randperm(damaged_size)[:num_damaged].tolist()
    
    correct_subset = Subset(correct_dataloader.dataset, correct_indices)
    damaged_subset = Subset(damaged_dataloader.dataset, damaged_indices)
    
    combined_dataset = ConcatDataset([correct_subset, damaged_subset])
    
    shuffled_loader = DataLoader(
        combined_dataset,
        batch_size=correct_dataloader.batch_size,
        shuffle=shuffle,
        collate_fn=concatenate_fn,
        num_workers=max(1, correct_dataloader.num_workers),
        pin_memory=False,
        persistent_workers=True
    )
    
    print(f"Utworzono dataset: {num_correct} poprawnych + {num_damaged} uszkodzonych = {len(combined_dataset)} obrazów")
    
    return shuffled_loader
