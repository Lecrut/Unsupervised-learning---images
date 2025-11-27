#%% Imports
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
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

#%% Preprocessing function with alpha channel
def preprocess_with_alpha(batch):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    batch['image'] = [transform(img.convert('RGBA')).to(device) for img in batch['image']]
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

#%% Load dataset and create DataLoaders
def load_data(train_split=0.7, test_split=0.15, batch_size=32, num_workers=4, add_fourth_channel=False):
    dataset = load_dataset(DATASET_NAME, split='train')
    if add_fourth_channel:
        dataset = dataset.with_transform(preprocess_with_alpha)
    else:
        dataset = dataset.with_transform(preprocess)
    
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

    batch_cpu = [x.cpu() for x in batch]
    return torch.stack(batch_cpu, dim=0)

#%% Function to shuffle data (correct images and damaged)
def shuffle_data(correct_dataloader, damaged_dataloader, damaged_percent=0.5):
    correct_size = len(correct_dataloader.dataset)
    damaged_size = len(damaged_dataloader.dataset)
    
    num_damaged = int(min(correct_size, damaged_size) * damaged_percent)
    num_correct = max(1, int(num_damaged * (1 - damaged_percent) / damaged_percent))
    
    correct_indices = torch.randperm(correct_size)[:num_correct].tolist()
    damaged_indices = torch.randperm(damaged_size)[:num_damaged].tolist()
    
    correct_subset = Subset(correct_dataloader.dataset, correct_indices)
    damaged_subset = Subset(damaged_dataloader.dataset, damaged_indices)
    
    combined_dataset = ConcatDataset([correct_subset, damaged_subset])
    
    shuffled_loader = DataLoader(
        combined_dataset,
        batch_size=correct_dataloader.batch_size,
        shuffle=True,
        collate_fn=concatenate_fn,
        num_workers=max(1, correct_dataloader.num_workers),
        pin_memory=False,
        persistent_workers=True
    )
    
    print(f"Utworzono dataset: {num_correct} poprawnych + {num_damaged} uszkodzonych = {len(combined_dataset)} obrazów")
    print(f"Stosunek uszkodzeń: {damaged_percent*100:.1f}%")
    
    return shuffled_loader
