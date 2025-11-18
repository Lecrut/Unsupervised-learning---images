from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

DATASET_NAME = "huggan/wikiart"
IMAGE_SIZE = 256


class WikiArtDataset(Dataset):
    """Wrapper dla HuggingFace dataset z transformacjami do PyTorch"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform or self._default_transform()
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.transform(image)
        return image


def load_data(train_split=0.7, test_split=0.15, max_samples=None, batch_size=32, num_workers=0):
    """
    Pobiera dataset WikiArt, dzieli na train/test/validation i zwraca DataLoadery
    
    Args:
        train_split: procent danych do treningu (domyślnie 70%)
        test_split: procent danych do testu (domyślnie 15%)
        max_samples: limit sampli (None = wszystkie)
        batch_size: rozmiar batcha dla DataLoader
        num_workers: liczba workerów do ładowania danych
    
    Returns:
        (train_loader, test_loader, validation_loader)
    """
    print(f"Pobieranie datasetu: {DATASET_NAME}")
    
    dataset_full = load_dataset(DATASET_NAME, split='train')
    total_size = len(dataset_full)
    
    print(f"Całkowity rozmiar datasetu: {total_size}")
    
    if max_samples and max_samples < total_size:
        dataset_full = dataset_full.select(range(max_samples))
        total_size = max_samples
        print(f"Ograniczono do: {total_size} sampli")
    
    train_size = int(total_size * train_split)
    test_size = int(total_size * test_split)
    val_size = total_size - train_size - test_size
    
    dataset_train = dataset_full.select(range(train_size))
    dataset_test = dataset_full.select(range(train_size, train_size + test_size))
    dataset_validation = dataset_full.select(range(train_size + test_size, total_size))

    print(f"Train set size: {len(dataset_train)}")
    print(f"Test set size: {len(dataset_test)}")
    print(f"Validation set size: {len(dataset_validation)}")
    
    if len(dataset_train) > 0:
        print(f"Dostępne klucze: {list(dataset_train[0].keys())}")
    
    train_dataset = WikiArtDataset(dataset_train)
    test_dataset = WikiArtDataset(dataset_test)
    val_dataset = WikiArtDataset(dataset_validation)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"DataLoadery utworzone (batch_size={batch_size})")
    
    return train_loader, test_loader, val_loader