#%% Imports
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.damage import make_damage_loader

#%% Constants directories
DATASET_DIR = 'data/dataset'
DAMAGED_DATASET_DIR = 'data/damaged_dataset'

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Custom Dataset for loading images from a folder
class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGBA')
        img_tensor = self.to_tensor(img)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor

#%% Create DataLoader from images in a folder
def create_image_dataloader(folder, transform=None, batch_size=32, num_workers=4):
    print(f"Tworzenie DataLoadera z obrazów w {folder}")
    dataset = ImageFolderDataset(folder, transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader

#%% Load or create damaged DataLoader
def load_or_create_damaged_loader(original_loader, damaged_dir, transform=None, batch_size=64):

    if os.path.exists(damaged_dir) and any(f.endswith('.png') for f in os.listdir(damaged_dir)):
        print(f"Ładowanie uszkodzonych obrazów z {damaged_dir}")
        return create_image_dataloader(damaged_dir, transform, batch_size)

    print("Generowanie uszkodzonych obrazów na GPU...")
    os.makedirs(damaged_dir, exist_ok=True)

    damaged_loader = make_damage_loader(original_loader)

    for batch_idx, (images, masks) in enumerate(damaged_loader):
        images = images.to(device)
        masks = masks.to(device)

        for img_idx, img in enumerate(images):
            img_path = os.path.join(damaged_dir, f"img_{batch_idx}_{img_idx}.png")
            save_image(img.cpu(), img_path)

    print(f"Zapisano uszkodzone obrazy do {damaged_dir}")

    return create_image_dataloader(damaged_dir, transform, batch_size)
