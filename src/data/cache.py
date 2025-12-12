#%% Imports
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import islice

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
def create_image_dataloader(folder, transform=None, batch_size=64, num_workers=10):
    print(f"Tworzenie DataLoadera z obrazów w {folder}")
    dataset = ImageFolderDataset(folder, transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader

#%% Save batch of images to disk
def process_batch_for_save(args):
    batch_idx, batch, damaged_dir = args
    images, masks = batch
    images = images.cpu()
    masks = masks.cpu()
    
    saved_paths = []
    for img_idx, img in enumerate(images):
        img_path = os.path.join(damaged_dir, f"img_{batch_idx}_{img_idx}.png")
        save_image(img, img_path)  
        saved_paths.append(img_path)
    return saved_paths

#%% Load or create damaged DataLoader
def load_or_create_damaged_loader(original_loader, damaged_dir, transform=None, batch_size=64):

    if os.path.exists(damaged_dir) and any(f.endswith('.png') for f in os.listdir(damaged_dir)):
        print(f"Ładowanie uszkodzonych obrazów z {damaged_dir}")
        return create_image_dataloader(damaged_dir, transform, batch_size)

    print("Generowanie uszkodzonych obrazów...")
    os.makedirs(damaged_dir, exist_ok=True)

    damaged_loader = make_damage_loader(original_loader)
    batch_size_chunk = 10  

    total_batches = len(damaged_loader)
    num_workers = min(cpu_count(), 10) 

    with Pool(num_workers) as pool:
        batch_idx = 0
        iterator = iter(damaged_loader)
        with tqdm(total=total_batches, desc="Zapisywanie obrazów") as pbar:
            while True:
                chunk = list(islice(iterator, batch_size_chunk))
                if not chunk:
                    break
                cpu_chunk = []
                for batch in chunk:
                    images, masks = batch
                    images = images.cpu()
                    masks = masks.cpu()
                    cpu_chunk.append((images, masks))
                args_list = [(batch_idx + i, cpu_batch, damaged_dir) for i, cpu_batch in enumerate(cpu_chunk)]
                list(pool.imap(process_batch_for_save, args_list))
                batch_idx += len(chunk)
                pbar.update(len(chunk))

    print(f"Zapisano uszkodzone obrazy do {damaged_dir}")

    return create_image_dataloader(damaged_dir, transform, batch_size)
