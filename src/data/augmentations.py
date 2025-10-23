"""
Augmentacje danych dla obrazów dzieł sztuki.

Zwiększa różnorodność zbioru treningowego bez konieczności 
dodawania nowych danych.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import ImageFilter


class RandomRotationWithBounds(object):
    """
    Losowy obrót z ograniczeniem kątów (żeby nie odwracać obrazów do góry nogami).
    """
    def __init__(self, max_angle=15):
        self.max_angle = max_angle
    
    def __call__(self, img):
        angle = random.uniform(-self.max_angle, self.max_angle)
        return TF.rotate(img, angle)


class RandomPerspective(object):
    """
    Losowa perspektywa - symuluje oglądanie obrazu pod kątem.
    """
    def __init__(self, distortion_scale=0.2, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            startpoints = [(0, 0), (width, 0), (width, height), (0, height)]
            
            # Losowe przemieszczenie narożników
            endpoints = []
            for point in startpoints:
                x = point[0] + random.randint(-int(width * self.distortion_scale), 
                                              int(width * self.distortion_scale))
                y = point[1] + random.randint(-int(height * self.distortion_scale), 
                                              int(height * self.distortion_scale))
                endpoints.append((x, y))
            
            return TF.perspective(img, startpoints, endpoints)
        return img


class ColorJitter(object):
    """
    Losowe zmiany kolorów - symuluje różne oświetlenie i warunki fotografowania.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img):
        return self.color_jitter(img)


class RandomGaussianBlur(object):
    """
    Losowe rozmycie gaussowskie - symuluje zdjęcia spoza ostrości.
    """
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class RandomNoise(object):
    """
    Dodaje losowy szum do obrazu - symuluje artefakty kompresji lub skanowania.
    """
    def __init__(self, noise_level=0.02, p=0.3):
        self.noise_level = noise_level
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            # Konwertuj do tensora jeśli jeszcze nie jest
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)
            
            noise = torch.randn_like(img) * self.noise_level
            img = img + noise
            img = torch.clamp(img, 0, 1)
            
            # Konwertuj z powrotem do PIL
            img = TF.to_pil_image(img)
        
        return img


class RandomErasing(object):
    """
    Losowe wymazywanie fragmentów - augmentacja dla inpaintingu.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        if random.random() < self.p:
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)
            
            return transforms.RandomErasing(
                p=1.0,
                scale=self.scale,
                ratio=self.ratio
            )(img)
        return img


def get_training_augmentation(image_size=256, mode='basic'):
    """
    Zwraca pipeline augmentacji dla treningu.
    
    Args:
        image_size: Rozmiar wyjściowy obrazów
        mode: 'basic', 'medium', 'aggressive'
        
    Returns:
        transforms.Compose z augmentacjami
    """
    
    if mode == 'basic':
        # Minimalne augmentacje - dla szybkiego prototypowania
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    
    elif mode == 'medium':
        # Średnie augmentacje - standardowe dla dzieł sztuki
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotationWithBounds(max_angle=10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            RandomGaussianBlur(p=0.2),
            transforms.ToTensor()
        ])
    
    elif mode == 'aggressive':
        # Silne augmentacje - dla małych datasetów
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotationWithBounds(max_angle=15),
            RandomPerspective(distortion_scale=0.2, p=0.3),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            RandomGaussianBlur(p=0.3),
            RandomNoise(noise_level=0.02, p=0.2),
            transforms.ToTensor(),
            RandomErasing(p=0.3)
        ])
    
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")


def get_validation_transform(image_size=256):
    """
    Zwraca pipeline transformacji dla walidacji/testu (bez augmentacji).
    
    Args:
        image_size: Rozmiar wyjściowy
        
    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])


def get_test_time_augmentation(image_size=256, n_augmentations=5):
    """
    Test-Time Augmentation (TTA) - kilka wersji tego samego obrazu.
    
    Podczas inferecji można uśrednić predykcje z różnych augmentacji
    dla lepszych wyników.
    
    Args:
        image_size: Rozmiar obrazu
        n_augmentations: Liczba różnych augmentacji
        
    Returns:
        Lista transforms.Compose
    """
    augmentations = []
    
    # Original
    augmentations.append(get_validation_transform(image_size))
    
    # Horizontal flip
    augmentations.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ]))
    
    # Slight rotations
    for angle in [-5, 5, 10]:
        if len(augmentations) >= n_augmentations:
            break
        augmentations.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: TF.rotate(img, angle)),
            transforms.ToTensor()
        ]))
    
    return augmentations[:n_augmentations]


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper dodający augmentacje.
    """
    def __init__(self, base_dataset, transform=None, mode='medium'):
        self.base = base_dataset
        self.transform = transform or get_training_augmentation(mode=mode)
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]
        
        # Jeśli obraz już jest tensorem, konwertuj do PIL
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Zastosuj augmentację
        img = self.transform(img)
        
        return img, label


class DualAugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset gdzie input i target mają różne augmentacje.
    
    Przydatne dla inpaintingu - target jest czysty, input ma uszkodzenia + augmentacje.
    """
    def __init__(self, base_dataset, input_transform=None, target_transform=None):
        self.base = base_dataset
        self.input_transform = input_transform or get_training_augmentation(mode='medium')
        self.target_transform = target_transform or get_validation_transform()
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Ta sama seed dla obu transformacji żeby miały te same geometryczne augmentacje
        seed = random.randint(0, 2**32)
        
        random.seed(seed)
        torch.manual_seed(seed)
        input_img = self.input_transform(img)
        
        random.seed(seed)
        torch.manual_seed(seed)
        target_img = self.target_transform(img)
        
        return input_img, target_img
