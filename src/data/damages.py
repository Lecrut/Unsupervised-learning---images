"""
Funkcje i klasy do generowania różnych typów uszkodzeń w obrazach (inpainting).

Ten moduł zawiera różne metody niszczenia obrazów:
- Losowe maski (random_mask)
- Prostokątne maski (rectangular_mask) 
- Szum (noise_mask)
- Linie i smugi (line_damage)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Any


def random_mask(image: torch.Tensor, mask_size: int = 32) -> torch.Tensor:
    """
    Dodaje białą maskę w losowym miejscu obrazu.
    
    Args:
        image: tensor z obrazem [C, H, W]
        mask_size: rozmiar kwadratu maski
        
    Returns:
        Obraz z nałożoną maską
    """
    img = image.clone()
    _, h, w = img.shape
    
    if mask_size >= min(h, w):
        mask_size = min(h, w) // 2
    
    y = np.random.randint(0, h - mask_size)
    x = np.random.randint(0, w - mask_size)
    img[:, y:y+mask_size, x:x+mask_size] = 1.0  # biała plama
    return img


def rectangular_mask(image: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
    """
    Dodaje prostokątną maskę w centrum obrazu.
    
    Args:
        image: tensor z obrazem [C, H, W]
        mask_ratio: stosunek maski do rozmiaru obrazu (0.0-1.0)
        
    Returns:
        Obraz z nałożoną maską
    """
    img = image.clone()
    _, h, w = img.shape
    
    mask_h = int(h * mask_ratio)
    mask_w = int(w * mask_ratio)
    
    start_h = (h - mask_h) // 2
    start_w = (w - mask_w) // 2
    
    img[:, start_h:start_h+mask_h, start_w:start_w+mask_w] = 1.0
    return img


def noise_mask(image: torch.Tensor, noise_level: float = 0.3) -> torch.Tensor:
    """
    Dodaje szum do losowych pikseli obrazu.
    
    Args:
        image: tensor z obrazem [C, H, W]
        noise_level: procent pikseli do zaszumienia (0.0-1.0)
        
    Returns:
        Obraz z nałożonym szumem
    """
    img = image.clone()
    mask = torch.rand_like(img) < noise_level
    img[mask] = torch.rand_like(img[mask])
    return img


def line_damage(image: torch.Tensor, num_lines: int = 3, line_width: int = 2) -> torch.Tensor:
    """
    Dodaje losowe linie przechodzące przez obraz.
    
    Args:
        image: tensor z obrazem [C, H, W]
        num_lines: liczba linii do narysowania
        line_width: grubość linii
        
    Returns:
        Obraz z nałożonymi liniami
    """
    img = image.clone()
    _, h, w = img.shape
    
    for _ in range(num_lines):
        # Losowe punkty początkowy i końcowy
        y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
        y2, x2 = np.random.randint(0, h), np.random.randint(0, w)
        
        # Rysuj linię (prosta implementacja)
        steps = max(abs(y2-y1), abs(x2-x1))
        if steps > 0:
            for i in range(steps):
                y = int(y1 + (y2-y1) * i / steps)
                x = int(x1 + (x2-x1) * i / steps)
                
                # Dodaj grubość linii
                for dy in range(-line_width//2, line_width//2 + 1):
                    for dx in range(-line_width//2, line_width//2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            img[:, ny, nx] = 1.0  # biała linia
    
    return img


def circular_mask(image: torch.Tensor, radius_ratio: float = 0.1) -> torch.Tensor:
    """
    Dodaje okrągłą maskę w losowym miejscu.
    
    Args:
        image: tensor z obrazem [C, H, W]
        radius_ratio: stosunek promienia do rozmiaru obrazu
        
    Returns:
        Obraz z nałożoną okrągłą maską
    """
    img = image.clone()
    _, h, w = img.shape
    
    radius = int(min(h, w) * radius_ratio)
    center_y = np.random.randint(radius, h - radius)
    center_x = np.random.randint(radius, w - radius)
    
    # Utworz maskę okrągłą
    y, x = np.ogrid[:h, :w]
    mask = (y - center_y)**2 + (x - center_x)**2 <= radius**2
    
    img[:, mask] = 1.0
    return img


class MaskedDataset(Dataset):
    """
    Dataset nakładający losowe uszkodzenia na obrazy.
    
    Obsługuje różne typy uszkodzeń i może losowo wybierać między nimi.
    """
    
    def __init__(self, 
                 base_dataset: Dataset, 
                 damage_functions: list = None,
                 damage_kwargs: dict = None,
                 random_damage: bool = True):
        """
        Args:
            base_dataset: podstawowy dataset z obrazami
            damage_functions: lista funkcji uszkadzających
            damage_kwargs: słownik z argumentami dla funkcji
            random_damage: czy losowo wybierać typ uszkodzenia
        """
        self.base = base_dataset
        
        # Domyślne funkcje uszkadzające
        if damage_functions is None:
            damage_functions = [random_mask, rectangular_mask, noise_mask]
        
        self.damage_functions = damage_functions
        self.damage_kwargs = damage_kwargs or {}
        self.random_damage = random_damage
        
    def __len__(self):
        return len(self.base)
        
    def __getitem__(self, idx):
        img, label = self.base[idx]
        
        if self.random_damage:
            # Losowo wybierz funkcję uszkadzającą
            damage_func = np.random.choice(self.damage_functions)
        else:
            # Użyj pierwszej funkcji z listy
            damage_func = self.damage_functions[0]
        
        # Pobierz argumenty dla tej funkcji
        func_name = damage_func.__name__
        kwargs = self.damage_kwargs.get(func_name, {})
        
        # Zastosuj uszkodzenie
        damaged_img = damage_func(img, **kwargs)
        
        return damaged_img, img  # (uszkodzony, oryginalny)


class MultiDamageDataset(Dataset):
    """
    Dataset stosujący kilka rodzajów uszkodzeń jednocześnie.
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 damage_pipeline: list,
                 damage_probability: float = 0.7):
        """
        Args:
            base_dataset: podstawowy dataset
            damage_pipeline: lista (funkcja, kwargs) do zastosowania
            damage_probability: prawdopodobieństwo zastosowania każdego uszkodzenia
        """
        self.base = base_dataset
        self.damage_pipeline = damage_pipeline
        self.damage_probability = damage_probability
        
    def __len__(self):
        return len(self.base)
        
    def __getitem__(self, idx):
        img, label = self.base[idx]
        damaged_img = img.clone()
        
        for damage_func, kwargs in self.damage_pipeline:
            if np.random.random() < self.damage_probability:
                damaged_img = damage_func(damaged_img, **kwargs)
        
        return damaged_img, img