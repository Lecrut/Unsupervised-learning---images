"""
Klasy pomocnicze do ograniczania i próbkowania datasetów
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import Optional, List, Union


class LimitedDataset(Dataset):
    """
    Wrapper który ogranicza liczbę próbek w datasecie
    """
    
    def __init__(self, 
                 base_dataset: Dataset, 
                 max_samples: Optional[int] = None,
                 random_sample: bool = True,
                 random_seed: int = 42):
        """
        Args:
            base_dataset: oryginalny dataset
            max_samples: maksymalna liczba próbek (None = wszystkie)
            random_sample: czy losowo próbkować czy brać pierwsze N
            random_seed: seed dla reprodukowalności
        """
        self.base_dataset = base_dataset
        self.max_samples = max_samples
        self.random_sample = random_sample
        
        # Określ które indeksy użyć
        total_samples = len(base_dataset)
        
        if max_samples is None or max_samples >= total_samples:
            self.indices = list(range(total_samples))
        else:
            if random_sample:
                # Losowe próbkowanie
                np.random.seed(random_seed)
                self.indices = np.random.choice(
                    total_samples, 
                    size=max_samples, 
                    replace=False
                ).tolist()
            else:
                # Pierwsze N próbek
                self.indices = list(range(max_samples))
        
        print(f"Dataset ograniczony do {len(self.indices)} próbek "
              f"(z {total_samples} dostępnych)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]


class StratifiedLimitedDataset(Dataset):
    """
    Ogranicza dataset zachowując proporcje klas (dla klasyfikacji)
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 max_samples: int,
                 samples_per_class: Optional[int] = None,
                 random_seed: int = 42):
        """
        Args:
            base_dataset: dataset z etykietami
            max_samples: maksymalna liczba próbek
            samples_per_class: próbek na klasę (None = automatycznie)
            random_seed: seed dla reprodukowalności
        """
        self.base_dataset = base_dataset
        np.random.seed(random_seed)
        
        # Zbierz informacje o klasach
        class_indices = {}
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        n_classes = len(class_indices)
        
        # Oblicz ile próbek na klasę
        if samples_per_class is None:
            samples_per_class = max_samples // n_classes
        
        # Próbkuj z każdej klasy
        selected_indices = []
        for class_label, indices in class_indices.items():
            n_available = len(indices)
            n_to_sample = min(samples_per_class, n_available)
            
            sampled = np.random.choice(
                indices, 
                size=n_to_sample, 
                replace=False
            )
            selected_indices.extend(sampled)
        
        self.indices = selected_indices[:max_samples]  # Ostateczne ograniczenie
        
        print(f"Stratyfikowany dataset: {len(self.indices)} próbek "
              f"z {n_classes} klas (≤{samples_per_class} na klasę)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]


def create_subset_by_classes(dataset: Dataset, 
                           target_classes: List[int],
                           max_samples_per_class: Optional[int] = None) -> Dataset:
    """
    Tworzy podzbiór datasetu zawierający tylko wybrane klasy
    
    Args:
        dataset: oryginalny dataset
        target_classes: lista klas do zachowania
        max_samples_per_class: max próbek na klasę
        
    Returns:
        Subset z wybranymi klasami
    """
    indices = []
    class_counts = {cls: 0 for cls in target_classes}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        
        if label in target_classes:
            if max_samples_per_class is None or class_counts[label] < max_samples_per_class:
                indices.append(idx)
                class_counts[label] += 1
    
    print(f"Wybrane klasy {target_classes}: {len(indices)} próbek")
    for cls, count in class_counts.items():
        print(f"  Klasa {cls}: {count} próbek")
    
    return Subset(dataset, indices)


def sample_dataset(dataset: Dataset,
                  sample_size: Union[int, float],
                  method: str = 'random',
                  random_seed: int = 42) -> Dataset:
    """
    Próbkuje dataset różnymi metodami
    
    Args:
        dataset: oryginalny dataset
        sample_size: liczba próbek (int) lub procent (float 0-1)
        method: metoda próbkowania ('random', 'first', 'stratified')
        random_seed: seed
        
    Returns:
        Dataset z próbkami
    """
    total_size = len(dataset)
    
    # Oblicz docelowy rozmiar
    if isinstance(sample_size, float):
        if 0 < sample_size <= 1:
            target_size = int(total_size * sample_size)
        else:
            raise ValueError("Float sample_size musi być między 0 a 1")
    else:
        target_size = min(sample_size, total_size)
    
    print(f"Próbkowanie: {target_size} z {total_size} próbek ({target_size/total_size*100:.1f}%)")
    
    if method == 'random':
        return LimitedDataset(dataset, target_size, random_sample=True, random_seed=random_seed)
    elif method == 'first':
        return LimitedDataset(dataset, target_size, random_sample=False)
    elif method == 'stratified':
        return StratifiedLimitedDataset(dataset, target_size, random_seed=random_seed)
    else:
        raise ValueError(f"Nieznana metoda: {method}. Dostępne: 'random', 'first', 'stratified'")


class QuickTestDataset(Dataset):
    """
    Bardzo mały dataset do szybkich testów
    """
    
    def __init__(self, base_dataset: Dataset, size: int = 50):
        """
        Args:
            base_dataset: oryginalny dataset
            size: rozmiar testowego datasetu
        """
        self.base_dataset = base_dataset
        self.size = min(size, len(base_dataset))
        
        # Weź pierwsze N próbek dla szybkości
        self.indices = list(range(self.size))
        
        print(f"⚡ Quick test dataset: {self.size} próbek dla szybkich testów")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]