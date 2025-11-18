"""
Funkcje do podziału danych na zbiory treningowe, walidacyjne i testowe.

Zgodnie z wymaganiami projektu: train, val, test split.
"""

import torch
from torch.utils.data import Dataset, Subset, random_split
import numpy as np
from typing import Tuple, List, Optional
from collections import defaultdict


def split_dataset(dataset: Dataset, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_seed: int = 42) -> Tuple[Subset, Subset, Subset]:
    """
    Dzieli dataset na zbiory treningowy, walidacyjny i testowy.
    
    Args:
        dataset: Dataset do podziału
        train_ratio: Proporcja danych treningowych (domyślnie 0.7 = 70%)
        val_ratio: Proporcja danych walidacyjnych (domyślnie 0.15 = 15%)
        test_ratio: Proporcja danych testowych (domyślnie 0.15 = 15%)
        random_seed: Seed dla reprodukowalności
        
    Returns:
        Tuple: (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Sumy proporcji muszą wynosić 1.0"
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Użyj torch.random_split dla prostoty
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    print(f"Podział datasetu:")
    print(f"  Train: {len(train_dataset)} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} ({test_ratio*100:.0f}%)")
    
    return train_dataset, val_dataset, test_dataset


def stratified_split(dataset: Dataset,
                     labels: List[int],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_seed: int = 42) -> Tuple[Subset, Subset, Subset]:
    """
    Stratified split - zachowuje proporcje klas w każdym zbiorze.
    
    Ważne dla datasetów z nierównomiernym rozkładem klas (np. style w WikiArt).
    
    Args:
        dataset: Dataset do podziału
        labels: Lista etykiet dla każdej próbki
        train_ratio: Proporcja train
        val_ratio: Proporcja val
        test_ratio: Proporcja test
        random_seed: Seed dla reprodukowalności
        
    Returns:
        Tuple: (train_dataset, val_dataset, test_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_seed)
    
    indices = np.arange(len(dataset))
    
    # Najpierw split train vs (val+test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # Potem split val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    temp_labels = [labels[i] for i in temp_indices]
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    # Stwórz Subsety
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Stratified split:")
    print(f"  Train: {len(train_dataset)} próbek")
    print(f"  Val:   {len(val_dataset)} próbek")
    print(f"  Test:  {len(test_dataset)} próbek")
    
    # Sprawdź czy proporcje klas są zachowane
    _verify_stratification(labels, train_indices, val_indices, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def _verify_stratification(labels: List[int], 
                           train_idx: np.ndarray,
                           val_idx: np.ndarray, 
                           test_idx: np.ndarray):
    """
    Weryfikuje czy stratyfikacja zachowała proporcje klas.
    """
    unique_labels = set(labels)
    
    print("\n  Rozkład klas w zbiorach:")
    for label in sorted(unique_labels)[:5]:  # Pokaż tylko 5 pierwszych
        train_count = sum(1 for i in train_idx if labels[i] == label)
        val_count = sum(1 for i in val_idx if labels[i] == label)
        test_count = sum(1 for i in test_idx if labels[i] == label)
        total = train_count + val_count + test_count
        
        print(f"    Klasa {label}: Train={train_count}/{total} "
              f"Val={val_count}/{total} Test={test_count}/{total}")


def temporal_split(dataset: Dataset,
                  timestamps: List,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15) -> Tuple[Subset, Subset, Subset]:
    """
    Podział czasowy - starsza dane do treningu, nowsze do testu.
    
    Przydatne jeśli dataset ma chronologię (np. daty utworzenia dzieł).
    
    Args:
        dataset: Dataset do podziału
        timestamps: Lista timestampów/dat dla każdej próbki
        train_ratio: Proporcja train
        val_ratio: Proporcja val
        test_ratio: Proporcja test
        
    Returns:
        Tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Sortuj indeksy według timestampów
    sorted_indices = np.argsort(timestamps)
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Podziel chronologicznie
    train_indices = sorted_indices[:train_size]
    val_indices = sorted_indices[train_size:train_size + val_size]
    test_indices = sorted_indices[train_size + val_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Temporal split (chronologiczny):")
    print(f"  Train: {len(train_dataset)} (najstarsze)")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)} (najnowsze)")
    
    return train_dataset, val_dataset, test_dataset


class CrossValidationSplitter:
    """
    K-fold cross validation dla bardziej wiarygodnych wyników.
    
    Przydatne przy małych datasetach.
    """
    
    def __init__(self, n_splits: int = 5, random_seed: int = 42):
        from sklearn.model_selection import KFold
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        self.n_splits = n_splits
        
    def split(self, dataset: Dataset) -> List[Tuple[Subset, Subset]]:
        """
        Zwraca listę (train, val) par dla każdego foldu.
        
        Returns:
            Lista tupli: [(train_dataset_fold1, val_dataset_fold1), ...]
        """
        indices = np.arange(len(dataset))
        folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(indices)):
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            folds.append((train_dataset, val_dataset))
            
            print(f"Fold {fold_idx + 1}/{self.n_splits}: "
                  f"Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        return folds


def create_subset_by_class(dataset: Dataset,
                           labels: List[int],
                           selected_classes: List[int]) -> Subset:
    """
    Tworzy subset zawierający tylko wybrane klasy.
    
    Przydatne do eksperymentów na podzbiorze klas (np. tylko wybrane style).
    
    Args:
        dataset: Pełny dataset
        labels: Lista etykiet
        selected_classes: Wybrane klasy do zachowania
        
    Returns:
        Subset zawierający tylko wybrane klasy
    """
    indices = [i for i, label in enumerate(labels) if label in selected_classes]
    subset = Subset(dataset, indices)
    
    print(f"Utworzono subset z klasami {selected_classes}")
    print(f"  Liczba próbek: {len(subset)}")
    
    return subset


def balance_dataset(dataset: Dataset,
                   labels: List[int],
                   strategy: str = 'undersample') -> Subset:
    """
    Balansuje dataset - wyrównuje liczebności klas.
    
    Args:
        dataset: Dataset do wyważenia
        labels: Lista etykiet
        strategy: 'undersample' (zmniejsz większe klasy) lub 'oversample' (powiel mniejsze)
        
    Returns:
        Zbalansowany Subset
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    
    if strategy == 'undersample':
        # Znajdź najmniejszą klasę
        min_count = min(label_counts.values())
        
        # Wybierz po min_count próbek z każdej klasy
        balanced_indices = []
        for label in label_counts.keys():
            label_indices = [i for i, l in enumerate(labels) if l == label]
            selected = np.random.choice(label_indices, min_count, replace=False)
            balanced_indices.extend(selected)
        
        print(f"Undersample: {len(balanced_indices)} próbek "
              f"(po {min_count} z każdej klasy)")
        
    elif strategy == 'oversample':
        # Znajdź największą klasę
        max_count = max(label_counts.values())
        
        # Powiel próbki z mniejszych klas
        balanced_indices = []
        for label in label_counts.keys():
            label_indices = [i for i, l in enumerate(labels) if l == label]
            selected = np.random.choice(label_indices, max_count, replace=True)
            balanced_indices.extend(selected)
        
        print(f"Oversample: {len(balanced_indices)} próbek "
              f"(po {max_count} z każdej klasy)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return Subset(dataset, balanced_indices)


def get_subset_indices(subset: Subset) -> List[int]:
    """
    Zwraca indeksy oryginalnego datasetu dla Subsetu.
    
    Przydatne przy pracy z zagnieżdżonymi Subsetami.
    """
    if isinstance(subset, Subset):
        return subset.indices
    else:
        return list(range(len(subset)))
