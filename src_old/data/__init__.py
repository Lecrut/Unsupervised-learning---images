"""
Funkcje do przetwarzania danych i generowania uszkodzeń
"""

from .damages import (
    MaskedDataset, random_mask, rectangular_mask, noise_mask,
    line_damage, circular_mask, MultiDamageDataset
)
from .sampling import LimitedDataset, StratifiedLimitedDataset, sample_dataset, QuickTestDataset
from .splitting import (
    split_dataset, stratified_split, temporal_split,
    CrossValidationSplitter, balance_dataset, create_subset_by_class
)
from .augmentations import (
    get_training_augmentation, get_validation_transform, get_test_time_augmentation,
    AugmentedDataset, DualAugmentedDataset
)

__all__ = [
    # Damages
    'MaskedDataset', 'random_mask', 'rectangular_mask', 'noise_mask',
    'line_damage', 'circular_mask', 'MultiDamageDataset',
    # Sampling
    'LimitedDataset', 'StratifiedLimitedDataset', 'sample_dataset', 'QuickTestDataset',
    # Splitting
    'split_dataset', 'stratified_split', 'temporal_split',
    'CrossValidationSplitter', 'balance_dataset', 'create_subset_by_class',
    # Augmentations
    'get_training_augmentation', 'get_validation_transform', 'get_test_time_augmentation',
    'AugmentedDataset', 'DualAugmentedDataset'
]