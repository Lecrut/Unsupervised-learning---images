"""
Funkcje pomocnicze do trenowania, wizualizacji i analizy
"""

from .training import train_autoencoder, validate_autoencoder, train_with_validation
from .analysis import (
    extract_latent_vectors, cluster_and_visualize, cluster_latent_space,
    reduce_dimensionality, analyze_cluster_characteristics
)
from .visualization import (
    visualize_reconstructions, plot_training_history,
    visualize_latent_space_2d, plot_cluster_analysis, compare_damage_types
)
from .local_logger import LocalLogger
from .metrics import (
    calculate_psnr, calculate_ssim, calculate_ms_ssim,
    calculate_mse, calculate_mae, evaluate_reconstruction,
    compare_models, CombinedLoss, PerceptualLoss
)

__all__ = [
    # Training
    'train_autoencoder', 'validate_autoencoder', 'train_with_validation',
    # Analysis
    'extract_latent_vectors', 'cluster_and_visualize', 'cluster_latent_space',
    'reduce_dimensionality', 'analyze_cluster_characteristics',
    # Visualization
    'visualize_reconstructions', 'plot_training_history',
    'visualize_latent_space_2d', 'plot_cluster_analysis', 'compare_damage_types',
    # Logging
    'LocalLogger',
    # Metrics
    'calculate_psnr', 'calculate_ssim', 'calculate_ms_ssim',
    'calculate_mse', 'calculate_mae', 'evaluate_reconstruction',
    'compare_models', 'CombinedLoss', 'PerceptualLoss'
]