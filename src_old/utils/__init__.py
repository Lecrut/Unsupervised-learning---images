"""
DeepCluster Architecture - Utilities
Funkcje pomocnicze zgodne z wymaganiami projektu
"""

# DeepCluster utilities - tylko potrzebne funkcje
from .training import train_with_validation  # Uniwersalne trenowanie dla DeepCluster
from .visualization import visualize_reconstructions, plot_training_history
from .local_logger import LocalLogger  # Logger lokalny
from .metrics import (
    calculate_psnr, calculate_ssim, calculate_ms_ssim,
    evaluate_reconstruction, CombinedLoss
)
# DeepCluster specific utilities
from .deepcluster_utils import (
    evaluate_deepcluster_pipeline,
    extract_deepcluster_representations, 
    deepcluster_clustering_analysis,
    visualize_deepcluster_clustering,
    deepcluster_stage_analysis,
    print_deepcluster_summary
)

# DeepCluster utilities - eksportowane funkcje
__all__ = [
    # Training (uniwersalne dla DeepCluster)
    'train_with_validation',
    
    # Visualization
    'visualize_reconstructions', 
    'plot_training_history',
    
    # Logging
    'LocalLogger',
    
    # Metrics (PSNR, SSIM - wymagane w projekcie)
    'calculate_psnr', 'calculate_ssim', 'calculate_ms_ssim',
    'evaluate_reconstruction', 'CombinedLoss',
    
    # DeepCluster specific functions
    'evaluate_deepcluster_pipeline',
    'extract_deepcluster_representations',
    'deepcluster_clustering_analysis', 
    'visualize_deepcluster_clustering',
    'deepcluster_stage_analysis',
    'print_deepcluster_summary'
]