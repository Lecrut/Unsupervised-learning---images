# DeepCluster Architecture - zgodna z wymaganiami projektu
from .deepcluster_modules import IMG, DMG, EMC, PCAModule, ClusA, IMP, DEC, DeepClusterPipeline
from .wrappers import EncoderModel, ClusteringModel, InpaintingModel, SuperResolutionModel, CometModel, ExperimentLogger
from .superres_model import LightweightSuperRes, create_lowres_highres_pairs

# DeepCluster Architecture - wszystkie moduły zgodne z wymaganiami
__all__ = [
    # Core DeepCluster modules (deepcluster_modules.py)
    "IMG",
    "DMG", 
    "EMC",
    "PCAModule",
    "ClusA",
    "IMP",
    "DEC",
    "DeepClusterPipeline",
    
    # Wrapper classes (wrappers.py)
    "EncoderModel",
    "ClusteringModel",
    "InpaintingModel", 
    "SuperResolutionModel",
    "CometModel",
    "ExperimentLogger",
    
    # Super-Resolution support
    "LightweightSuperRes",
    "create_lowres_highres_pairs"
]
