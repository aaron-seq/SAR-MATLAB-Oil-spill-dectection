# SAR Oil Spill Detection System
# Modern Python package for oil spill detection in SAR images

__version__ = "2.0.0"
__author__ = "Aaron Sequeira"
__email__ = "aaronsequeira12@gmail.com"

from .core import SARImageProcessor, OilSpillDetector
from .models import DeepLearningSegmentation, TraditionalSegmentation
from .utils import PerformanceEvaluator, DataVisualizer

__all__ = [
    "SARImageProcessor",
    "OilSpillDetector", 
    "DeepLearningSegmentation",
    "TraditionalSegmentation",
    "PerformanceEvaluator",
    "DataVisualizer"
]