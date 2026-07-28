"""Oil spill detection and segmentation in Synthetic Aperture Radar (SAR) imagery.

The package is a Python port and modernisation of an original MATLAB research
project (still shipped under ``matlab/``). It offers:

* :mod:`sar_oil_spill.core` -- SAR preprocessing and the detection pipeline.
* :mod:`sar_oil_spill.models` -- traditional and deep-learning segmenters.
* :mod:`sar_oil_spill.data` -- dataset loading and a synthetic SAR generator.
* :mod:`sar_oil_spill.utils` -- evaluation metrics and plotting helpers.
"""

from sar_oil_spill.config import Settings, load_settings
from sar_oil_spill.core import OilSpillDetector, SARImageProcessor
from sar_oil_spill.models import TraditionalSegmentation
from sar_oil_spill.utils import DataVisualizer, PerformanceEvaluator

__version__ = "3.0.0"
__author__ = "Aaron Sequeira"
__email__ = "aaronsequeira12@gmail.com"

__all__ = [
    "DataVisualizer",
    "OilSpillDetector",
    "PerformanceEvaluator",
    "SARImageProcessor",
    "Settings",
    "TraditionalSegmentation",
    "__version__",
    "load_settings",
]
