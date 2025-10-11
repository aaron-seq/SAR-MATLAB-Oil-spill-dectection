"""
Core processing modules for SAR oil spill detection.
"""

from .sar_image_processor import SARImageProcessor
from .oil_spill_detector import OilSpillDetector

__all__ = ["SARImageProcessor", "OilSpillDetector"]