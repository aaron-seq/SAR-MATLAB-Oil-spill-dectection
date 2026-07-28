"""SAR preprocessing and the end-to-end detection pipeline."""

from sar_oil_spill.core.oil_spill_detector import DetectionResult, OilSpillDetector
from sar_oil_spill.core.sar_image_processor import SARImageProcessor

__all__ = ["DetectionResult", "OilSpillDetector", "SARImageProcessor"]
