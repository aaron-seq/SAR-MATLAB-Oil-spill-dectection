"""Segmentation models: classical methods, plus an optional PyTorch U-Net.

The deep-learning symbols are exported lazily so that importing this package
never requires PyTorch. Accessing them without the ``dl`` extra installed
raises an :class:`ImportError` that names the extra.
"""

from typing import Any

from sar_oil_spill.models.traditional_segmentation import (
    METHOD_NAMES,
    SegmentationResult,
    TraditionalSegmentation,
)

_LAZY = {
    "DeepLearningSegmentation",
    "ImprovedUNet",
    "SARDataset",
    "dice_loss",
}

__all__ = [
    "METHOD_NAMES",
    "DeepLearningSegmentation",
    "ImprovedUNet",
    "SARDataset",
    "SegmentationResult",
    "TraditionalSegmentation",
    "dice_loss",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from sar_oil_spill.models import deep_learning_segmentation as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
