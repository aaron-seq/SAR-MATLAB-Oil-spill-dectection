"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from sar_oil_spill.core import OilSpillDetector, SARImageProcessor
from sar_oil_spill.data import generate_sar_scene


@pytest.fixture
def processor() -> SARImageProcessor:
    """A processor targeting a small size, to keep tests fast."""
    return SARImageProcessor(target_image_size=(128, 128))


@pytest.fixture
def detector() -> OilSpillDetector:
    """A detector using the built-in default settings."""
    return OilSpillDetector()


@pytest.fixture
def scene():
    """A deterministic synthetic SAR scene with a known oil mask."""
    return generate_sar_scene(size=(256, 256), n_slicks=1, seed=7)


@pytest.fixture
def speckled_image() -> np.ndarray:
    """A small speckled image with two dark patches standing in for slicks."""
    rng = np.random.default_rng(42)
    base = np.full((128, 128), 120.0, dtype=np.float32)
    base[40:60, 40:60] = 35.0
    base[80:100, 20:40] = 45.0
    speckle = rng.gamma(shape=4.0, scale=0.25, size=base.shape)
    return (base * speckle).astype(np.float32)
