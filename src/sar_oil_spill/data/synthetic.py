"""Synthetic SAR scene generator.

The original dataset is not redistributable, which historically made this
repository impossible to run without one. This module builds physically
plausible SAR scenes with known ground truth so the pipeline, the tests, the
API and the documentation figures all work from a clean clone.

The model follows standard SAR imaging assumptions:

* Backscatter is **multiplicative** speckle, not additive Gaussian noise, and
  the intensity of an L-look image follows a Gamma distribution with shape L.
* Oil dampens capillary waves, lowering the normalised radar cross-section by
  roughly 5-12 dB relative to the surrounding sea.
* Land is a strong, rough scatterer and appears bright.
* Wind-driven variation makes the sea's mean brightness drift across the swath.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticScene:
    """A generated SAR scene with its ground truth."""

    image: np.ndarray
    """``float32`` amplitude image scaled to ``[0, 255]``."""

    oil_mask: np.ndarray
    """Boolean ground truth for the oil slick."""

    land_mask: np.ndarray
    """Boolean ground truth for land; all ``False`` for open-sea scenes."""

    metadata: dict[str, float]

    @property
    def oil_fraction(self) -> float:
        """Fraction of the scene covered by oil."""
        return float(self.oil_mask.mean())


def _smooth_noise(
    shape: tuple[int, int], scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Band-limited noise in ``[0, 1]``, used to build organic shapes."""
    field = rng.standard_normal(shape)
    field = ndimage.gaussian_filter(field, sigma=scale)
    spread = field.max() - field.min()
    return (field - field.min()) / spread if spread > 0 else np.zeros(shape)


def _blob(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    irregularity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """An irregular filled blob, closer to a real slick than a plain ellipse."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    distance = np.hypot(rows - center[0], cols - center[1])

    # Perturb the radius with smooth noise so the outline is ragged.
    warp = _smooth_noise(shape, scale=max(2.0, radius / 6.0), rng=rng)
    effective_radius = radius * (1.0 + irregularity * (warp - 0.5) * 2.0)
    return distance < effective_radius


def generate_sar_scene(
    size: tuple[int, int] = (512, 512),
    *,
    with_land: bool = False,
    n_slicks: int = 2,
    oil_contrast_db: float = 8.0,
    looks: int = 4,
    seed: int | None = None,
) -> SyntheticScene:
    """Generate one synthetic SAR scene containing oil slicks.

    Args:
        size: Scene shape as ``(height, width)``.
        with_land: Add a bright land mass along one edge.
        n_slicks: Number of separate slicks; the first is the largest.
        oil_contrast_db: Backscatter drop inside the slick, in decibels.
            Real Sentinel-1 slicks typically sit between 5 and 12 dB.
        looks: Number of looks (Gamma shape). Higher means less speckle.
        seed: Seed for reproducible scenes.

    Returns:
        A :class:`SyntheticScene` with the image and both ground-truth masks.
    """
    rng = np.random.default_rng(seed)
    height, width = size

    # --- Sea background: mean backscatter drifts with wind across the swath.
    gradient = np.linspace(0.85, 1.15, width, dtype=np.float32)[None, :]
    wind_texture = 0.9 + 0.2 * _smooth_noise((height, width), scale=height / 8.0, rng=rng)
    sea_sigma0 = (gradient * wind_texture).astype(np.float32)

    # --- Land: a bright, rough region occupying one edge.
    # Annotated so the type stays "boolean array" whatever shape numpy's stubs
    # infer: numpy 2.2 types np.zeros as fixed 2-D and rejects the reassignment
    # below, while numpy 2.4 does not -- mypy passed on 3.11 and failed on 3.10.
    land_mask: npt.NDArray[np.bool_] = np.zeros((height, width), dtype=bool)
    if with_land:
        coastline = (
            0.22 * width
            + 0.10 * width * _smooth_noise((height, width), scale=height / 10.0, rng=rng)[:, :1]
        )
        columns = np.arange(width)[None, :]
        land_mask = columns < coastline
        # Land returns several times more energy than a calm sea surface.
        sea_sigma0 = np.where(land_mask, sea_sigma0 * 3.2, sea_sigma0)

    # --- Oil slicks: multiplicative damping of the sea's backscatter.
    damping = 10.0 ** (-oil_contrast_db / 10.0)
    oil_mask = np.zeros((height, width), dtype=bool)

    for index in range(max(0, n_slicks)):
        # Successive slicks are smaller satellites of the main one.
        radius = (min(height, width) / 6.0) * (0.85**index) * rng.uniform(0.7, 1.2)
        margin = radius * 1.4
        low_col = margin + (0.35 * width if with_land else 0.0)
        center = (
            rng.uniform(margin, height - margin),
            rng.uniform(min(low_col, width - margin - 1), width - margin),
        )
        blob = _blob((height, width), center, radius, irregularity=0.45, rng=rng)
        oil_mask |= blob & ~land_mask

    # Slick edges are gradual, not step changes, so feather the damping.
    feathered = ndimage.gaussian_filter(oil_mask.astype(np.float32), sigma=2.0)
    sigma0 = sea_sigma0 * (1.0 - feathered * (1.0 - damping))

    # --- Speckle: Gamma-distributed intensity with unit mean, L looks.
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=(height, width))
    intensity = sigma0 * speckle
    amplitude = np.sqrt(np.clip(intensity, 0.0, None))

    # Scale to 8-bit display range against a robust upper percentile so a few
    # bright land pixels do not crush the sea's dynamic range.
    upper = float(np.percentile(amplitude, 99.5)) or 1.0
    image = np.clip(amplitude / upper, 0.0, 1.0) * 255.0

    return SyntheticScene(
        image=image.astype(np.float32),
        oil_mask=oil_mask,
        land_mask=land_mask,
        metadata={
            "oil_contrast_db": float(oil_contrast_db),
            "looks": float(looks),
            "n_slicks": float(n_slicks),
            "oil_fraction": float(oil_mask.mean()),
            "land_fraction": float(land_mask.mean()),
        },
    )


def generate_dataset(
    count: int = 10,
    size: tuple[int, int] = (512, 512),
    *,
    land_ratio: float = 0.3,
    seed: int = 42,
) -> list[SyntheticScene]:
    """Generate a reproducible list of scenes for benchmarking.

    Args:
        count: How many scenes to produce.
        size: Scene shape as ``(height, width)``.
        land_ratio: Fraction of scenes that include a land mass.
        seed: Base seed; scene ``i`` uses ``seed + i`` so runs are repeatable.
    """
    scenes = []
    for index in range(count):
        rng = np.random.default_rng(seed + index)
        scenes.append(
            generate_sar_scene(
                size=size,
                with_land=bool(rng.random() < land_ratio),
                n_slicks=int(rng.integers(1, 4)),
                oil_contrast_db=float(rng.uniform(6.0, 11.0)),
                looks=int(rng.integers(3, 8)),
                seed=seed + index,
            )
        )
    logger.info("Generated %d synthetic SAR scenes at %s", count, size)
    return scenes


def save_scene(scene: SyntheticScene, image_path: str, mask_path: str) -> None:
    """Write a scene and its oil mask to disk as 8-bit PNGs."""
    import cv2

    cv2.imwrite(image_path, scene.image.astype(np.uint8))
    cv2.imwrite(mask_path, (scene.oil_mask.astype(np.uint8) * 255))
