"""Typed configuration for the SAR oil spill detection system.

Configuration is read from a YAML file (``config/model_config.yaml`` by
default) into frozen dataclasses, so a typo in the YAML surfaces immediately
rather than as a ``KeyError`` deep inside a processing loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/model_config.yaml")


@dataclass(frozen=True)
class ImageProcessingSettings:
    """Preprocessing behaviour applied before any segmentation runs."""

    target_size: tuple[int, int] = (512, 512)
    normalize: bool = True
    despeckle: bool = True
    despeckle_filter: str = "lee"
    despeckle_window: int = 7

    enhance_contrast: bool = False
    """Off by default. CLAHE and histogram equalisation are *display*
    transforms: they normalise local contrast, which erases the very
    slick-versus-sea intensity gap that detection depends on. Measured on the
    synthetic benchmark, enabling CLAHE drops mean IoU from 0.85 to 0.32."""

    enhancement_method: str = "none"


@dataclass(frozen=True)
class AdaptiveThresholdSettings:
    background_window: int = 251
    """Window for the large-scale background estimate. Must be considerably
    wider than the slicks themselves, otherwise the background tracks the
    slick and the contrast cancels out."""

    offset: float = 0.0
    """Extra margin subtracted from the Otsu threshold on the ratio image."""

    min_blob_area: int = 100
    median_filter_size: int = 3


@dataclass(frozen=True)
class KMeansSettings:
    n_clusters: int = 5
    max_iter: int = 350
    n_init: int = 5
    random_state: int = 42
    n_largest_blobs: int = 3
    gaussian_sigma: float = 1.5


@dataclass(frozen=True)
class SuperpixelSettings:
    n_segments: int = 1200
    compactness: float = 10.0
    sigma: float = 1.0
    max_centroid_distance: float = 450.0


@dataclass(frozen=True)
class FuzzyEdgeSettings:
    lee_window: int = 5
    sigma: float = 0.1
    """Width of the "gradient is zero" membership function."""

    binarize_threshold: float = 0.7
    """Minimum uniformity for a pixel to count as smooth (i.e. candidate oil)."""


@dataclass(frozen=True)
class TraditionalMethodSettings:
    adaptive_threshold: AdaptiveThresholdSettings = field(
        default_factory=AdaptiveThresholdSettings
    )
    kmeans: KMeansSettings = field(default_factory=KMeansSettings)
    superpixel: SuperpixelSettings = field(default_factory=SuperpixelSettings)
    fuzzy_edge: FuzzyEdgeSettings = field(default_factory=FuzzyEdgeSettings)


@dataclass(frozen=True)
class DeepLearningSettings:
    batch_size: int = 8
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 15
    architecture: str = "unet"
    base_channels: int = 32
    num_classes: int = 1
    dropout: float = 0.2
    checkpoint_dir: str = "models/saved_models"


@dataclass(frozen=True)
class EvaluationSettings:
    metrics: tuple[str, ...] = (
        "jaccard",
        "dice",
        "pixel_accuracy",
        "precision",
        "recall",
        "boundary_f1",
    )
    boundary_tolerance: int = 2
    save_predictions: bool = True


@dataclass(frozen=True)
class PathSettings:
    results_dir: str = "results"
    logs_dir: str = "logs"
    model_save_dir: str = "models/saved_models"


@dataclass(frozen=True)
class SystemSettings:
    log_level: str = "INFO"
    random_seed: int = 42
    use_gpu: bool = True


@dataclass(frozen=True)
class Settings:
    """Root configuration object."""

    image_processing: ImageProcessingSettings = field(
        default_factory=ImageProcessingSettings
    )
    traditional_methods: TraditionalMethodSettings = field(
        default_factory=TraditionalMethodSettings
    )
    deep_learning: DeepLearningSettings = field(default_factory=DeepLearningSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    system: SystemSettings = field(default_factory=SystemSettings)


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    """Recursively instantiate ``cls`` from a plain mapping, ignoring extras.

    ``from __future__ import annotations`` turns field annotations into strings,
    so the real types are resolved with :func:`typing.get_type_hints`.
    """
    if raw is None:
        return cls()

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", cls.__name__, sorted(unknown))

    kwargs: dict[str, Any] = {}
    for name in known & set(raw):
        value = raw[name]
        hint = hints[name]
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = _build(hint, value)
        elif get_origin(hint) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load :class:`Settings` from YAML, falling back to built-in defaults.

    A missing file is not an error -- the defaults are usable out of the box,
    which keeps the demo and test paths free of required setup.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.info("No config file at %s; using built-in defaults.", path)
        return Settings()

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration at {path} must be a YAML mapping.")

    settings = Settings(
        image_processing=_build(ImageProcessingSettings, raw.get("image_processing")),
        traditional_methods=TraditionalMethodSettings(
            adaptive_threshold=_build(
                AdaptiveThresholdSettings,
                (raw.get("traditional_methods") or {}).get("adaptive_threshold"),
            ),
            kmeans=_build(
                KMeansSettings, (raw.get("traditional_methods") or {}).get("kmeans")
            ),
            superpixel=_build(
                SuperpixelSettings,
                (raw.get("traditional_methods") or {}).get("superpixel"),
            ),
            fuzzy_edge=_build(
                FuzzyEdgeSettings,
                (raw.get("traditional_methods") or {}).get("fuzzy_edge"),
            ),
        ),
        deep_learning=_build(DeepLearningSettings, raw.get("deep_learning")),
        evaluation=_build(EvaluationSettings, raw.get("evaluation")),
        paths=_build(PathSettings, raw.get("paths")),
        system=_build(SystemSettings, raw.get("system")),
    )
    logger.info("Loaded configuration from %s", path)
    return settings


def configure_logging(level: str = "INFO") -> None:
    """Install a consistent log format for CLI and API entry points."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
