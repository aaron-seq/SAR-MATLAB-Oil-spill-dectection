"""Loading and describing SAR oil spill datasets from disk.

Expects the layout used by the original MATLAB project::

    dataset_root/
      train/
        images/              # open-sea scenes
        labels/
        images_with_land/    # scenes containing a coastline
        labels_with_land/

Loading is lazy: :meth:`SARDatasetHandler.load_dataset` only indexes paths, and
pixels are read on demand, so an index over thousands of scenes stays cheap.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from sar_oil_spill.config import Settings, as_size
from sar_oil_spill.core.sar_image_processor import SARImageProcessor

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


@dataclass(frozen=True)
class DatasetStatistics:
    """Summary of an indexed dataset."""

    total_images: int
    images_with_land: int
    images_without_land: int
    target_size: tuple[int, int]
    oil_spill_percentage: float
    sampled_images: int
    class_distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SamplePaths:
    """One image/mask pair on disk."""

    name: str
    image_path: Path
    mask_path: Path
    has_land: bool


class SARDatasetHandler:
    """Index and read a SAR oil spill dataset."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.target_size = as_size(self.settings.image_processing.target_size)
        self.processor = SARImageProcessor(target_image_size=self.target_size)
        self.dataset_root: Path | None = None
        self.samples: list[SamplePaths] = []

    # ------------------------------------------------------------- loading

    def load_dataset(self, dataset_root_path: str | Path) -> bool:
        """Index every valid image/mask pair under ``dataset_root_path``.

        Returns:
            ``True`` when at least one pair was found, ``False`` otherwise --
            callers treat a false return as "nothing to do", not a crash.
        """
        root = Path(dataset_root_path)
        if not root.exists():
            logger.error("Dataset root does not exist: %s", root)
            return False

        self.dataset_root = root
        self.samples = []

        # Accept both `root/train/images` and a flat `root/images` layout.
        bases = [root / "train", root] if (root / "train").is_dir() else [root]
        for base in bases:
            self.samples.extend(
                self._index_pairs(base / "images", base / "labels", has_land=False)
            )
            self.samples.extend(
                self._index_pairs(
                    base / "images_with_land", base / "labels_with_land", has_land=True
                )
            )
            if self.samples:
                break

        if not self.samples:
            logger.error("No valid image/mask pairs found under %s", root)
            return False

        logger.info("Indexed %d image/mask pairs from %s", len(self.samples), root)
        return True

    def _index_pairs(
        self, images_dir: Path, labels_dir: Path, has_land: bool
    ) -> list[SamplePaths]:
        """Pair every image with its identically-named mask, by stem."""
        if not images_dir.is_dir() or not labels_dir.is_dir():
            return []

        # Masks are often PNG while images are JPG, so match on stem not name.
        masks_by_stem = {
            path.stem: path
            for path in sorted(labels_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_FORMATS
        }

        pairs: list[SamplePaths] = []
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_FORMATS:
                continue
            mask_path = masks_by_stem.get(image_path.stem)
            if mask_path is None:
                logger.warning("No mask for %s; skipping.", image_path.name)
                continue
            pairs.append(
                SamplePaths(
                    name=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    has_land=has_land,
                )
            )
        return pairs

    # ------------------------------------------------------------- reading

    def get_available_images(self, has_land: bool | None = None) -> list[str]:
        """Names of indexed samples, optionally filtered by land presence."""
        return [s.name for s in self.samples if has_land is None or s.has_land == has_land]

    def load_image_pair(self, image_name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Read and preprocess one sample by name.

        Returns ``(None, None)`` when the sample is unknown or unreadable.
        """
        sample = next((s for s in self.samples if s.name == image_name), None)
        if sample is None:
            logger.error("Unknown sample: %s", image_name)
            return None, None
        return self.read_sample(sample)

    def read_sample(self, sample: SamplePaths) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Read one indexed pair into a preprocessed image and a binary mask."""
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            logger.error("Failed to decode %s", sample.name)
            return None, None

        image = self.processor.resize_image(image.astype(np.float32), self.target_size)
        mask = cv2.resize(
            mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST
        )
        return image, self._binarize_mask(mask)

    def iter_samples(
        self, limit: int | None = None, has_land: bool | None = None
    ) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        """Yield ``(name, image, mask)`` triples, skipping unreadable files."""
        selected = [s for s in self.samples if has_land is None or s.has_land == has_land]
        for sample in selected[:limit]:
            image, mask = self.read_sample(sample)
            if image is not None and mask is not None:
                yield sample.name, image, mask

    @staticmethod
    def _binarize_mask(mask: np.ndarray) -> np.ndarray:
        """Convert a label image to a boolean oil mask.

        The reference labels mark oil as the bright class, so a mid-range
        threshold is enough; anything darker is sea or land.
        """
        return (mask > 127).astype(bool)

    # ---------------------------------------------------------- statistics

    def get_dataset_statistics(self, sample_size: int = 20) -> DatasetStatistics:
        """Summarise the dataset, estimating oil coverage from a subsample."""
        with_land = sum(1 for s in self.samples if s.has_land)

        oil_pixels = 0
        total_pixels = 0
        sampled = 0
        for _, _, mask in self.iter_samples(limit=sample_size):
            oil_pixels += int(np.count_nonzero(mask))
            total_pixels += int(mask.size)
            sampled += 1

        percentage = (oil_pixels / total_pixels * 100) if total_pixels else 0.0

        return DatasetStatistics(
            total_images=len(self.samples),
            images_with_land=with_land,
            images_without_land=len(self.samples) - with_land,
            target_size=self.target_size,
            oil_spill_percentage=percentage,
            sampled_images=sampled,
            class_distribution={
                "sea": total_pixels - oil_pixels,
                "oil_spill": oil_pixels,
            },
        )

    def export_statistics(self, output_dir: str | Path) -> Path:
        """Write the dataset statistics to ``dataset_statistics.json``."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "dataset_statistics.json"
        target.write_text(
            json.dumps(self.get_dataset_statistics().as_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Wrote dataset statistics to %s", target)
        return target
