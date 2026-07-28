"""End-to-end oil spill detection pipeline.

Ties preprocessing, segmentation and evaluation together behind one object, so
the CLI, the API and the Streamlit app all share exactly one code path -- and
therefore produce identical results for identical inputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sar_oil_spill.config import Settings
from sar_oil_spill.core.sar_image_processor import SARImageProcessor
from sar_oil_spill.models.traditional_segmentation import (
    METHOD_NAMES,
    SegmentationResult,
    TraditionalSegmentation,
)
from sar_oil_spill.utils.performance_evaluator import PerformanceEvaluator, SegmentationMetrics

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Everything one detection run produced."""

    mask: np.ndarray
    """Boolean oil mask at the processed resolution."""

    method: str
    oil_detected: bool
    affected_area_pixels: int
    coverage_fraction: float
    confidence: float
    """Mean margin by which flagged pixels fall below the sea's median level."""

    processing_time: float
    preprocessed_image: np.ndarray
    land_mask: np.ndarray | None = None
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: SegmentationMetrics | None = None
    processing_history: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float | str | bool]:
        """A compact, JSON-safe view suitable for API responses and logs."""
        payload: dict[str, float | str | bool] = {
            "method": self.method,
            "oil_spill_detected": self.oil_detected,
            "affected_area_pixels": self.affected_area_pixels,
            "coverage_percent": round(self.coverage_fraction * 100, 3),
            "confidence_score": round(self.confidence, 4),
            "processing_time_seconds": round(self.processing_time, 4),
        }
        if self.metrics is not None:
            payload |= {
                "jaccard_index": round(self.metrics.jaccard_index, 4),
                "dice_coefficient": round(self.metrics.dice_coefficient, 4),
            }
        return payload


class OilSpillDetector:
    """Preprocess a SAR scene, segment it and optionally score the result."""

    def __init__(
        self,
        settings: Settings | None = None,
        processor: SARImageProcessor | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.processor = processor or SARImageProcessor(
            target_image_size=tuple(self.settings.image_processing.target_size)
        )
        self.segmenter = TraditionalSegmentation(
            settings=self.settings.traditional_methods, processor=self.processor
        )
        self.evaluator = PerformanceEvaluator(
            boundary_tolerance=self.settings.evaluation.boundary_tolerance
        )

    @property
    def available_methods(self) -> tuple[str, ...]:
        """Names accepted by the ``method`` argument of :meth:`detect`."""
        return METHOD_NAMES

    # ------------------------------------------------------------ pipeline

    def preprocess(self, image: np.ndarray, resize: bool = True) -> np.ndarray:
        """Despeckle, enhance and (optionally) resize a raw SAR image.

        Steps are driven entirely by ``settings.image_processing``, so a run is
        reproducible from its config file alone.
        """
        config = self.settings.image_processing
        self.processor.reset_processing_history()
        working = np.asarray(image, dtype=np.float32)

        if config.despeckle:
            working = self.processor.apply_despeckling_filter(
                working,
                filter_type=config.despeckle_filter,
                window_size=config.despeckle_window,
            )
        if config.enhance_contrast:
            working = self.processor.enhance_contrast(working, method=config.enhancement_method)
        if resize:
            working = self.processor.resize_image(working, tuple(config.target_size))
        return working

    def detect(
        self,
        image: np.ndarray,
        method: str = "adaptive_threshold",
        *,
        ground_truth: np.ndarray | None = None,
        mask_land: bool = False,
        resize: bool = True,
        min_area_pixels: int = 100,
    ) -> DetectionResult:
        """Run the full pipeline on one scene.

        Args:
            image: Raw SAR image, any dtype.
            method: One of :attr:`available_methods`.
            ground_truth: Reference mask. When supplied, metrics are computed
                and attached to the result.
            mask_land: Detect land first and exclude it from the oil mask.
                Land is bright and oil is dark, so this mostly guards against
                dark inland features such as lakes and shadowed terrain.
            resize: Resize to the configured target size before segmenting.
            min_area_pixels: Detections smaller than this are discarded.

        Returns:
            A :class:`DetectionResult`.

        Raises:
            ValueError: If ``method`` is unknown or the image is empty.
        """
        if method not in METHOD_NAMES:
            raise ValueError(f"Unknown method '{method}'. Choose from {METHOD_NAMES}.")

        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("Cannot run detection on an empty image.")

        started = time.perf_counter()
        preprocessed = self.preprocess(array, resize=resize)

        segmentation: SegmentationResult = self.segmenter.segment(preprocessed, method)
        mask = segmentation.mask

        land_mask = None
        if mask_land:
            land_mask = self.segmenter.detect_land(preprocessed)
            mask = mask & ~land_mask

        if min_area_pixels > 0:
            mask = self.processor.remove_small_objects(mask, min_area_pixels).astype(bool)

        metrics = None
        if ground_truth is not None:
            metrics = self.evaluator.evaluate(mask, self._align(ground_truth, mask.shape))

        return DetectionResult(
            mask=mask,
            method=method,
            oil_detected=bool(mask.any()),
            affected_area_pixels=int(np.count_nonzero(mask)),
            coverage_fraction=float(mask.mean()) if mask.size else 0.0,
            confidence=self._confidence(preprocessed, mask),
            processing_time=time.perf_counter() - started,
            preprocessed_image=preprocessed,
            land_mask=land_mask,
            stages=segmentation.stages,
            metrics=metrics,
            processing_history=self.processor.get_processing_summary(),
        )

    def detect_from_file(
        self, image_path: str | Path, method: str = "adaptive_threshold", **kwargs: object
    ) -> DetectionResult:
        """Load an image from disk and run :meth:`detect` on it.

        Raises:
            FileNotFoundError: If the image cannot be read or decoded.
        """
        image = self.processor.load_sar_image(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read a SAR image from {image_path}")
        return self.detect(image, method=method, **kwargs)  # type: ignore[arg-type]

    def compare_methods(
        self,
        image: np.ndarray,
        methods: list[str] | None = None,
        ground_truth: np.ndarray | None = None,
        **kwargs: object,
    ) -> dict[str, DetectionResult]:
        """Run several methods on the same scene for side-by-side comparison.

        A method that fails is logged and skipped rather than aborting the
        whole comparison.
        """
        selected = methods or list(METHOD_NAMES)
        results: dict[str, DetectionResult] = {}

        for name in selected:
            try:
                results[name] = self.detect(
                    image, method=name, ground_truth=ground_truth, **kwargs
                )  # type: ignore[arg-type]
            except Exception as error:
                logger.error("Method '%s' failed: %s", name, error)

        return results

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _confidence(image: np.ndarray, mask: np.ndarray) -> float:
        """How confidently the flagged pixels read as oil, in ``[0, 1]``.

        Oil is dark, so confidence is the normalised gap between the sea's
        median level and the mean level inside the detection. A detection only
        marginally darker than the background scores near zero.
        """
        if not mask.any() or image.size == 0:
            return 0.0

        background = image[~mask]
        if background.size == 0:
            return 0.0

        sea_level = float(np.median(background))
        detection_level = float(image[mask].mean())
        spread = float(np.percentile(image, 99) - np.percentile(image, 1)) or 1.0
        return float(np.clip((sea_level - detection_level) / spread, 0.0, 1.0))

    @staticmethod
    def _align(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Resize a ground-truth mask to the prediction's shape if needed."""
        array = np.asarray(mask)
        if array.shape == shape:
            return array

        import cv2

        resized = cv2.resize(
            array.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return resized.astype(bool)
