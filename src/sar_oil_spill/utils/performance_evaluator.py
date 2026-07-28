"""Segmentation metrics for oil spill masks.

Covers the three metrics of the original ``segmentation_evaluation.m`` (Jaccard,
Dice, boundary F1) plus the pixel-level and object-level measures needed to
compare methods meaningfully.

Oil spill masks are heavily class-imbalanced -- a slick often covers a few
percent of a scene -- so pixel accuracy alone is misleading and is always
reported alongside IoU, Dice and recall.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import measure, segmentation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationMetrics:
    """Metric bundle for a single predicted/ground-truth mask pair."""

    jaccard_index: float
    """Intersection over union. The headline segmentation metric."""

    dice_coefficient: float
    """2|A n B| / (|A| + |B|); equals the F1 score over pixels."""

    pixel_accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    boundary_f1: float
    """F1 over boundary pixels within a tolerance, i.e. contour agreement."""

    hausdorff_distance: float
    """Worst-case boundary deviation in pixels; ``inf`` if one mask is empty."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    predicted_area: int
    ground_truth_area: int
    area_error_ratio: float
    """Signed relative area error: ``(pred - truth) / truth``."""

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain JSON-serialisable mapping."""
        return {k: (float(v) if not isinstance(v, int) else v) for k, v in asdict(self).items()}


class PerformanceEvaluator:
    """Compute and aggregate segmentation metrics."""

    def __init__(self, boundary_tolerance: int = 2) -> None:
        """Args:
        boundary_tolerance: Radius in pixels within which a predicted
            boundary pixel counts as matching the ground truth boundary.
        """
        self.boundary_tolerance = max(0, int(boundary_tolerance))

    def calculate_segmentation_metrics(
        self, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray
    ) -> dict[str, float]:
        """Compute every metric for one mask pair, returned as a dict."""
        return self.evaluate(predicted_mask, ground_truth_mask).as_dict()

    def evaluate(
        self, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray
    ) -> SegmentationMetrics:
        """Compute every metric for one mask pair.

        Raises:
            ValueError: If the two masks have different shapes.
        """
        prediction = self._as_binary(predicted_mask)
        truth = self._as_binary(ground_truth_mask)

        if prediction.shape != truth.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs ground truth {truth.shape}"
            )

        true_positives = int(np.count_nonzero(prediction & truth))
        false_positives = int(np.count_nonzero(prediction & ~truth))
        false_negatives = int(np.count_nonzero(~prediction & truth))
        true_negatives = int(np.count_nonzero(~prediction & ~truth))

        union = true_positives + false_positives + false_negatives
        # A pair of empty masks is perfect agreement, not a division by zero.
        jaccard = true_positives / union if union else 1.0
        dice_denominator = 2 * true_positives + false_positives + false_negatives
        dice = 2 * true_positives / dice_denominator if dice_denominator else 1.0

        precision = self._ratio(true_positives, true_positives + false_positives)
        recall = self._ratio(true_positives, true_positives + false_negatives)
        specificity = self._ratio(true_negatives, true_negatives + false_positives)
        f1 = self._ratio(2 * precision * recall, precision + recall)

        total = prediction.size
        accuracy = (true_positives + true_negatives) / total if total else 1.0

        predicted_area = int(np.count_nonzero(prediction))
        truth_area = int(np.count_nonzero(truth))
        area_error = (
            (predicted_area - truth_area) / truth_area if truth_area else float(predicted_area > 0)
        )

        return SegmentationMetrics(
            jaccard_index=jaccard,
            dice_coefficient=dice,
            pixel_accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            boundary_f1=self.boundary_f1_score(prediction, truth),
            hausdorff_distance=self.hausdorff_distance(prediction, truth),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=true_negatives,
            predicted_area=predicted_area,
            ground_truth_area=truth_area,
            area_error_ratio=area_error,
        )

    def boundary_f1_score(self, prediction: np.ndarray, truth: np.ndarray) -> float:
        """F1 over boundary pixels, matching MATLAB's ``bfscore``.

        A predicted boundary pixel counts as correct when a ground-truth
        boundary pixel lies within :attr:`boundary_tolerance`, so a contour
        that is right in shape but off by a pixel is not punished as a miss.
        """
        predicted_boundary = segmentation.find_boundaries(prediction, mode="inner")
        truth_boundary = segmentation.find_boundaries(truth, mode="inner")

        if not predicted_boundary.any() and not truth_boundary.any():
            return 1.0
        if not predicted_boundary.any() or not truth_boundary.any():
            return 0.0

        # Distance from every pixel to the nearest boundary pixel of each mask.
        distance_to_truth = ndimage.distance_transform_edt(~truth_boundary)
        distance_to_prediction = ndimage.distance_transform_edt(~predicted_boundary)

        precision = float(
            np.mean(distance_to_truth[predicted_boundary] <= self.boundary_tolerance)
        )
        recall = float(np.mean(distance_to_prediction[truth_boundary] <= self.boundary_tolerance))
        return self._ratio(2 * precision * recall, precision + recall)

    @staticmethod
    def hausdorff_distance(prediction: np.ndarray, truth: np.ndarray) -> float:
        """Symmetric Hausdorff distance between the two mask boundaries."""
        predicted_boundary = segmentation.find_boundaries(prediction, mode="inner")
        truth_boundary = segmentation.find_boundaries(truth, mode="inner")

        if not predicted_boundary.any() and not truth_boundary.any():
            return 0.0
        if not predicted_boundary.any() or not truth_boundary.any():
            return float("inf")

        distance_to_truth = ndimage.distance_transform_edt(~truth_boundary)
        distance_to_prediction = ndimage.distance_transform_edt(~predicted_boundary)
        return float(
            max(
                distance_to_truth[predicted_boundary].max(),
                distance_to_prediction[truth_boundary].max(),
            )
        )

    @staticmethod
    def object_detection_rate(
        prediction: np.ndarray, truth: np.ndarray, overlap_threshold: float = 0.5
    ) -> dict[str, float]:
        """Object-level detection rate.

        A ground-truth slick counts as detected when at least
        ``overlap_threshold`` of its pixels are covered by the prediction.
        Pixel metrics can look healthy while a small second slick is missed
        entirely -- this catches that.
        """
        truth_labels, n_truth = ndimage.label(truth)
        predicted_labels, n_predicted = ndimage.label(prediction)

        if n_truth == 0:
            return {
                "detected_objects": 0.0,
                "total_objects": 0.0,
                "detection_rate": 1.0 if n_predicted == 0 else 0.0,
                "false_positive_objects": float(n_predicted),
            }

        detected = 0
        matched_predictions: set[int] = set()
        for region in measure.regionprops(truth_labels):
            coords = tuple(region.coords.T)
            covered = prediction[coords]
            if covered.mean() >= overlap_threshold:
                detected += 1
                matched_predictions.update(np.unique(predicted_labels[coords]).tolist())

        matched_predictions.discard(0)
        return {
            "detected_objects": float(detected),
            "total_objects": float(n_truth),
            "detection_rate": detected / n_truth,
            "false_positive_objects": float(max(0, n_predicted - len(matched_predictions))),
        }

    def aggregate(self, metrics: list[SegmentationMetrics]) -> dict[str, float]:
        """Mean of each metric over a batch, ignoring non-finite values."""
        if not metrics:
            return {}

        aggregated: dict[str, float] = {}
        for key in asdict(metrics[0]):
            values = np.array([getattr(m, key) for m in metrics], dtype=float)
            finite = values[np.isfinite(values)]
            aggregated[f"mean_{key}"] = float(finite.mean()) if finite.size else float("nan")
        aggregated["sample_count"] = float(len(metrics))
        return aggregated

    def generate_evaluation_report(
        self,
        metrics: dict[str, float] | SegmentationMetrics,
        image_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Wrap metrics with context and a plain-language quality verdict."""
        values = metrics.as_dict() if isinstance(metrics, SegmentationMetrics) else dict(metrics)
        iou = values.get("jaccard_index", 0.0)

        if iou >= 0.75:
            quality = "excellent"
        elif iou >= 0.5:
            quality = "good"
        elif iou >= 0.25:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "metrics": values,
            "quality": quality,
            "summary": (
                f"IoU {iou:.3f}, Dice {values.get('dice_coefficient', 0.0):.3f}, "
                f"recall {values.get('recall', 0.0):.3f} ({quality})"
            ),
            "image_info": dict(image_info or {}),
        }

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _as_binary(mask: np.ndarray) -> np.ndarray:
        """Coerce a mask of any dtype into ``bool``.

        Integer masks in the 0-255 range are thresholded at the midpoint;
        anything else is treated as truthy/falsy.
        """
        array = np.asarray(mask)
        if array.dtype == bool:
            return array
        if np.issubdtype(array.dtype, np.floating):
            return array > 0.5 if array.max() <= 1.0 else array > 127.5
        return array > (127 if array.max() > 1 else 0)

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        """Safe division that returns 0.0 instead of raising on a zero denominator."""
        return float(numerator / denominator) if denominator else 0.0
