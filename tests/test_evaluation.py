"""Tests for :mod:`sar_oil_spill.utils.performance_evaluator`."""

from __future__ import annotations

import numpy as np
import pytest

from sar_oil_spill.utils import PerformanceEvaluator


@pytest.fixture
def evaluator() -> PerformanceEvaluator:
    return PerformanceEvaluator(boundary_tolerance=2)


@pytest.fixture
def square_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    return mask


class TestCoreMetrics:
    def test_identical_masks_score_perfectly(self, evaluator, square_mask):
        metrics = evaluator.evaluate(square_mask, square_mask)

        assert metrics.jaccard_index == 1.0
        assert metrics.dice_coefficient == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.hausdorff_distance == 0.0

    def test_disjoint_masks_score_zero(self, evaluator):
        a = np.zeros((32, 32), dtype=bool)
        a[0:8, 0:8] = True
        b = np.zeros((32, 32), dtype=bool)
        b[20:28, 20:28] = True

        metrics = evaluator.evaluate(a, b)

        assert metrics.jaccard_index == 0.0
        assert metrics.dice_coefficient == 0.0
        assert metrics.precision == 0.0

    def test_two_empty_masks_are_perfect_agreement(self, evaluator):
        """Predicting no oil where there is none is correct, not a zero score."""
        empty = np.zeros((32, 32), dtype=bool)
        metrics = evaluator.evaluate(empty, empty)

        assert metrics.jaccard_index == 1.0
        assert metrics.dice_coefficient == 1.0
        assert metrics.boundary_f1 == 1.0

    def test_known_overlap_matches_hand_computation(self, evaluator):
        """Half-overlapping 10x20 blocks: intersection 100, union 300."""
        prediction = np.zeros((32, 32), dtype=bool)
        prediction[0:10, 0:20] = True
        truth = np.zeros((32, 32), dtype=bool)
        truth[0:10, 10:30] = True

        metrics = evaluator.evaluate(prediction, truth)

        assert metrics.true_positives == 100
        assert metrics.false_positives == 100
        assert metrics.false_negatives == 100
        assert metrics.jaccard_index == pytest.approx(1 / 3)
        assert metrics.dice_coefficient == pytest.approx(0.5)

    def test_dice_equals_pixel_f1(self, evaluator):
        rng = np.random.default_rng(0)
        prediction = rng.random((48, 48)) > 0.7
        truth = rng.random((48, 48)) > 0.6

        metrics = evaluator.evaluate(prediction, truth)

        assert metrics.dice_coefficient == pytest.approx(metrics.f1_score, abs=1e-9)

    def test_shape_mismatch_raises(self, evaluator):
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluator.evaluate(np.zeros((8, 8), bool), np.zeros((16, 16), bool))

    def test_area_error_ratio_is_signed(self, evaluator):
        truth = np.zeros((32, 32), dtype=bool)
        truth[0:10, 0:10] = True
        over = np.zeros((32, 32), dtype=bool)
        over[0:15, 0:10] = True

        assert evaluator.evaluate(over, truth).area_error_ratio == pytest.approx(0.5)
        assert evaluator.evaluate(truth, over).area_error_ratio < 0


class TestMaskCoercion:
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            np.array([[False, True], [True, False]]),
            np.array([[0, 1], [1, 0]], dtype=np.int32),
        ],
    )
    def test_common_mask_encodings_are_equivalent(self, evaluator, mask):
        """uint8 0/255, float 0/1, bool and int 0/1 must all score identically."""
        reference = np.array([[False, True], [True, False]])
        assert evaluator.evaluate(mask, reference).jaccard_index == 1.0


class TestBoundaryMetrics:
    def test_boundary_f1_perfect_for_identical_masks(self, evaluator, square_mask):
        assert evaluator.boundary_f1_score(square_mask, square_mask) == 1.0

    def test_boundary_f1_tolerates_small_shifts(self, evaluator, square_mask):
        """A one-pixel shift is a near-perfect contour, not a failure."""
        shifted = np.roll(square_mask, 1, axis=0)
        assert evaluator.boundary_f1_score(shifted, square_mask) > 0.9

    def test_tighter_tolerance_is_stricter(self, square_mask):
        shifted = np.roll(square_mask, 3, axis=0)

        strict = PerformanceEvaluator(boundary_tolerance=1).boundary_f1_score(shifted, square_mask)
        loose = PerformanceEvaluator(boundary_tolerance=5).boundary_f1_score(shifted, square_mask)

        assert strict < loose

    def test_hausdorff_is_infinite_when_one_mask_is_empty(self, evaluator, square_mask):
        assert evaluator.hausdorff_distance(np.zeros_like(square_mask), square_mask) == float("inf")

    def test_hausdorff_grows_with_displacement(self, evaluator, square_mask):
        near = evaluator.hausdorff_distance(np.roll(square_mask, 2, axis=0), square_mask)
        far = evaluator.hausdorff_distance(np.roll(square_mask, 10, axis=0), square_mask)

        assert far > near


class TestObjectMetrics:
    def test_detects_all_objects(self, evaluator):
        truth = np.zeros((64, 64), dtype=bool)
        truth[5:15, 5:15] = True
        truth[40:50, 40:50] = True

        stats = evaluator.object_detection_rate(truth, truth)

        assert stats["total_objects"] == 2
        assert stats["detection_rate"] == 1.0

    def test_missed_object_lowers_rate(self, evaluator):
        """Pixel metrics can look fine while a whole second slick is missed."""
        truth = np.zeros((64, 64), dtype=bool)
        truth[5:15, 5:15] = True
        truth[40:50, 40:50] = True
        prediction = np.zeros((64, 64), dtype=bool)
        prediction[5:15, 5:15] = True

        stats = evaluator.object_detection_rate(prediction, truth)

        assert stats["detection_rate"] == 0.5

    def test_counts_false_positive_objects(self, evaluator):
        truth = np.zeros((64, 64), dtype=bool)
        truth[5:15, 5:15] = True
        prediction = truth.copy()
        prediction[40:50, 40:50] = True

        assert evaluator.object_detection_rate(prediction, truth)["false_positive_objects"] == 1


class TestAggregationAndReporting:
    def test_aggregate_averages_each_metric(self, evaluator, square_mask):
        metrics = [
            evaluator.evaluate(square_mask, square_mask),
            evaluator.evaluate(np.zeros_like(square_mask), square_mask),
        ]

        aggregated = evaluator.aggregate(metrics)

        assert aggregated["sample_count"] == 2
        assert aggregated["mean_jaccard_index"] == pytest.approx(0.5)

    def test_aggregate_ignores_infinite_values(self, evaluator, square_mask):
        """An empty prediction yields an infinite Hausdorff; the mean must stay finite."""
        metrics = [
            evaluator.evaluate(square_mask, square_mask),
            evaluator.evaluate(np.zeros_like(square_mask), square_mask),
        ]

        assert np.isfinite(evaluator.aggregate(metrics)["mean_hausdorff_distance"])

    def test_aggregate_of_nothing_is_empty(self, evaluator):
        assert evaluator.aggregate([]) == {}

    @pytest.mark.parametrize(
        ("iou", "expected"),
        [(0.9, "excellent"), (0.6, "good"), (0.3, "fair"), (0.1, "poor")],
    )
    def test_report_assigns_quality_band(self, evaluator, iou, expected):
        report = evaluator.generate_evaluation_report({"jaccard_index": iou})
        assert report["quality"] == expected

    def test_report_accepts_dataclass_or_dict(self, evaluator, square_mask):
        metrics = evaluator.evaluate(square_mask, square_mask)

        from_object = evaluator.generate_evaluation_report(metrics)
        from_dict = evaluator.generate_evaluation_report(metrics.as_dict())

        assert from_object["metrics"] == from_dict["metrics"]

    def test_metrics_serialise_to_json(self, evaluator, square_mask):
        import json

        payload = evaluator.evaluate(square_mask, square_mask).as_dict()
        assert json.loads(json.dumps(payload)) == payload
