"""Tests for the classical segmentation methods and the detection pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from sar_oil_spill.config import Settings
from sar_oil_spill.core import OilSpillDetector
from sar_oil_spill.data import generate_sar_scene
from sar_oil_spill.models import METHOD_NAMES, TraditionalSegmentation

# Every method should comfortably clear this on clean synthetic scenes; the
# measured values sit well above it, so a regression is unambiguous.
MIN_ACCEPTABLE_IOU = 0.6


@pytest.fixture(scope="module")
def segmenter() -> TraditionalSegmentation:
    return TraditionalSegmentation()


class TestSegmentationMethods:
    @pytest.mark.parametrize("method", METHOD_NAMES)
    def test_returns_boolean_mask_of_matching_shape(self, segmenter, scene, method):
        result = segmenter.segment(scene.image, method)

        assert result.mask.dtype == bool
        assert result.mask.shape == scene.image.shape
        assert result.method == method
        assert result.elapsed_seconds > 0

    @pytest.mark.parametrize("method", METHOD_NAMES)
    def test_records_named_pipeline_stages(self, segmenter, scene, method):
        result = segmenter.segment(scene.image, method)

        assert "input" in result.stages
        assert "cleaned mask" in result.stages
        assert all(isinstance(v, np.ndarray) for v in result.stages.values())

    def test_unknown_method_raises(self, segmenter, scene):
        with pytest.raises(ValueError, match="Unknown method"):
            segmenter.segment(scene.image, "telepathy")

    @pytest.mark.parametrize("method", METHOD_NAMES)
    def test_detects_a_known_slick(self, segmenter, scene, method):
        """Each method must recover the synthetic slick it was given."""
        from sar_oil_spill.utils import PerformanceEvaluator

        result = segmenter.segment(scene.image, method)
        iou = PerformanceEvaluator().evaluate(result.mask, scene.oil_mask).jaccard_index

        assert iou > MIN_ACCEPTABLE_IOU, f"{method} scored IoU {iou:.3f}"

    def test_uniform_image_yields_no_detection(self, segmenter):
        """A flat scene has no slick, so nothing should be flagged."""
        flat = np.full((128, 128), 100.0, dtype=np.float32)
        result = segmenter.segment(flat, "adaptive_threshold")

        assert not result.mask.any()

    def test_land_is_detected_as_bright_region(self, segmenter):
        scene = generate_sar_scene(size=(256, 256), with_land=True, seed=3)
        land = segmenter.detect_land(scene.image)

        # Land occupies the left edge of the generated scene.
        assert land[:, :40].mean() > land[:, -40:].mean()


class TestOilSpillDetector:
    def test_available_methods_match_registry(self, detector):
        assert set(detector.available_methods) == set(METHOD_NAMES)

    def test_detect_populates_result_fields(self, detector, scene):
        result = detector.detect(scene.image, method="adaptive_threshold")

        assert result.oil_detected
        assert result.affected_area_pixels > 0
        assert 0.0 <= result.coverage_fraction <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time > 0
        assert result.processing_history

    def test_detect_computes_metrics_when_given_ground_truth(self, detector, scene):
        result = detector.detect(
            scene.image, method="adaptive_threshold", ground_truth=scene.oil_mask
        )

        assert result.metrics is not None
        assert result.metrics.jaccard_index > MIN_ACCEPTABLE_IOU

    def test_detect_without_ground_truth_has_no_metrics(self, detector, scene):
        assert detector.detect(scene.image).metrics is None

    def test_resizes_to_configured_target(self, detector, scene):
        result = detector.detect(scene.image, resize=True)
        assert result.mask.shape == tuple(detector.settings.image_processing.target_size)

    def test_ground_truth_of_different_size_is_aligned(self, detector, scene):
        """A 256x256 truth mask must be matched against a 512x512 prediction."""
        result = detector.detect(scene.image, ground_truth=scene.oil_mask, resize=True)

        assert result.metrics is not None
        assert result.mask.shape != scene.oil_mask.shape

    def test_unknown_method_raises(self, detector, scene):
        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect(scene.image, method="telepathy")

    def test_empty_image_raises(self, detector):
        with pytest.raises(ValueError, match="empty image"):
            detector.detect(np.array([], dtype=np.float32))

    def test_min_area_filter_drops_small_detections(self, detector, scene):
        permissive = detector.detect(scene.image, min_area_pixels=0)
        strict = detector.detect(scene.image, min_area_pixels=100_000)

        assert strict.affected_area_pixels < permissive.affected_area_pixels

    def test_land_masking_excludes_land(self, detector):
        scene = generate_sar_scene(size=(256, 256), with_land=True, seed=5)

        without = detector.detect(scene.image, mask_land=False)
        with_masking = detector.detect(scene.image, mask_land=True)

        assert with_masking.land_mask is not None
        assert with_masking.affected_area_pixels <= without.affected_area_pixels

    def test_compare_methods_runs_every_method(self, detector, scene):
        results = detector.compare_methods(scene.image, ground_truth=scene.oil_mask)

        assert set(results) == set(METHOD_NAMES)
        assert all(r.metrics is not None for r in results.values())

    def test_compare_methods_survives_a_failing_method(self, detector, scene, monkeypatch):
        """One broken method must not abort the whole comparison."""

        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(detector.segmenter, "kmeans_segmentation", explode)
        results = detector.compare_methods(scene.image)

        assert "kmeans_clustering" not in results
        assert "adaptive_threshold" in results

    def test_detect_from_missing_file_raises(self, detector):
        with pytest.raises(FileNotFoundError):
            detector.detect_from_file("no_such_image.png")

    def test_summary_is_json_safe(self, detector, scene):
        import json

        summary = detector.detect(scene.image, ground_truth=scene.oil_mask).summary()
        assert json.loads(json.dumps(summary)) == summary


class TestPreprocessingChoices:
    def test_contrast_enhancement_is_off_by_default(self):
        """Regression guard: CLAHE destroys slick contrast (IoU 0.85 -> 0.32)."""
        assert Settings().image_processing.enhance_contrast is False

    def test_enabling_clahe_degrades_detection(self):
        """Documents *why* the default is off, so nobody flips it back blindly."""
        from dataclasses import replace

        from sar_oil_spill.utils import PerformanceEvaluator

        scene = generate_sar_scene(size=(256, 256), n_slicks=1, seed=11)
        evaluator = PerformanceEvaluator()
        base = Settings()

        without = OilSpillDetector(base).detect(scene.image, ground_truth=scene.oil_mask)
        with_clahe = OilSpillDetector(
            replace(
                base,
                image_processing=replace(
                    base.image_processing, enhance_contrast=True, enhancement_method="clahe"
                ),
            )
        ).detect(scene.image, ground_truth=scene.oil_mask)

        assert without.metrics is not None and with_clahe.metrics is not None
        assert without.metrics.jaccard_index > with_clahe.metrics.jaccard_index
        _ = evaluator
