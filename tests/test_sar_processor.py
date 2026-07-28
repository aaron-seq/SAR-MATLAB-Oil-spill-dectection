"""Tests for :mod:`sar_oil_spill.core.sar_image_processor`."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sar_oil_spill.core.sar_image_processor import (
    SARImageProcessor,
    remove_small_components,
)


class TestInitialisation:
    def test_stores_target_size(self):
        processor = SARImageProcessor(target_image_size=(512, 512))
        assert processor.target_size == (512, 512)
        assert processor.get_processing_summary() == []

    @pytest.mark.parametrize("bad_size", [(0, 10), (-5, 5), (64,), (1, 2, 3)])
    def test_rejects_invalid_target_size(self, bad_size):
        with pytest.raises(ValueError, match="two positive ints"):
            SARImageProcessor(target_image_size=bad_size)


class TestLoading:
    def test_loads_png_as_float32(self, processor, tmp_path):
        path = tmp_path / "scene.png"
        cv2.imwrite(str(path), np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8))

        image = processor.load_sar_image(path)

        assert image is not None
        assert image.dtype == np.float32
        assert image.shape == (64, 64)
        assert len(processor.get_processing_summary()) == 1

    def test_missing_file_returns_none(self, processor):
        assert processor.load_sar_image("does_not_exist.png") is None

    def test_undecodable_file_returns_none(self, processor, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"this is not a PNG")
        assert processor.load_sar_image(path) is None


class TestDespeckling:
    @pytest.mark.parametrize("filter_type", ["lee", "frost", "kuan", "bilateral", "median"])
    def test_all_filters_preserve_shape_and_dtype(self, processor, speckled_image, filter_type):
        filtered = processor.apply_despeckling_filter(
            speckled_image, filter_type=filter_type, window_size=5
        )

        assert filtered.shape == speckled_image.shape
        assert filtered.dtype == np.float32
        assert f"{filter_type} despeckling filter" in processor.get_processing_summary()[-1]

    def test_unknown_filter_falls_back_to_lee(self, processor, speckled_image):
        processor.apply_despeckling_filter(speckled_image, filter_type="nonsense")
        assert "lee despeckling filter" in processor.get_processing_summary()[-1]

    @pytest.mark.parametrize("filter_type", ["lee", "kuan", "median"])
    def test_filtering_reduces_speckle(self, processor, speckled_image, filter_type):
        """Despeckling must lower variance without erasing the dark patches."""
        filtered = processor.apply_despeckling_filter(
            speckled_image, filter_type=filter_type, window_size=7
        )

        assert np.std(filtered) < np.std(speckled_image)
        # The dark square must remain clearly darker than the background.
        assert filtered[45:55, 45:55].mean() < filtered[5:15, 5:15].mean()

    def test_even_window_size_is_made_odd(self, processor, speckled_image):
        """An even window would be rejected by OpenCV; it is rounded up instead."""
        filtered = processor.apply_despeckling_filter(
            speckled_image, filter_type="median", window_size=6
        )
        assert filtered.shape == speckled_image.shape

    def test_noise_variance_estimate_is_positive(self, processor, speckled_image):
        assert processor.estimate_noise_variance(speckled_image) > 0

    def test_empty_image_is_returned_unchanged(self, processor):
        assert processor.apply_despeckling_filter(np.array([], dtype=np.float32)).size == 0


class TestContrastEnhancement:
    @pytest.mark.parametrize(
        "method", ["clahe", "histogram_equalization", "gamma", "none"]
    )
    def test_methods_preserve_shape(self, processor, speckled_image, method):
        enhanced = processor.enhance_contrast(speckled_image, method=method)

        assert enhanced.shape == speckled_image.shape
        assert enhanced.dtype == np.float32
        assert f"{method} contrast enhancement" in processor.get_processing_summary()[-1]

    def test_unknown_method_falls_back_to_clahe(self, processor, speckled_image):
        processor.enhance_contrast(speckled_image, method="nonsense")
        assert "clahe contrast enhancement" in processor.get_processing_summary()[-1]

    def test_clahe_increases_dynamic_range(self, processor):
        low_contrast = np.full((64, 64), 100.0, dtype=np.float32)
        low_contrast[20:40, 20:40] = 110.0

        enhanced = processor.enhance_contrast(low_contrast, method="clahe")

        assert enhanced.max() - enhanced.min() > low_contrast.max() - low_contrast.min()

    def test_empty_image_is_returned_unchanged(self, processor):
        assert processor.enhance_contrast(np.array([], dtype=np.float32)).shape == (0,)


class TestMorphology:
    @pytest.fixture
    def mask_with_speck(self) -> np.ndarray:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1  # a solid block
        mask[5, 5] = 1  # an isolated speck
        return mask

    @pytest.mark.parametrize("operation", ["opening", "closing", "erosion", "dilation"])
    def test_operations_return_uint8_masks(self, processor, mask_with_speck, operation):
        result = processor.apply_morphological_operations(
            mask_with_speck, operation=operation, kernel_size=2
        )

        assert result.shape == mask_with_speck.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)) <= {0, 1}

    def test_opening_removes_isolated_speck(self, processor, mask_with_speck):
        opened = processor.apply_morphological_operations(
            mask_with_speck, operation="opening", kernel_size=2
        )

        assert opened[5, 5] == 0
        assert opened[30, 30] == 1

    def test_unknown_operation_leaves_mask_unchanged(self, processor, mask_with_speck):
        result = processor.apply_morphological_operations(mask_with_speck, operation="nonsense")
        np.testing.assert_array_equal(result, mask_with_speck)

    def test_fill_holes_closes_interior_gap(self, processor):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[18:22, 18:22] = 0

        filled = processor.fill_holes(mask)

        assert filled[20, 20] == 1
        assert filled[0, 0] == 0


class TestRemoveSmallComponents:
    def test_removes_only_components_below_threshold(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True  # 400 px
        mask[50:52, 50:52] = True  # 4 px

        cleaned = remove_small_components(mask, min_size=100)

        assert cleaned[20, 20]
        assert not cleaned[50, 50]

    def test_keeps_component_exactly_at_threshold(self):
        """`min_size` is inclusive, so a component of exactly that size survives."""
        mask = np.zeros((32, 32), dtype=bool)
        mask[4:6, 4:6] = True  # exactly 4 px

        assert remove_small_components(mask, min_size=4).any()
        assert not remove_small_components(mask, min_size=5).any()

    def test_empty_mask_is_handled(self):
        assert not remove_small_components(np.zeros((8, 8), dtype=bool), 10).any()

    def test_processor_wrapper_records_history(self, processor):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:6, 4:6] = 1

        processor.remove_small_objects(mask, minimum_size=100)

        assert "Removed objects smaller than 100 pixels" in processor.get_processing_summary()[-1]


class TestResize:
    def test_resizes_to_explicit_size(self, processor, speckled_image):
        assert processor.resize_image(speckled_image, target_size=(64, 32)).shape == (64, 32)

    def test_resizes_to_configured_default(self, processor, speckled_image):
        assert processor.resize_image(speckled_image).shape == processor.target_size

    def test_non_square_target_keeps_row_major_order(self, processor):
        """A (height, width) tuple must not come back transposed."""
        image = np.zeros((100, 50), dtype=np.float32)
        assert processor.resize_image(image, target_size=(200, 80)).shape == (200, 80)

    def test_empty_image_is_returned_unchanged(self, processor):
        assert processor.resize_image(np.array([], dtype=np.float32)).shape == (0,)


class TestNormalisation:
    def test_minmax_maps_onto_unit_interval(self, processor, speckled_image):
        normalized = processor.normalize_intensity(speckled_image, method="minmax")

        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_zscore_centres_on_zero(self, processor, speckled_image):
        normalized = processor.normalize_intensity(speckled_image, method="zscore")

        assert normalized.mean() == pytest.approx(0.0, abs=1e-5)
        assert normalized.std() == pytest.approx(1.0, abs=1e-4)

    def test_percentile_clips_outliers(self, processor):
        image = np.full((32, 32), 50.0, dtype=np.float32)
        image[0, 0] = 10_000.0  # a specular return

        normalized = processor.normalize_intensity(image, method="percentile")

        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0

    def test_constant_image_does_not_divide_by_zero(self, processor):
        normalized = processor.normalize_intensity(np.full((16, 16), 7.0), method="minmax")
        assert np.all(np.isfinite(normalized))

    def test_unknown_method_leaves_image_unchanged(self, processor, speckled_image):
        np.testing.assert_array_equal(
            processor.normalize_intensity(speckled_image, method="nonsense"), speckled_image
        )


class TestProcessingHistory:
    def test_records_each_step_in_order(self, processor, speckled_image):
        processor.apply_despeckling_filter(speckled_image, "lee")
        processor.enhance_contrast(speckled_image, "clahe")
        processor.resize_image(speckled_image)

        assert len(processor.get_processing_summary()) == 3

    def test_summary_is_a_copy(self, processor, speckled_image):
        processor.resize_image(speckled_image)
        processor.get_processing_summary().clear()

        assert len(processor.get_processing_summary()) == 1

    def test_reset_clears_history(self, processor, speckled_image):
        processor.resize_image(speckled_image)
        processor.reset_processing_history()

        assert processor.get_processing_summary() == []


class TestFullPipeline:
    def test_end_to_end_produces_normalised_output(self, processor, speckled_image):
        image = processor.apply_despeckling_filter(speckled_image, "lee", window_size=5)
        image = processor.enhance_contrast(image, "clahe")
        image = processor.resize_image(image)
        image = processor.normalize_intensity(image, "percentile")

        assert image.shape == processor.target_size
        assert image.dtype == np.float32
        assert 0.0 <= image.min() <= image.max() <= 1.0
        assert len(processor.get_processing_summary()) == 4

    def test_rgb_input_is_converted_to_single_channel(self, processor):
        rgb = np.random.default_rng(1).random((32, 32, 3)).astype(np.float32) * 255
        assert processor.enhance_contrast(rgb, "clahe").ndim == 2
