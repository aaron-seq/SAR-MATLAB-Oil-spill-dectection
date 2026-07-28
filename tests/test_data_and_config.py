"""Tests for the synthetic generator, dataset handler, config and CLI."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sar_oil_spill.cli import main
from sar_oil_spill.config import Settings, load_settings
from sar_oil_spill.data import SARDatasetHandler, generate_dataset, generate_sar_scene


class TestSyntheticGenerator:
    def test_scene_has_expected_shape_and_range(self):
        scene = generate_sar_scene(size=(128, 128), seed=1)

        assert scene.image.shape == (128, 128)
        assert scene.image.dtype == np.float32
        assert 0.0 <= scene.image.min() <= scene.image.max() <= 255.0

    def test_masks_are_boolean_and_aligned(self):
        scene = generate_sar_scene(size=(128, 128), with_land=True, seed=2)

        assert scene.oil_mask.dtype == bool
        assert scene.land_mask.dtype == bool
        assert scene.oil_mask.shape == scene.image.shape

    def test_open_sea_scene_has_no_land(self):
        assert not generate_sar_scene(size=(64, 64), with_land=False, seed=3).land_mask.any()

    def test_land_scene_has_land(self):
        assert generate_sar_scene(size=(128, 128), with_land=True, seed=3).land_mask.any()

    def test_oil_and_land_never_overlap(self):
        """A slick sits on water, so the two ground-truth masks must be disjoint."""
        scene = generate_sar_scene(size=(128, 128), with_land=True, seed=4)
        assert not (scene.oil_mask & scene.land_mask).any()

    def test_same_seed_reproduces_the_scene(self):
        a = generate_sar_scene(size=(64, 64), seed=99)
        b = generate_sar_scene(size=(64, 64), seed=99)

        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.oil_mask, b.oil_mask)

    def test_different_seeds_give_different_scenes(self):
        a = generate_sar_scene(size=(64, 64), seed=1)
        b = generate_sar_scene(size=(64, 64), seed=2)

        assert not np.array_equal(a.image, b.image)

    def test_slicks_are_darker_than_the_sea(self):
        """The core physical assumption the detectors rely on."""
        scene = generate_sar_scene(size=(256, 256), oil_contrast_db=10.0, seed=5)

        assert scene.image[scene.oil_mask].mean() < scene.image[~scene.oil_mask].mean()

    def test_higher_contrast_darkens_the_slick(self):
        low = generate_sar_scene(size=(128, 128), oil_contrast_db=4.0, n_slicks=1, seed=6)
        high = generate_sar_scene(size=(128, 128), oil_contrast_db=12.0, n_slicks=1, seed=6)

        assert high.image[high.oil_mask].mean() < low.image[low.oil_mask].mean()

    def test_more_looks_means_less_speckle(self):
        """Speckle variance falls as 1/L, so a 16-look scene is smoother."""
        few = generate_sar_scene(size=(128, 128), looks=1, n_slicks=0, seed=8)
        many = generate_sar_scene(size=(128, 128), looks=16, n_slicks=0, seed=8)

        assert many.image.std() < few.image.std()

    def test_zero_slicks_produces_an_empty_mask(self):
        assert not generate_sar_scene(size=(64, 64), n_slicks=0, seed=9).oil_mask.any()

    def test_generate_dataset_is_reproducible(self):
        first = generate_dataset(count=3, size=(64, 64), seed=7)
        second = generate_dataset(count=3, size=(64, 64), seed=7)

        assert len(first) == 3
        np.testing.assert_array_equal(first[1].image, second[1].image)


@pytest.fixture
def dataset_root(tmp_path):
    """A miniature on-disk dataset in the layout the handler expects."""
    for split in ("images", "labels", "images_with_land", "labels_with_land"):
        (tmp_path / "train" / split).mkdir(parents=True)

    for index in range(3):
        scene = generate_sar_scene(size=(64, 64), seed=index)
        cv2.imwrite(
            str(tmp_path / "train" / "images" / f"s{index}.jpg"), scene.image.astype(np.uint8)
        )
        cv2.imwrite(
            str(tmp_path / "train" / "labels" / f"s{index}.png"),
            scene.oil_mask.astype(np.uint8) * 255,
        )

    land = generate_sar_scene(size=(64, 64), with_land=True, seed=50)
    cv2.imwrite(
        str(tmp_path / "train" / "images_with_land" / "L0.jpg"), land.image.astype(np.uint8)
    )
    cv2.imwrite(
        str(tmp_path / "train" / "labels_with_land" / "L0.png"),
        land.oil_mask.astype(np.uint8) * 255,
    )
    return tmp_path


class TestDatasetHandler:
    def test_indexes_both_categories(self, dataset_root):
        handler = SARDatasetHandler()

        assert handler.load_dataset(dataset_root) is True
        assert len(handler.samples) == 4
        assert len(handler.get_available_images(has_land=True)) == 1
        assert len(handler.get_available_images(has_land=False)) == 3

    def test_matches_jpg_images_to_png_masks(self, dataset_root):
        """Images and labels routinely use different extensions."""
        handler = SARDatasetHandler()
        handler.load_dataset(dataset_root)

        image, mask = handler.load_image_pair("s0")

        assert image is not None
        assert mask is not None and mask.dtype == bool

    def test_missing_root_returns_false(self, tmp_path):
        assert SARDatasetHandler().load_dataset(tmp_path / "nope") is False

    def test_empty_root_returns_false(self, tmp_path):
        assert SARDatasetHandler().load_dataset(tmp_path) is False

    def test_unknown_sample_returns_none(self, dataset_root):
        handler = SARDatasetHandler()
        handler.load_dataset(dataset_root)

        assert handler.load_image_pair("missing") == (None, None)

    def test_statistics_summarise_the_dataset(self, dataset_root):
        handler = SARDatasetHandler()
        handler.load_dataset(dataset_root)

        stats = handler.get_dataset_statistics()

        assert stats.total_images == 4
        assert stats.images_with_land == 1
        assert 0.0 < stats.oil_spill_percentage < 100.0

    def test_export_writes_json(self, dataset_root, tmp_path):
        import json

        handler = SARDatasetHandler()
        handler.load_dataset(dataset_root)

        path = handler.export_statistics(tmp_path / "out")

        assert json.loads(path.read_text())["total_images"] == 4

    def test_iter_samples_respects_limit(self, dataset_root):
        handler = SARDatasetHandler()
        handler.load_dataset(dataset_root)

        assert len(list(handler.iter_samples(limit=2))) == 2


class TestConfiguration:
    def test_defaults_load_without_a_file(self, tmp_path):
        settings = load_settings(tmp_path / "absent.yaml")
        assert settings == Settings()

    def test_shipped_config_matches_the_defaults_it_documents(self):
        """The YAML in config/ must not silently drift from the dataclasses."""
        settings = load_settings("config/model_config.yaml")

        assert settings.image_processing.enhance_contrast is False
        assert settings.traditional_methods.adaptive_threshold.background_window == 251

    def test_yaml_overrides_are_applied(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "image_processing:\n  despeckle_window: 11\n  target_size: [256, 256]\n"
        )

        settings = load_settings(path)

        assert settings.image_processing.despeckle_window == 11
        assert settings.image_processing.target_size == (256, 256)

    def test_unknown_keys_are_ignored_not_fatal(self, tmp_path, caplog):
        path = tmp_path / "extra.yaml"
        path.write_text("image_processing:\n  invented_key: 3\n  despeckle_window: 9\n")

        settings = load_settings(path)

        assert settings.image_processing.despeckle_window == 9
        assert "invented_key" in caplog.text

    def test_non_mapping_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a list\n")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_settings(path)


class TestCommandLine:
    def test_demo_runs_and_writes_figures(self, tmp_path):
        exit_code = main(
            ["demo", "--size", "128", "--seed", "3", "--output-dir", str(tmp_path)]
        )

        assert exit_code == 0
        assert (tmp_path / "demo_adaptive_threshold.png").exists()
        assert (tmp_path / "demo_adaptive_threshold_stages.png").exists()

    def test_benchmark_writes_a_report(self, tmp_path):
        exit_code = main(
            [
                "benchmark",
                "--samples", "2",
                "--size", "128",
                "--methods", "adaptive_threshold",
                "--output-dir", str(tmp_path),
            ]
        )

        assert exit_code == 0
        assert (tmp_path / "benchmark.json").exists()

    def test_dataset_command_summarises(self, dataset_root):
        assert main(["dataset", str(dataset_root)]) == 0

    def test_dataset_command_fails_on_missing_root(self, tmp_path):
        assert main(["dataset", str(tmp_path / "nope")]) == 1

    def test_detect_on_missing_file_returns_error(self, tmp_path):
        assert main(["detect", str(tmp_path / "nope.png")]) == 1

    def test_detect_on_a_real_file(self, tmp_path):
        scene = generate_sar_scene(size=(128, 128), seed=4)
        image_path = tmp_path / "scene.png"
        cv2.imwrite(str(image_path), scene.image.astype(np.uint8))

        assert main(["detect", str(image_path), "--output-dir", str(tmp_path)]) == 0

    def test_unknown_method_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            main(["demo", "--method", "telepathy"])


class TestOptionalDeepLearning:
    def test_module_imports_without_torch(self):
        """The package must remain usable when the `dl` extra is absent."""
        from sar_oil_spill.models import deep_learning_segmentation as dl

        assert isinstance(dl.TORCH_AVAILABLE, bool)

    def test_helpful_error_when_torch_is_missing(self):
        from sar_oil_spill.models import deep_learning_segmentation as dl

        if dl.TORCH_AVAILABLE:
            pytest.skip("PyTorch is installed in this environment.")

        with pytest.raises(ImportError, match=r"sar-oil-spill\[dl\]"):
            dl.DeepLearningSegmentation()
