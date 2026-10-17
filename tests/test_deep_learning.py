"""Tests for the optional PyTorch U-Net.

Skipped entirely when PyTorch is not installed, which is the default: the `dl`
extra is optional. CI runs these in a dedicated job with CPU-only torch.

Before these tests existed the deep-learning module had never been executed,
and the first run found that ``predict`` crashed on any image whose sides were
not divisible by 16 -- which is most real SAR scenes.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sar_oil_spill.data import generate_sar_scene  # noqa: E402

# Imported through the public, lazily-resolved path the README documents --
# not the internal module -- so the lazy export is exercised too.
from sar_oil_spill.models import (  # noqa: E402
    DeepLearningSegmentation,
    ImprovedUNet,
    SARDataset,
    dice_loss,
)

# Narrow networks keep the suite fast on CPU; behaviour does not depend on width.
BASE_CHANNELS = 4


@pytest.fixture
def segmenter() -> DeepLearningSegmentation:
    torch.manual_seed(0)
    model = DeepLearningSegmentation(base_channels=BASE_CHANNELS, device="cpu")
    model.create_model()
    return model


@pytest.fixture(scope="module")
def scene():
    return generate_sar_scene(size=(64, 64), n_slicks=1, seed=3)


class TestModelConstruction:
    def test_builds_an_attention_unet_with_trainable_parameters(self, segmenter):
        assert isinstance(segmenter.model, ImprovedUNet)
        assert segmenter.parameter_count > 0

    def test_parameter_count_is_zero_before_the_model_exists(self):
        assert DeepLearningSegmentation(device="cpu").parameter_count == 0

    def test_unknown_architecture_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            DeepLearningSegmentation(device="cpu").create_model("transformer")

    def test_forward_pass_preserves_spatial_shape(self, segmenter):
        output = segmenter.model(torch.zeros(2, 1, 64, 32))
        assert tuple(output.shape) == (2, 1, 64, 32)


class TestPrediction:
    def test_predicting_without_a_model_is_an_error(self):
        with pytest.raises(RuntimeError, match="No model loaded"):
            DeepLearningSegmentation(device="cpu").predict(np.zeros((32, 32)))

    def test_returns_a_probability_map(self, segmenter, scene):
        probabilities = segmenter.predict(scene.image)

        assert probabilities.shape == scene.image.shape
        assert probabilities.dtype == np.float32
        assert 0.0 <= probabilities.min() <= probabilities.max() <= 1.0

    def test_threshold_returns_a_boolean_mask(self, segmenter, scene):
        mask = segmenter.predict(scene.image, threshold=0.5)

        assert mask.dtype == bool
        assert mask.shape == scene.image.shape

    @pytest.mark.parametrize("shape", [(100, 100), (129, 77), (17, 31), (3, 5), (1, 40)])
    def test_arbitrary_scene_sizes_are_supported(self, segmenter, shape):
        """Regression: sides not divisible by 2**depth used to crash the
        skip-connection concatenation (e.g. 25 vs 24 at the deepest level)."""
        image = np.random.default_rng(0).random(shape).astype(np.float32)

        assert segmenter.predict(image).shape == shape

    def test_padding_does_not_change_predictions_inside_the_original_area(self, segmenter):
        """Cropping must return exactly the original region, not a shifted one."""
        image = np.random.default_rng(1).random((64, 64)).astype(np.float32)

        aligned = segmenter.predict(image)
        cropped = segmenter.predict(image[:60, :60])

        # Same pixels, different padding context: identical far from the border.
        np.testing.assert_allclose(aligned[:40, :40], cropped[:40, :40], atol=1e-2)

    def test_8bit_and_unit_range_inputs_are_equivalent(self, segmenter, scene):
        scaled = segmenter.predict(scene.image)
        unit = segmenter.predict(scene.image / 255.0)

        np.testing.assert_allclose(scaled, unit, atol=1e-6)


class TestCheckpoints:
    def test_round_trip_reproduces_predictions_exactly(self, segmenter, scene, tmp_path):
        path = tmp_path / "unet.pt"
        torch.save(segmenter.model.state_dict(), path)

        restored = DeepLearningSegmentation(base_channels=BASE_CHANNELS, device="cpu")
        restored.load_checkpoint(path)

        np.testing.assert_array_equal(segmenter.predict(scene.image), restored.predict(scene.image))

    def test_accepts_training_checkpoints_that_wrap_the_state_dict(
        self, segmenter, scene, tmp_path
    ):
        path = tmp_path / "training.pt"
        torch.save({"model_state_dict": segmenter.model.state_dict(), "epoch": 7}, path)

        restored = DeepLearningSegmentation(base_channels=BASE_CHANNELS, device="cpu")
        restored.load_checkpoint(path)

        np.testing.assert_array_equal(segmenter.predict(scene.image), restored.predict(scene.image))

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DeepLearningSegmentation(device="cpu").load_checkpoint(tmp_path / "absent.pt")

    def test_checkpoints_cannot_execute_arbitrary_code(self, segmenter, tmp_path):
        """Security: checkpoints are loaded with ``weights_only=True``, so a
        file carrying a pickled callable is refused instead of executed."""
        path = tmp_path / "malicious.pt"
        payload = {"model_state_dict": segmenter.model.state_dict(), "hook": _Exploit()}
        torch.save(payload, path)

        with pytest.raises(pickle.UnpicklingError):
            DeepLearningSegmentation(base_channels=BASE_CHANNELS, device="cpu").load_checkpoint(
                path
            )


class _Exploit:
    """A stand-in for a malicious object embedded in a checkpoint."""


class TestTraining:
    def test_dice_loss_is_zero_for_a_perfect_prediction(self):
        loss = dice_loss(torch.full((1, 1, 8, 8), 20.0), torch.ones(1, 1, 8, 8))
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_dice_loss_is_near_one_for_a_completely_wrong_prediction(self):
        loss = dice_loss(torch.full((1, 1, 8, 8), -20.0), torch.ones(1, 1, 8, 8))
        assert float(loss) == pytest.approx(1.0, abs=1e-3)

    def test_the_network_can_learn(self):
        """Gradients must flow through the attention gates: a few optimiser
        steps on one scene should cut the loss substantially."""
        torch.manual_seed(0)
        model = ImprovedUNet(base_channels=BASE_CHANNELS, depth=2, dropout=0.0)
        scene = generate_sar_scene(size=(32, 32), n_slicks=1, seed=5)
        image, mask = SARDataset([scene.image], [scene.oil_mask])[0]
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

        def step() -> float:
            optimiser.zero_grad()
            loss = dice_loss(model(image[None]), mask[None])
            loss.backward()
            optimiser.step()
            return loss.item()

        initial = step()
        for _ in range(40):
            final = step()

        assert final < initial * 0.5


class TestDataset:
    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="images but"):
            SARDataset([np.zeros((8, 8))], [])

    def test_8bit_images_are_scaled_to_unit_range(self, scene):
        image, _ = SARDataset([scene.image], [scene.oil_mask])[0]
        assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0

    def test_augmentation_keeps_image_and_mask_aligned(self):
        """Flips and rotations must apply identically to both, or the labels
        silently stop matching the pixels they describe."""
        rng = np.random.default_rng(0)
        image = rng.random((16, 12)).astype(np.float32)
        mask = (image > 0.5).astype(np.float32)
        dataset = SARDataset([image] * 20, [mask] * 20, augment=True, seed=1)

        for index in range(len(dataset)):
            augmented_image, augmented_mask = dataset[index]
            np.testing.assert_array_equal(
                (augmented_image.numpy() > 0.5).astype(np.float32), augmented_mask.numpy()
            )
