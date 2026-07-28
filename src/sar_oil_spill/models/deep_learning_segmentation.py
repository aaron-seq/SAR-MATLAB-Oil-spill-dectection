"""PyTorch U-Net for SAR oil spill segmentation.

PyTorch is an **optional** dependency (``pip install 'sar-oil-spill[dl]'``).
Importing this module without it raises a clear :class:`ImportError` naming the
extra, and the rest of the package -- including the whole traditional pipeline,
the API and the CLI -- keeps working without it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly by the import guard test
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

_INSTALL_HINT = (
    "PyTorch is required for deep-learning segmentation. "
    "Install it with: pip install 'sar-oil-spill[dl]'"
)


def require_torch() -> None:
    """Raise a helpful :class:`ImportError` when PyTorch is missing."""
    if not TORCH_AVAILABLE:
        raise ImportError(_INSTALL_HINT)


if TORCH_AVAILABLE:

    class DoubleConvolution(nn.Module):
        """Two 3x3 convolutions with batch norm and ReLU -- the U-Net block."""

        def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            self.block = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class AttentionGate(nn.Module):
        """Additive attention gate on a skip connection.

        Suppresses skip-connection features that the decoder does not need,
        which matters for SAR: sea clutter and land edges are high-contrast but
        irrelevant, and would otherwise leak straight into the output.
        """

        def __init__(self, gate_channels: int, skip_channels: int, hidden: int) -> None:
            super().__init__()
            self.gate = nn.Sequential(
                nn.Conv2d(gate_channels, hidden, kernel_size=1), nn.BatchNorm2d(hidden)
            )
            self.skip = nn.Sequential(
                nn.Conv2d(skip_channels, hidden, kernel_size=1), nn.BatchNorm2d(hidden)
            )
            self.attention = nn.Sequential(
                nn.Conv2d(hidden, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            weights = self.attention(self.relu(self.gate(gate) + self.skip(skip)))
            return skip * weights

    class ImprovedUNet(nn.Module):
        """U-Net with attention-gated skip connections.

        Args:
            in_channels: Input bands; 1 for single-polarisation SAR amplitude.
            num_classes: Output channels. 1 means a single oil/no-oil logit.
            base_channels: Width of the first encoder stage; doubles per level.
            depth: Number of down/up-sampling stages.
            dropout: Dropout applied in the bottleneck.
        """

        def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 1,
            base_channels: int = 32,
            depth: int = 4,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.depth = depth

            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()
            channels = in_channels
            widths: list[int] = []
            for level in range(depth):
                width = base_channels * (2**level)
                self.encoders.append(DoubleConvolution(channels, width))
                self.pools.append(nn.MaxPool2d(2))
                widths.append(width)
                channels = width

            self.bottleneck = DoubleConvolution(channels, channels * 2, dropout=dropout)
            channels *= 2

            self.upsamples = nn.ModuleList()
            self.attentions = nn.ModuleList()
            self.decoders = nn.ModuleList()
            for width in reversed(widths):
                self.upsamples.append(nn.ConvTranspose2d(channels, width, kernel_size=2, stride=2))
                self.attentions.append(AttentionGate(width, width, max(1, width // 2)))
                self.decoders.append(DoubleConvolution(width * 2, width))
                channels = width

            self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skips: list[torch.Tensor] = []
            for encoder, pool in zip(self.encoders, self.pools, strict=True):
                x = encoder(x)
                skips.append(x)
                x = pool(x)

            x = self.bottleneck(x)

            for upsample, attention, decoder, skip in zip(
                self.upsamples, self.attentions, self.decoders, reversed(skips), strict=True
            ):
                x = upsample(x)
                x = decoder(torch.cat([attention(x, skip), x], dim=1))

            return self.head(x)

    class SARDataset(Dataset):
        """Image/mask pairs as tensors, with optional augmentation.

        Augmentation is limited to flips and 90-degree rotations: SAR geometry
        has no canonical orientation, but elastic or brightness distortions
        would break the radiometric relationship the model relies on.
        """

        def __init__(
            self,
            images: Iterable[np.ndarray],
            masks: Iterable[np.ndarray],
            augment: bool = False,
            seed: int = 42,
        ) -> None:
            self.images = [np.asarray(i, dtype=np.float32) for i in images]
            self.masks = [np.asarray(m, dtype=np.float32) for m in masks]
            if len(self.images) != len(self.masks):
                raise ValueError(
                    f"Got {len(self.images)} images but {len(self.masks)} masks."
                )
            self.augment = augment
            self.rng = np.random.default_rng(seed)

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            image = self.images[index]
            mask = self.masks[index]

            if image.max() > 1.0:
                image = image / 255.0

            if self.augment:
                if self.rng.random() > 0.5:
                    image, mask = np.fliplr(image), np.fliplr(mask)
                if self.rng.random() > 0.5:
                    image, mask = np.flipud(image), np.flipud(mask)
                turns = int(self.rng.integers(0, 4))
                if turns:
                    image, mask = np.rot90(image, turns), np.rot90(mask, turns)

            return (
                torch.from_numpy(np.ascontiguousarray(image)[None]).float(),
                torch.from_numpy(np.ascontiguousarray(mask)[None]).float(),
            )

    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Soft Dice loss -- robust to the heavy class imbalance of oil masks."""
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        cardinality = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (1.0 - (2.0 * intersection + eps) / (cardinality + eps)).mean()

else:  # pragma: no cover - placeholders so names exist for import guards

    class ImprovedUNet:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    class SARDataset:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()

    def dice_loss(*args: object, **kwargs: object) -> object:  # type: ignore[misc]
        require_torch()


class DeepLearningSegmentation:
    """Build, load and run a U-Net segmenter.

    Training is intentionally out of scope for this class -- see
    ``scripts/train.py`` for the training loop. Here the focus is inference,
    which is what the API and CLI need.
    """

    def __init__(
        self,
        architecture: str = "unet",
        num_classes: int = 1,
        base_channels: int = 32,
        device: str | None = None,
    ) -> None:
        require_torch()
        self.architecture = architecture
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: nn.Module | None = None

    def create_model(self, architecture: str | None = None) -> nn.Module:
        """Instantiate the network and move it onto :attr:`device`.

        Raises:
            ValueError: If the architecture name is not recognised.
        """
        name = (architecture or self.architecture).lower()
        if name not in {"unet", "improved_unet"}:
            raise ValueError(
                f"Unsupported architecture '{name}'. Available: 'unet', 'improved_unet'."
            )

        self.model = ImprovedUNet(
            in_channels=1, num_classes=self.num_classes, base_channels=self.base_channels
        ).to(self.device)
        logger.info(
            "Built %s on %s (%s parameters)", name, self.device, f"{self.parameter_count:,}"
        )
        return self.model

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters, or 0 before the model is built."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load weights from a ``.pt``/``.pth`` file.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if self.model is None:
            self.create_model()

        state = torch.load(path, map_location=self.device, weights_only=True)
        # Accept both bare state dicts and training checkpoints that wrap one.
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        assert self.model is not None
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("Loaded checkpoint %s", path)

    def predict(self, image: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Run inference on a single image.

        Args:
            image: 2-D SAR image. Values above 1.0 are assumed to be 0-255.
            threshold: When given, return a boolean mask at this probability;
                otherwise return the raw probability map.

        Returns:
            A ``float32`` probability map, or a boolean mask if thresholded.

        Raises:
            RuntimeError: If no model has been created or loaded.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call create_model() or load_checkpoint() first.")

        array = np.asarray(image, dtype=np.float32)
        if array.max() > 1.0:
            array = array / 255.0

        tensor = torch.from_numpy(array[None, None]).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(tensor))

        result = probabilities.squeeze().cpu().numpy().astype(np.float32)
        return result > threshold if threshold is not None else result
