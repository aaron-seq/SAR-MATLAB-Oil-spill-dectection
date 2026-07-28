"""SAR image preprocessing: despeckling, contrast enhancement, morphology.

Speckle is multiplicative noise inherent to coherent radar imaging, so the
filters here are the SAR-specific adaptive ones (Lee, Frost, Kuan) rather than
the additive-noise filters used for optical imagery. Every filter is expressed
with separable box filters over integral-image style accumulations, which keeps
them O(pixels) regardless of window size.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, morphology, restoration

logger = logging.getLogger(__name__)

VALID_DESPECKLE_FILTERS = ("lee", "frost", "kuan", "bilateral", "median")
VALID_ENHANCEMENTS = ("clahe", "histogram_equalization", "gamma", "none")
VALID_MORPHOLOGY = ("opening", "closing", "erosion", "dilation")


def _box_mean(image: np.ndarray, window_size: int) -> np.ndarray:
    """Local arithmetic mean over a square window, with reflected borders."""
    kernel = (window_size, window_size)
    return cv2.blur(image, kernel, borderType=cv2.BORDER_REFLECT)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than ``min_size`` pixels.

    Implemented on :mod:`scipy.ndimage` rather than
    ``skimage.morphology.remove_small_objects`` because that function's
    parameter renamed from ``min_size`` to ``max_size`` in scikit-image 0.26
    *and* changed from a strict to an inclusive comparison. Doing the count
    directly keeps one behaviour across every supported version.
    """
    binary = np.asarray(mask, dtype=bool)
    if not binary.any() or min_size <= 1:
        return binary

    labels, count = ndimage.label(binary)
    if count == 0:
        return binary

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False  # label 0 is the background
    return keep[labels]


def _local_statistics(
    image: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the local mean and local variance over a square window.

    Uses ``E[x^2] - E[x]^2``; the result is clipped at zero because floating
    point cancellation can otherwise produce tiny negative variances.
    """
    mean = _box_mean(image, window_size)
    mean_of_squares = _box_mean(image * image, window_size)
    variance = np.clip(mean_of_squares - mean * mean, 0.0, None)
    return mean, variance


class SARImageProcessor:
    """Preprocessing pipeline for Synthetic Aperture Radar imagery.

    Every operation appends a human-readable line to :attr:`processing_history`
    so a run can be traced end to end, which matters when comparing methods.
    """

    def __init__(self, target_image_size: tuple[int, int] = (512, 512)) -> None:
        if len(target_image_size) != 2 or any(s <= 0 for s in target_image_size):
            raise ValueError(
                f"target_image_size must be two positive ints, got {target_image_size}"
            )

        self.target_size = tuple(int(s) for s in target_image_size)
        self.processing_history: list[str] = []
        logger.debug("SARImageProcessor ready (target size %s)", self.target_size)

    # ------------------------------------------------------------------ I/O

    def load_sar_image(self, image_path: str | Path) -> np.ndarray | None:
        """Load a single-channel SAR image as ``float32``.

        Returns ``None`` rather than raising when the file is missing or
        undecodable, so batch loops can skip bad inputs and carry on.
        """
        path = Path(image_path)
        if not path.exists():
            logger.error("Image file not found: %s", path)
            return None

        sar_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if sar_image is None:
            logger.error("Failed to decode image: %s", path)
            return None

        self.processing_history.append(f"Loaded image from {path}")
        return sar_image.astype(np.float32)

    # ------------------------------------------------------------ despeckle

    def apply_despeckling_filter(
        self,
        sar_image: np.ndarray,
        filter_type: str = "lee",
        window_size: int = 7,
        noise_variance: float | None = None,
    ) -> np.ndarray:
        """Reduce speckle noise while preserving oil/water boundaries.

        Args:
            sar_image: Single-channel SAR image.
            filter_type: One of :data:`VALID_DESPECKLE_FILTERS`.
            window_size: Side length of the local window; forced odd and >= 3.
            noise_variance: Speckle variance. When ``None`` it is estimated
                from the image itself as the squared coefficient of
                variation of the most homogeneous regions.

        Returns:
            The despeckled image as ``float32`` with the input's shape.
        """
        image = self._as_float(sar_image)
        if image.size == 0:
            return image

        window_size = max(3, int(window_size) | 1)

        if filter_type not in VALID_DESPECKLE_FILTERS:
            logger.warning("Unknown filter '%s'; falling back to 'lee'.", filter_type)
            filter_type = "lee"

        if filter_type == "lee":
            filtered = self._lee_filter(image, window_size, noise_variance)
        elif filter_type == "frost":
            filtered = self._frost_filter(image, window_size)
        elif filter_type == "kuan":
            filtered = self._kuan_filter(image, window_size, noise_variance)
        elif filter_type == "median":
            # scipy rather than cv2.medianBlur: the OpenCV version only accepts
            # float32 input at kernel sizes 3 and 5, and SAR windows are wider.
            filtered = ndimage.median_filter(image, size=window_size, mode="reflect")
        else:  # bilateral
            filtered = cv2.bilateralFilter(image, window_size, 75, 75)

        self.processing_history.append(f"Applied {filter_type} despeckling filter")
        return filtered.astype(np.float32)

    @staticmethod
    def estimate_noise_variance(image: np.ndarray, window_size: int = 7) -> float:
        """Estimate the speckle variance of the multiplicative noise model.

        For fully developed speckle the noise variance equals the squared
        coefficient of variation measured over homogeneous areas. The 5th
        percentile of the local coefficient of variation approximates those
        areas without needing a hand-picked region of interest.
        """
        mean, variance = _local_statistics(image, window_size)
        safe_mean = np.where(mean > 1e-6, mean, np.nan)
        coefficient_of_variation = np.sqrt(variance) / safe_mean
        finite = coefficient_of_variation[np.isfinite(coefficient_of_variation)]
        if finite.size == 0:
            return 0.25
        return float(np.percentile(finite, 5) ** 2) or 0.25

    def _lee_filter(
        self, image: np.ndarray, window_size: int, noise_variance: float | None
    ) -> np.ndarray:
        """Vectorised Lee filter.

        ``out = mean + k * (pixel - mean)`` where the weight ``k`` approaches 1
        in high-variance neighbourhoods (edges, kept sharp) and 0 in
        homogeneous ones (smoothed towards the local mean).
        """
        if noise_variance is None:
            noise_variance = self.estimate_noise_variance(image, window_size)

        mean, variance = _local_statistics(image, window_size)
        # Variance of the underlying signal under the multiplicative model.
        signal_variance = np.clip(
            (variance - (mean**2) * noise_variance) / (1.0 + noise_variance), 0.0, None
        )
        denominator = signal_variance + (mean**2) * noise_variance
        weight = np.divide(
            signal_variance,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
        return mean + weight * (image - mean)

    def _frost_filter(
        self, image: np.ndarray, window_size: int, damping: float = 2.0
    ) -> np.ndarray:
        """Frost filter: exponentially weighted local mean.

        The decay rate is driven by the local coefficient of variation, so
        homogeneous regions average over the whole window while textured ones
        collapse towards the centre pixel.
        """
        mean, variance = _local_statistics(image, window_size)
        safe_mean = np.where(np.abs(mean) > 1e-6, mean, 1e-6)
        coefficient_of_variation = variance / (safe_mean**2)

        half = window_size // 2
        offsets = np.arange(-half, half + 1, dtype=np.float32)
        distance = np.sqrt(offsets[:, None] ** 2 + offsets[None, :] ** 2)

        padded = np.pad(image, half, mode="reflect")
        weight_sum = np.zeros_like(image)
        value_sum = np.zeros_like(image)

        alpha = damping * coefficient_of_variation
        for row in range(window_size):
            for col in range(window_size):
                shifted = padded[
                    row : row + image.shape[0], col : col + image.shape[1]
                ]
                weight = np.exp(-alpha * distance[row, col])
                weight_sum += weight
                value_sum += weight * shifted

        return value_sum / np.maximum(weight_sum, 1e-12)

    def _kuan_filter(
        self, image: np.ndarray, window_size: int, noise_variance: float | None
    ) -> np.ndarray:
        """Kuan filter -- like Lee but derived from a linear MMSE criterion."""
        if noise_variance is None:
            noise_variance = self.estimate_noise_variance(image, window_size)

        mean, variance = _local_statistics(image, window_size)
        safe_mean = np.where(np.abs(mean) > 1e-6, mean, 1e-6)
        observed_cv = variance / (safe_mean**2)
        weight = np.clip(
            (observed_cv - noise_variance) / (observed_cv * (1.0 + noise_variance)),
            0.0,
            1.0,
        )
        return mean + weight * (image - mean)

    def denoise_total_variation(self, image: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """Total-variation denoising, useful as an edge-preserving baseline."""
        result = restoration.denoise_tv_chambolle(self._as_float(image), weight=weight)
        self.processing_history.append("Applied total-variation denoising")
        return result.astype(np.float32)

    # ---------------------------------------------------------- enhancement

    def enhance_contrast(
        self, sar_image: np.ndarray, method: str = "clahe", **kwargs: object
    ) -> np.ndarray:
        """Stretch contrast so dark oil slicks separate from the sea clutter.

        Args:
            sar_image: Single-channel SAR image.
            method: One of :data:`VALID_ENHANCEMENTS`.
            **kwargs: ``clip_limit`` and ``tile_grid_size`` for CLAHE, or
                ``gamma`` for gamma correction.

        Returns:
            Enhanced image as ``float32`` in the range ``[0, 255]``.
        """
        image = self._as_float(sar_image)
        if image.size == 0:
            return image

        if method not in VALID_ENHANCEMENTS:
            logger.warning("Unknown enhancement '%s'; falling back to 'clahe'.", method)
            method = "clahe"

        if method == "none":
            enhanced = image
        elif method == "clahe":
            clip_limit = float(kwargs.get("clip_limit", 3.0))  # type: ignore[arg-type]
            tile_grid_size = tuple(kwargs.get("tile_grid_size", (8, 8)))  # type: ignore[arg-type]
            uint8_image = self.to_uint8(image)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(uint8_image).astype(np.float32)
        elif method == "histogram_equalization":
            enhanced = exposure.equalize_hist(image) * 255.0
        else:  # gamma
            gamma = float(kwargs.get("gamma", 1.5))  # type: ignore[arg-type]
            rescaled = exposure.rescale_intensity(image, out_range=(0.0, 1.0))
            enhanced = exposure.adjust_gamma(rescaled, gamma=gamma) * 255.0

        self.processing_history.append(f"Applied {method} contrast enhancement")
        return enhanced.astype(np.float32)

    # ----------------------------------------------------------- morphology

    def apply_morphological_operations(
        self,
        binary_mask: np.ndarray,
        operation: str = "opening",
        kernel_size: int = 5,
        iterations: int = 1,
    ) -> np.ndarray:
        """Clean a binary mask with a disk-shaped structuring element."""
        if operation not in VALID_MORPHOLOGY:
            logger.warning("Unknown morphological operation '%s'; mask unchanged.", operation)
            return binary_mask

        mask = binary_mask.astype(bool)
        if mask.size == 0:
            return binary_mask.astype(np.uint8)

        footprint = morphology.disk(max(1, int(kernel_size)))
        operations = {
            "opening": ndimage.binary_opening,
            "closing": ndimage.binary_closing,
            "erosion": ndimage.binary_erosion,
            "dilation": ndimage.binary_dilation,
        }
        mask = operations[operation](
            mask, structure=footprint, iterations=max(1, int(iterations))
        )

        self.processing_history.append(f"Applied {operation} morphological operation")
        return mask.astype(np.uint8)

    def fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """Fill enclosed background holes, matching MATLAB's ``imfill(...,'holes')``."""
        filled = ndimage.binary_fill_holes(binary_mask.astype(bool))
        self.processing_history.append("Filled holes in binary mask")
        return filled.astype(np.uint8)

    def remove_small_objects(self, binary_mask: np.ndarray, minimum_size: int = 100) -> np.ndarray:
        """Drop connected components below ``minimum_size`` pixels."""
        if binary_mask.size == 0:
            return binary_mask.astype(np.uint8)

        cleaned = remove_small_components(binary_mask, max(1, int(minimum_size)))
        self.processing_history.append(f"Removed objects smaller than {minimum_size} pixels")
        return cleaned.astype(np.uint8)

    # ------------------------------------------------------- geometry/range

    def resize_image(
        self,
        image: np.ndarray,
        target_size: tuple[int, int] | None = None,
        interpolation_method: str = "lanczos",
    ) -> np.ndarray:
        """Resize to ``(height, width)``.

        ``cv2.resize`` takes ``(width, height)``, so the tuple is swapped here;
        callers work in NumPy's row-major convention throughout.
        """
        if image.size == 0:
            return image

        size = tuple(target_size) if target_size is not None else self.target_size
        interpolation = {
            "lanczos": cv2.INTER_LANCZOS4,
            "cubic": cv2.INTER_CUBIC,
            "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
        }.get(interpolation_method, cv2.INTER_LANCZOS4)

        resized = cv2.resize(image, (size[1], size[0]), interpolation=interpolation)
        self.processing_history.append(f"Resized image to {size}")
        return resized

    def normalize_intensity(
        self,
        image: np.ndarray,
        method: str = "minmax",
        percentile_range: tuple[float, float] = (2.0, 98.0),
    ) -> np.ndarray:
        """Normalise intensities to a comparable range across scenes.

        ``percentile`` is the robust default for SAR: a handful of very bright
        specular returns would otherwise compress the whole dynamic range.
        """
        image = self._as_float(image)
        if image.size == 0:
            return image

        if method == "minmax":
            low, high = float(np.min(image)), float(np.max(image))
            normalized = (image - low) / (high - low) if high > low else np.zeros_like(image)
        elif method == "zscore":
            mean, std = float(np.mean(image)), float(np.std(image))
            normalized = (image - mean) / std if std > 0 else image - mean
        elif method == "percentile":
            low = float(np.percentile(image, percentile_range[0]))
            high = float(np.percentile(image, percentile_range[1]))
            normalized = (
                np.clip((image - low) / (high - low), 0.0, 1.0)
                if high > low
                else np.zeros_like(image)
            )
        else:
            logger.warning("Unknown normalization method '%s'; image unchanged.", method)
            return image

        self.processing_history.append(f"Applied {method} normalization")
        return normalized.astype(np.float32)

    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        """Rescale any float range to ``uint8`` for OpenCV operations."""
        if image.size == 0:
            return image.astype(np.uint8)
        low, high = float(np.min(image)), float(np.max(image))
        if high <= low:
            return np.zeros(image.shape, dtype=np.uint8)
        return (((image - low) / (high - low)) * 255.0).astype(np.uint8)

    @staticmethod
    def _as_float(image: np.ndarray) -> np.ndarray:
        """Coerce to a contiguous ``float32`` single-channel array."""
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array.astype(np.float32), cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(array, dtype=np.float32)

    # -------------------------------------------------------------- history

    def get_processing_summary(self) -> list[str]:
        """Return a copy of every step applied since the last reset."""
        return self.processing_history.copy()

    def reset_processing_history(self) -> None:
        """Clear the recorded processing steps."""
        self.processing_history.clear()
