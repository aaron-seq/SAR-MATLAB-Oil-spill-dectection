"""Classical segmentation methods, ported from the original MATLAB project.

All four methods exploit the same physical fact: oil dampens capillary waves,
so a slick backscatters less energy and appears as a *dark* region against the
brighter sea clutter. They differ in how they decide where "dark" begins.

Each method returns a :class:`SegmentationResult` with a boolean mask plus the
intermediate stage images, so callers can visualise or debug the pipeline
without re-running it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation
from sklearn.cluster import KMeans

from sar_oil_spill.config import (
    AdaptiveThresholdSettings,
    FuzzyEdgeSettings,
    KMeansSettings,
    SuperpixelSettings,
    TraditionalMethodSettings,
)
from sar_oil_spill.core.sar_image_processor import SARImageProcessor, remove_small_components

logger = logging.getLogger(__name__)

METHOD_NAMES = (
    "adaptive_threshold",
    "kmeans_clustering",
    "superpixel_clustering",
    "fuzzy_edge_detection",
)


@dataclass
class SegmentationResult:
    """Outcome of one segmentation run."""

    mask: np.ndarray
    """Boolean array, ``True`` where oil is predicted."""

    method: str
    elapsed_seconds: float
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    """Named intermediate images, in pipeline order, for visualisation."""

    details: dict[str, float] = field(default_factory=dict)
    """Scalar diagnostics such as the threshold that was chosen."""

    @property
    def coverage_fraction(self) -> float:
        """Fraction of the scene flagged as oil."""
        return float(self.mask.mean()) if self.mask.size else 0.0


class TraditionalSegmentation:
    """Threshold, clustering, superpixel and fuzzy-edge oil spill segmenters."""

    def __init__(
        self,
        settings: TraditionalMethodSettings | None = None,
        processor: SARImageProcessor | None = None,
    ) -> None:
        self.settings = settings or TraditionalMethodSettings()
        self.processor = processor or SARImageProcessor()

    # ------------------------------------------------------------ dispatch

    def segment(self, image: np.ndarray, method: str, **overrides: object) -> SegmentationResult:
        """Run ``method`` by name. Raises ``ValueError`` for unknown methods."""
        dispatch = {
            "adaptive_threshold": self.adaptive_threshold_segmentation,
            "kmeans_clustering": self.kmeans_segmentation,
            "superpixel_clustering": self.superpixel_segmentation,
            "fuzzy_edge_detection": self.fuzzy_edge_segmentation,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose from {METHOD_NAMES}.")
        return dispatch[method](image, **overrides)  # type: ignore[operator]

    # -------------------------------------------------- adaptive threshold

    def adaptive_threshold_segmentation(
        self,
        image: np.ndarray,
        settings: AdaptiveThresholdSettings | None = None,
    ) -> SegmentationResult:
        """Background-normalised Otsu thresholding (``local_threshold.m``).

        A single global threshold fails on SAR scenes because the sea's mean
        backscatter drifts with wind speed and incidence angle across the
        swath. A purely *local* threshold fails too, in the opposite way: over
        a slick wider than its window the local mean sinks with the slick and
        the contrast cancels out, leaving only the rim detected.

        So the background is estimated over a window much wider than any
        plausible slick and the image is divided by it. Under SAR's
        multiplicative model that ratio is the local backscatter *ratio* --
        near 1.0 over sea, near 0.3-0.5 over oil, independent of absolute
        brightness -- and Otsu then finds the split between those two modes.
        """
        config = settings or self.settings.adaptive_threshold
        started = time.perf_counter()
        working = self._prepare(image)
        stages: dict[str, np.ndarray] = {"input": working.copy()}

        smoothed = ndimage.median_filter(
            working, size=max(3, config.median_filter_size | 1), mode="reflect"
        )
        stages["median filtered"] = smoothed.copy()

        # Window must stay odd and fit inside the image.
        window = max(3, min(config.background_window | 1, (min(smoothed.shape) - 1) | 1))
        background = cv2.blur(smoothed, (window, window), borderType=cv2.BORDER_REFLECT)
        stages["background estimate"] = background.copy()

        ratio = smoothed / np.maximum(background, 1e-6)
        stages["backscatter ratio"] = ratio.astype(np.float32)

        threshold = float(filters.threshold_otsu(ratio)) - config.offset
        # Oil backscatters less than the surrounding sea, hence "<".
        mask = ratio < threshold
        stages["raw mask"] = mask.astype(np.float32)

        mask = self._clean_mask(mask, min_area=config.min_blob_area, disk_radius=2)
        stages["cleaned mask"] = mask.astype(np.float32)

        return SegmentationResult(
            mask=mask,
            method="adaptive_threshold",
            elapsed_seconds=time.perf_counter() - started,
            stages=stages,
            details={"background_window": float(window), "ratio_threshold": threshold},
        )

    # ----------------------------------------------------------- k-means

    def kmeans_segmentation(
        self, image: np.ndarray, settings: KMeansSettings | None = None
    ) -> SegmentationResult:
        """Intensity clustering (``kmeansSegment.m``).

        Pixels are clustered on grey level alone; the cluster with the lowest
        centroid is taken as oil, then only the ``n_largest_blobs`` biggest
        components survive -- an oil slick is a large contiguous region, while
        residual speckle is not.
        """
        config = settings or self.settings.kmeans
        started = time.perf_counter()
        working = self._prepare(image)
        stages: dict[str, np.ndarray] = {"input": working.copy()}

        blurred = cv2.GaussianBlur(working, (0, 0), sigmaX=config.gaussian_sigma)
        stages["gaussian smoothed"] = blurred.copy()

        samples = blurred.reshape(-1, 1)
        n_clusters = min(config.n_clusters, max(2, len(np.unique(samples))))
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        labels = kmeans.fit_predict(samples).reshape(blurred.shape)
        stages["cluster labels"] = labels.astype(np.float32)

        darkest_cluster = int(np.argmin(kmeans.cluster_centers_.ravel()))
        mask = labels == darkest_cluster

        mask = self._keep_largest_blobs(mask, config.n_largest_blobs)
        mask = ndimage.binary_fill_holes(mask)
        stages["cleaned mask"] = mask.astype(np.float32)

        return SegmentationResult(
            mask=mask.astype(bool),
            method="kmeans_clustering",
            elapsed_seconds=time.perf_counter() - started,
            stages=stages,
            details={
                "n_clusters": float(n_clusters),
                "darkest_centroid": float(kmeans.cluster_centers_.ravel()[darkest_cluster]),
            },
        )

    # -------------------------------------------------------- superpixels

    def superpixel_segmentation(
        self, image: np.ndarray, settings: SuperpixelSettings | None = None
    ) -> SegmentationResult:
        """SLIC superpixels + Otsu (``superpixel.m``).

        Averaging within superpixels suppresses speckle far more effectively
        than a fixed-window filter, because superpixel borders follow real
        image structure. Otsu then separates the two resulting intensity modes.
        Finally, detections too far from the main slick are discarded, which is
        the original project's way of rejecting unrelated dark features.
        """
        config = settings or self.settings.superpixel
        started = time.perf_counter()
        working = self._prepare(image)
        stages: dict[str, np.ndarray] = {"input": working.copy()}

        normalized = working / 255.0
        labels = segmentation.slic(
            normalized,
            n_segments=config.n_segments,
            compactness=config.compactness,
            sigma=config.sigma,
            channel_axis=None,
            start_label=1,
        )
        stages["superpixel boundaries"] = segmentation.find_boundaries(labels).astype(np.float32)

        # Replace every superpixel by its mean intensity.
        mean_image = ndimage.mean(normalized, labels=labels, index=np.arange(1, labels.max() + 1))
        averaged = mean_image[labels - 1].astype(np.float32)
        stages["superpixel means"] = averaged.copy()

        threshold = float(filters.threshold_otsu(averaged))
        # The MATLAB original nudges Otsu downwards; oil is the darker tail and
        # Otsu sits between the modes, which over-segments the sea.
        threshold -= 0.25 if threshold > 0.5 else 0.15
        mask = averaged < threshold
        stages["otsu mask"] = mask.astype(np.float32)

        mask = self._keep_largest_blobs(mask, 3)
        mask = self._filter_by_distance_to_main_blob(mask, config.max_centroid_distance)
        stages["cleaned mask"] = mask.astype(np.float32)

        return SegmentationResult(
            mask=mask.astype(bool),
            method="superpixel_clustering",
            elapsed_seconds=time.perf_counter() - started,
            stages=stages,
            details={"otsu_threshold": threshold, "n_superpixels": float(labels.max())},
        )

    # ------------------------------------------------------- fuzzy edges

    def fuzzy_edge_segmentation(
        self, image: np.ndarray, settings: FuzzyEdgeSettings | None = None
    ) -> SegmentationResult:
        """Fuzzy-logic texture segmentation (``fuzzy_edgeDetect.m``).

        A Mamdani fuzzy inference system with two rules -- "if both image
        gradients are near zero the pixel is uniform, otherwise it is an edge"
        -- reduces analytically to a product of Gaussian memberships, so it is
        evaluated in closed form instead of running an inference engine per
        pixel. That is the whole reason this runs in milliseconds rather than
        the minutes the original ``evalfis`` loop took.

        The output is then thresholded on *uniformity*, not on edges, which is
        what the MATLAB original did too (``imbinarize(Ieval, 0.8)`` keeps the
        white, i.e. uniform, class). This is the physically meaningful choice:
        oil damps the capillary waves that give the sea its speckle texture, so
        a slick is markedly smoother than the water around it. Darkness is then
        used as a second, independent check.
        """
        config = settings or self.settings.fuzzy_edge
        started = time.perf_counter()
        working = self._prepare(image)
        stages: dict[str, np.ndarray] = {"input": working.copy()}

        despeckled = self.processor.apply_despeckling_filter(
            working, filter_type="lee", window_size=config.lee_window
        )
        equalized = self.processor.enhance_contrast(despeckled, method="histogram_equalization")
        stages["lee + equalised"] = equalized.copy()

        scaled = self._prepare(equalized) / 255.0
        gradient_x = cv2.Sobel(scaled, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(scaled, cv2.CV_32F, 0, 1, ksize=3)
        stages["sobel magnitude"] = np.hypot(gradient_x, gradient_y).astype(np.float32)

        # mu_zero(g) = exp(-g^2 / 2 sigma^2), applied to both gradients; the
        # product is the firing strength of rule 1 ("Iout is white").
        sigma_sq = 2.0 * config.sigma**2
        uniformity = np.exp(-(gradient_x**2) / sigma_sq) * np.exp(-(gradient_y**2) / sigma_sq)
        stages["fuzzy uniformity"] = uniformity.astype(np.float32)

        smooth_regions = uniformity > config.binarize_threshold
        smooth_regions = ndimage.binary_closing(smooth_regions, structure=morphology.disk(3))
        smooth_regions = ndimage.binary_fill_holes(smooth_regions)
        stages["smooth regions"] = smooth_regions.astype(np.float32)

        # Calm water is smooth as well, so require the region to be dark too.
        mask = self._keep_dark_regions(smooth_regions, despeckled)
        mask = self._clean_mask(mask, min_area=100, disk_radius=2)
        stages["cleaned mask"] = mask.astype(np.float32)

        return SegmentationResult(
            mask=mask,
            method="fuzzy_edge_detection",
            elapsed_seconds=time.perf_counter() - started,
            stages=stages,
            details={"sigma": float(config.sigma), "threshold": float(config.binarize_threshold)},
        )

    # ---------------------------------------------------------- land mask

    def detect_land(self, image: np.ndarray, land_threshold: float = 0.55) -> np.ndarray:
        """Segment land, which returns strongly and so appears bright (``land_mask.m``).

        Args:
            image: SAR image.
            land_threshold: Brightness cut-off in ``[0, 1]`` after normalisation.

        Returns:
            Boolean land mask.
        """
        working = self._prepare(image)
        smoothed = cv2.GaussianBlur(working, (0, 0), sigmaX=2.0) / 255.0
        land = smoothed > land_threshold
        footprint = morphology.diamond(2)
        land = ndimage.binary_closing(land, structure=footprint)
        land = ndimage.binary_opening(land, structure=footprint)
        return remove_small_components(land, 200)

    # ----------------------------------------------------------- helpers

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """Normalise any input into a ``float32`` image on ``[0, 255]``."""
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array.astype(np.float32), cv2.COLOR_RGB2GRAY)
        array = array.astype(np.float32)
        if array.size == 0:
            return array
        low, high = float(array.min()), float(array.max())
        if high <= low:
            return np.zeros_like(array)
        return ((array - low) / (high - low) * 255.0).astype(np.float32)

    @staticmethod
    def _clean_mask(mask: np.ndarray, min_area: int, disk_radius: int) -> np.ndarray:
        """Open, fill holes and drop small components -- the shared tail of every method."""
        cleaned = ndimage.binary_opening(mask, structure=morphology.disk(disk_radius))
        cleaned = ndimage.binary_fill_holes(cleaned)
        return remove_small_components(cleaned, max(1, min_area))

    @staticmethod
    def _keep_largest_blobs(mask: np.ndarray, count: int) -> np.ndarray:
        """Retain only the ``count`` largest connected components."""
        labels, n_labels = ndimage.label(mask)
        if n_labels == 0:
            return mask.astype(bool)
        sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n_labels + 1))
        keep = np.argsort(sizes)[::-1][: max(1, count)] + 1
        return np.isin(labels, keep)

    @staticmethod
    def _keep_dark_regions(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Drop candidate regions whose mean intensity is above the scene median."""
        labels, n_labels = ndimage.label(mask)
        if n_labels == 0:
            return mask.astype(bool)
        scene_median = float(np.median(image))
        means = ndimage.mean(image, labels=labels, index=np.arange(1, n_labels + 1))
        keep = np.nonzero(means < scene_median)[0] + 1
        return np.isin(labels, keep)

    @staticmethod
    def _filter_by_distance_to_main_blob(mask: np.ndarray, max_distance: float) -> np.ndarray:
        """Keep blobs within ``max_distance`` pixels of the largest one.

        Oil released from a single source stays spatially coherent; look-alikes
        (low-wind cells, biogenic films) tend to sit far from the main slick.
        """
        labels, n_labels = ndimage.label(mask)
        if n_labels <= 1:
            return mask.astype(bool)

        regions = measure.regionprops(labels)
        main = max(regions, key=lambda r: r.area)
        main_centroid = np.array(main.centroid)

        keep = [
            region.label
            for region in regions
            if np.linalg.norm(np.array(region.centroid) - main_centroid) <= max_distance
        ]
        return np.isin(labels, keep)
