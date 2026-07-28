"""Figure generation for SAR oil spill results.

Uses the Agg backend so figures render identically in headless CI, containers
and notebooks. Every function returns the :class:`~matplotlib.figure.Figure`
and optionally writes it to disk.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# Colour-blind-safe palette; oil is cyan to match the original MATLAB overlays.
OIL_COLOR = "#00c2d1"
TRUTH_COLOR = "#f0a202"
LAND_COLOR = "#4c9f70"
FALSE_POSITIVE_COLOR = "#d64550"
FALSE_NEGATIVE_COLOR = "#8b5cf6"


class DataVisualizer:
    """Render SAR imagery, masks and metrics as publication-ready figures."""

    def __init__(self, dpi: int = 130, style: str = "dark_background") -> None:
        self.dpi = dpi
        self.style = style

    # ------------------------------------------------------------ overlays

    def plot_detection_result(
        self,
        original: np.ndarray,
        mask: np.ndarray,
        ground_truth: np.ndarray | None = None,
        title: str = "Oil spill detection",
        save_path: str | Path | None = None,
    ) -> Figure:
        """Original image, predicted overlay and (optionally) the ground truth.

        Args:
            original: SAR image.
            mask: Predicted boolean oil mask.
            ground_truth: Reference mask; when given, a fourth panel shows the
                per-pixel agreement so errors are visible at a glance.
            title: Figure suptitle.
            save_path: Where to write the PNG; skipped when ``None``.
        """
        panel_count = 4 if ground_truth is not None else 3
        with plt.style.context(self.style):
            figure, axes = plt.subplots(
                1, panel_count, figsize=(4.1 * panel_count, 4.4), dpi=self.dpi
            )

            self._show(axes[0], original, "SAR image")

            self._show(axes[1], original, "Detected slick")
            self._overlay(axes[1], mask, OIL_COLOR)

            self._show(axes[2], mask.astype(float), "Predicted mask", cmap="gray")

            if ground_truth is not None:
                self._plot_agreement(axes[3], mask, ground_truth)

            figure.suptitle(title, fontsize=15, weight="bold")
            figure.tight_layout(rect=(0, 0, 1, 0.94))

        return self._finish(figure, save_path)

    def _plot_agreement(
        self, axis: plt.Axes, mask: np.ndarray, ground_truth: np.ndarray
    ) -> None:
        """Colour-code true positives, false positives and misses."""
        prediction = mask.astype(bool)
        truth = ground_truth.astype(bool)

        agreement = np.zeros((*prediction.shape, 4), dtype=float)
        agreement[prediction & truth] = matplotlib.colors.to_rgba(OIL_COLOR)
        agreement[prediction & ~truth] = matplotlib.colors.to_rgba(FALSE_POSITIVE_COLOR)
        agreement[~prediction & truth] = matplotlib.colors.to_rgba(FALSE_NEGATIVE_COLOR)

        axis.imshow(agreement, interpolation="nearest")
        axis.set_title("Agreement", fontsize=11)
        axis.axis("off")
        axis.legend(
            handles=[
                Patch(color=OIL_COLOR, label="correct"),
                Patch(color=FALSE_POSITIVE_COLOR, label="false alarm"),
                Patch(color=FALSE_NEGATIVE_COLOR, label="missed"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            frameon=False,
            fontsize=8,
        )

    # -------------------------------------------------------------- stages

    def plot_pipeline_stages(
        self,
        stages: Mapping[str, np.ndarray],
        title: str = "Processing pipeline",
        save_path: str | Path | None = None,
        columns: int = 3,
    ) -> Figure:
        """Lay out the intermediate images of a segmentation run in order."""
        items = list(stages.items())
        rows = max(1, -(-len(items) // columns))

        with plt.style.context(self.style):
            figure, axes = plt.subplots(
                rows, columns, figsize=(4.0 * columns, 4.0 * rows), dpi=self.dpi
            )
            flat_axes = np.atleast_1d(axes).ravel()

            for axis, (name, image) in zip(flat_axes, items, strict=False):
                self._show(axis, image, name.capitalize())
            for axis in flat_axes[len(items) :]:
                axis.axis("off")

            figure.suptitle(title, fontsize=15, weight="bold")
            figure.tight_layout(rect=(0, 0, 1, 0.94))

        return self._finish(figure, save_path)

    def plot_method_comparison(
        self,
        original: np.ndarray,
        results: Mapping[str, np.ndarray],
        ground_truth: np.ndarray | None = None,
        title: str = "Method comparison",
        save_path: str | Path | None = None,
    ) -> Figure:
        """One panel per method, all overlaid on the same scene."""
        panels: list[tuple[str, np.ndarray | None]] = [("SAR image", None)]
        if ground_truth is not None:
            panels.append(("Ground truth", ground_truth))
        panels.extend((name.replace("_", " "), mask) for name, mask in results.items())

        columns = min(3, len(panels))
        rows = max(1, -(-len(panels) // columns))

        with plt.style.context(self.style):
            figure, axes = plt.subplots(
                rows, columns, figsize=(4.2 * columns, 4.3 * rows), dpi=self.dpi
            )
            flat_axes = np.atleast_1d(axes).ravel()

            for axis, (name, mask) in zip(flat_axes, panels, strict=False):
                self._show(axis, original, name)
                if mask is not None:
                    self._overlay(
                        axis, mask, TRUTH_COLOR if name == "Ground truth" else OIL_COLOR
                    )
            for axis in flat_axes[len(panels) :]:
                axis.axis("off")

            figure.suptitle(title, fontsize=15, weight="bold")
            figure.tight_layout(rect=(0, 0, 1, 0.94))

        return self._finish(figure, save_path)

    # ------------------------------------------------------------- metrics

    def plot_metrics(
        self,
        metrics_by_method: Mapping[str, Mapping[str, float]],
        metric_names: Sequence[str] = ("jaccard_index", "dice_coefficient", "boundary_f1"),
        title: str = "Segmentation quality by method",
        save_path: str | Path | None = None,
    ) -> Figure:
        """Grouped bar chart comparing methods across the chosen metrics."""
        methods = list(metrics_by_method)
        positions = np.arange(len(methods), dtype=float)
        bar_width = 0.8 / max(1, len(metric_names))
        palette = ["#00c2d1", "#f0a202", "#8b5cf6", "#4c9f70", "#d64550"]

        with plt.style.context(self.style):
            figure, axis = plt.subplots(figsize=(1.9 * len(methods) + 4.0, 5.0), dpi=self.dpi)

            for index, metric in enumerate(metric_names):
                values = [float(metrics_by_method[m].get(metric, 0.0)) for m in methods]
                offset = (index - (len(metric_names) - 1) / 2) * bar_width
                bars = axis.bar(
                    positions + offset,
                    values,
                    bar_width,
                    label=metric.replace("_", " "),
                    color=palette[index % len(palette)],
                )
                axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

            axis.set_xticks(positions)
            axis.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=9)
            axis.set_ylim(0, 1.12)
            axis.set_ylabel("Score")
            axis.set_title(title, fontsize=14, weight="bold")
            axis.legend(frameon=False, ncol=len(metric_names), loc="upper center")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            figure.tight_layout()

        return self._finish(figure, save_path)

    def plot_runtime(
        self,
        runtimes: Mapping[str, float],
        title: str = "Runtime by method",
        save_path: str | Path | None = None,
    ) -> Figure:
        """Horizontal bar chart of per-method wall-clock time in milliseconds."""
        methods = list(runtimes)
        values = [runtimes[m] * 1000.0 for m in methods]

        with plt.style.context(self.style):
            figure, axis = plt.subplots(figsize=(8.0, 0.7 * len(methods) + 2.4), dpi=self.dpi)
            bars = axis.barh([m.replace("_", " ") for m in methods], values, color=OIL_COLOR)
            axis.bar_label(bars, fmt="%.0f ms", fontsize=9, padding=3)
            axis.set_xlabel("Milliseconds per 512x512 scene")
            axis.set_xlim(0, max(values) * 1.25 if values else 1)
            axis.set_title(title, fontsize=14, weight="bold")
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
            figure.tight_layout()

        return self._finish(figure, save_path)

    # ------------------------------------------------------------ internals

    @staticmethod
    def _show(axis: plt.Axes, image: np.ndarray, title: str, cmap: str = "gray") -> None:
        axis.imshow(np.asarray(image, dtype=float), cmap=cmap, interpolation="nearest")
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    @staticmethod
    def _overlay(axis: plt.Axes, mask: np.ndarray, color: str, alpha: float = 0.6) -> None:
        """Tint the ``True`` pixels of ``mask`` over whatever the axis shows.

        Built as an explicit RGBA array rather than a masked array plus a
        two-entry colormap: a mask holds a single distinct value, so matplotlib
        would collapse ``vmin`` onto ``vmax`` and map every pixel to the first
        (fully transparent) colour, silently drawing nothing at all.
        """
        selected = np.asarray(mask).astype(bool)
        rgba = np.zeros((*selected.shape, 4), dtype=float)
        rgba[selected] = matplotlib.colors.to_rgba(color, alpha)
        axis.imshow(rgba, interpolation="nearest")

    def _finish(self, figure: Figure, save_path: str | Path | None) -> Figure:
        """Persist the figure when a path is given, then return it."""
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
            logger.info("Wrote figure to %s", path)
        return figure
