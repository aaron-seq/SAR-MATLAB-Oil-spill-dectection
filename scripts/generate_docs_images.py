"""Regenerate every figure embedded in the documentation.

Run from the repository root::

    python scripts/generate_docs_images.py

Everything is produced from real runs of the pipeline over deterministic
synthetic scenes, so the figures in the README always match what the code
actually does. Re-run this after changing any algorithm.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from sar_oil_spill.config import configure_logging, load_settings
from sar_oil_spill.core import OilSpillDetector
from sar_oil_spill.data import generate_dataset, generate_sar_scene
from sar_oil_spill.models import METHOD_NAMES
from sar_oil_spill.utils import DataVisualizer, PerformanceEvaluator

logger = logging.getLogger("generate_docs_images")

OUTPUT_DIR = Path("docs/images")
BENCHMARK_SCENES = 12
SCENE_SIZE = (512, 512)
SHOWCASE_SEED = 42


def figure_hero(detector: OilSpillDetector, visualizer: DataVisualizer) -> None:
    """Headline figure: one scene, detection overlay and agreement map."""
    scene = generate_sar_scene(size=SCENE_SIZE, n_slicks=2, seed=SHOWCASE_SEED)
    result = detector.detect(scene.image, method="kmeans_clustering", ground_truth=scene.oil_mask)
    assert result.metrics is not None

    visualizer.plot_detection_result(
        result.preprocessed_image,
        result.mask,
        ground_truth=detector._align(scene.oil_mask, result.mask.shape),
        title=(
            f"K-means clustering  |  IoU {result.metrics.jaccard_index:.3f}  "
            f"|  Dice {result.metrics.dice_coefficient:.3f}"
        ),
        save_path=OUTPUT_DIR / "detection-overview.png",
    )


def figure_pipeline(detector: OilSpillDetector, visualizer: DataVisualizer) -> None:
    """Every intermediate stage of the adaptive-threshold pipeline."""
    scene = generate_sar_scene(size=SCENE_SIZE, n_slicks=2, seed=SHOWCASE_SEED)
    result = detector.detect(scene.image, method="adaptive_threshold")

    visualizer.plot_pipeline_stages(
        result.stages,
        title="Adaptive threshold: background-normalised Otsu, stage by stage",
        save_path=OUTPUT_DIR / "pipeline-stages.png",
        columns=3,
    )


def figure_method_comparison(detector: OilSpillDetector, visualizer: DataVisualizer) -> None:
    """All four methods on one scene, against the ground truth."""
    scene = generate_sar_scene(size=SCENE_SIZE, n_slicks=2, seed=SHOWCASE_SEED)
    results = detector.compare_methods(scene.image, ground_truth=scene.oil_mask)
    reference = next(iter(results.values()))

    visualizer.plot_method_comparison(
        reference.preprocessed_image,
        {
            f"{name}\nIoU {r.metrics.jaccard_index:.3f}": r.mask
            for name, r in results.items()
            if r.metrics is not None
        },
        ground_truth=detector._align(scene.oil_mask, reference.mask.shape),
        title="Segmentation methods compared on one synthetic scene",
        save_path=OUTPUT_DIR / "method-comparison.png",
    )


def figure_land_scene(detector: OilSpillDetector, visualizer: DataVisualizer) -> None:
    """A coastal scene, showing land exclusion at work."""
    scene = generate_sar_scene(size=SCENE_SIZE, with_land=True, n_slicks=2, seed=17)
    result = detector.detect(
        scene.image, method="adaptive_threshold", ground_truth=scene.oil_mask, mask_land=True
    )
    assert result.metrics is not None and result.land_mask is not None

    visualizer.plot_pipeline_stages(
        {
            "SAR scene with coastline": result.preprocessed_image,
            "detected land (excluded)": result.land_mask.astype(float),
            "oil ground truth": detector._align(scene.oil_mask, result.mask.shape).astype(float),
            "detected oil": result.mask.astype(float),
        },
        title=f"Coastal scene with land masking  |  IoU {result.metrics.jaccard_index:.3f}",
        save_path=OUTPUT_DIR / "land-masking.png",
        columns=4,
    )


def figure_preprocessing_effect(visualizer: DataVisualizer) -> None:
    """Why contrast enhancement is disabled: CLAHE erases the slick."""
    from dataclasses import replace

    scene = generate_sar_scene(size=SCENE_SIZE, n_slicks=1, seed=5)
    base = load_settings()

    plain = OilSpillDetector(base)
    with_clahe = OilSpillDetector(
        replace(
            base,
            image_processing=replace(
                base.image_processing, enhance_contrast=True, enhancement_method="clahe"
            ),
        )
    )

    without = plain.detect(scene.image, ground_truth=scene.oil_mask)
    boosted = with_clahe.detect(scene.image, ground_truth=scene.oil_mask)
    assert without.metrics is not None and boosted.metrics is not None

    visualizer.plot_pipeline_stages(
        {
            "despeckled only": without.preprocessed_image,
            f"detection  IoU {without.metrics.jaccard_index:.3f}": without.mask.astype(float),
            "despeckled + CLAHE": boosted.preprocessed_image,
            f"detection  IoU {boosted.metrics.jaccard_index:.3f}": boosted.mask.astype(float),
        },
        title="CLAHE normalises away the slick-versus-sea contrast detection needs",
        save_path=OUTPUT_DIR / "preprocessing-effect.png",
        columns=4,
    )


def figure_speckle_filters(visualizer: DataVisualizer) -> None:
    """Side-by-side comparison of the despeckling filters."""
    from sar_oil_spill.core import SARImageProcessor

    scene = generate_sar_scene(size=(384, 384), n_slicks=1, looks=2, seed=23)
    processor = SARImageProcessor()

    stages = {"raw (2-look speckle)": scene.image}
    for filter_type in ("lee", "frost", "kuan", "median", "bilateral"):
        stages[filter_type] = processor.apply_despeckling_filter(
            scene.image, filter_type=filter_type, window_size=7
        )

    visualizer.plot_pipeline_stages(
        stages,
        title="Despeckling filters on a 2-look SAR scene",
        save_path=OUTPUT_DIR / "despeckling-filters.png",
        columns=3,
    )


def figure_benchmark(detector: OilSpillDetector, visualizer: DataVisualizer) -> dict:
    """Aggregate metrics and runtimes over a set of scenes."""
    evaluator = PerformanceEvaluator()
    scenes = generate_dataset(count=BENCHMARK_SCENES, size=SCENE_SIZE, seed=SHOWCASE_SEED)

    metrics: dict[str, list] = {name: [] for name in METHOD_NAMES}
    runtimes: dict[str, list[float]] = {name: [] for name in METHOD_NAMES}

    for index, scene in enumerate(scenes, start=1):
        for name in METHOD_NAMES:
            result = detector.detect(scene.image, method=name, ground_truth=scene.oil_mask)
            if result.metrics is not None:
                metrics[name].append(result.metrics)
            runtimes[name].append(result.processing_time)
        logger.info("Benchmark scene %d/%d", index, len(scenes))

    aggregated = {name: evaluator.aggregate(values) for name, values in metrics.items()}
    mean_runtimes = {name: float(np.mean(times)) for name, times in runtimes.items()}

    visualizer.plot_metrics(
        {
            name: {key.removeprefix("mean_"): value for key, value in values.items()}
            for name, values in aggregated.items()
        },
        metric_names=("jaccard_index", "dice_coefficient", "precision", "recall"),
        title=f"Accuracy over {BENCHMARK_SCENES} synthetic scenes (512x512)",
        save_path=OUTPUT_DIR / "benchmark-metrics.png",
    )
    visualizer.plot_runtime(
        mean_runtimes,
        title="Mean end-to-end runtime per scene",
        save_path=OUTPUT_DIR / "benchmark-runtime.png",
    )

    return {
        "scenes": BENCHMARK_SCENES,
        "scene_size": list(SCENE_SIZE),
        "seed": SHOWCASE_SEED,
        "metrics": aggregated,
        "mean_runtime_seconds": mean_runtimes,
    }


def main() -> int:
    configure_logging("INFO")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = OilSpillDetector(load_settings())
    visualizer = DataVisualizer(dpi=120)

    figure_hero(detector, visualizer)
    figure_pipeline(detector, visualizer)
    figure_method_comparison(detector, visualizer)
    figure_land_scene(detector, visualizer)
    figure_preprocessing_effect(visualizer)
    figure_speckle_filters(visualizer)
    report = figure_benchmark(detector, visualizer)

    results_path = Path("docs/benchmark-results.json")
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Figures written to %s", OUTPUT_DIR.resolve())
    logger.info("Benchmark data written to %s", results_path.resolve())

    print("\nMeasured results (mean over "
          f"{BENCHMARK_SCENES} scenes):\n")
    print(f"{'method':<24}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Recall':>8}{'BF1':>8}{'Time':>11}")
    print("-" * 75)
    for name, values in sorted(
        report["metrics"].items(), key=lambda kv: -kv[1]["mean_jaccard_index"]
    ):
        print(
            f"{name:<24}"
            f"{values['mean_jaccard_index']:>8.3f}"
            f"{values['mean_dice_coefficient']:>8.3f}"
            f"{values['mean_precision']:>8.3f}"
            f"{values['mean_recall']:>8.3f}"
            f"{values['mean_boundary_f1']:>8.3f}"
            f"{report['mean_runtime_seconds'][name] * 1000:>8.0f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
