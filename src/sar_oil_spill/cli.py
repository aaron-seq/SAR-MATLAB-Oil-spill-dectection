"""Command-line interface for the SAR oil spill detection system.

Subcommands::

    sar-oil-spill demo        # generate a synthetic scene and detect on it
    sar-oil-spill detect      # run one method on one image file
    sar-oil-spill benchmark   # score every method over a set of scenes
    sar-oil-spill dataset     # summarise a dataset on disk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from sar_oil_spill import __version__
from sar_oil_spill.config import configure_logging, load_settings
from sar_oil_spill.core import OilSpillDetector
from sar_oil_spill.data import SARDatasetHandler, generate_dataset, generate_sar_scene
from sar_oil_spill.models.traditional_segmentation import METHOD_NAMES
from sar_oil_spill.utils import DataVisualizer, PerformanceEvaluator

logger = logging.getLogger("sar_oil_spill.cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sar-oil-spill",
        description="Detect and segment oil spills in SAR imagery.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to a YAML configuration file."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="Run the pipeline on a generated synthetic scene (no dataset needed)."
    )
    demo.add_argument("--method", default="adaptive_threshold", choices=METHOD_NAMES)
    demo.add_argument("--with-land", action="store_true", help="Include a coastline.")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--size", type=int, default=512, help="Square scene side length.")
    demo.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="Where to write figures."
    )
    demo.set_defaults(handler=_run_demo)

    detect = subparsers.add_parser("detect", help="Detect oil in a SAR image file.")
    detect.add_argument("image", type=Path, help="Path to the SAR image.")
    detect.add_argument("--method", default="adaptive_threshold", choices=METHOD_NAMES)
    detect.add_argument("--ground-truth", type=Path, default=None, help="Reference mask.")
    detect.add_argument("--mask-land", action="store_true", help="Exclude bright land areas.")
    detect.add_argument("--output-dir", type=Path, default=Path("results"))
    detect.set_defaults(handler=_run_detect)

    benchmark = subparsers.add_parser(
        "benchmark", help="Score every method across a set of scenes."
    )
    benchmark.add_argument(
        "--dataset", type=Path, default=None, help="Dataset root; omit for synthetic."
    )
    benchmark.add_argument("--samples", type=int, default=10, help="Number of scenes.")
    benchmark.add_argument("--size", type=int, default=512)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument(
        "--methods", nargs="*", default=list(METHOD_NAMES), choices=METHOD_NAMES
    )
    benchmark.add_argument("--output-dir", type=Path, default=Path("results"))
    benchmark.set_defaults(handler=_run_benchmark)

    dataset = subparsers.add_parser("dataset", help="Summarise a dataset directory.")
    dataset.add_argument("root", type=Path, help="Dataset root directory.")
    dataset.add_argument("--sample-size", type=int, default=20)
    dataset.set_defaults(handler=_run_dataset)

    return parser


# --------------------------------------------------------------- handlers


def _run_demo(args: argparse.Namespace) -> int:
    """Generate a synthetic scene, detect on it, and write figures."""
    settings = load_settings(args.config)
    detector = OilSpillDetector(settings)
    visualizer = DataVisualizer()

    scene = generate_sar_scene(
        size=(args.size, args.size), with_land=args.with_land, seed=args.seed
    )
    logger.info("Generated scene with %.2f%% oil coverage", scene.oil_fraction * 100)

    result = detector.detect(
        scene.image, method=args.method, ground_truth=scene.oil_mask, mask_land=args.with_land
    )

    output_dir = Path(args.output_dir)
    visualizer.plot_detection_result(
        result.preprocessed_image,
        result.mask,
        ground_truth=detector._align(scene.oil_mask, result.mask.shape),
        title=f"{args.method.replace('_', ' ').title()} on a synthetic SAR scene",
        save_path=output_dir / f"demo_{args.method}.png",
    )
    visualizer.plot_pipeline_stages(
        result.stages,
        title=f"Pipeline stages: {args.method.replace('_', ' ')}",
        save_path=output_dir / f"demo_{args.method}_stages.png",
    )

    _print_report(result)
    print(f"\nFigures written to {output_dir.resolve()}")
    return 0


def _run_detect(args: argparse.Namespace) -> int:
    """Detect oil in a single image file."""
    settings = load_settings(args.config)
    detector = OilSpillDetector(settings)

    if not args.image.exists():
        logger.error("Image not found: %s", args.image)
        return 1

    ground_truth = None
    if args.ground_truth is not None:
        import cv2

        loaded = cv2.imread(str(args.ground_truth), cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            logger.error("Could not read ground truth: %s", args.ground_truth)
            return 1
        ground_truth = loaded > 127

    result = detector.detect_from_file(
        args.image, method=args.method, ground_truth=ground_truth, mask_land=args.mask_land
    )

    output_dir = Path(args.output_dir)
    DataVisualizer().plot_detection_result(
        result.preprocessed_image,
        result.mask,
        ground_truth=None
        if ground_truth is None
        else detector._align(ground_truth, result.mask.shape),
        title=f"{args.image.name} - {args.method.replace('_', ' ')}",
        save_path=output_dir / f"{args.image.stem}_{args.method}.png",
    )

    _print_report(result)
    print(f"\nFigure written to {output_dir.resolve()}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    """Score every requested method over synthetic or on-disk scenes."""
    settings = load_settings(args.config)
    detector = OilSpillDetector(settings)
    evaluator = PerformanceEvaluator()
    visualizer = DataVisualizer()

    scenes = _collect_scenes(args)
    if not scenes:
        logger.error("No scenes to benchmark.")
        return 1

    logger.info("Benchmarking %d methods over %d scenes", len(args.methods), len(scenes))

    per_method_metrics: dict[str, list] = {m: [] for m in args.methods}
    per_method_runtime: dict[str, list[float]] = {m: [] for m in args.methods}

    for index, (image, truth) in enumerate(scenes, start=1):
        for method in args.methods:
            try:
                result = detector.detect(image, method=method, ground_truth=truth)
            except Exception as error:
                logger.error("Scene %d, method %s failed: %s", index, method, error)
                continue
            if result.metrics is not None:
                per_method_metrics[method].append(result.metrics)
            per_method_runtime[method].append(result.processing_time)
        logger.info("Scene %d/%d done", index, len(scenes))

    summary = {
        method: evaluator.aggregate(metrics)
        for method, metrics in per_method_metrics.items()
        if metrics
    }
    runtimes = {
        method: float(np.mean(times)) for method, times in per_method_runtime.items() if times
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scenes": len(scenes),
        "source": "dataset" if args.dataset else "synthetic",
        "metrics": summary,
        "mean_runtime_seconds": runtimes,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    chart_data = {
        method: {key.removeprefix("mean_"): value for key, value in values.items()}
        for method, values in summary.items()
    }
    if chart_data:
        visualizer.plot_metrics(chart_data, save_path=output_dir / "benchmark_metrics.png")
    if runtimes:
        visualizer.plot_runtime(runtimes, save_path=output_dir / "benchmark_runtime.png")

    _print_benchmark_table(summary, runtimes)
    print(f"\nReport written to {(output_dir / 'benchmark.json').resolve()}")
    return 0


def _run_dataset(args: argparse.Namespace) -> int:
    """Print a summary of a dataset directory."""
    handler = SARDatasetHandler(load_settings(args.config))
    if not handler.load_dataset(args.root):
        return 1

    stats = handler.get_dataset_statistics(sample_size=args.sample_size)
    print("\n" + "=" * 58)
    print(f"{'DATASET SUMMARY':^58}")
    print("=" * 58)
    print(f"  Total images         {stats.total_images}")
    print(f"  With land            {stats.images_with_land}")
    print(f"  Without land         {stats.images_without_land}")
    print(f"  Target size          {stats.target_size[0]} x {stats.target_size[1]}")
    print(f"  Oil coverage         {stats.oil_spill_percentage:.2f}%  "
          f"(from {stats.sampled_images} sampled scenes)")
    print("=" * 58)
    return 0


# ---------------------------------------------------------------- helpers


def _collect_scenes(args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    """Gather ``(image, ground_truth)`` pairs from disk or the generator."""
    if args.dataset is not None:
        handler = SARDatasetHandler(load_settings(args.config))
        if not handler.load_dataset(args.dataset):
            return []
        return [(image, mask) for _, image, mask in handler.iter_samples(limit=args.samples)]

    return [
        (scene.image, scene.oil_mask)
        for scene in generate_dataset(
            count=args.samples, size=(args.size, args.size), seed=args.seed
        )
    ]


def _print_report(result) -> None:
    """Print a single detection result as an aligned block."""
    print("\n" + "=" * 58)
    print(f"{'DETECTION RESULT':^58}")
    print("=" * 58)
    print(f"  Method               {result.method}")
    print(f"  Oil detected         {'yes' if result.oil_detected else 'no'}")
    print(f"  Affected area        {result.affected_area_pixels:,} px "
          f"({result.coverage_fraction * 100:.2f}% of scene)")
    print(f"  Confidence           {result.confidence:.3f}")
    print(f"  Processing time      {result.processing_time * 1000:.0f} ms")
    if result.metrics is not None:
        m = result.metrics
        print("-" * 58)
        print(f"  IoU (Jaccard)        {m.jaccard_index:.3f}")
        print(f"  Dice                 {m.dice_coefficient:.3f}")
        print(f"  Precision / Recall   {m.precision:.3f} / {m.recall:.3f}")
        print(f"  Boundary F1          {m.boundary_f1:.3f}")
        print(f"  Pixel accuracy       {m.pixel_accuracy:.3f}")
    print("=" * 58)


def _print_benchmark_table(
    summary: dict[str, dict[str, float]], runtimes: dict[str, float]
) -> None:
    """Print aggregated benchmark results as a fixed-width table."""
    print("\n" + "=" * 78)
    print(f"{'BENCHMARK SUMMARY':^78}")
    print("=" * 78)
    print(f"{'method':<24}{'IoU':>9}{'Dice':>9}{'Precision':>11}{'Recall':>9}{'Time':>12}")
    print("-" * 78)
    for method, values in sorted(
        summary.items(), key=lambda kv: kv[1].get("mean_jaccard_index", 0), reverse=True
    ):
        print(
            f"{method:<24}"
            f"{values.get('mean_jaccard_index', 0):>9.3f}"
            f"{values.get('mean_dice_coefficient', 0):>9.3f}"
            f"{values.get('mean_precision', 0):>11.3f}"
            f"{values.get('mean_recall', 0):>9.3f}"
            f"{runtimes.get(method, 0) * 1000:>9.0f} ms"
        )
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sar-oil-spill`` console script."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        logger.error("%s", error, exc_info=args.log_level == "DEBUG")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
