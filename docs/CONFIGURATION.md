# Configuration reference

Every key in `config/model_config.yaml`, what it does, and when to change it.

Configuration loads into frozen dataclasses in `sar_oil_spill.config`. Two
properties are deliberate:

- **A missing config file is not an error.** Built-in defaults are usable, which
  is why the demo and the tests need no setup.
- **Unknown keys warn and are ignored.** An old config keeps working after a
  rename instead of crashing.

```python
from sar_oil_spill.config import load_settings

settings = load_settings()                          # config/model_config.yaml
settings = load_settings("experiments/wide.yaml")   # or anywhere else
```

```bash
sar-oil-spill benchmark --config experiments/wide.yaml
```

## `image_processing`

Applied to every image before segmentation.

| Key | Default | Notes |
|---|---|---|
| `target_size` | `[512, 512]` | `[height, width]`. Larger costs roughly quadratic time. |
| `normalize` | `true` | Rescale intensities before segmentation. |
| `despeckle` | `true` | Turning this off degrades every method badly — speckle is the dominant noise. |
| `despeckle_filter` | `lee` | `lee`, `frost`, `kuan`, `bilateral`, `median`. |
| `despeckle_window` | `7` | Forced odd. Larger removes more speckle and more small slicks. |
| `enhance_contrast` | `false` | **Leave off.** See below. |
| `enhancement_method` | `none` | `none`, `clahe`, `histogram_equalization`, `gamma`. |

### Why `enhance_contrast` is off

CLAHE and histogram equalisation normalise contrast *locally*. A slick wider
than a CLAHE tile has its interior brightened to match the surrounding sea,
which removes the exact intensity gap detection relies on.

> Measured across the benchmark: mean IoU **0.85 → 0.32**.

`docs/images/preprocessing-effect.png` shows what this looks like. Enhancement
is for display, not detection. A test pins the default.

### Choosing a despeckle filter

| Filter | Character |
|---|---|
| `lee` | The default. Good edge preservation, cheap. |
| `frost` | Exponentially weighted; slightly better on strong texture, slower. |
| `kuan` | Similar to Lee, derived from a linear MMSE criterion. |
| `median` | Robust to outliers; blunts corners. |
| `bilateral` | Edge-preserving but not SAR-specific; treats speckle as additive. |

Lee, Frost and Kuan use separable box filters, so window size does not affect
cost. See `docs/images/despeckling-filters.png`.

## `traditional_methods`

### `adaptive_threshold`

| Key | Default | Notes |
|---|---|---|
| `background_window` | `251` | Must be much wider than any slick. |
| `offset` | `0.0` | Extra margin below the Otsu threshold on the ratio image. Raise to trade recall for precision. |
| `min_blob_area` | `100` | Detections smaller than this are dropped. |
| `median_filter_size` | `3` | Pre-smoothing before the background estimate. |

`background_window` is the one to get right. The method divides the image by a
wide-window local mean and thresholds the ratio. If the window is comparable to
the slick, the background sinks with the slick and the contrast cancels — that
failure mode scored recall 0.009. Rule of thumb: at least three times the widest
slick you expect. It is clamped to the image size, so oversizing is safe.

### `kmeans`

| Key | Default | Notes |
|---|---|---|
| `n_clusters` | `5` | Grey-level clusters. Too few merges oil with dark sea; too many fragments the slick. |
| `max_iter` | `350` | Lloyd iterations. |
| `n_init` | `5` | Restarts. Lower is faster, less stable. |
| `random_state` | `42` | Fixed for reproducibility. |
| `n_largest_blobs` | `3` | Keep only the N biggest components. Raise if you expect many separate slicks. |
| `gaussian_sigma` | `1.5` | Pre-smoothing. |

The most accurate method on the benchmark but the slowest, since it clusters
every pixel. `n_init` is the main cost lever.

### `superpixel`

| Key | Default | Notes |
|---|---|---|
| `n_segments` | `1200` | SLIC superpixels. More means finer boundaries and more time. |
| `compactness` | `10.0` | Higher gives squarer superpixels; lower follows intensity more closely. |
| `sigma` | `1.0` | Pre-smoothing before SLIC. |
| `max_centroid_distance` | `450.0` | Pixels. Blobs further than this from the main slick are discarded. |

The most conservative method — highest precision, lowest recall. The distance
filter assumes one dominant slick with satellites nearby; raise it or set it very
large if you expect spatially unrelated spills.

### `fuzzy_edge`

| Key | Default | Notes |
|---|---|---|
| `lee_window` | `5` | Despeckling window inside the method. |
| `sigma` | `0.1` | Width of the "gradient is zero" membership. Sensitive — see below. |
| `binarize_threshold` | `0.7` | Minimum uniformity for a pixel to count as smooth. |

This method thresholds **uniformity, not edges**: oil damps the waves that give
the sea its texture, so a slick is smoother than the water around it. Darkness
is then applied as a second check.

`sigma` and `binarize_threshold` interact strongly. Measured mean IoU over 6
scenes while tuning:

| `sigma` | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| 0.02 | 0.094 | 0.031 | 0.015 | 0.006 |
| 0.05 | 0.871 | 0.579 | 0.336 | 0.086 |
| **0.10** | 0.871 | **0.907** | 0.900 | 0.705 |
| 0.20 | 0.431 | 0.729 | 0.837 | 0.902 |

(columns are `binarize_threshold`). The defaults sit at the broad optimum. If
you change one, re-sweep both.

## `deep_learning`

Only used when the `dl` extra is installed. No trained weights ship with the
repository.

| Key | Default | Notes |
|---|---|---|
| `architecture` | `unet` | `unet` or `improved_unet` (same attention-gated network). |
| `batch_size` | `8` | 512×512 scenes at base width fit ~8 per 8 GB of VRAM. |
| `learning_rate` | `0.001` | |
| `epochs` | `100` | |
| `early_stopping_patience` | `15` | |
| `base_channels` | `32` | Encoder width; doubles per level. Halve it if memory-bound. |
| `num_classes` | `1` | One logit — oil versus not-oil. |
| `dropout` | `0.2` | Applied in the bottleneck. |
| `checkpoint_dir` | `models/saved_models` | |

## `evaluation`

| Key | Default | Notes |
|---|---|---|
| `metrics` | jaccard, dice, pixel_accuracy, precision, recall, boundary_f1 | Informational; the evaluator computes all of them regardless. |
| `boundary_tolerance` | `2` | Pixels. How far a predicted boundary may sit from the truth and still count as matching. |
| `save_predictions` | `true` | |

Raise `boundary_tolerance` to be lenient about contour placement, lower it to be
strict. It only affects `boundary_f1`.

## `system` and `paths`

| Key | Default | Notes |
|---|---|---|
| `system.log_level` | `INFO` | The CLI's `--log-level` overrides this. |
| `system.random_seed` | `42` | |
| `system.use_gpu` | `true` | Only consulted by the deep-learning path. |
| `paths.results_dir` | `results` | |
| `paths.logs_dir` | `logs` | |
| `paths.model_save_dir` | `models/saved_models` | |

## Overriding in Python

Settings are frozen dataclasses, so use `dataclasses.replace` for experiments
rather than mutating:

```python
from dataclasses import replace
from sar_oil_spill import OilSpillDetector
from sar_oil_spill.config import load_settings

base = load_settings()
wide = replace(
    base,
    traditional_methods=replace(
        base.traditional_methods,
        adaptive_threshold=replace(
            base.traditional_methods.adaptive_threshold, background_window=401
        ),
    ),
)

detector = OilSpillDetector(wide)
```
