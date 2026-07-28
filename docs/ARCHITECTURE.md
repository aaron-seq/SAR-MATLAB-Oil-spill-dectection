# Architecture

Contributor-facing design notes. The [README](../README.md) covers what the
system does and how to use it; this covers how it is put together and why.

## Layering

```
                       ┌──────────┬──────────┬────────────┐
   entry points        │   CLI    │ FastAPI  │  Streamlit │
                       └────┬─────┴────┬─────┴──────┬─────┘
                            └──────────┼────────────┘
                                       ▼
                            ┌────────────────────┐
   orchestration            │  OilSpillDetector  │
                            └─────────┬──────────┘
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
        ┌───────────────────┐ ┌───────────────┐ ┌──────────────────────┐
   work │ SARImageProcessor │ │ Traditional-  │ │ PerformanceEvaluator │
        │  (preprocessing)  │ │ Segmentation  │ │      (metrics)       │
        └───────────────────┘ └───────────────┘ └──────────────────────┘
                    ▲                 ▲                  ▲
                    └─────────────────┴──────────────────┘
                                       │
                            ┌──────────┴──────────┐
   inputs                   │  data/  +  config/  │
                            └─────────────────────┘
```

The rule that matters: **all three entry points go through
`OilSpillDetector`.** Nothing reimplements the pipeline, so the same image and
settings produce byte-identical masks from the CLI, the API and the browser UI.
When adding a surface, call the detector.

## Why the modules split where they do

**`SARImageProcessor` knows nothing about oil.** It despeckles, enhances,
resizes and cleans masks. That keeps it reusable for any SAR task and testable
without ground truth.

**`TraditionalSegmentation` knows nothing about files, config loading or
output.** Each method takes an array and returns a `SegmentationResult`. That is
why the parametrised tests can iterate `METHOD_NAMES` and treat every method
identically.

**`OilSpillDetector` is the only place that knows the whole story** — which
settings apply, whether to mask land, when to score against ground truth. It is
also where cross-cutting concerns live (resizing ground truth to match a
prediction, computing a confidence score).

**`PerformanceEvaluator` never sees an image**, only two masks. That makes every
metric trivially unit-testable against hand-computed values, which
`tests/test_evaluation.py` does.

## The single registry

`METHOD_NAMES` in `models/traditional_segmentation.py` is the one list of
available methods. It drives:

- the CLI's `--method` choices
- the API's validation and `/api/v1/methods` response
- `OilSpillDetector.compare_methods`
- the parametrised tests
- the comparison figure

Adding a method means adding one tuple entry. Anything that hardcodes a method
list somewhere else is a bug.

## Data flow through a detection

```
np.ndarray (any dtype, any size)
  │
  │ OilSpillDetector.preprocess
  ├─→ despeckle          driven by settings.image_processing
  ├─→ enhance contrast   (off by default — see below)
  └─→ resize             to settings.image_processing.target_size
  │
  │ TraditionalSegmentation.segment
  ├─→ method-specific work, recording every intermediate into `stages`
  └─→ morphological cleanup, hole filling, small-object removal
  │
  │ OilSpillDetector (post)
  ├─→ land mask subtraction    if mask_land=True
  ├─→ minimum area filter
  └─→ metrics                  if ground_truth was supplied
  │
  ▼
DetectionResult (mask, stages, metrics, timing, processing history)
```

`stages` is deliberately carried on the result rather than being drawn inline.
Segmentation stays free of plotting code, and callers can visualise a run
without re-executing it — which is what the pipeline figures and the Streamlit
expander both do.

## Design decisions worth knowing

### Preprocessing does not enhance contrast

CLAHE and histogram equalisation normalise contrast *locally*. A slick wider
than a CLAHE tile has its interior lifted to match the surrounding sea, which
removes the exact intensity gap every detector keys on. Measured across the
benchmark, enabling it takes mean IoU from 0.85 to 0.32.

Enhancement is still implemented and still useful — for *display*, where the
goal is for a human to see the scene, not for a threshold to separate classes.

### Thresholding is done on a ratio, not on intensity

The sea's mean backscatter drifts across the swath with wind speed and incidence
angle, so a global threshold cannot work. But a *local* threshold fails the
other way: over a slick wider than its window, the local mean sinks with the
slick and the contrast cancels.

The resolution is to estimate the background over a window much wider than any
plausible slick and divide. Under SAR's multiplicative noise model that ratio is
the meaningful quantity — roughly 1.0 over sea and 0.3–0.5 over oil, independent
of absolute brightness — and Otsu then separates the two modes.

### Speckle filters are vectorised over box filters

Lee, Frost and Kuan all need a local mean and local variance. Both come from
`cv2.blur` over the image and its square (`E[x²] − E[x]²`), so a 7×7 and a 31×31
window cost the same. This is what makes the filters usable at full scene size.

### Fuzzy inference is evaluated in closed form

The MATLAB original builds a Mamdani FIS and calls `evalfis` per image row. With
two rules and Gaussian input memberships, the inference collapses analytically to

```
uniformity(x, y) = exp(−Gx² / 2σ²) · exp(−Gy² / 2σ²)
```

which is evaluated directly. Same semantics, no inference engine, milliseconds
instead of minutes.

### Optional dependencies are guarded at import

`deep_learning_segmentation.py` wraps its PyTorch import in a `try`, sets
`TORCH_AVAILABLE`, and defines stub classes that raise a naming `ImportError`
when instantiated. `models/__init__.py` exports the deep-learning symbols
lazily through `__getattr__`, so importing the package never pulls in torch.

A CI job installs the base package only and asserts both that the import
succeeds and that the error message names the extra.

### Configuration is typed and forgiving

YAML loads into frozen dataclasses. Unknown keys log a warning and are ignored
rather than raising, so an old config file keeps working after a rename. A
missing config file is not an error either — the built-in defaults are usable,
which is what lets the demo and the tests run with no setup.

A test asserts that the shipped YAML matches the dataclass defaults it claims to
document, so the two cannot silently drift.

## Testing strategy

| File | Covers |
|---|---|
| `test_sar_processor.py` | Filters, enhancement, morphology, normalisation, edge cases |
| `test_segmentation.py` | Each method end to end; detector orchestration; the CLAHE regression guard |
| `test_evaluation.py` | Metrics against hand-computed values; mask encoding equivalence |
| `test_api.py` | Every endpoint, including rejection paths |
| `test_data_and_config.py` | Synthetic generator physics, dataset indexing, config loading, CLI |

The synthetic generator is what makes this tractable: tests get scenes with
*known* ground truth, so accuracy assertions are meaningful rather than
smoke-test-shaped. `tests/test_segmentation.py` requires every method to clear
IoU 0.6, well below the measured 0.86–0.94, so the threshold catches real
regressions without being flaky.

The generator itself is tested against the physics it claims to model: slicks
must be darker than the sea, higher contrast must darken them further, more
looks must reduce speckle variance, oil and land must not overlap.
