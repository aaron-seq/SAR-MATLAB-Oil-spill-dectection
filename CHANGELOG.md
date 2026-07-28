# Changelog

Notable changes to this project. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0]

A rebuild. The previous version did not execute — every entry point failed at
import and the test suite could not be collected.

### Fixed

- **The package did not import.** `sar_image_processor.py` imported
  `skimage.feature.peak_local_maxima`, which does not exist (the function is
  `peak_local_max`). `src/__init__.py` imported four modules that were never
  written. `api/main.py` called `pd.Timestamp` without importing pandas and
  mounted a `static/` directory that did not exist. `main.py` imported three
  modules from a package path that did not exist.
- **Contrast enhancement was destroying detection.** CLAHE was applied by
  default. It normalises contrast within local tiles, so a slick wider than a
  tile has its interior brightened to match the sea — erasing the signal every
  detector depends on. Mean IoU **0.85 → 0.32** with it enabled; it is now off
  by default, with a regression test pinning the default.
- **Local thresholding could not detect a slick wider than its window.** The
  local mean sinks with the slick and the contrast cancels, leaving only the rim
  detected. Replaced with a wide-window background estimate and Otsu on the
  ratio image. Recall **0.009 → 0.971**.
- **Fuzzy edge detection was mistranslated from MATLAB.** The original's
  `imbinarize(Ieval, 0.8)` keeps the *uniform* class, not the edges — which is
  physically right, since oil damps the waves that give the sea its texture.
  IoU **0.004 → 0.941**.
- The Lee filter looped over every pixel in Python (262,144 iterations per
  512×512 scene). Now vectorised over separable box filters, so cost is
  independent of window size.
- The fuzzy inference system was evaluated row by row through an inference
  engine. With two rules and Gaussian memberships it reduces analytically to a
  product of exponentials, now computed directly.
- `lee_filter.m`'s variance term divided a single pixel's squared deviation by
  the window area, which is not a local standard deviation. The port uses
  `E[x²] − E[x]²`.

### Added

- **Synthetic SAR scene generator** — Gamma-distributed speckle, decibel-
  specified slick damping, swath wind gradient, optional land. The original
  dataset is not distributable, so the project was previously unrunnable without
  it; `sar-oil-spill demo` now works on a clean clone.
- **Real implementations** for six modules that were placeholders returning
  zeros: the four classical segmentation methods, the metrics suite and the
  visualiser.
- `OilSpillDetector` — the orchestrator that `src/core/__init__.py` had imported
  but which was never written. Now the single pipeline shared by the CLI, API
  and Streamlit app.
- Full metrics: IoU, Dice, boundary F1, Hausdorff distance, per-object detection
  rate and false-positive count.
- `sar-oil-spill` CLI with `demo`, `detect`, `benchmark` and `dataset`.
- Attention-gated U-Net behind an optional `dl` extra. No trained weights ship
  with the repository.
- 165 tests (84% coverage), ruff, CI across Python 3.10–3.13 plus a job that
  verifies the package works with base dependencies only.
- Non-root multi-stage Dockerfile; `LICENSE`; `CONTRIBUTING.md`; `CLAUDE.md`;
  architecture, configuration and MATLAB reference docs; eight generated figures.

### Changed

- Restructured into an installable src-layout package (`src/sar_oil_spill/`)
  with `pyproject.toml`. MATLAB sources moved to `matlab/` unchanged.
- Configuration is now typed frozen dataclasses. Unknown YAML keys warn rather
  than crash, and a missing file falls back to usable defaults.
- FastAPI rewritten on `lifespan`, with CPU work moved off the event loop,
  upload size and extension limits, a bounded job registry and correct
  multipart form parsing.
- Streamlit app moved off the removed `st.cache` and `use_column_width`.
- `remove_small_components` reimplemented on scipy: scikit-image 0.26 renamed
  `min_size` → `max_size` *and* changed the comparison from strict to inclusive,
  so counting directly keeps one behaviour across supported versions.
- README rewritten against measured output. The previous version advertised IoU
  0.847, "10x performance" and 91% test coverage with per-module breakdowns —
  for code that could not run — and documented a directory structure that did
  not match the tree.

### Removed

- `deployment/vercel.json`. The dependency set (OpenCV, SciPy, scikit-image,
  scikit-learn) exceeds Vercel's serverless bundle limit, so that configuration
  could never have deployed. `deployment/README.md` explains why.
- Duplicate placeholder modules `src/evaluation_metrics.py`,
  `src/traditional_segmentation.py`, `src/visualization.py`.
- `OLDREADME.md`.

### Known gaps

- All accuracy figures are measured on synthetic scenes. Nothing here has been
  validated against real Sentinel-1 imagery.
- No look-alike discrimination — low-wind cells, biogenic films and rain cells
  are dark and smooth too.
- No georeferencing; areas are reported in pixels, not m².
- The U-Net is architecture only.
- API batch jobs are in-process and not safe across replicas.
- The Docker build has not been executed (no daemon in the development
  environment); CI is its first real exercise.

## [2.0.0] and earlier

See the git history. Released versions did not execute.
