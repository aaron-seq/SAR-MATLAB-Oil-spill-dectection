# Dispatch prompt

A ready-to-paste prompt for starting a Claude Code session on this repository —
from claude.ai/code, the CLI, or a GitHub Action. Copy the block below, delete
the tasks you don't want, and send it.

The prompt front-loads the things a fresh session would otherwise waste a lot of
turns rediscovering: that the benchmarks are synthetic, that three
counter-intuitive defaults are deliberate, and that documented numbers are
measured rather than aspirational.

---

## The prompt

````text
You are working on aaron-seq/SAR-MATLAB-Oil-spill-dectection, a Python system
that detects oil spills in Synthetic Aperture Radar imagery. Read CLAUDE.md
first — it covers the architecture, commands and conventions.

## Context you need before changing anything

The physics: oil damps the capillary waves that scatter radar back to the
satellite, so a slick is DARKER and SMOOTHER than the sea around it. Every
detector here rests on that. If a change breaks that assumption, it is wrong
regardless of what the metrics say.

Three defaults look like bugs and are not. Each is explained in a docstring and
pinned by a test. Do not "fix" them without new measurements:

1. Contrast enhancement is OFF (`config.py`). CLAHE normalises contrast within
   local tiles, which erases the slick-versus-sea gap detection depends on.
   Enabling it drops mean IoU from 0.85 to 0.32.
2. The adaptive threshold's background window is 251 px, far wider than any
   slick. A narrow window makes the background sink with the slick, cancelling
   the contrast — that bug scored recall 0.009.
3. Fuzzy edge detection thresholds UNIFORMITY, not edges. Slicks are smooth;
   the sea is speckled. Thresholding edges scored IoU 0.004.

Benchmarks run on SYNTHETIC scenes from `sar_oil_spill.data.synthetic`. They
compare methods fairly and catch regressions, but they do NOT predict accuracy
on real Sentinel-1 data. Never present them as if they do.

## Rules

- Every performance or accuracy number you write into docs must come from
  `python scripts/generate_docs_images.py` or `sar-oil-spill benchmark`. Never
  copy figures from papers or estimate them. If you did not measure it, do not
  claim it.
- `pytest` and `ruff check .` must be green before you commit.
- If you change any algorithm, re-run `python scripts/generate_docs_images.py`
  and update the results table in README.md to match.
- PyTorch is optional. The package must import and the API must serve with only
  the base dependencies installed.
- Do not edit `matlab/` — it is the historical record. Record any divergence in
  `docs/MATLAB.md` instead.
- Report honestly. If something is untested or you could not verify it, say so
  plainly rather than implying it works.

## Your task

<REPLACE THIS WITH ONE OF THE TASKS BELOW, OR YOUR OWN>

Work on the branch `claude/<short-description>`. Commit with a descriptive
message and push. Do not open a pull request unless I ask.
````

---

## Task blocks

Paste one of these into the `## Your task` slot.

### Load real SAR bit depths (#18)

````text
Issue #18. Every image loader assumes 8-bit greyscale. Measured: a float32
sigma-nought TIFF fails to load on the CLI and API and becomes an all-zero
image in Streamlit; a uint16 scene with GRD-like values (0-600) collapses to
three grey levels on the CLI and API and is clipped at 255 in Streamlit. The
issue lists the five call sites.

Add one shared loader that reads native bit depth and scales into the
pipeline's 0-255 working range by a documented, configurable rule (dB window
for sigma-nought, percentile stretch for integer DN). Route the CLI, dataset,
processor, API and Streamlit through it. The API's header-only pixel check
(`_check_dimensions`) must still run before any decode.

Prove it with tests: the same synthetic scene saved as uint8, uint16 and
float32 must give the same loaded array on every surface and detect within
0.02 IoU of the uint8 result.
````

### Make batch status work across workers (#17)

````text
Issue #17. Batch jobs live in a per-process dict, and the production image
runs two gunicorn workers, so /api/v1/batch-status 404s on about half of polls
(measured 56 of 100) even in one container. TestClient is single-process, so
the test suite cannot see it.

First write a test or CI step that starts the shipped server configuration and
polls a batch repeatedly; confirm it fails. Then fix it -- either one worker
(measure throughput before and after) or a shared job store. Changing the
worker model is an infrastructure decision: present the measurements and ask
before committing to one.
````

### Validate against real SAR data (#13)

````text
Every accuracy figure in this repository is measured on synthetic scenes. That
is the single biggest gap in the project's credibility.

Add a reproducible evaluation path against real Sentinel-1 imagery:

  1. A downloader for a small public, redistributable set of Sentinel-1 GRD
     scenes with oil spill annotations. Do NOT commit imagery to the repository.
  2. A loader that handles real GRD products, building on #18 — calibration
     to sigma-nought, multi-look, and the fact that real scenes are far larger
     than 512x512 (tile them).
  3. A `sar-oil-spill benchmark --real` path that scores the existing four
     methods on that data.
  4. A results table in the README, clearly separated from the synthetic one.

I expect the real numbers to be much worse than the synthetic ones. Report them
exactly as measured. A large drop is a finding, not a failure — the interesting
output is which methods degrade least and why.
````

### Discriminate look-alikes (#14)

````text
The methods here detect dark, smooth regions. Low-wind cells, biogenic films,
rain cells and wave shadows are also dark and smooth, so they all produce false
alarms on real imagery. This is the main reason radiometric thresholding alone
is not operationally usable.

Add a false-alarm reduction stage:

  1. Extract per-region features known to separate oil from look-alikes:
     shape complexity, edge gradient sharpness, contrast to local background,
     texture statistics inside versus outside the region.
  2. Train a lightweight classifier over those features to score each candidate
     region as oil or look-alike.
  3. Wire it into `OilSpillDetector` as an optional post-processing stage,
     defaulting to off until it is validated.
  4. Extend `synthetic.py` to generate look-alike scenes (low-wind cells are
     dark and smooth but have soft, diffuse edges), so there is something to
     train and test against.

Report the precision/recall trade-off the classifier actually buys. If it does
not help, say so and leave it disabled.
````

### Train the U-Net (#16)

````text
`models/deep_learning_segmentation.py` defines an attention-gated U-Net but
ships no trained weights, so it is architecture only and is excluded from the
benchmark.

Add `scripts/train.py`:

  - Dice + BCE loss (oil masks are heavily class-imbalanced, so plain BCE
    collapses to predicting all-sea).
  - Train/validation split, early stopping, checkpointing.
  - Trains on synthetic scenes out of the box, with a flag to point at a real
    dataset.
  - Deterministic given a seed.

Then evaluate it against the four traditional methods on the same benchmark and
add it to the README table. Do not commit checkpoint files.

Be honest in the write-up about what training on synthetic data proves: it
shows the architecture and training loop work, not that the model generalises.
````

### Add georeferencing (#15)

````text
Input is currently treated as a plain raster, so detections are reported in
pixels. For any operational use, responders need square kilometres and
coordinates.

Add optional GeoTIFF support behind a `geo` extra (rasterio):

  - Read geotransform and CRS when present.
  - Report affected area in m^2 and km^2 alongside pixel counts.
  - Emit detections as GeoJSON polygons in EPSG:4326.
  - Expose it through the CLI and the API.

Keep it optional and gracefully degrade: a plain PNG must still work exactly as
it does today, and the package must import without rasterio installed.
````
