# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Oil spill detection and segmentation in Synthetic Aperture Radar imagery. A
Python port of an original MATLAB research project, which is preserved unchanged
in `matlab/`.

The whole system rests on one physical fact: **oil damps the capillary waves
that scatter radar energy back to the satellite, so a slick is darker and
smoother than the sea around it.** All four detection methods are different ways
of deciding where "dark and smooth" begins. A change that breaks that assumption
is wrong no matter what the metrics say.

## Commands

```bash
pip install -e '.[api,dev]'          # set up

pytest                               # 165 tests, ~9 s
pytest tests/test_segmentation.py -v # one file
ruff check .                         # must be clean
mypy src/sar_oil_spill               # advisory, not enforced in CI

sar-oil-spill demo                   # end-to-end on a synthetic scene
sar-oil-spill benchmark --samples 20 # score every method
uvicorn api.main:app --reload        # API on :8000, docs at /api/docs
streamlit run app.py                 # browser UI

python scripts/generate_docs_images.py   # regenerate README figures + numbers
```

## Architecture

```
src/sar_oil_spill/
├── config.py          Frozen dataclasses loaded from config/model_config.yaml.
│                      Unknown YAML keys warn and are ignored, never crash.
├── cli.py             `sar-oil-spill`: demo, detect, benchmark, dataset.
├── core/
│   ├── sar_image_processor.py   Despeckling, contrast, morphology, resizing.
│   └── oil_spill_detector.py    Orchestrates preprocess → segment → evaluate.
├── models/
│   ├── traditional_segmentation.py   The four methods. METHOD_NAMES is the
│   │                                 single registry — CLI, API and figures
│   │                                 all read from it.
│   └── deep_learning_segmentation.py Optional PyTorch U-Net, no trained weights.
├── data/
│   ├── dataset.py     Lazy on-disk dataset indexing.
│   └── synthetic.py   Generates SAR scenes with known ground truth.
└── utils/
    ├── performance_evaluator.py   IoU, Dice, boundary F1, Hausdorff, object rates.
    └── data_visualizer.py         All figures. Agg backend, headless-safe.
```

`OilSpillDetector` is the single entry point used by the CLI, the API and the
Streamlit app, so identical inputs give identical results everywhere. New
surfaces should call it rather than reimplementing the pipeline.

## Things that look like bugs but are not

Each of these is deliberate, explained in a docstring, and pinned by a test.
Do not change one without new measurements.

**Contrast enhancement is disabled by default** (`config.py`). CLAHE normalises
contrast within local tiles; a slick wider than a tile has its interior
brightened to match the sea, erasing the signal. Enabling it drops mean IoU from
0.85 to 0.32. Enhancement is for display, not detection.

**The adaptive threshold's background window is 251 px**, much wider than any
plausible slick. With a narrow window the background estimate sinks with the
slick and the contrast cancels out — that was the original bug, at recall 0.009.

**Fuzzy edge detection thresholds uniformity, not edges.** Slicks are smooth and
the sea is speckled, so the smooth class is the candidate class. This matches
the MATLAB original's `imbinarize(Ieval, 0.8)`, which keeps the white (uniform)
class. Thresholding edges instead scored IoU 0.004.

**`remove_small_components` uses scipy, not skimage.** scikit-image 0.26 renamed
`remove_small_objects(min_size=)` to `max_size=` *and* changed the comparison
from strict to inclusive. Counting directly keeps one behaviour across all
supported versions.

**Despeckling filters use `cv2.blur`, not loops.** Local mean and variance come
from separable box filters, so cost is independent of window size. An earlier
version looped over every pixel — 262,144 iterations per 512×512 scene.

## Conventions

**Measure every claim.** Any accuracy or timing number in the docs must come
from `scripts/generate_docs_images.py` or `sar-oil-spill benchmark`. The README
table is measured output. Never copy numbers from papers, and never estimate.

**Benchmarks are synthetic.** They compare methods fairly and catch regressions.
They do *not* predict accuracy on real Sentinel-1 imagery, where look-alikes
(low-wind cells, biogenic films, rain cells) are genuinely dark and smooth. Say
so whenever quoting them.

**Keep PyTorch optional.** The package must import and the API must serve with
only base dependencies. Deep-learning code goes behind the `TORCH_AVAILABLE`
guard. CI has a job that verifies this.

**Never edit `matlab/`.** It is the historical record. Document divergences in
`docs/MATLAB.md`.

**Tests are named for behaviour**, not for the function they call —
`test_slicks_are_darker_than_the_sea`, not `test_generate_sar_scene`.

**No datasets or checkpoints in git.** `.gitignore` covers `*.tif`, `*.pt`,
`*.pth`, `data/`, `results/`.

## Adding a segmentation method

1. Implement it in `models/traditional_segmentation.py`, returning a
   `SegmentationResult` with `stages` populated for visualisation.
2. Add its name to `METHOD_NAMES`. The CLI, API and comparison figures pick it
   up automatically — nothing else needs wiring.
3. Add settings to `config.py` and `config/model_config.yaml` if tunable.
4. The parametrised tests in `tests/test_segmentation.py` will find it and
   require IoU > 0.6 on synthetic scenes.
5. Re-run `scripts/generate_docs_images.py` and update the README table.

## Known gaps

- No validation against real SAR imagery — all numbers are synthetic.
- No look-alike discrimination, which is the main source of false alarms in
  practice.
- No georeferencing; areas are reported in pixels, not m².
- The U-Net has no trained weights.
- API batch jobs are in-process, so they are not safe across replicas.
- The Docker build has never run in this environment (no daemon available); CI
  is the first real exercise of it.

`docs/dispatch-prompt.md` has ready-made task blocks for several of these.
