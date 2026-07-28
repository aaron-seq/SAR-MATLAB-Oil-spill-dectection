# Contributing

## Setup

```bash
git clone https://github.com/aaronseq12/SAR-MATLAB-Oil-spill-dectection.git
cd SAR-MATLAB-Oil-spill-dectection

python -m venv .venv && source .venv/bin/activate
pip install -e '.[api,dev]'

pytest && ruff check .
```

## Before opening a pull request

```bash
pytest                          # all tests must pass
ruff check .                    # no lint errors
mypy src/sar_oil_spill          # advisory, not enforced in CI
```

If you changed anything that affects detection output, also run:

```bash
python scripts/generate_docs_images.py
```

This regenerates every figure in the README plus `docs/benchmark-results.json`,
and prints the results table. **Update the table in the README to match** —
the numbers there are measured, not aspirational, and a stale table is worse
than none.

## House rules

**Measure claims.** Any accuracy or performance number in the docs must come
from `scripts/generate_docs_images.py` or `sar-oil-spill benchmark`. Do not
copy figures from papers and present them as this project's results.

**Explain non-obvious choices.** Several defaults here look wrong until you know
why — contrast enhancement is disabled, the background window is 251 px, the
fuzzy method thresholds uniformity rather than edges. Each is explained in a
comment or docstring, and each has a test pinning it. If you change one, update
both.

**Tests describe behaviour.** Name them for what they assert
(`test_slicks_are_darker_than_the_sea`), not for the function they call. Prefer
one clear assertion over five incidental ones.

**Keep PyTorch optional.** The package must import and the API must serve with
only the base dependencies installed. Deep-learning code belongs behind the
`TORCH_AVAILABLE` guard in `models/deep_learning_segmentation.py`.

**Do not edit `matlab/`.** It is the historical record. Document divergences in
`docs/MATLAB.md` instead.

## Adding a segmentation method

1. Add the method to `models/traditional_segmentation.py`, returning a
   `SegmentationResult` with populated `stages`.
2. Add its name to `METHOD_NAMES` — the CLI, API and comparison figures all
   read from that tuple, so nothing else needs wiring up.
3. Add settings to `config.py` and `config/model_config.yaml` if it is tunable.
4. The parametrised tests in `tests/test_segmentation.py` will pick it up
   automatically and require IoU > 0.6 on synthetic scenes.
5. Re-run `scripts/generate_docs_images.py` and update the README table.

## Reporting bugs

Include the command you ran, the full traceback, and `pip list` output. If it
concerns detection quality, attach the scene or the `--seed` that reproduces it.
