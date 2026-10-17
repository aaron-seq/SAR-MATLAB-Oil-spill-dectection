# Original MATLAB implementation

The original research code, preserved as written. It is kept for reference and
reproducibility.

**New work belongs in the Python package** (`src/sar_oil_spill/`), which is
tested, benchmarked, and runs without a MATLAB licence. Please do not modify the
files here — record divergences in [`../docs/MATLAB.md`](../docs/MATLAB.md)
instead.

## Running it

```matlab
cd matlab
Main            % directory picker, then a text menu
```

Requires MATLAB R2020b+ with the Image Processing, Statistics and Machine
Learning, and Fuzzy Logic toolboxes. It also needs the original dataset, which
is not distributable and is not included.

To try the same algorithms without MATLAB or the dataset:

```bash
sar-oil-spill demo --method kmeans_clustering
```

## What is here

Entry points are `Main.m` (interactive menu) and
`ElicioDom__Progetto_Oil_Spill_Detection.m` (the earlier single-file version).
Segmentation lives in `local_threshold.m`, `automatic_threshold.m`,
`manual_thresholding.m`, `kmeansSegment.m`, `superpixel.m` and
`fuzzy_edgeDetect.m`, with land-aware variants alongside. `lee_filter.m`,
`segmentation_evaluation.m` and the `visualizeImages*.m` files are support code.

[`../docs/MATLAB.md`](../docs/MATLAB.md) maps every file to its Python
equivalent and explains the four places where the port deliberately behaves
differently — including two that fix measurable bugs in the original.
