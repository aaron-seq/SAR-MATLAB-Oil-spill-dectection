# The original MATLAB implementation

The `matlab/` directory holds the original research code, preserved as written.
The Python package in `src/sar_oil_spill/` is a port of it — this document maps
one onto the other and records where the port deliberately diverges.

The MATLAB code is kept for reference and reproducibility. **New work should go
into the Python package**, which is tested, benchmarked and runnable without a
MATLAB licence.

## Requirements

- MATLAB R2020b or newer
- Image Processing Toolbox — `imfill`, `bwareafilt`, `regionprops`,
  `adaptthresh`, `superpixels`, `imoverlay`, `wiener2`
- Statistics and Machine Learning Toolbox — `kmeans`, `pdist2`
- Fuzzy Logic Toolbox — `mamfis`, `addMF`, `evalfis` (only `fuzzy_edgeDetect.m`)

## Running it

```matlab
cd matlab
Main            % opens a directory picker, then a text menu
```

`Main.m` prompts for the dataset root and expects this layout:

```
dataset_root/train/
├── images/             % open-sea scenes (.jpg)
├── labels/             % ground truth (.png)
├── images_with_land/   % scenes containing a coastline
└── labels_with_land/
```

The dataset is not distributable and is not included here. To try the
algorithms without it, use the Python package, which generates synthetic SAR
scenes with known ground truth:

```bash
sar-oil-spill demo --method kmeans_clustering
```

## File map

### Entry points

| File | Purpose |
|---|---|
| `Main.m` | Interactive menu: pick scene type, image and algorithm |
| `ElicioDom__Progetto_Oil_Spill_Detection.m` | The original single-file version, superseded by `Main.m` |

### Segmentation

| File | Python equivalent |
|---|---|
| `local_threshold.m` | `TraditionalSegmentation.adaptive_threshold_segmentation` |
| `automatic_threshold.m` | same (the two MATLAB variants were merged) |
| `manual_thresholding.m` | *not ported* — needs interactive `impixel` pixel picking |
| `kmeansSegment.m` | `TraditionalSegmentation.kmeans_segmentation` |
| `superpixel.m` | `TraditionalSegmentation.superpixel_segmentation` |
| `fuzzy_edgeDetect.m` | `TraditionalSegmentation.fuzzy_edge_segmentation` |
| `land_mask.m` | `TraditionalSegmentation.detect_land` |
| `automatic_threshold_for_land.m` | folded into `detect` with `mask_land=True` |
| `kmeansSegment_for_land.m` | folded into `detect` with `mask_land=True` |

### Support

| File | Python equivalent |
|---|---|
| `lee_filter.m` | `SARImageProcessor._lee_filter` |
| `segmentation_evaluation.m` | `PerformanceEvaluator` |
| `visualizeImages.m` | `DataVisualizer.plot_detection_result` |
| `visualizeImages_for_land.m` | `DataVisualizer.plot_pipeline_stages` |

## Where the port diverges, and why

The port is not line-for-line. Four changes were made because the original
behaviour was measurably wrong or impractical; each is covered by a test.

### 1. Local thresholding replaced by background-normalised Otsu

`local_threshold.m` calls `adaptthresh(..., 'NeighborhoodSize', 41)`. A
41-pixel window is far smaller than a typical slick, so inside a large slick the
local threshold sinks with the slick and the contrast cancels out — only the rim
is detected. The direct Python translation scored **recall 0.009**.

The port estimates the background over a 251-pixel window and thresholds the
*ratio* image with Otsu. Under SAR's multiplicative noise model that ratio is
the physically meaningful quantity: near 1.0 over sea, 0.3–0.5 over oil,
independent of absolute brightness. Recall went from 0.009 to **0.971**.

### 2. Histogram equalisation dropped from the detection path

`automatic_threshold.m` and `manual_thresholding.m` both call `histeq` before
thresholding. Contrast normalisation removes exactly the slick-versus-sea
intensity gap that detection depends on. With CLAHE enabled, mean IoU across
the benchmark falls from **0.85 to 0.32**. Enhancement is available in the port
but off by default, and is intended for display.

### 3. Fuzzy inference evaluated in closed form

`fuzzy_edgeDetect.m` builds a Mamdani FIS and calls `evalfis` row by row over
the image. With two rules and Gaussian input memberships, the inference reduces
analytically to

```
uniformity(x, y) = exp(-Gx² / 2σ²) · exp(-Gy² / 2σ²)
```

so the port computes that expression directly. Same result, no inference engine,
and it runs in ~150 ms instead of minutes.

The port also keeps the MATLAB original's key insight, which is easy to
misread: `imbinarize(Ieval, 0.8)` keeps the **white** (uniform) class, not the
edges. That is physically right — oil damps the capillary waves that give the
sea its speckle texture, so a slick is *smoother* than the water around it. The
port adds a darkness check on top, since calm water is smooth too.

### 4. Lee filter vectorised

`lee_filter.m` is already vectorised in MATLAB via `imfilter`. An intermediate
Python version reintroduced a per-pixel loop — 262,144 iterations for a 512×512
scene. The port computes local statistics with separable box filters, so cost is
independent of window size.

Note that `lee_filter.m` computes `sigmas = sqrt((I-means).^2/window_size^2)`,
which is not the local standard deviation (it divides the squared deviation of a
single pixel by the window area). The port uses the proper local variance,
`E[x²] − E[x]²`.

## Not ported

- **`manual_thresholding.m`** — depends on `impixel`, which requires a human
  clicking pixels in a figure window. There is no batch equivalent, and the
  automatic methods outperform it.
- **The blob standard-deviation feature analysis** — commented out in the
  original and never enabled.
- **`bfscore`, `jaccard`, `dice`** — MATLAB built-ins; reimplemented in
  `PerformanceEvaluator` with the same definitions. Boundary F1 uses a distance
  transform with a configurable tolerance, matching `bfscore`'s semantics.
