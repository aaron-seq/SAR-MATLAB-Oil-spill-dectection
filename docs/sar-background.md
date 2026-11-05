# Background: SAR and oil spill sensing

Oil can damp the short surface waves that return radar energy, producing a dark
patch in SAR imagery. SAR can image at night and through cloud, but slick
visibility depends on wind and sea conditions. Low winds can hide the contrast;
high winds can also reduce detection. Biological films and other natural
phenomena can produce similar dark regions. A dark patch is therefore a
candidate, not confirmation of oil
([EMSA explanation](https://www.emsa.europa.eu/earth-observation-products/how-does-sar-detection-work.html),
checked 2026-11-04).

Operational assessment needs more context. EMSA's CleanSeaNet operators combine
imagery with meteorological, oceanographic, and vessel information when assessing
possible pollution. ESA SNAP's oil-spill operator assumes calibration and speckle
filtering before dark-spot detection. These are useful references for the gaps
between a plain-raster experiment and a monitoring workflow
([CleanSeaNet](https://www.emsa.europa.eu/csn-menu.html),
[SNAP 9 operator](https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-toolboxes/org.esa.s1tbx.s1tbx.op.feature.extraction.ui/operators/OilSpillDetectionOp.html),
checked 2026-11-04).

This repository compares dark-region segmentation on synthetic scenes. It has
no real-data validation, look-alike classifier, or georeferencing, and does not
estimate oil thickness or concentration. Its confidence score measures image
contrast, not the probability of oil. See the [README](../README.md) for measured
synthetic results and limitations, [product plan](product/PRODUCT.md) for evidence
gates, and [MATLAB guide](MATLAB.md) for the historical implementation.
