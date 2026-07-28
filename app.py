"""Streamlit front-end for SAR oil spill detection.

Run with::

    streamlit run app.py

Uses the same :class:`~sar_oil_spill.core.OilSpillDetector` as the CLI and the
API, so a scene analysed here produces exactly the results it would anywhere
else. No trained model or dataset is required -- the built-in synthetic
generator provides a scene to try immediately.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from sar_oil_spill import __version__
from sar_oil_spill.config import load_settings
from sar_oil_spill.core import OilSpillDetector
from sar_oil_spill.data import generate_sar_scene
from sar_oil_spill.models.traditional_segmentation import METHOD_NAMES
from sar_oil_spill.utils import DataVisualizer

st.set_page_config(
    page_title="SAR Oil Spill Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_detector() -> OilSpillDetector:
    """Build the detector once and reuse it across reruns."""
    return OilSpillDetector(load_settings())


@st.cache_data(show_spinner=False)
def make_synthetic_scene(seed: int, with_land: bool, size: int):
    """Generate a reproducible demo scene; cached so the slider feels instant."""
    scene = generate_sar_scene(size=(size, size), with_land=with_land, seed=seed)
    return scene.image, scene.oil_mask


def to_display(array: np.ndarray) -> np.ndarray:
    """Rescale any array to ``uint8`` for display."""
    values = np.asarray(array, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return (((values - low) / (high - low)) * 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint detected pixels cyan over the greyscale scene."""
    base = np.stack([to_display(image)] * 3, axis=-1).astype(np.float32)
    tint = np.array([0, 194, 209], dtype=np.float32)
    selected = mask.astype(bool)
    base[selected] = 0.45 * base[selected] + 0.55 * tint
    return base.astype(np.uint8)


# ------------------------------------------------------------------ sidebar

st.sidebar.title("🛰️ SAR Oil Spill Detection")
st.sidebar.caption(f"version {__version__}")

source = st.sidebar.radio("Image source", ["Synthetic demo scene", "Upload your own"])

method = st.sidebar.selectbox(
    "Segmentation method",
    METHOD_NAMES,
    format_func=lambda name: name.replace("_", " ").title(),
)
mask_land = st.sidebar.checkbox("Exclude land", value=False)
min_area = st.sidebar.slider("Minimum detection size (px)", 0, 2000, 100, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "Oil dampens the short capillary waves that scatter radar energy back to "
    "the satellite, so a slick reflects less and reads as a dark, smooth patch "
    "against the brighter, speckled sea."
)

# --------------------------------------------------------------- main panel

st.title("Oil Spill Detection in SAR Imagery")

image: np.ndarray | None = None
ground_truth: np.ndarray | None = None

if source == "Synthetic demo scene":
    columns = st.columns(3)
    seed = columns[0].number_input("Scene seed", min_value=0, max_value=9999, value=42)
    size = columns[1].select_slider("Scene size", options=[256, 512, 768], value=512)
    with_land = columns[2].checkbox("Include coastline", value=False)
    image, ground_truth = make_synthetic_scene(int(seed), bool(with_land), int(size))
else:
    uploaded = st.file_uploader("SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    truth_upload = st.file_uploader(
        "Ground truth mask (optional)", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
    if uploaded is not None:
        image = np.array(Image.open(uploaded).convert("L"), dtype=np.float32)
    if truth_upload is not None:
        ground_truth = np.array(Image.open(truth_upload).convert("L")) > 127

if image is None:
    st.info("Upload a SAR image, or switch to the synthetic demo scene to try it out.")
    st.stop()

detector = get_detector()
with st.spinner(f"Running {method.replace('_', ' ')}..."):
    result = detector.detect(
        image,
        method=method,
        ground_truth=ground_truth,
        mask_land=mask_land,
        min_area_pixels=int(min_area),
    )

# ------------------------------------------------------------------ results

metric_columns = st.columns(4)
metric_columns[0].metric("Oil detected", "Yes" if result.oil_detected else "No")
metric_columns[1].metric("Affected area", f"{result.affected_area_pixels:,} px")
metric_columns[2].metric("Scene coverage", f"{result.coverage_fraction * 100:.2f}%")
metric_columns[3].metric("Processing time", f"{result.processing_time * 1000:.0f} ms")

left, right = st.columns(2)
left.image(
    to_display(result.preprocessed_image),
    caption="Preprocessed SAR image",
    use_container_width=True,
)
right.image(
    overlay_mask(result.preprocessed_image, result.mask),
    caption=f"Detection — {method.replace('_', ' ')}",
    use_container_width=True,
)

if result.metrics is not None:
    st.subheader("Accuracy against ground truth")
    m = result.metrics
    score_columns = st.columns(5)
    score_columns[0].metric("IoU", f"{m.jaccard_index:.3f}")
    score_columns[1].metric("Dice", f"{m.dice_coefficient:.3f}")
    score_columns[2].metric("Precision", f"{m.precision:.3f}")
    score_columns[3].metric("Recall", f"{m.recall:.3f}")
    score_columns[4].metric("Boundary F1", f"{m.boundary_f1:.3f}")

    st.pyplot(
        DataVisualizer(style="default").plot_detection_result(
            result.preprocessed_image,
            result.mask,
            ground_truth=detector._align(ground_truth, result.mask.shape),
            title="",
        )
    )

with st.expander("Processing pipeline stages"):
    stage_names = list(result.stages)
    for row_start in range(0, len(stage_names), 3):
        for column, name in zip(
            st.columns(3), stage_names[row_start : row_start + 3], strict=False
        ):
            column.image(
                to_display(result.stages[name]),
                caption=name.capitalize(),
                use_container_width=True,
            )

with st.expander("Processing history"):
    for step in result.processing_history:
        st.text(f"• {step}")
