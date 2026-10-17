"""End-to-end tests for the Streamlit front-end (`app.py`).

Uses Streamlit's own headless harness, ``streamlit.testing.v1.AppTest``, which
executes the real script -- widgets, caching and all -- without a browser.
Skipped when the optional ``app`` extra is not installed; CI runs it in the
optional-extras job.

Until these existed, app.py had never actually been executed, only compiled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from sar_oil_spill.models import METHOD_NAMES

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
# The first run builds the detector and a 512x512 scene; leave generous room
# on slow CI runners rather than flake.
TIMEOUT_SECONDS = 120


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT_SECONDS).run()


def _metrics(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


class TestSyntheticDemo:
    def test_default_scene_runs_without_errors(self, app):
        assert not app.exception
        assert not app.error

    def test_reports_a_detection(self, app):
        metrics = _metrics(app)

        assert metrics["Oil detected"] == "Yes"
        assert metrics["Affected area"].endswith("px")

    def test_scores_against_the_synthetic_ground_truth(self, app):
        """The demo scene carries its own truth mask, so accuracy is shown."""
        metrics = _metrics(app)

        assert {"IoU", "Dice", "Precision", "Recall", "Boundary F1"} <= metrics.keys()
        assert float(metrics["IoU"]) > 0.6

    @pytest.mark.parametrize("method", METHOD_NAMES)
    def test_every_method_in_the_sidebar_runs(self, app, method):
        """The selectbox is driven by METHOD_NAMES, so each entry must work."""
        app.sidebar.selectbox[0].select(method).run()

        assert not app.exception
        assert _metrics(app)["Oil detected"] in {"Yes", "No"}

    def test_land_masking_with_a_coastline_runs(self, app):
        coastline = next(box for box in app.checkbox if box.label == "Include coastline")
        coastline.check().run()
        exclude_land = next(box for box in app.sidebar.checkbox if box.label == "Exclude land")
        exclude_land.check().run()

        assert not app.exception


class TestUploadMode:
    def test_prompts_for_an_image_instead_of_failing(self, app):
        """With nothing uploaded the app must explain itself, not crash or
        show stale results from the demo scene."""
        app.sidebar.radio[0].set_value("Upload your own").run()

        assert not app.exception
        assert any("Upload a SAR image" in info.value for info in app.info)
        assert not app.metric


class TestStreamlitApiUsage:
    def test_does_not_use_the_removed_use_container_width_argument(self):
        """Streamlit deprecated `use_container_width` with removal scheduled
        after 2025-12-31; any release may now drop it and break the app.
        `width="stretch"` replaces it (available from Streamlit 1.49, which is
        why the `app` extra pins that floor)."""
        assert "use_container_width" not in APP_PATH.read_text(encoding="utf-8")
