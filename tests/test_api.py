"""Tests for the FastAPI service."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from sar_oil_spill.data import generate_sar_scene
from sar_oil_spill.models import METHOD_NAMES


@pytest.fixture(scope="module")
def client():
    """A test client with the lifespan run, so the detector is initialised."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sar_png() -> bytes:
    """A synthetic SAR scene encoded as PNG bytes."""
    scene = generate_sar_scene(size=(256, 256), n_slicks=1, seed=7)
    return cv2.imencode(".png", scene.image.astype(np.uint8))[1].tobytes()


@pytest.fixture(scope="module")
def truth_png() -> bytes:
    scene = generate_sar_scene(size=(256, 256), n_slicks=1, seed=7)
    return cv2.imencode(".png", scene.oil_mask.astype(np.uint8) * 255)[1].tobytes()


def upload(data: bytes, name: str = "scene.png"):
    return (name, io.BytesIO(data), "image/png")


class TestMetaEndpoints:
    def test_root_lists_endpoints(self, client):
        body = client.get("/").json()
        assert body["service"] == "SAR Oil Spill Detection API"
        assert "detect" in body["endpoints"]

    def test_health_reports_ready(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["detector_ready"] is True

    def test_methods_endpoint_lists_all_methods(self, client):
        body = client.get("/api/v1/methods").json()
        assert "adaptive_threshold" in body["available_methods"]
        assert body["default_method"] == "adaptive_threshold"

    def test_openapi_schema_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_openapi_advertises_methods_as_an_enum(self, client):
        """Issue #7: the method parameter is a closed set in the schema, driven
        by the same METHOD_NAMES registry as the CLI -- not a free string."""
        schemas = client.get("/openapi.json").json()["components"]["schemas"]

        enums = [schema["enum"] for schema in schemas.values() if "enum" in schema]
        assert sorted(METHOD_NAMES) in [sorted(values) for values in enums]


class TestDetectEndpoint:
    def test_detects_oil_in_a_synthetic_scene(self, client, sar_png):
        response = client.post("/api/v1/detect", files={"image_file": upload(sar_png)})

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["results"]["oil_spill_detected"] is True
        assert body["results"]["affected_area_pixels"] > 0

    def test_returned_mask_decodes_to_an_image(self, client, sar_png):
        body = client.post("/api/v1/detect", files={"image_file": upload(sar_png)}).json()

        raw = base64.b64decode(body["results"]["detection_mask_png_base64"])
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

        assert mask is not None
        assert set(np.unique(mask)) <= {0, 255}

    @pytest.mark.parametrize(
        "method",
        [
            "adaptive_threshold",
            "kmeans_clustering",
            "superpixel_clustering",
            "fuzzy_edge_detection",
        ],
    )
    def test_every_method_is_accepted(self, client, sar_png, method):
        response = client.post(
            "/api/v1/detect", files={"image_file": upload(sar_png)}, data={"method": method}
        )

        assert response.status_code == 200
        assert response.json()["results"]["method"] == method

    def test_unknown_method_is_rejected_with_the_permitted_values(self, client, sar_png):
        """Issue #7: invalid parameters fail validation at the boundary (422),
        and the message tells the caller what *would* have been accepted."""
        response = client.post(
            "/api/v1/detect", files={"image_file": upload(sar_png)}, data={"method": "telepathy"}
        )

        assert response.status_code == 422
        message = str(response.json()["detail"])
        assert all(name in message for name in METHOD_NAMES)

    def test_unsupported_extension_is_rejected(self, client, sar_png):
        response = client.post(
            "/api/v1/detect", files={"image_file": upload(sar_png, "notes.txt")}
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_empty_upload_is_rejected(self, client):
        response = client.post("/api/v1/detect", files={"image_file": upload(b"")})
        assert response.status_code == 400

    def test_oversized_file_is_rejected(self, client, sar_png, monkeypatch):
        import api.main

        monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", len(sar_png) - 1)

        response = client.post("/api/v1/detect", files={"image_file": upload(sar_png)})

        assert response.status_code == 413
        assert "MB limit" in response.json()["detail"]

    def test_undecodable_upload_is_rejected(self, client):
        response = client.post("/api/v1/detect", files={"image_file": upload(b"not an image")})

        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    def test_missing_file_is_a_validation_error(self, client):
        assert client.post("/api/v1/detect").status_code == 422


def uniform_png(height: int, width: int) -> bytes:
    """A blank PNG. Compresses ~1000:1, so huge images are tiny on the wire."""
    return cv2.imencode(
        ".png", np.zeros((height, width), np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )[1].tobytes()


class TestDecompressionBombs:
    """The 25 MB upload cap bounds compressed bytes, not pixels. Measured: a
    400-megapixel blank PNG is 389 kB, and the pipeline needs ~29 MB per
    megapixel because it despeckles before resizing -- ~11.6 GB from one
    request. Dimensions must be checked from the header, before decoding."""

    def test_oversized_image_is_rejected_before_decoding(self, client):
        bomb = uniform_png(6_000, 6_000)  # 36 MP, ~35 kB on the wire
        assert len(bomb) < 100_000

        response = client.post("/api/v1/detect", files={"image_file": upload(bomb)})

        assert response.status_code == 413
        assert "megapixel" in response.json()["detail"]

    def test_limit_is_inclusive_and_exact(self, client, monkeypatch):
        """An image of exactly MAX_IMAGE_PIXELS is accepted; one more row is not."""
        import api.main

        monkeypatch.setattr(api.main, "MAX_IMAGE_PIXELS", 64 * 64)

        at_limit = client.post("/api/v1/detect", files={"image_file": upload(uniform_png(64, 64))})
        over = client.post("/api/v1/detect", files={"image_file": upload(uniform_png(65, 64))})

        assert at_limit.status_code == 200
        assert over.status_code == 413

    def test_batch_rejects_a_bomb_before_queueing(self, client, sar_png):
        """Otherwise the background task would decode it after the 200."""
        response = client.post(
            "/api/v1/batch-process",
            files=[
                ("images", upload(sar_png, "fine.png")),
                ("images", upload(uniform_png(6_000, 6_000), "bomb.png")),
            ],
        )
        assert response.status_code == 413

    def test_ground_truth_uploads_are_checked_too(self, client, sar_png):
        response = client.post(
            "/api/v1/evaluate",
            files={
                "image_file": upload(sar_png),
                "ground_truth_file": upload(uniform_png(6_000, 6_000), "truth.png"),
            },
        )
        assert response.status_code == 413

    def test_formats_the_header_check_cannot_read_are_refused(self, client):
        """Fail closed. OpenCV sniffs content, not the filename, and decodes
        formats Pillow cannot read -- Radiance HDR among them -- so an image
        whose size the header check cannot see must not reach OpenCV."""
        hdr = cv2.imencode(".hdr", np.full((8, 8, 3), 0.5, np.float32))[1].tobytes()
        assert cv2.imdecode(np.frombuffer(hdr, np.uint8), cv2.IMREAD_GRAYSCALE) is not None

        response = client.post("/api/v1/detect", files={"image_file": upload(hdr)})

        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()


class TestEvaluateEndpoint:
    def test_scores_a_prediction_against_ground_truth(self, client, sar_png, truth_png):
        response = client.post(
            "/api/v1/evaluate",
            files={
                "image_file": upload(sar_png),
                "ground_truth_file": upload(truth_png, "truth.png"),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["evaluation_metrics"]["jaccard_index"] > 0.6
        assert body["detailed_report"]["quality"] in {"excellent", "good", "fair", "poor"}

    def test_requires_both_files(self, client, sar_png):
        response = client.post("/api/v1/evaluate", files={"image_file": upload(sar_png)})
        assert response.status_code == 422

    def test_unknown_method_is_rejected(self, client, sar_png, truth_png):
        response = client.post(
            "/api/v1/evaluate",
            files={
                "image_file": upload(sar_png),
                "ground_truth_file": upload(truth_png, "truth.png"),
            },
            data={"method": "telepathy"},
        )
        assert response.status_code == 422


class TestBatchEndpoints:
    def test_batch_runs_to_completion(self, client, sar_png):
        response = client.post(
            "/api/v1/batch-process",
            files=[("images", upload(sar_png, f"scene_{i}.png")) for i in range(3)],
        )

        assert response.status_code == 200
        batch_id = response.json()["batch_id"]

        status = client.get(f"/api/v1/batch-status/{batch_id}").json()
        assert status["total_images"] == 3
        assert status["status"] in {"processing", "completed"}

    def test_batch_rejects_oversized_request(self, client, sar_png):
        response = client.post(
            "/api/v1/batch-process",
            files=[("images", upload(sar_png, f"s_{i}.png")) for i in range(11)],
        )

        assert response.status_code == 400
        assert "At most 10" in response.json()["detail"]

    def test_batch_rejects_unknown_method_before_queueing(self, client, sar_png):
        """Validation must happen up front, not inside the background task
        where the caller would only discover it by polling."""
        response = client.post(
            "/api/v1/batch-process",
            files=[("images", upload(sar_png))],
            data={"method": "telepathy"},
        )
        assert response.status_code == 422

    def test_unknown_batch_id_returns_404(self, client):
        assert client.get("/api/v1/batch-status/does-not-exist").status_code == 404
