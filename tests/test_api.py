"""Tests for the FastAPI service."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import pytest
from api.main import app
from fastapi.testclient import TestClient

from sar_oil_spill.data import generate_sar_scene


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

    def test_unknown_method_is_rejected(self, client, sar_png):
        response = client.post(
            "/api/v1/detect", files={"image_file": upload(sar_png)}, data={"method": "telepathy"}
        )

        assert response.status_code == 400
        assert "Unknown method" in response.json()["detail"]

    def test_unsupported_extension_is_rejected(self, client, sar_png):
        response = client.post(
            "/api/v1/detect", files={"image_file": upload(sar_png, "notes.txt")}
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_empty_upload_is_rejected(self, client):
        response = client.post("/api/v1/detect", files={"image_file": upload(b"")})
        assert response.status_code == 400

    def test_undecodable_upload_is_rejected(self, client):
        response = client.post("/api/v1/detect", files={"image_file": upload(b"not an image")})

        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    def test_missing_file_is_a_validation_error(self, client):
        assert client.post("/api/v1/detect").status_code == 422


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

    def test_unknown_batch_id_returns_404(self, client):
        assert client.get("/api/v1/batch-status/does-not-exist").status_code == 404
