"""FastAPI service for SAR oil spill detection.

Endpoints are versioned under ``/api/v1``. Interactive docs live at
``/api/docs``. The service is stateless apart from an in-process batch job
registry, so it scales horizontally behind any load balancer -- with the
caveat noted on :data:`_batch_jobs`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sar_oil_spill import __version__
from sar_oil_spill.config import configure_logging, load_settings
from sar_oil_spill.core import OilSpillDetector
from sar_oil_spill.models.traditional_segmentation import METHOD_NAMES
from sar_oil_spill.utils import PerformanceEvaluator

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_BATCH_SIZE = 10
MAX_TRACKED_JOBS = 100
VALID_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})

# Batch jobs live in memory, so a job is only visible to the worker that
# created it and is lost on restart. That is deliberate for a single-node
# deployment; put Redis behind this before running multiple replicas.
_batch_jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()

# asyncio only holds weak references to running tasks, so a batch task that is
# not referenced anywhere can be garbage-collected mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()

_detector: OilSpillDetector | None = None
_evaluator = PerformanceEvaluator()


DetectionMethod = Enum(  # type: ignore[misc]
    "DetectionMethod", {name.upper(): name for name in METHOD_NAMES}, type=str
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the detector once at startup rather than per request.

    Constructing it involves reading config and allocating the processing
    stack, which is wasteful to repeat on every call.
    """
    global _detector
    configure_logging()
    settings = load_settings()
    _detector = OilSpillDetector(settings)
    logger.info("Detector ready with methods: %s", ", ".join(METHOD_NAMES))
    yield
    _detector = None
    _batch_jobs.clear()
    _background_tasks.clear()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="SAR Oil Spill Detection API",
    description=__doc__,
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Tighten this to your front-end origins before exposing the service.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_detector() -> OilSpillDetector:
    """Dependency returning the process-wide detector."""
    if _detector is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detector is not initialised.",
        )
    return _detector


DetectorDep = Annotated[OilSpillDetector, Depends(get_detector)]


# ------------------------------------------------------------------ schemas


class DetectionParameters(BaseModel):
    """Tunable parameters accepted by the detection endpoints."""

    method: str = Field(default="adaptive_threshold", description="Segmentation method.")
    mask_land: bool = Field(default=False, description="Exclude bright land areas.")
    min_area_pixels: int = Field(default=100, ge=0, description="Drop smaller detections.")


class DetectionResults(BaseModel):
    oil_spill_detected: bool
    confidence_score: float
    affected_area_pixels: int
    coverage_percent: float
    detection_mask_png_base64: str
    image_dimensions: list[int]
    method: str


class DetectionResponse(BaseModel):
    success: bool
    message: str
    job_id: str
    processing_time_seconds: float | None = None
    results: DetectionResults | None = None


class EvaluationResponse(BaseModel):
    success: bool
    evaluation_metrics: dict[str, float]
    detailed_report: dict[str, Any]
    detection: DetectionResults


class MethodInfo(BaseModel):
    available_methods: list[str]
    default_method: str
    deep_learning_available: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    detector_ready: bool
    active_batch_jobs: int


class BatchStatus(BaseModel):
    batch_id: str
    status: str
    total_images: int
    completed: int
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------- endpoints


@app.get("/", tags=["meta"])
async def root() -> dict[str, Any]:
    """Service metadata and endpoint index."""
    return {
        "service": "SAR Oil Spill Detection API",
        "version": __version__,
        "docs": "/api/docs",
        "endpoints": {
            "health": "/health",
            "methods": "/api/v1/methods",
            "detect": "/api/v1/detect",
            "evaluate": "/api/v1/evaluate",
            "batch": "/api/v1/batch-process",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health_check() -> HealthResponse:
    """Liveness probe used by Docker, Railway and Render health checks."""
    return HealthResponse(
        status="healthy" if _detector is not None else "starting",
        version=__version__,
        timestamp=datetime.now(UTC).isoformat(),
        detector_ready=_detector is not None,
        active_batch_jobs=sum(1 for j in _batch_jobs.values() if j["status"] == "processing"),
    )


@app.get("/api/v1/methods", response_model=MethodInfo, tags=["detection"])
async def get_available_methods() -> MethodInfo:
    """List the segmentation methods this build supports."""
    from sar_oil_spill.models.deep_learning_segmentation import TORCH_AVAILABLE

    return MethodInfo(
        available_methods=list(METHOD_NAMES),
        default_method="adaptive_threshold",
        deep_learning_available=TORCH_AVAILABLE,
    )


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["detection"])
async def detect_oil_spill(
    detector: DetectorDep,
    image_file: Annotated[UploadFile, File(description="SAR image file.")],
    method: Annotated[str, Form()] = "adaptive_threshold",
    mask_land: Annotated[bool, Form()] = False,
    min_area_pixels: Annotated[int, Form(ge=0)] = 100,
) -> DetectionResponse:
    """Detect oil slicks in a single uploaded SAR image."""
    job_id = str(uuid.uuid4())
    started = time.perf_counter()

    _validate_method(method)
    image = await _decode_upload(image_file)

    # The pipeline is CPU-bound NumPy work; off-thread it so one large scene
    # cannot block the event loop for every other client.
    result = await asyncio.to_thread(
        detector.detect,
        image,
        method=method,
        mask_land=mask_land,
        min_area_pixels=min_area_pixels,
    )

    logger.info("Job %s: %s in %.0f ms", job_id, method, (time.perf_counter() - started) * 1000)
    return DetectionResponse(
        success=True,
        message="Detection completed.",
        job_id=job_id,
        processing_time_seconds=round(time.perf_counter() - started, 4),
        results=_to_results(result),
    )


@app.post("/api/v1/evaluate", response_model=EvaluationResponse, tags=["detection"])
async def evaluate_detection(
    detector: DetectorDep,
    image_file: Annotated[UploadFile, File(description="SAR image file.")],
    ground_truth_file: Annotated[UploadFile, File(description="Reference mask.")],
    method: Annotated[str, Form()] = "adaptive_threshold",
    mask_land: Annotated[bool, Form()] = False,
) -> EvaluationResponse:
    """Detect, then score the prediction against an uploaded reference mask."""
    _validate_method(method)
    image = await _decode_upload(image_file)
    ground_truth = await _decode_upload(ground_truth_file) > 127

    result = await asyncio.to_thread(
        detector.detect, image, method=method, ground_truth=ground_truth, mask_land=mask_land
    )

    if result.metrics is None:  # pragma: no cover - ground truth is always supplied here
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Metrics were not computed.")

    metrics = result.metrics.as_dict()
    return EvaluationResponse(
        success=True,
        evaluation_metrics=metrics,
        detailed_report=_evaluator.generate_evaluation_report(
            result.metrics, {"method": method, "shape": list(result.mask.shape)}
        ),
        detection=_to_results(result),
    )


@app.post("/api/v1/batch-process", response_model=BatchStatus, tags=["batch"])
async def batch_process_images(
    detector: DetectorDep,
    images: Annotated[list[UploadFile], File(description="Up to 10 SAR images.")],
    method: Annotated[str, Form()] = "adaptive_threshold",
    mask_land: Annotated[bool, Form()] = False,
) -> BatchStatus:
    """Queue several images for background processing.

    Uploads are read to bytes *before* returning, because FastAPI closes the
    temporary files once the response is sent -- a background task holding
    ``UploadFile`` objects would find them already closed.
    """
    _validate_method(method)
    if not images:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No images supplied.")
    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"At most {MAX_BATCH_SIZE} images per batch, got {len(images)}.",
        )

    payloads = [(f.filename or "unnamed", await _read_upload(f)) for f in images]

    batch_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "batch_id": batch_id,
        "status": "processing",
        "total_images": len(payloads),
        "completed": 0,
        "results": [],
        "error": None,
    }
    _register_job(batch_id, job)

    task = asyncio.create_task(_process_batch(detector, batch_id, payloads, method, mask_land))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return BatchStatus(**job)


@app.get("/api/v1/batch-status/{batch_id}", response_model=BatchStatus, tags=["batch"])
async def get_batch_status(batch_id: str) -> BatchStatus:
    """Poll the progress of a queued batch."""
    job = _batch_jobs.get(batch_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown batch id: {batch_id}")
    return BatchStatus(**job)


# ------------------------------------------------------------------ helpers


def _validate_method(method: str) -> None:
    """Reject unknown method names with a 400 that lists the valid ones."""
    if method not in METHOD_NAMES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown method '{method}'. Available: {', '.join(METHOD_NAMES)}.",
        )


async def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload, enforcing the extension and size limits."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in VALID_SUFFIXES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(VALID_SUFFIXES))}.",
        )

    data = await upload.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return data


async def _decode_upload(upload: UploadFile) -> np.ndarray:
    """Read and decode an upload into a single-channel array."""
    return _decode_bytes(await _read_upload(upload))


def _decode_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to greyscale, raising a 400 if undecodable."""
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Could not decode the file as an image."
        )
    return decoded


def _to_results(result: Any) -> DetectionResults:
    """Convert a ``DetectionResult`` into the API response schema."""
    encoded = cv2.imencode(".png", result.mask.astype(np.uint8) * 255)[1]
    return DetectionResults(
        oil_spill_detected=result.oil_detected,
        confidence_score=round(result.confidence, 4),
        affected_area_pixels=result.affected_area_pixels,
        coverage_percent=round(result.coverage_fraction * 100, 3),
        detection_mask_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        image_dimensions=list(result.mask.shape),
        method=result.method,
    )


def _register_job(batch_id: str, job: dict[str, Any]) -> None:
    """Store a job, evicting the oldest so the registry cannot grow unbounded."""
    _batch_jobs[batch_id] = job
    while len(_batch_jobs) > MAX_TRACKED_JOBS:
        _batch_jobs.popitem(last=False)


async def _process_batch(
    detector: OilSpillDetector,
    batch_id: str,
    payloads: list[tuple[str, bytes]],
    method: str,
    mask_land: bool,
) -> None:
    """Run detection over a batch, recording per-image success or failure."""
    job = _batch_jobs[batch_id]
    try:
        for index, (filename, data) in enumerate(payloads, start=1):
            try:
                image = _decode_bytes(data)
                result = await asyncio.to_thread(
                    detector.detect, image, method=method, mask_land=mask_land
                )
                job["results"].append(
                    {"filename": filename, "success": True, "results": result.summary()}
                )
            except Exception as error:
                logger.error("Batch %s: %s failed: %s", batch_id, filename, error)
                job["results"].append(
                    {"filename": filename, "success": False, "error": str(error)}
                )
            job["completed"] = index
        job["status"] = "completed"
    except Exception as error:
        logger.exception("Batch %s failed", batch_id)
        job["status"] = "failed"
        job["error"] = str(error)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Any, error: ValueError) -> JSONResponse:
    """Surface bad arguments from the pipeline as 400s, not 500s."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(error)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
