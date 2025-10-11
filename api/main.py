"""
Modern FastAPI Application for SAR Oil Spill Detection
Provides REST API endpoints for image upload, processing, and results.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# Import our custom modules
import sys
sys.path.append('..')
from src.core.sar_image_processor import SARImageProcessor
from src.models.deep_learning_segmentation import DeepLearningSegmentation
from src.utils.performance_evaluator import PerformanceEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SAR Oil Spill Detection API",
    description="Advanced API for detecting oil spills in SAR satellite imagery",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global components initialization
sar_processor = SARImageProcessor(target_image_size=(512, 512))
evaluator = PerformanceEvaluator()
model_instance = None  # Will be loaded on demand

# Processing status tracking
processing_jobs = {}


class DetectionRequest(BaseModel):
    """Request model for oil spill detection."""
    model_type: str = Field(default="improved_unet", description="Type of model to use")
    enhancement_method: str = Field(default="clahe", description="Contrast enhancement method")
    despeckling_filter: str = Field(default="lee", description="Despeckling filter type")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold for detection")


class DetectionResponse(BaseModel):
    """Response model for detection results."""
    success: bool
    message: str
    results: Optional[Dict] = None
    processing_time: Optional[float] = None
    job_id: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information response."""
    available_models: List[str]
    current_model: Optional[str]
    model_parameters: Optional[Dict] = None


@app.get("/", response_class=JSONResponse)
async def root():
    """API root endpoint with basic information."""
    return {
        "message": "SAR Oil Spill Detection API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/api/docs",
            "health": "/health",
            "detect": "/api/v1/detect",
            "models": "/api/v1/models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    try:
        # Perform basic system checks
        system_status = {
            "api_status": "healthy",
            "processor_ready": sar_processor is not None,
            "evaluator_ready": evaluator is not None,
            "active_jobs": len(processing_jobs)
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "timestamp": str(pd.Timestamp.now()),
                "system": system_status
            }
        )
    except Exception as error:
        logger.error(f"Health check failed: {error}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(error)
            }
        )


@app.get("/api/v1/models", response_model=ModelInfo)
async def get_available_models():
    """Get information about available models."""
    try:
        available_models = [
            "improved_unet",
            "smp_unet", 
            "smp_deeplabv3plus",
            "smp_fpn"
        ]
        
        current_model = None
        model_params = None
        
        if model_instance is not None:
            current_model = "loaded_model"
            model_params = {
                "device": str(model_instance.device),
                "input_channels": 1,
                "num_classes": 2
            }
        
        return ModelInfo(
            available_models=available_models,
            current_model=current_model,
            model_parameters=model_params
        )
    
    except Exception as error:
        logger.error(f"Error getting model info: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_oil_spill(
    image_file: UploadFile = File(..., description="SAR image file"),
    request_params: DetectionRequest = DetectionRequest()
):
    """Main endpoint for oil spill detection in SAR images."""
    import time
    import uuid
    
    start_time = time.time()
    job_id = str(uuid.uuid4())
    
    try:
        # Validate file type
        if not _is_valid_image_file(image_file.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Supported: JPG, PNG, TIFF"
            )
        
        # Read and preprocess image
        image_data = await image_file.read()
        processed_image = await _preprocess_uploaded_image(
            image_data, request_params
        )
        
        # Load model if not already loaded
        global model_instance
        if model_instance is None:
            model_instance = await _load_detection_model(request_params.model_type)
        
        # Perform detection
        detection_result = await _perform_detection(
            processed_image, model_instance, request_params
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "oil_spill_detected": detection_result["oil_detected"],
            "confidence_score": detection_result["confidence"],
            "affected_area": detection_result["area_pixels"],
            "detection_mask": detection_result["mask_base64"],
            "processing_details": {
                "model_used": request_params.model_type,
                "enhancement_method": request_params.enhancement_method,
                "despeckling_filter": request_params.despeckling_filter,
                "image_size": detection_result["image_dimensions"]
            }
        }
        
        logger.info(f"Detection completed for job {job_id} in {processing_time:.2f}s")
        
        return DetectionResponse(
            success=True,
            message="Oil spill detection completed successfully",
            results=response_data,
            processing_time=processing_time,
            job_id=job_id
        )
    
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Detection failed for job {job_id}: {error}")
        return DetectionResponse(
            success=False,
            message=f"Detection failed: {str(error)}",
            job_id=job_id
        )


@app.post("/api/v1/evaluate")
async def evaluate_detection(
    image_file: UploadFile = File(..., description="SAR image file"),
    ground_truth_file: UploadFile = File(..., description="Ground truth mask file"),
    request_params: DetectionRequest = DetectionRequest()
):
    """Evaluate detection performance against ground truth."""
    try:
        # Read images
        image_data = await image_file.read()
        gt_data = await ground_truth_file.read()
        
        # Process images
        processed_image = await _preprocess_uploaded_image(image_data, request_params)
        ground_truth = _load_ground_truth_mask(gt_data)
        
        # Perform detection
        global model_instance
        if model_instance is None:
            model_instance = await _load_detection_model(request_params.model_type)
        
        detection_result = await _perform_detection(
            processed_image, model_instance, request_params
        )
        
        # Evaluate performance
        predicted_mask = _decode_mask_from_base64(detection_result["mask_base64"])
        metrics = evaluator.calculate_segmentation_metrics(
            predicted_mask, ground_truth
        )
        
        evaluation_report = evaluator.generate_evaluation_report(metrics)
        
        return JSONResponse({
            "success": True,
            "evaluation_metrics": metrics,
            "detailed_report": evaluation_report,
            "detection_results": detection_result
        })
    
    except Exception as error:
        logger.error(f"Evaluation failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/v1/batch-process")
async def batch_process_images(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    request_params: DetectionRequest = DetectionRequest()
):
    """Process multiple images in batch mode."""
    import uuid
    
    batch_id = str(uuid.uuid4())
    
    try:
        # Validate file count
        if len(images) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per batch"
            )
        
        # Start background processing
        background_tasks.add_task(
            _process_image_batch, batch_id, images, request_params
        )
        
        processing_jobs[batch_id] = {
            "status": "processing",
            "total_images": len(images),
            "completed": 0,
            "results": []
        }
        
        return JSONResponse({
            "success": True,
            "message": "Batch processing started",
            "batch_id": batch_id,
            "status_endpoint": f"/api/v1/batch-status/{batch_id}"
        })
    
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Batch processing failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/v1/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of batch processing job."""
    if batch_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    return JSONResponse(processing_jobs[batch_id])


# Helper functions

def _is_valid_image_file(filename: str) -> bool:
    """Check if uploaded file is a valid image format."""
    if not filename:
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    file_ext = Path(filename).suffix.lower()
    return file_ext in valid_extensions


async def _preprocess_uploaded_image(image_data: bytes, 
                                   params: DetectionRequest) -> np.ndarray:
    """Preprocess uploaded image data."""
    try:
        # Load image from bytes
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Apply preprocessing
        image = image.astype(np.float32)
        
        # Apply despeckling filter
        image = sar_processor.apply_despeckling_filter(
            image, 
            filter_type=params.despeckling_filter
        )
        
        # Apply contrast enhancement
        image = sar_processor.enhance_contrast(
            image, 
            method=params.enhancement_method
        )
        
        # Resize to target size
        image = sar_processor.resize_image(image)
        
        # Normalize
        image = sar_processor.normalize_intensity(image)
        
        return image
        
    except Exception as error:
        logger.error(f"Image preprocessing failed: {error}")
        raise


async def _load_detection_model(model_type: str):
    """Load the specified detection model."""
    try:
        # Model configuration
        config = {
            'learning_rate': 0.001,
            'model_save_dir': 'models',
            'early_stopping_patience': 15
        }
        
        model = DeepLearningSegmentation(config)
        model.create_model(architecture=model_type)
        
        # Try to load pre-trained weights if available
        model_path = Path('models') / f'{model_type}_best.pth'
        if model_path.exists():
            model.load_checkpoint(str(model_path))
            logger.info(f"Loaded pre-trained {model_type} model")
        else:
            logger.warning(f"No pre-trained weights found for {model_type}")
        
        return model
        
    except Exception as error:
        logger.error(f"Model loading failed: {error}")
        raise


async def _perform_detection(image: np.ndarray, model, 
                           params: DetectionRequest) -> Dict:
    """Perform oil spill detection on processed image."""
    try:
        # Make prediction
        predicted_mask = model.predict(image)
        
        # Apply confidence threshold
        confidence_mask = (predicted_mask >= params.confidence_threshold).astype(np.uint8)
        
        # Calculate metrics
        oil_detected = np.sum(confidence_mask) > 0
        confidence_score = float(np.max(predicted_mask)) if oil_detected else 0.0
        affected_area = int(np.sum(confidence_mask))
        
        # Encode mask as base64
        import base64
        mask_bytes = cv2.imencode('.png', confidence_mask * 255)[1].tobytes()
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
        
        return {
            "oil_detected": oil_detected,
            "confidence": confidence_score,
            "area_pixels": affected_area,
            "mask_base64": mask_base64,
            "image_dimensions": image.shape
        }
        
    except Exception as error:
        logger.error(f"Detection failed: {error}")
        raise


def _load_ground_truth_mask(gt_data: bytes) -> np.ndarray:
    """Load ground truth mask from uploaded data."""
    try:
        image_array = np.frombuffer(gt_data, np.uint8)
        gt_mask = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        
        if gt_mask is None:
            raise ValueError("Failed to decode ground truth mask")
        
        # Convert to binary mask (assuming oil spill pixels are white/bright)
        binary_mask = (gt_mask > 128).astype(np.uint8)
        
        return binary_mask
        
    except Exception as error:
        logger.error(f"Ground truth loading failed: {error}")
        raise


def _decode_mask_from_base64(mask_base64: str) -> np.ndarray:
    """Decode base64 encoded mask."""
    import base64
    
    mask_bytes = base64.b64decode(mask_base64)
    mask_array = np.frombuffer(mask_bytes, np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    
    return (mask > 128).astype(np.uint8)


async def _process_image_batch(batch_id: str, images: List[UploadFile], 
                             params: DetectionRequest):
    """Background task for batch image processing."""
    try:
        global model_instance
        if model_instance is None:
            model_instance = await _load_detection_model(params.model_type)
        
        results = []
        
        for idx, image_file in enumerate(images):
            try:
                # Process individual image
                image_data = await image_file.read()
                processed_image = await _preprocess_uploaded_image(image_data, params)
                detection_result = await _perform_detection(
                    processed_image, model_instance, params
                )
                
                results.append({
                    "filename": image_file.filename,
                    "success": True,
                    "results": detection_result
                })
                
                # Update progress
                processing_jobs[batch_id]["completed"] = idx + 1
                processing_jobs[batch_id]["results"] = results
                
            except Exception as error:
                logger.error(f"Failed to process {image_file.filename}: {error}")
                results.append({
                    "filename": image_file.filename,
                    "success": False,
                    "error": str(error)
                })
        
        # Mark batch as completed
        processing_jobs[batch_id]["status"] = "completed"
        processing_jobs[batch_id]["results"] = results
        
    except Exception as error:
        logger.error(f"Batch processing failed: {error}")
        processing_jobs[batch_id]["status"] = "failed"
        processing_jobs[batch_id]["error"] = str(error)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
