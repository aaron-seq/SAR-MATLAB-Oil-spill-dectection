"""
Deep Learning Segmentation Module for SAR Oil Spill Detection
Provides PyTorch-based semantic segmentation models optimized for SAR imagery.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    logging.warning("segmentation_models_pytorch not available, limited model support")

logger = logging.getLogger(__name__)


class DeepLearningSegmentation:
    """
    Deep learning-based segmentation for oil spill detection in SAR images.
    Supports multiple architectures: U-Net, DeepLabV3+, FPN.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the deep learning segmentation model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.architecture = None
        self.input_channels = 1  # Grayscale SAR images
        self.num_classes = 2  # Binary segmentation (oil/no-oil)
        
        logger.info(f"DeepLearningSegmentation initialized on device: {self.device}")
    
    def create_model(self, architecture: str = "unet", 
                    encoder_name: str = "resnet34",
                    encoder_weights: Optional[str] = "imagenet") -> None:
        """
        Create segmentation model with specified architecture.
        
        Args:
            architecture: Model architecture ('unet', 'deeplabv3plus', 'fpn', 'improved_unet')
            encoder_name: Encoder backbone ('resnet34', 'resnet50', 'efficientnet-b0')
            encoder_weights: Pre-trained weights ('imagenet' or None)
        """
        try:
            self.architecture = architecture
            
            if architecture == "improved_unet" or architecture == "smp_unet":
                self.model = self._create_unet_model(encoder_name, encoder_weights)
            
            elif architecture == "smp_deeplabv3plus":
                if not SMP_AVAILABLE:
                    raise ImportError("segmentation_models_pytorch required for DeepLabV3+")
                self.model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=self.input_channels,
                    classes=self.num_classes
                )
            
            elif architecture == "smp_fpn":
                if not SMP_AVAILABLE:
                    raise ImportError("segmentation_models_pytorch required for FPN")
                self.model = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=self.input_channels,
                    classes=self.num_classes
                )
            
            else:
                logger.warning(f"Unknown architecture: {architecture}. Using U-Net.")
                self.model = self._create_unet_model(encoder_name, encoder_weights)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Created {architecture} model with {encoder_name} encoder")
            
        except Exception as error:
            logger.error(f"Error creating model: {error}")
            # Fallback to simple U-Net
            self.model = self._create_simple_unet()
            logger.info("Fallback: Created simple U-Net model")
    
    def _create_unet_model(self, encoder_name: str, encoder_weights: Optional[str]) -> nn.Module:
        """Create U-Net model using segmentation_models_pytorch."""
        if not SMP_AVAILABLE:
            return self._create_simple_unet()
        
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.input_channels,
            classes=self.num_classes,
            activation=None  # We'll apply sigmoid separately
        )
    
    def _create_simple_unet(self) -> nn.Module:
        """
        Create a simple U-Net model from scratch (fallback when SMP not available).
        """
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=1, num_classes=2):
                super().__init__()
                
                # Encoder
                self.enc1 = self._conv_block(in_channels, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                
                # Bottleneck
                self.bottleneck = self._conv_block(256, 512)
                
                # Decoder
                self.dec3 = self._conv_block(512 + 256, 256)
                self.dec2 = self._conv_block(256 + 128, 128)
                self.dec1 = self._conv_block(128 + 64, 64)
                
                # Output
                self.out = nn.Conv2d(64, num_classes, kernel_size=1)
                
                self.pool = nn.MaxPool2d(2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
            def _conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                
                # Bottleneck
                b = self.bottleneck(self.pool(e3))
                
                # Decoder with skip connections
                d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1))
                d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
                
                return self.out(d1)
        
        return SimpleUNet(self.input_channels, self.num_classes)
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint file
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as error:
            logger.error(f"Error loading checkpoint: {error}")
            raise
    
    def predict(self, image: np.ndarray, apply_sigmoid: bool = True) -> np.ndarray:
        """
        Perform inference on input image.
        
        Args:
            image: Input SAR image (H, W) or (H, W, 1)
            apply_sigmoid: Whether to apply sigmoid activation
            
        Returns:
            Predicted segmentation mask (H, W) with values in [0, 1]
        """
        try:
            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")
            
            # Prepare input tensor
            input_tensor = self._preprocess_image(image)
            
            # Inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # Apply sigmoid for binary segmentation
                if apply_sigmoid:
                    output = torch.sigmoid(output)
                
                # Take the foreground class (oil spill)
                if output.shape[1] == 2:
                    prediction = output[0, 1].cpu().numpy()
                else:
                    prediction = output[0, 0].cpu().numpy()
            
            # Resize to original image size if needed
            if prediction.shape != image.shape[:2]:
                import cv2
                prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]))
            
            return prediction
            
        except Exception as error:
            logger.error(f"Error during prediction: {error}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Handle different input shapes
        if image.ndim == 2:
            image = image[np.newaxis, :, :]  # Add channel dimension
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image.transpose(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        elif image.ndim == 3:
            image = image[:, :, 0][np.newaxis, :, :]  # Take first channel
        
        # Normalize to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(image.copy()).float()
        
        # Add batch dimension
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def save_checkpoint(self, save_path: Union[str, Path], 
                       additional_info: Optional[Dict] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save checkpoint
            additional_info: Additional information to save with checkpoint
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'architecture': self.architecture,
                'config': self.config
            }
            
            if additional_info:
                checkpoint.update(additional_info)
            
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
        except Exception as error:
            logger.error(f"Error saving checkpoint: {error}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "architecture": self.architecture,
            "device": str(self.device),
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }
