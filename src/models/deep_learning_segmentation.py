"""
Deep Learning Segmentation Module for SAR Oil Spill Detection
Implements U-Net and other architectures for semantic segmentation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    logging.warning("segmentation_models_pytorch not available. Using custom U-Net only.")

logger = logging.getLogger(__name__)


class UNetBlock(nn.Module):
    """Basic U-Net building block with double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ImprovedUNet(nn.Module):
    """Improved U-Net architecture with attention mechanisms for SAR segmentation."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_filters: int = 64):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = UNetBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = UNetBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = UNetBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = UNetBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(base_filters * 8, base_filters * 16)
        
        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(base_filters * 16, base_filters * 8)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(base_filters * 2, base_filters)
        
        # Output layer
        self.out_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out_conv(dec1)
        return out


class DeepLearningSegmentation:
    """
    Deep Learning Segmentation wrapper for multiple architectures.
    Supports custom U-Net and segmentation-models-pytorch architectures.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the segmentation model.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config or {}
        self.model: Optional[nn.Module] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = 1  # Grayscale SAR images
        self.num_classes = 2  # Binary segmentation (oil/no-oil)
        
        logger.info(f"Using device: {self.device}")
    
    def create_model(self, architecture: str = "improved_unet", pretrained: bool = False):
        """
        Create the segmentation model architecture.
        
        Args:
            architecture: Model architecture name
            pretrained: Whether to use pretrained encoder (ImageNet)
        """
        logger.info(f"Creating model: {architecture}")
        
        try:
            if architecture == "improved_unet":
                self.model = ImprovedUNet(
                    in_channels=self.in_channels,
                    num_classes=self.num_classes,
                    base_filters=64
                )
            
            elif architecture.startswith("smp_") and SMP_AVAILABLE:
                # Use segmentation-models-pytorch
                arch_name = architecture.replace("smp_", "")
                
                if arch_name == "unet":
                    self.model = smp.Unet(
                        encoder_name="resnet34",
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=self.in_channels,
                        classes=self.num_classes
                    )
                elif arch_name == "deeplabv3plus":
                    self.model = smp.DeepLabV3Plus(
                        encoder_name="resnet50",
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=self.in_channels,
                        classes=self.num_classes
                    )
                elif arch_name == "fpn":
                    self.model = smp.FPN(
                        encoder_name="resnet34",
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=self.in_channels,
                        classes=self.num_classes
                    )
                else:
                    raise ValueError(f"Unknown SMP architecture: {arch_name}")
            
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            self.model = self.model.to(self.device)
            logger.info(f"Model created successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info(f"Model weights loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inference on a single image.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, 1)
        
        Returns:
            Predicted segmentation mask (H, W) with values [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.model.eval()
        
        try:
            # Preprocess image
            if image.ndim == 2:
                image = image[np.newaxis, ...]  # Add channel dim
            elif image.ndim == 3 and image.shape[2] == 1:
                image = image.transpose(2, 0, 1)  # HWC to CHW
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            
            # Add batch dimension and convert to tensor
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(image_tensor)
                
                # Apply softmax for multi-class or sigmoid for binary
                if self.num_classes == 2:
                    # Binary segmentation - take channel 1 (oil spill class)
                    probs = torch.softmax(output, dim=1)
                    mask_probs = probs[0, 1, :, :].cpu().numpy()
                else:
                    mask_probs = torch.sigmoid(output[0, 0, :, :]).cpu().numpy()
            
            return mask_probs
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Perform inference on a batch of images.
        
        Args:
            images: Batch of images (B, H, W) or (B, H, W, 1)
        
        Returns:
            Predicted masks (B, H, W)
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.model.eval()
        
        try:
            # Preprocess batch
            if images.ndim == 3:
                images = images[:, np.newaxis, :, :]  # Add channel dim
            elif images.ndim == 4 and images.shape[3] == 1:
                images = images.transpose(0, 3, 1, 2)  # BHWC to BCHW
            
            images_tensor = torch.from_numpy(images).float().to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images_tensor)
                probs = torch.softmax(outputs, dim=1)
                masks = probs[:, 1, :, :].cpu().numpy()
            
            return masks
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def save_checkpoint(self, save_path: str, epoch: Optional[int] = None, 
                       metrics: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save checkpoint
            epoch: Training epoch number
            metrics: Training/validation metrics
        """
        if self.model is None:
            raise ValueError("Model not created.")
        
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
                'config': self.config
            }
            
            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
