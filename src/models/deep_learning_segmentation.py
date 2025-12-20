"""
Deep Learning Segmentation Module for SAR Oil Spill Detection
Implements U-Net and other segmentation architectures using PyTorch.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UNetBlock(nn.Module):
    """Basic U-Net convolutional block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleUNet(nn.Module):
    """Lightweight U-Net implementation for SAR image segmentation."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        # Final classification
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


class DeepLearningSegmentation:
    """
    Main class for deep learning-based oil spill segmentation.
    Handles model creation, training, inference, and checkpoint management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the segmentation model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = None
        
        logger.info(f"Initialized DeepLearningSegmentation on device: {self.device}")
    
    def create_model(self, architecture: str = 'improved_unet', 
                    in_channels: int = 1, num_classes: int = 2):
        """
        Create a segmentation model with specified architecture.
        
        Args:
            architecture: Model architecture type
            in_channels: Number of input channels
            num_classes: Number of output classes
        """
        self.model_type = architecture
        
        if architecture in ['unet', 'improved_unet']:
            self.model = SimpleUNet(in_channels, num_classes)
        elif architecture == 'smp_unet':
            try:
                import segmentation_models_pytorch as smp
                self.model = smp.Unet(
                    encoder_name='resnet34',
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes
                )
            except ImportError:
                logger.warning("segmentation_models_pytorch not available, using SimpleUNet")
                self.model = SimpleUNet(in_channels, num_classes)
        elif architecture == 'smp_deeplabv3plus':
            try:
                import segmentation_models_pytorch as smp
                self.model = smp.DeepLabV3Plus(
                    encoder_name='resnet50',
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes
                )
            except ImportError:
                logger.warning("segmentation_models_pytorch not available, using SimpleUNet")
                self.model = SimpleUNet(in_channels, num_classes)
        elif architecture == 'smp_fpn':
            try:
                import segmentation_models_pytorch as smp
                self.model = smp.FPN(
                    encoder_name='resnet34',
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes
                )
            except ImportError:
                logger.warning("segmentation_models_pytorch not available, using SimpleUNet")
                self.model = SimpleUNet(in_channels, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self.model = self.model.to(self.device)
        logger.info(f"Created model: {architecture}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Model will use random initialization")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Model will use random initialization")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inference on a single image.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, 1)
        
        Returns:
            Predicted mask probabilities as numpy array (H, W)
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.model.eval()
        
        # Prepare input
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]  # Add batch dim
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Apply softmax for multi-class or sigmoid for binary
            if output.shape[1] > 1:
                probs = F.softmax(output, dim=1)
                # Get oil spill class probability (class 1)
                mask_probs = probs[:, 1, :, :]
            else:
                mask_probs = torch.sigmoid(output[:, 0, :, :])
        
        # Convert back to numpy
        mask = mask_probs.cpu().numpy()[0]
        
        return mask
    
    def save_checkpoint(self, save_path: Union[str, Path], 
                       epoch: Optional[int] = None, 
                       metrics: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Training metrics dictionary
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'device': str(self.device)
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
