"""
Modern Deep Learning Models for SAR Oil Spill Segmentation
Implements state-of-the-art architectures with PyTorch and TensorFlow backends.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

logger = logging.getLogger(__name__)


class SARDataset(Dataset):
    """PyTorch dataset for SAR oil spill segmentation."""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transform=None, target_size: Tuple[int, int] = (512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess SAR image."""
        import cv2
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.target_size)
        return image.astype(np.float32) / 255.0
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load and preprocess segmentation mask."""
        import cv2
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        # Convert to class indices (0: sea, 1: oil spill, 2: land if applicable)
        return (mask > 128).astype(np.uint8)


class ImprovedUNet(nn.Module):
    """Enhanced U-Net architecture with attention mechanisms and residual connections."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 2, 
                 base_features: int = 64):
        super(ImprovedUNet, self).__init__()
        
        # Encoder path with residual blocks
        self.encoder1 = self._make_encoder_block(input_channels, base_features)
        self.encoder2 = self._make_encoder_block(base_features, base_features * 2)
        self.encoder3 = self._make_encoder_block(base_features * 2, base_features * 4)
        self.encoder4 = self._make_encoder_block(base_features * 4, base_features * 8)
        
        # Bottleneck with attention
        self.bottleneck = self._make_encoder_block(base_features * 8, base_features * 16)
        self.attention_bottleneck = AttentionBlock(base_features * 16, base_features * 8)
        
        # Decoder path with skip connections
        self.decoder4 = self._make_decoder_block(base_features * 16, base_features * 8)
        self.decoder3 = self._make_decoder_block(base_features * 8, base_features * 4)
        self.decoder2 = self._make_decoder_block(base_features * 4, base_features * 2)
        self.decoder1 = self._make_decoder_block(base_features * 2, base_features)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(base_features, num_classes, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.3)
        
    def _make_encoder_block(self, in_channels: int, out_channels: int):
        """Create an encoder block with residual connections."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Create a decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.dropout(bottleneck)
        
        # Decoder path with skip connections
        dec4 = F.interpolate(bottleneck, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)


class AttentionBlock(nn.Module):
    """Attention mechanism for focusing on relevant features."""
    
    def __init__(self, gate_channels: int, feature_channels: int):
        super(AttentionBlock, self).__init__()
        
        self.gate_conv = nn.Conv2d(gate_channels, feature_channels, 1)
        self.feature_conv = nn.Conv2d(feature_channels, feature_channels, 1)
        self.attention_conv = nn.Conv2d(feature_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, gate, feature):
        gate_processed = self.gate_conv(gate)
        feature_processed = self.feature_conv(feature)
        
        combined = F.relu(gate_processed + feature_processed)
        attention_weights = self.sigmoid(self.attention_conv(combined))
        
        return feature * attention_weights


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in segmentation."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DeepLearningSegmentation:
    """Main class for deep learning-based SAR oil spill segmentation."""
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        logger.info(f"Deep Learning Segmentation initialized on {self.device}")
        
    def create_model(self, architecture: str = 'improved_unet', 
                    input_channels: int = 1, num_classes: int = 2):
        """Create and initialize the segmentation model."""
        try:
            if architecture == 'improved_unet':
                self.model = ImprovedUNet(input_channels, num_classes)
                
            elif architecture == 'smp_unet':
                # Using segmentation_models_pytorch
                self.model = smp.Unet(
                    encoder_name='resnet34',
                    encoder_weights='imagenet',
                    in_channels=input_channels,
                    classes=num_classes,
                    activation=None
                )
                
            elif architecture == 'smp_deeplabv3plus':
                self.model = smp.DeepLabV3Plus(
                    encoder_name='resnet50',
                    encoder_weights='imagenet',
                    in_channels=input_channels,
                    classes=num_classes,
                    activation=None
                )
                
            elif architecture == 'smp_fpn':
                self.model = smp.FPN(
                    encoder_name='resnet34',
                    encoder_weights='imagenet',
                    in_channels=input_channels,
                    classes=num_classes,
                    activation=None
                )
                
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            self.model.to(self.device)
            
            # Initialize optimizer and loss function
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.get('learning_rate', 0.001),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
            
            # Use Focal Loss for class imbalance
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
            
            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5, verbose=True
            )
            
            logger.info(f"Created {architecture} model with {sum(p.numel() for p in self.model.parameters())} parameters")
            
        except Exception as error:
            logger.error(f"Error creating model: {error}")
            raise
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   num_epochs: int = 100):
        """Train the segmentation model."""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.get('early_stopping_patience', 15)
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        return training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate metrics
                predictions = torch.argmax(outputs, dim=1)
                iou = self._calculate_iou(predictions, targets)
                dice = self._calculate_dice(predictions, targets)
                
                total_iou += iou
                total_dice += dice
        
        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        metrics = {
            'iou': avg_iou,
            'dice': avg_dice
        }
        
        return avg_loss, metrics
    
    def _calculate_iou(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Intersection over Union (IoU) metric."""
        intersection = torch.logical_and(predictions, targets).sum().float()
        union = torch.logical_or(predictions, targets).sum().float()
        return (intersection / (union + 1e-8)).item()
    
    def _calculate_dice(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Dice coefficient."""
        intersection = torch.logical_and(predictions, targets).sum().float()
        total = predictions.sum() + targets.sum()
        return (2.0 * intersection / (total + 1e-8)).item()
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Make prediction on a single image."""
        self.model.eval()
        
        # Preprocess image
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            predictions = torch.softmax(outputs, dim=1)
            predicted_mask = torch.argmax(predictions, dim=1)
        
        return predicted_mask.cpu().numpy().squeeze()
    
    def _save_checkpoint(self, epoch: int, loss: float, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        save_path = Path(self.config.get('model_save_dir', 'models')) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"Model loaded from {checkpoint_path}")
            return checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)
            
        except Exception as error:
            logger.error(f"Error loading checkpoint: {error}")
            raise


def create_tensorflow_unet(input_shape: Tuple[int, int, int], num_classes: int = 2) -> Model:
    """Create U-Net model using TensorFlow/Keras."""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    
    # Decoder
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
