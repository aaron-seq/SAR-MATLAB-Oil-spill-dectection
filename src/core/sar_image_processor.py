"""
Advanced SAR Image Processing Module
Provides comprehensive preprocessing and enhancement capabilities for SAR imagery.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters, morphology, restoration
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


class SARImageProcessor:
    """
    Advanced processor for Synthetic Aperture Radar images with specialized
    despeckling, enhancement, and preprocessing capabilities.
    """
    
    def __init__(self, target_image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the SAR image processor.
        
        Args:
            target_image_size: Target dimensions for processed images
        """
        self.target_size = target_image_size
        self.processing_history = []
        
        logger.info(f"SAR Image Processor initialized with target size {target_image_size}")
    
    def load_sar_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load SAR image with error handling and format validation.
        
        Args:
            image_path: Path to the SAR image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
                
            sar_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if sar_image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            self.processing_history.append(f"Loaded image from {image_path}")
            logger.debug(f"Successfully loaded SAR image: {image_path}")
            
            return sar_image.astype(np.float32)
            
        except Exception as error:
            logger.error(f"Error loading SAR image: {error}")
            return None
    
    def apply_despeckling_filter(self, sar_image: np.ndarray, 
                                filter_type: str = "lee",
                                window_size: int = 7,
                                noise_variance: float = 0.5) -> np.ndarray:
        """
        Apply advanced despeckling filters to reduce SAR noise.
        
        Args:
            sar_image: Input SAR image
            filter_type: Type of filter ('lee', 'frost', 'kuan', 'bilateral')
            window_size: Size of the filtering window
            noise_variance: Estimated noise variance for adaptive filtering
            
        Returns:
            Despeckled image
        """
        try:
            if filter_type == "lee":
                filtered_image = self._apply_lee_filter(sar_image, window_size, noise_variance)
            elif filter_type == "frost":
                filtered_image = self._apply_frost_filter(sar_image, window_size)
            elif filter_type == "kuan":
                filtered_image = self._apply_kuan_filter(sar_image, window_size)
            elif filter_type == "bilateral":
                filtered_image = cv2.bilateralFilter(
                    sar_image.astype(np.uint8), window_size, 75, 75
                ).astype(np.float32)
            else:
                logger.warning(f"Unknown filter type: {filter_type}. Using Lee filter.")
                filtered_image = self._apply_lee_filter(sar_image, window_size, noise_variance)
            
            self.processing_history.append(f"Applied {filter_type} despeckling filter")
            return filtered_image
            
        except Exception as error:
            logger.error(f"Error applying despeckling filter: {error}")
            return sar_image
    
    def _apply_lee_filter(self, image: np.ndarray, window_size: int, 
                         noise_variance: float) -> np.ndarray:
        """
        Apply Lee filter for SAR despeckling.
        
        The Lee filter is specifically designed for speckle reduction in SAR images,
        preserving edges while smoothing homogeneous regions.
        """
        height, width = image.shape
        filtered_image = np.zeros_like(image)
        half_window = window_size // 2
        
        # Pad image for border handling
        padded_image = np.pad(image, half_window, mode='reflect')
        
        for row in range(height):
            for col in range(width):
                # Extract local window
                window = padded_image[row:row + window_size, col:col + window_size]
                
                # Calculate local statistics
                local_mean = np.mean(window)
                local_variance = np.var(window)
                
                # Lee filter coefficient
                if local_variance > 0:
                    lee_coefficient = local_variance / (local_variance + noise_variance)
                else:
                    lee_coefficient = 0
                
                # Apply Lee filter
                center_pixel = padded_image[row + half_window, col + half_window]
                filtered_image[row, col] = local_mean + lee_coefficient * (center_pixel - local_mean)
        
        return filtered_image
    
    def _apply_frost_filter(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply Frost filter for edge-preserving despeckling.
        """
        return filters.rank.mean(image, morphology.disk(window_size // 2))
    
    def _apply_kuan_filter(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply Kuan filter for adaptive despeckling.
        """
        return restoration.denoise_tv_bregman(image, weight=0.1)
    
    def enhance_contrast(self, sar_image: np.ndarray, 
                        method: str = "clahe",
                        **kwargs) -> np.ndarray:
        """
        Apply contrast enhancement techniques optimized for SAR images.
        
        Args:
            sar_image: Input SAR image
            method: Enhancement method ('clahe', 'histogram_equalization', 'gamma')
            **kwargs: Additional parameters for enhancement methods
            
        Returns:
            Contrast-enhanced image
        """
        try:
            if method == "clahe":
                # Contrast Limited Adaptive Histogram Equalization
                clip_limit = kwargs.get('clip_limit', 3.0)
                tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
                
                # Convert to uint8 for CLAHE
                normalized_image = exposure.rescale_intensity(sar_image, out_range=(0, 255))
                uint8_image = normalized_image.astype(np.uint8)
                
                clahe_processor = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced_image = clahe_processor.apply(uint8_image).astype(np.float32)
                
            elif method == "histogram_equalization":
                enhanced_image = exposure.equalize_hist(sar_image) * 255
                
            elif method == "gamma":
                gamma_value = kwargs.get('gamma', 1.5)
                enhanced_image = exposure.adjust_gamma(sar_image, gamma=gamma_value)
                
            else:
                logger.warning(f"Unknown enhancement method: {method}. Using CLAHE.")
                enhanced_image = self.enhance_contrast(sar_image, "clahe")
            
            self.processing_history.append(f"Applied {method} contrast enhancement")
            return enhanced_image
            
        except Exception as error:
            logger.error(f"Error enhancing contrast: {error}")
            return sar_image
    
    def apply_morphological_operations(self, binary_mask: np.ndarray,
                                     operation: str = "opening",
                                     kernel_size: int = 5,
                                     iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations to clean up binary masks.
        
        Args:
            binary_mask: Binary segmentation mask
            operation: Type of operation ('opening', 'closing', 'erosion', 'dilation')
            kernel_size: Size of the morphological kernel
            iterations: Number of iterations to apply
            
        Returns:
            Processed binary mask
        """
        try:
            kernel = morphology.disk(kernel_size)
            
            if operation == "opening":
                processed_mask = morphology.opening(binary_mask, kernel)
            elif operation == "closing":
                processed_mask = morphology.closing(binary_mask, kernel)
            elif operation == "erosion":
                processed_mask = morphology.erosion(binary_mask, kernel)
            elif operation == "dilation":
                processed_mask = morphology.dilation(binary_mask, kernel)
            else:
                logger.warning(f"Unknown morphological operation: {operation}")
                return binary_mask
            
            # Apply iterations if specified
            for _ in range(iterations - 1):
                if operation == "opening":
                    processed_mask = morphology.opening(processed_mask, kernel)
                elif operation == "closing":
                    processed_mask = morphology.closing(processed_mask, kernel)
                elif operation == "erosion":
                    processed_mask = morphology.erosion(processed_mask, kernel)
                elif operation == "dilation":
                    processed_mask = morphology.dilation(processed_mask, kernel)
            
            self.processing_history.append(f"Applied {operation} morphological operation")
            return processed_mask.astype(np.uint8)
            
        except Exception as error:
            logger.error(f"Error applying morphological operations: {error}")
            return binary_mask
    
    def remove_small_objects(self, binary_mask: np.ndarray, 
                           minimum_size: int = 100) -> np.ndarray:
        """
        Remove small connected components from binary mask.
        
        Args:
            binary_mask: Binary segmentation mask
            minimum_size: Minimum size of objects to keep (in pixels)
            
        Returns:
            Cleaned binary mask
        """
        try:
            cleaned_mask = morphology.remove_small_objects(
                binary_mask.astype(bool), min_size=minimum_size
            )
            
            self.processing_history.append(f"Removed objects smaller than {minimum_size} pixels")
            return cleaned_mask.astype(np.uint8)
            
        except Exception as error:
            logger.error(f"Error removing small objects: {error}")
            return binary_mask
    
    def resize_image(self, image: np.ndarray, 
                    target_size: Optional[Tuple[int, int]] = None,
                    interpolation_method: str = "lanczos") -> np.ndarray:
        """
        Resize image with high-quality interpolation.
        
        Args:
            image: Input image
            target_size: Target dimensions (height, width)
            interpolation_method: Interpolation method ('lanczos', 'cubic', 'linear')
            
        Returns:
            Resized image
        """
        try:
            if target_size is None:
                target_size = self.target_size
            
            # Map interpolation methods
            interpolation_map = {
                "lanczos": cv2.INTER_LANCZOS4,
                "cubic": cv2.INTER_CUBIC,
                "linear": cv2.INTER_LINEAR,
                "nearest": cv2.INTER_NEAREST
            }
            
            interpolation = interpolation_map.get(interpolation_method, cv2.INTER_LANCZOS4)
            resized_image = cv2.resize(image, target_size, interpolation=interpolation)
            
            self.processing_history.append(f"Resized image to {target_size}")
            return resized_image
            
        except Exception as error:
            logger.error(f"Error resizing image: {error}")
            return image
    
    def normalize_intensity(self, image: np.ndarray, 
                          method: str = "minmax",
                          percentile_range: Tuple[int, int] = (2, 98)) -> np.ndarray:
        """
        Normalize image intensity values.
        
        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore', 'percentile')
            percentile_range: Percentile range for robust normalization
            
        Returns:
            Normalized image
        """
        try:
            if method == "minmax":
                min_val = np.min(image)
                max_val = np.max(image)
                if max_val > min_val:
                    normalized_image = (image - min_val) / (max_val - min_val)
                else:
                    normalized_image = image
                    
            elif method == "zscore":
                mean_val = np.mean(image)
                std_val = np.std(image)
                if std_val > 0:
                    normalized_image = (image - mean_val) / std_val
                else:
                    normalized_image = image - mean_val
                    
            elif method == "percentile":
                lower_percentile = np.percentile(image, percentile_range[0])
                upper_percentile = np.percentile(image, percentile_range[1])
                
                normalized_image = np.clip(
                    (image - lower_percentile) / (upper_percentile - lower_percentile),
                    0, 1
                )
            else:
                logger.warning(f"Unknown normalization method: {method}")
                return image
            
            self.processing_history.append(f"Applied {method} normalization")
            return normalized_image.astype(np.float32)
            
        except Exception as error:
            logger.error(f"Error normalizing image: {error}")
            return image
    
    def get_processing_summary(self) -> List[str]:
        """
        Get a summary of all processing steps applied.
        
        Returns:
            List of processing steps
        """
        return self.processing_history.copy()
    
    def reset_processing_history(self) -> None:
        """
        Clear the processing history.
        """
        self.processing_history.clear()
        logger.debug("Processing history reset")
