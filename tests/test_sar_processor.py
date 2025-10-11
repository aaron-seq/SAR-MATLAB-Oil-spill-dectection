"""
Comprehensive tests for SAR image processor module.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Import the module to test
import sys
sys.path.append('../src')
from src.core.sar_image_processor import SARImageProcessor


class TestSARImageProcessor:
    """Test suite for SAR Image Processor."""
    
    @pytest.fixture
    def processor(self):
        """Create a SAR processor instance for testing."""
        return SARImageProcessor(target_image_size=(256, 256))
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample SAR image for testing."""
        # Generate a synthetic SAR-like image
        np.random.seed(42)
        base_image = np.random.gamma(2, 2, (128, 128)).astype(np.float32)
        
        # Add some "oil spill" features (dark regions)
        base_image[40:60, 40:60] *= 0.3  # Dark patch simulating oil
        base_image[80:100, 20:40] *= 0.4  # Another dark patch
        
        # Add speckle noise
        speckle = np.random.gamma(1, 0.1, base_image.shape)
        noisy_image = base_image * speckle
        
        return noisy_image.astype(np.float32)
    
    def test_processor_initialization(self):
        """Test proper initialization of the processor."""
        processor = SARImageProcessor(target_image_size=(512, 512))
        
        assert processor.target_size == (512, 512)
        assert len(processor.processing_history) == 0
    
    def test_load_sar_image_success(self, processor, tmp_path):
        """Test successful image loading."""
        # Create a temporary image file
        import cv2
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_file = tmp_path / "test_image.png"
        cv2.imwrite(str(test_file), test_image)
        
        loaded_image = processor.load_sar_image(test_file)
        
        assert loaded_image is not None
        assert loaded_image.dtype == np.float32
        assert len(processor.processing_history) == 1
    
    def test_load_sar_image_file_not_found(self, processor):
        """Test handling of non-existent file."""
        result = processor.load_sar_image("nonexistent_file.png")
        assert result is None
    
    def test_apply_despeckling_filter_lee(self, processor, sample_image):
        """Test Lee filter application."""
        filtered_image = processor.apply_despeckling_filter(
            sample_image, filter_type="lee", window_size=5
        )
        
        assert filtered_image.shape == sample_image.shape
        assert filtered_image.dtype == np.float32
        assert "lee despeckling filter" in processor.processing_history[-1]
        
        # Check that filtering reduces noise (should be smoother)
        assert np.std(filtered_image) <= np.std(sample_image)
    
    def test_apply_despeckling_filter_bilateral(self, processor, sample_image):
        """Test bilateral filter application."""
        filtered_image = processor.apply_despeckling_filter(
            sample_image, filter_type="bilateral", window_size=5
        )
        
        assert filtered_image.shape == sample_image.shape
        assert "bilateral despeckling filter" in processor.processing_history[-1]
    
    def test_apply_despeckling_filter_invalid_type(self, processor, sample_image):
        """Test handling of invalid filter type."""
        # Should default to Lee filter
        filtered_image = processor.apply_despeckling_filter(
            sample_image, filter_type="invalid_filter"
        )
        
        assert filtered_image.shape == sample_image.shape
        assert "lee despeckling filter" in processor.processing_history[-1]
    
    def test_enhance_contrast_clahe(self, processor, sample_image):
        """Test CLAHE contrast enhancement."""
        enhanced_image = processor.enhance_contrast(
            sample_image, method="clahe", clip_limit=2.0
        )
        
        assert enhanced_image.shape == sample_image.shape
        assert "clahe contrast enhancement" in processor.processing_history[-1]
        
        # Enhanced image should have different statistics
        assert not np.array_equal(enhanced_image, sample_image)
    
    def test_enhance_contrast_histogram_equalization(self, processor, sample_image):
        """Test histogram equalization."""
        enhanced_image = processor.enhance_contrast(
            sample_image, method="histogram_equalization"
        )
        
        assert enhanced_image.shape == sample_image.shape
        assert "histogram_equalization contrast enhancement" in processor.processing_history[-1]
    
    def test_enhance_contrast_gamma(self, processor, sample_image):
        """Test gamma correction."""
        enhanced_image = processor.enhance_contrast(
            sample_image, method="gamma", gamma=1.5
        )
        
        assert enhanced_image.shape == sample_image.shape
        assert "gamma contrast enhancement" in processor.processing_history[-1]
    
    def test_apply_morphological_operations_opening(self, processor):
        """Test morphological opening operation."""
        # Create a binary mask
        binary_mask = np.zeros((50, 50), dtype=np.uint8)
        binary_mask[20:30, 20:30] = 1  # Small square
        binary_mask[25:27, 25:27] = 0  # Small hole
        
        processed_mask = processor.apply_morphological_operations(
            binary_mask, operation="opening", kernel_size=3
        )
        
        assert processed_mask.shape == binary_mask.shape
        assert processed_mask.dtype == np.uint8
        assert "opening morphological operation" in processor.processing_history[-1]
    
    def test_apply_morphological_operations_closing(self, processor):
        """Test morphological closing operation."""
        binary_mask = np.zeros((50, 50), dtype=np.uint8)
        binary_mask[20:30, 20:30] = 1
        binary_mask[25:27, 25:27] = 0  # Small hole
        
        processed_mask = processor.apply_morphological_operations(
            binary_mask, operation="closing", kernel_size=3
        )
        
        assert processed_mask.shape == binary_mask.shape
        assert "closing morphological operation" in processor.processing_history[-1]
    
    def test_remove_small_objects(self, processor):
        """Test removal of small connected components."""
        # Create binary mask with small and large objects
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[10:50, 10:50] = 1  # Large object (40x40 = 1600 pixels)
        binary_mask[70:72, 70:72] = 1  # Small object (2x2 = 4 pixels)
        
        cleaned_mask = processor.remove_small_objects(
            binary_mask, minimum_size=100
        )
        
        assert cleaned_mask.shape == binary_mask.shape
        assert np.sum(cleaned_mask) < np.sum(binary_mask)  # Small objects removed
        assert "Removed objects smaller than 100 pixels" in processor.processing_history[-1]
    
    def test_resize_image(self, processor, sample_image):
        """Test image resizing."""
        target_size = (64, 64)
        resized_image = processor.resize_image(
            sample_image, target_size=target_size
        )
        
        assert resized_image.shape == target_size
        assert f"Resized image to {target_size}" in processor.processing_history[-1]
    
    def test_resize_image_default_size(self, processor, sample_image):
        """Test resizing with default target size."""
        resized_image = processor.resize_image(sample_image)
        
        assert resized_image.shape == processor.target_size
    
    def test_normalize_intensity_minmax(self, processor, sample_image):
        """Test min-max normalization."""
        normalized_image = processor.normalize_intensity(
            sample_image, method="minmax"
        )
        
        assert normalized_image.shape == sample_image.shape
        assert normalized_image.min() >= 0.0
        assert normalized_image.max() <= 1.0
        assert "minmax normalization" in processor.processing_history[-1]
    
    def test_normalize_intensity_zscore(self, processor, sample_image):
        """Test z-score normalization."""
        normalized_image = processor.normalize_intensity(
            sample_image, method="zscore"
        )
        
        assert normalized_image.shape == sample_image.shape
        assert abs(np.mean(normalized_image)) < 1e-6  # Mean should be ~0
        assert "zscore normalization" in processor.processing_history[-1]
    
    def test_normalize_intensity_percentile(self, processor, sample_image):
        """Test percentile normalization."""
        normalized_image = processor.normalize_intensity(
            sample_image, method="percentile", percentile_range=(5, 95)
        )
        
        assert normalized_image.shape == sample_image.shape
        assert normalized_image.min() >= 0.0
        assert normalized_image.max() <= 1.0
        assert "percentile normalization" in processor.processing_history[-1]
    
    def test_processing_history_management(self, processor, sample_image):
        """Test processing history tracking and management."""
        # Perform several operations
        processor.apply_despeckling_filter(sample_image, "lee")
        processor.enhance_contrast(sample_image, "clahe")
        processor.resize_image(sample_image)
        
        history = processor.get_processing_summary()
        assert len(history) == 3
        
        # Reset history
        processor.reset_processing_history()
        assert len(processor.get_processing_summary()) == 0
    
    def test_full_processing_pipeline(self, processor, sample_image):
        """Test a complete processing pipeline."""
        # Simulate a full processing workflow
        processed_image = sample_image.copy()
        
        # Step 1: Despeckling
        processed_image = processor.apply_despeckling_filter(
            processed_image, filter_type="lee", window_size=5
        )
        
        # Step 2: Contrast enhancement
        processed_image = processor.enhance_contrast(
            processed_image, method="clahe", clip_limit=3.0
        )
        
        # Step 3: Resize
        processed_image = processor.resize_image(processed_image)
        
        # Step 4: Normalize
        processed_image = processor.normalize_intensity(
            processed_image, method="percentile"
        )
        
        # Verify final result
        assert processed_image.shape == processor.target_size
        assert processed_image.dtype == np.float32
        assert processed_image.min() >= 0.0
        assert processed_image.max() <= 1.0
        
        # Check that all steps were recorded
        history = processor.get_processing_summary()
        assert len(history) == 4
        
    def test_error_handling_invalid_image(self, processor):
        """Test error handling with invalid image input."""
        invalid_image = None
        
        # These should handle errors gracefully
        result = processor.enhance_contrast(np.array([]))
        assert result.shape == (0,)  # Should return the empty array
        
        result = processor.resize_image(np.array([]))
        assert result.shape == (0,)  # Should return the empty array
    
    @pytest.mark.parametrize("filter_type", ["lee", "frost", "kuan", "bilateral"])
    def test_all_despeckling_filters(self, processor, sample_image, filter_type):
        """Test all available despeckling filters."""
        filtered_image = processor.apply_despeckling_filter(
            sample_image, filter_type=filter_type
        )
        
        assert filtered_image.shape == sample_image.shape
        assert f"{filter_type} despeckling filter" in processor.processing_history[-1]
    
    @pytest.mark.parametrize("enhancement_method", ["clahe", "histogram_equalization", "gamma"])
    def test_all_enhancement_methods(self, processor, sample_image, enhancement_method):
        """Test all available contrast enhancement methods."""
        enhanced_image = processor.enhance_contrast(
            sample_image, method=enhancement_method
        )
        
        assert enhanced_image.shape == sample_image.shape
        assert f"{enhancement_method} contrast enhancement" in processor.processing_history[-1]


if __name__ == "__main__":
    pytest.main([__file__])
