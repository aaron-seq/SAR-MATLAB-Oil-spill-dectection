"""
Performance Evaluation Module for SAR Oil Spill Detection
Provides comprehensive metrics for segmentation quality assessment.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from skimage import morphology, measure

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for oil spill detection.
    Provides 15+ metrics including IoU, Dice, boundary metrics, and SAR-specific measures.
    """
    
    def __init__(self):
        """Initialize the performance evaluator."""
        self.metrics_history = []
        logger.info("PerformanceEvaluator initialized")
    
    def calculate_segmentation_metrics(
        self, 
        predicted_mask: np.ndarray, 
        ground_truth_mask: np.ndarray,
        pixel_spacing: Optional[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive segmentation metrics.
        
        Args:
            predicted_mask: Binary prediction mask (H, W)
            ground_truth_mask: Binary ground truth mask (H, W)
            pixel_spacing: Physical pixel spacing (dy, dx) for area calculations
            
        Returns:
            Dictionary containing all computed metrics
        """
        try:
            # Ensure binary masks
            pred = (predicted_mask > 0.5).astype(np.uint8)
            gt = (ground_truth_mask > 0.5).astype(np.uint8)
            
            # Core segmentation metrics
            metrics = {}
            
            # Intersection and Union
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            pred_sum = pred.sum()
            gt_sum = gt.sum()
            
            # IoU (Jaccard Index)
            metrics['jaccard_index'] = float(intersection / union) if union > 0 else 0.0
            metrics['iou'] = metrics['jaccard_index']  # Alias
            
            # Dice Coefficient (F1 Score for segmentation)
            metrics['dice_coefficient'] = float(2 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else 0.0
            
            # Pixel Accuracy
            correct_pixels = np.sum(pred == gt)
            total_pixels = pred.size
            metrics['pixel_accuracy'] = float(correct_pixels / total_pixels)
            
            # True Positives, False Positives, True Negatives, False Negatives
            tp = intersection
            fp = pred_sum - intersection
            tn = np.sum(np.logical_and(pred == 0, gt == 0))
            fn = gt_sum - intersection
            
            # Precision, Recall, F1
            metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics['sensitivity'] = metrics['recall']  # Alias
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0.0
            
            # Specificity
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            # Boundary-based metrics
            boundary_metrics = self._calculate_boundary_metrics(pred, gt)
            metrics.update(boundary_metrics)
            
            # Object-level metrics
            object_metrics = self._calculate_object_metrics(pred, gt)
            metrics.update(object_metrics)
            
            # Area-based metrics
            if pixel_spacing is not None:
                area_metrics = self._calculate_area_metrics(pred, gt, pixel_spacing)
                metrics.update(area_metrics)
            
            # Shape similarity
            shape_metrics = self._calculate_shape_metrics(pred, gt)
            metrics.update(shape_metrics)
            
            self.metrics_history.append(metrics)
            logger.debug(f"Calculated {len(metrics)} metrics")
            
            return metrics
            
        except Exception as error:
            logger.error(f"Error calculating metrics: {error}")
            return {"error": str(error)}
    
    def _calculate_boundary_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Calculate boundary-based evaluation metrics.
        
        Args:
            pred: Binary prediction mask
            gt: Binary ground truth mask
            
        Returns:
            Dictionary with boundary metrics
        """
        metrics = {}
        
        try:
            # Extract boundaries
            pred_boundary = morphology.binary_erosion(pred) ^ pred
            gt_boundary = morphology.binary_erosion(gt) ^ gt
            
            # Boundary Precision and Recall
            pred_boundary_pixels = np.sum(pred_boundary)
            gt_boundary_pixels = np.sum(gt_boundary)
            
            if pred_boundary_pixels == 0 or gt_boundary_pixels == 0:
                metrics['boundary_precision'] = 0.0
                metrics['boundary_recall'] = 0.0
                metrics['boundary_f1'] = 0.0
            else:
                # Dilate boundaries slightly for tolerance
                tolerance = morphology.disk(2)
                pred_boundary_dilated = morphology.binary_dilation(pred_boundary, tolerance)
                gt_boundary_dilated = morphology.binary_dilation(gt_boundary, tolerance)
                
                # Calculate boundary matches
                boundary_tp = np.sum(pred_boundary & gt_boundary_dilated)
                boundary_precision = boundary_tp / pred_boundary_pixels
                boundary_recall = np.sum(gt_boundary & pred_boundary_dilated) / gt_boundary_pixels
                
                metrics['boundary_precision'] = float(boundary_precision)
                metrics['boundary_recall'] = float(boundary_recall)
                
                if boundary_precision + boundary_recall > 0:
                    metrics['boundary_f1'] = 2 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
                else:
                    metrics['boundary_f1'] = 0.0
            
            # Hausdorff Distance
            if np.any(pred) and np.any(gt):
                pred_points = np.argwhere(pred)
                gt_points = np.argwhere(gt)
                
                hausdorff_dist = max(
                    directed_hausdorff(pred_points, gt_points)[0],
                    directed_hausdorff(gt_points, pred_points)[0]
                )
                metrics['hausdorff_distance'] = float(hausdorff_dist)
            else:
                metrics['hausdorff_distance'] = float('inf')
            
        except Exception as error:
            logger.warning(f"Error calculating boundary metrics: {error}")
            metrics['boundary_precision'] = 0.0
            metrics['boundary_recall'] = 0.0
            metrics['boundary_f1'] = 0.0
            metrics['hausdorff_distance'] = float('inf')
        
        return metrics
    
    def _calculate_object_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Calculate object-level detection metrics.
        
        Args:
            pred: Binary prediction mask
            gt: Binary ground truth mask
            
        Returns:
            Dictionary with object-level metrics
        """
        metrics = {}
        
        try:
            # Label connected components
            pred_labeled = measure.label(pred, connectivity=2)
            gt_labeled = measure.label(gt, connectivity=2)
            
            pred_objects = pred_labeled.max()
            gt_objects = gt_labeled.max()
            
            metrics['predicted_objects'] = int(pred_objects)
            metrics['ground_truth_objects'] = int(gt_objects)
            
            # Count correctly detected objects (IoU > 0.5 threshold)
            detected_objects = 0
            if gt_objects > 0:
                for gt_label in range(1, gt_objects + 1):
                    gt_obj_mask = (gt_labeled == gt_label)
                    
                    # Find overlapping predicted objects
                    overlapping_labels = np.unique(pred_labeled[gt_obj_mask])
                    overlapping_labels = overlapping_labels[overlapping_labels > 0]
                    
                    for pred_label in overlapping_labels:
                        pred_obj_mask = (pred_labeled == pred_label)
                        
                        # Calculate IoU for this pair
                        obj_intersection = np.logical_and(pred_obj_mask, gt_obj_mask).sum()
                        obj_union = np.logical_or(pred_obj_mask, gt_obj_mask).sum()
                        obj_iou = obj_intersection / obj_union if obj_union > 0 else 0
                        
                        if obj_iou > 0.5:
                            detected_objects += 1
                            break
            
            metrics['detected_objects'] = detected_objects
            metrics['object_detection_rate'] = float(detected_objects / gt_objects) if gt_objects > 0 else 0.0
            
            # False positive rate (predicted objects not matching ground truth)
            false_positive_objects = max(0, pred_objects - detected_objects)
            metrics['false_positive_objects'] = int(false_positive_objects)
            
        except Exception as error:
            logger.warning(f"Error calculating object metrics: {error}")
            metrics['predicted_objects'] = 0
            metrics['ground_truth_objects'] = 0
            metrics['detected_objects'] = 0
            metrics['object_detection_rate'] = 0.0
        
        return metrics
    
    def _calculate_area_metrics(self, pred: np.ndarray, gt: np.ndarray, 
                               pixel_spacing: Tuple[float, float]) -> Dict[str, float]:
        """
        Calculate area-based metrics with physical units.
        
        Args:
            pred: Binary prediction mask
            gt: Binary ground truth mask
            pixel_spacing: Physical pixel spacing (dy, dx)
            
        Returns:
            Dictionary with area metrics
        """
        metrics = {}
        
        try:
            pixel_area = pixel_spacing[0] * pixel_spacing[1]
            
            pred_area = np.sum(pred) * pixel_area
            gt_area = np.sum(gt) * pixel_area
            
            metrics['predicted_area'] = float(pred_area)
            metrics['ground_truth_area'] = float(gt_area)
            
            # Area estimation error
            area_error = abs(pred_area - gt_area)
            metrics['area_error'] = float(area_error)
            metrics['area_error_percentage'] = float(area_error / gt_area * 100) if gt_area > 0 else 0.0
            
        except Exception as error:
            logger.warning(f"Error calculating area metrics: {error}")
        
        return metrics
    
    def _calculate_shape_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Calculate shape similarity metrics.
        
        Args:
            pred: Binary prediction mask
            gt: Binary ground truth mask
            
        Returns:
            Dictionary with shape metrics
        """
        metrics = {}
        
        try:
            # Get region properties
            pred_props = measure.regionprops(measure.label(pred))
            gt_props = measure.regionprops(measure.label(gt))
            
            if len(pred_props) > 0 and len(gt_props) > 0:
                # Use largest object for comparison
                pred_largest = max(pred_props, key=lambda x: x.area)
                gt_largest = max(gt_props, key=lambda x: x.area)
                
                # Solidity (convexity)
                pred_solidity = pred_largest.solidity
                gt_solidity = gt_largest.solidity
                metrics['shape_solidity_error'] = abs(pred_solidity - gt_solidity)
                
                # Eccentricity
                pred_eccentricity = pred_largest.eccentricity
                gt_eccentricity = gt_largest.eccentricity
                metrics['shape_eccentricity_error'] = abs(pred_eccentricity - gt_eccentricity)
                
        except Exception as error:
            logger.warning(f"Error calculating shape metrics: {error}")
        
        return metrics
    
    def generate_evaluation_report(
        self, 
        metrics: Dict[str, float],
        image_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Calculated metrics dictionary
            image_info: Optional image metadata
            
        Returns:
            Structured evaluation report
        """
        try:
            report = {
                "summary": {
                    "overall_performance": self._categorize_performance(metrics),
                    "primary_metrics": {
                        "iou": metrics.get('iou', 0.0),
                        "dice": metrics.get('dice_coefficient', 0.0),
                        "f1_score": metrics.get('f1_score', 0.0)
                    }
                },
                "segmentation_quality": {
                    "pixel_level": {
                        "accuracy": metrics.get('pixel_accuracy', 0.0),
                        "precision": metrics.get('precision', 0.0),
                        "recall": metrics.get('recall', 0.0),
                        "specificity": metrics.get('specificity', 0.0)
                    },
                    "boundary_accuracy": {
                        "boundary_f1": metrics.get('boundary_f1', 0.0),
                        "hausdorff_distance": metrics.get('hausdorff_distance', 0.0)
                    }
                },
                "object_detection": {
                    "detected_objects": metrics.get('detected_objects', 0),
                    "ground_truth_objects": metrics.get('ground_truth_objects', 0),
                    "detection_rate": metrics.get('object_detection_rate', 0.0),
                    "false_positives": metrics.get('false_positive_objects', 0)
                },
                "all_metrics": metrics
            }
            
            if image_info:
                report["image_info"] = image_info
            
            logger.debug("Generated evaluation report")
            return report
            
        except Exception as error:
            logger.error(f"Error generating report: {error}")
            return {"error": str(error), "metrics": metrics}
    
    def _categorize_performance(self, metrics: Dict[str, float]) -> str:
        """
        Categorize overall performance based on IoU.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Performance category string
        """
        iou = metrics.get('iou', 0.0)
        
        if iou >= 0.8:
            return "excellent"
        elif iou >= 0.65:
            return "good"
        elif iou >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def get_metrics_history(self) -> list:
        """
        Get history of all calculated metrics.
        
        Returns:
            List of metrics dictionaries
        """
        return self.metrics_history.copy()
    
    def reset_history(self) -> None:
        """
        Clear metrics history.
        """
        self.metrics_history.clear()
        logger.debug("Metrics history reset")
