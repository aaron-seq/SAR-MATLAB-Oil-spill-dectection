"""
Performance Evaluation Module for SAR Oil Spill Detection
Provides comprehensive metrics for segmentation quality assessment.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Comprehensive evaluation metrics for oil spill segmentation.
    Includes pixel-wise metrics, object-level metrics, and SAR-specific measures.
    """
    
    def __init__(self):
        """Initialize the performance evaluator."""
        self.metrics_cache = {}
    
    @staticmethod
    def calculate_segmentation_metrics(predicted_mask: np.ndarray, 
                                      ground_truth_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive segmentation metrics.
        
        Args:
            predicted_mask: Binary predicted mask (H, W) with values 0 or 1
            ground_truth_mask: Binary ground truth mask (H, W) with values 0 or 1
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Ensure binary masks
        pred = (predicted_mask > 0.5).astype(np.uint8).flatten()
        true = (ground_truth_mask > 0.5).astype(np.uint8).flatten()
        
        # Basic confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
        
        # Pixel-wise metrics
        pixel_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Intersection over Union (IoU / Jaccard Index)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        
        # Dice Coefficient (F1 Score)
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0
        
        # Boundary-based metrics
        boundary_f1 = PerformanceEvaluator._calculate_boundary_f1(
            predicted_mask, ground_truth_mask
        )
        
        # Area metrics
        pred_area = np.sum(predicted_mask > 0.5)
        true_area = np.sum(ground_truth_mask > 0.5)
        area_error = abs(pred_area - true_area) / true_area if true_area > 0 else float('inf')
        
        # Object-level metrics
        object_metrics = PerformanceEvaluator._calculate_object_metrics(
            predicted_mask, ground_truth_mask
        )
        
        metrics = {
            # Core segmentation metrics
            'jaccard_index': float(iou),
            'dice_coefficient': float(dice),
            'pixel_accuracy': float(pixel_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'specificity': float(specificity),
            
            # Additional metrics
            'matthews_correlation': float(mcc),
            'boundary_f1': float(boundary_f1),
            
            # Area metrics
            'predicted_area': int(pred_area),
            'ground_truth_area': int(true_area),
            'area_error': float(area_error),
            
            # Confusion matrix
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Object metrics
            'object_detection_rate': object_metrics['detection_rate'],
            'false_positive_rate': object_metrics['fp_rate'],
            'hausdorff_distance': object_metrics['hausdorff']
        }
        
        return metrics
    
    @staticmethod
    def _calculate_boundary_f1(pred_mask: np.ndarray, 
                               true_mask: np.ndarray, 
                               threshold: float = 2.0) -> float:
        """
        Calculate boundary F1 score.
        
        Args:
            pred_mask: Predicted binary mask
            true_mask: Ground truth binary mask
            threshold: Distance threshold for boundary matching
        
        Returns:
            Boundary F1 score
        """
        try:
            from scipy.ndimage import distance_transform_edt
            
            # Get boundaries
            pred_boundary = PerformanceEvaluator._get_boundary(pred_mask)
            true_boundary = PerformanceEvaluator._get_boundary(true_mask)
            
            if not (np.any(pred_boundary) and np.any(true_boundary)):
                return 0.0
            
            # Distance transforms
            pred_dist = distance_transform_edt(~pred_boundary)
            true_dist = distance_transform_edt(~true_boundary)
            
            # Precision: how many predicted boundary pixels are close to true boundary
            pred_points = np.argwhere(pred_boundary)
            if len(pred_points) == 0:
                boundary_precision = 0.0
            else:
                boundary_precision = np.mean(true_dist[pred_boundary] <= threshold)
            
            # Recall: how many true boundary pixels are close to predicted boundary
            true_points = np.argwhere(true_boundary)
            if len(true_points) == 0:
                boundary_recall = 0.0
            else:
                boundary_recall = np.mean(pred_dist[true_boundary] <= threshold)
            
            # F1 score
            if boundary_precision + boundary_recall > 0:
                boundary_f1 = 2 * (boundary_precision * boundary_recall) / \
                             (boundary_precision + boundary_recall)
            else:
                boundary_f1 = 0.0
            
            return boundary_f1
            
        except Exception as e:
            logger.warning(f"Boundary F1 calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def _get_boundary(mask: np.ndarray) -> np.ndarray:
        """
        Extract boundary pixels from binary mask using morphological operations.
        
        Args:
            mask: Binary mask
        
        Returns:
            Boundary mask
        """
        from scipy.ndimage import binary_erosion
        
        binary_mask = mask > 0.5
        eroded = binary_erosion(binary_mask)
        boundary = binary_mask & ~eroded
        
        return boundary
    
    @staticmethod
    def _calculate_object_metrics(pred_mask: np.ndarray, 
                                 true_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate object-level detection metrics.
        
        Args:
            pred_mask: Predicted binary mask
            true_mask: Ground truth binary mask
        
        Returns:
            Dictionary with object-level metrics
        """
        try:
            from scipy.ndimage import label
            from scipy.spatial.distance import directed_hausdorff
            
            # Label connected components
            true_labels, true_num = label(true_mask > 0.5)
            pred_labels, pred_num = label(pred_mask > 0.5)
            
            if true_num == 0:
                return {
                    'detection_rate': 0.0,
                    'fp_rate': 1.0 if pred_num > 0 else 0.0,
                    'hausdorff': 0.0
                }
            
            # Detection rate: fraction of true objects that overlap with predictions
            detected = 0
            for i in range(1, true_num + 1):
                true_obj = (true_labels == i)
                if np.any(true_obj & (pred_mask > 0.5)):
                    detected += 1
            
            detection_rate = detected / true_num
            
            # False positive rate
            fp_rate = max(0, pred_num - detected) / max(1, pred_num)
            
            # Hausdorff distance
            true_points = np.argwhere(true_mask > 0.5)
            pred_points = np.argwhere(pred_mask > 0.5)
            
            if len(true_points) > 0 and len(pred_points) > 0:
                hausdorff = max(
                    directed_hausdorff(true_points, pred_points)[0],
                    directed_hausdorff(pred_points, true_points)[0]
                )
            else:
                hausdorff = float('inf')
            
            return {
                'detection_rate': detection_rate,
                'fp_rate': fp_rate,
                'hausdorff': float(hausdorff) if hausdorff != float('inf') else 999.9
            }
            
        except Exception as e:
            logger.warning(f"Object metrics calculation failed: {e}")
            return {
                'detection_rate': 0.0,
                'fp_rate': 0.0,
                'hausdorff': 0.0
            }
    
    @staticmethod
    def generate_evaluation_report(metrics: Dict[str, float], 
                                  image_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of calculated metrics
            image_info: Optional image metadata
        
        Returns:
            Formatted evaluation report
        """
        report = {
            'summary': {
                'overall_performance': 'Excellent' if metrics.get('dice_coefficient', 0) > 0.85 
                                      else 'Good' if metrics.get('dice_coefficient', 0) > 0.70
                                      else 'Fair' if metrics.get('dice_coefficient', 0) > 0.50
                                      else 'Poor',
                'dice_score': metrics.get('dice_coefficient', 0),
                'iou_score': metrics.get('jaccard_index', 0),
                'pixel_accuracy': metrics.get('pixel_accuracy', 0)
            },
            'detailed_metrics': metrics,
            'recommendations': []
        }
        
        # Add recommendations based on metrics
        if metrics.get('precision', 0) < 0.7:
            report['recommendations'].append(
                "Low precision detected. Consider increasing confidence threshold to reduce false positives."
            )
        
        if metrics.get('recall', 0) < 0.7:
            report['recommendations'].append(
                "Low recall detected. Model may be missing oil spills. Consider retraining with more data."
            )
        
        if metrics.get('boundary_f1', 0) < 0.6:
            report['recommendations'].append(
                "Boundary detection could be improved. Consider post-processing with morphological operations."
            )
        
        if image_info:
            report['image_info'] = image_info
        
        return report
