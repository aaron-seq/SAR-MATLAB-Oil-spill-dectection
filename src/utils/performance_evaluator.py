"""
Performance Evaluation Module for Segmentation Metrics
Provides comprehensive metrics for SAR oil spill detection evaluation.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """Evaluates segmentation performance with multiple metrics."""
    
    def __init__(self):
        """Initialize the performance evaluator."""
        self.epsilon = 1e-7  # Small constant to avoid division by zero
    
    def calculate_segmentation_metrics(self, 
                                      predicted_mask: np.ndarray,
                                      ground_truth_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive segmentation metrics.
        
        Args:
            predicted_mask: Binary prediction mask (values 0 or 1)
            ground_truth_mask: Binary ground truth mask (values 0 or 1)
        
        Returns:
            Dictionary containing various metrics
        """
        try:
            # Ensure binary masks
            pred = (predicted_mask > 0.5).astype(np.uint8)
            gt = (ground_truth_mask > 0.5).astype(np.uint8)
            
            # Flatten for pixel-wise metrics
            pred_flat = pred.flatten()
            gt_flat = gt.flatten()
            
            # True Positive, False Positive, True Negative, False Negative
            tp = np.sum((pred_flat == 1) & (gt_flat == 1))
            fp = np.sum((pred_flat == 1) & (gt_flat == 0))
            tn = np.sum((pred_flat == 0) & (gt_flat == 0))
            fn = np.sum((pred_flat == 0) & (gt_flat == 1))
            
            # Core metrics
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + self.epsilon)
            accuracy = (tp + tn) / (tp + fp + tn + fn + self.epsilon)
            
            # IoU (Jaccard Index)
            intersection = np.sum(pred_flat & gt_flat)
            union = np.sum(pred_flat | gt_flat)
            iou = intersection / (union + self.epsilon)
            
            # Dice Coefficient (F1 Score alternative formulation)
            dice = (2 * intersection) / (np.sum(pred_flat) + np.sum(gt_flat) + self.epsilon)
            
            # Specificity
            specificity = tn / (tn + fp + self.epsilon)
            
            # Boundary-based metrics
            boundary_f1 = self._calculate_boundary_f1(pred, gt)
            hausdorff_dist = self._calculate_hausdorff_distance(pred, gt)
            
            # Area-based metrics
            pred_area = np.sum(pred)
            gt_area = np.sum(gt)
            area_error = abs(pred_area - gt_area) / (gt_area + self.epsilon)
            
            metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'accuracy': float(accuracy),
                'jaccard_index': float(iou),  # IoU
                'dice_coefficient': float(dice),
                'specificity': float(specificity),
                'boundary_f1': float(boundary_f1),
                'hausdorff_distance': float(hausdorff_dist),
                'area_estimation_error': float(area_error),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
            
            logger.info(f"Metrics calculated - IoU: {iou:.3f}, Dice: {dice:.3f}, F1: {f1_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            raise
    
    def _calculate_boundary_f1(self, pred: np.ndarray, gt: np.ndarray, 
                              tolerance: int = 2) -> float:
        """
        Calculate boundary F1 score with tolerance for boundary pixels.
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            tolerance: Pixel tolerance for boundary matching
        
        Returns:
            Boundary F1 score
        """
        try:
            from scipy.ndimage import binary_erosion, binary_dilation
            
            # Extract boundaries
            pred_boundary = pred - binary_erosion(pred)
            gt_boundary = gt - binary_erosion(gt)
            
            # Dilate GT boundary for tolerance
            gt_boundary_dilated = binary_dilation(gt_boundary, iterations=tolerance)
            pred_boundary_dilated = binary_dilation(pred_boundary, iterations=tolerance)
            
            # Calculate boundary precision and recall
            tp_boundary = np.sum(pred_boundary & gt_boundary_dilated)
            fp_boundary = np.sum(pred_boundary & ~gt_boundary_dilated)
            fn_boundary = np.sum(gt_boundary & ~pred_boundary_dilated)
            
            precision_b = tp_boundary / (tp_boundary + fp_boundary + self.epsilon)
            recall_b = tp_boundary / (tp_boundary + fn_boundary + self.epsilon)
            
            boundary_f1 = 2 * (precision_b * recall_b) / (precision_b + recall_b + self.epsilon)
            
            return boundary_f1
            
        except Exception as e:
            logger.warning(f"Boundary F1 calculation failed: {e}")
            return 0.0
    
    def _calculate_hausdorff_distance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Hausdorff distance between prediction and ground truth.
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
        
        Returns:
            Hausdorff distance in pixels
        """
        try:
            # Compute distance transforms
            dist_pred = distance_transform_edt(~pred.astype(bool))
            dist_gt = distance_transform_edt(~gt.astype(bool))
            
            # Hausdorff distance is max of min distances
            hausdorff = max(
                np.max(dist_pred[gt.astype(bool)]) if np.any(gt) else 0,
                np.max(dist_gt[pred.astype(bool)]) if np.any(pred) else 0
            )
            
            return hausdorff
            
        except Exception as e:
            logger.warning(f"Hausdorff distance calculation failed: {e}")
            return float('inf')
    
    def generate_evaluation_report(self, 
                                  metrics: Dict[str, float],
                                  image_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of calculated metrics
            image_info: Optional image metadata
        
        Returns:
            Formatted evaluation report
        """
        try:
            report = {
                'summary': {
                    'overall_performance': self._categorize_performance(metrics['f1_score']),
                    'primary_metrics': {
                        'IoU': metrics['jaccard_index'],
                        'Dice': metrics['dice_coefficient'],
                        'F1_Score': metrics['f1_score'],
                        'Accuracy': metrics['accuracy']
                    }
                },
                'detailed_metrics': metrics,
                'interpretation': self._generate_interpretation(metrics)
            }
            
            if image_info:
                report['image_info'] = image_info
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': str(e), 'metrics': metrics}
    
    def _categorize_performance(self, f1_score: float) -> str:
        """Categorize model performance based on F1 score."""
        if f1_score >= 0.9:
            return 'Excellent'
        elif f1_score >= 0.8:
            return 'Good'
        elif f1_score >= 0.7:
            return 'Fair'
        elif f1_score >= 0.5:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _generate_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable interpretation of metrics."""
        interpretation = {}
        
        # Precision interpretation
        if metrics['precision'] >= 0.9:
            interpretation['precision'] = "High precision - few false positives"
        elif metrics['precision'] < 0.7:
            interpretation['precision'] = "Low precision - many false alarms"
        
        # Recall interpretation
        if metrics['recall'] >= 0.9:
            interpretation['recall'] = "High recall - detecting most oil spills"
        elif metrics['recall'] < 0.7:
            interpretation['recall'] = "Low recall - missing many oil spills"
        
        # IoU interpretation
        if metrics['jaccard_index'] >= 0.8:
            interpretation['iou'] = "Strong spatial overlap with ground truth"
        elif metrics['jaccard_index'] < 0.5:
            interpretation['iou'] = "Weak spatial overlap - poor localization"
        
        # Boundary quality
        if metrics['boundary_f1'] >= 0.8:
            interpretation['boundary'] = "Accurate boundary detection"
        elif metrics['boundary_f1'] < 0.6:
            interpretation['boundary'] = "Poor boundary alignment"
        
        return interpretation
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Compare performance of multiple models.
        
        Args:
            results: Dictionary mapping model names to their metrics
        
        Returns:
            Comparison report with rankings
        """
        try:
            comparison = {
                'models': list(results.keys()),
                'rankings': {},
                'best_model': None,
                'metric_comparison': {}
            }
            
            # Rank by each metric
            key_metrics = ['jaccard_index', 'dice_coefficient', 'f1_score', 'accuracy']
            
            for metric in key_metrics:
                sorted_models = sorted(
                    results.items(),
                    key=lambda x: x[1].get(metric, 0),
                    reverse=True
                )
                comparison['rankings'][metric] = [m[0] for m in sorted_models]
            
            # Determine best overall model (by average rank)
            rank_scores = {model: 0 for model in results.keys()}
            for rankings in comparison['rankings'].values():
                for idx, model in enumerate(rankings):
                    rank_scores[model] += idx
            
            comparison['best_model'] = min(rank_scores.items(), key=lambda x: x[1])[0]
            
            # Metric comparison table
            for metric in key_metrics:
                comparison['metric_comparison'][metric] = {
                    model: results[model].get(metric, 0) for model in results.keys()
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {'error': str(e)}
