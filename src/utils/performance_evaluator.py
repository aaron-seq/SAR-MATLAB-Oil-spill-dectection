"""
Comprehensive Performance Evaluation Module for SAR Oil Spill Segmentation
Provides multiple metrics and visualization capabilities for model assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve, 
    roc_curve, auc, jaccard_score, f1_score
)
from scipy import ndimage
from skimage.measure import regionprops, label

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Comprehensive evaluation suite for oil spill segmentation performance.
    Includes standard metrics plus SAR-specific evaluation criteria.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['Sea/Water', 'Oil Spill', 'Land']
        self.evaluation_history = []
        
        logger.info("Performance Evaluator initialized")
    
    def calculate_segmentation_metrics(self, predicted_mask: np.ndarray, 
                                     ground_truth_mask: np.ndarray,
                                     class_of_interest: int = 1) -> Dict[str, float]:
        """
        Calculate comprehensive segmentation metrics.
        
        Args:
            predicted_mask: Predicted segmentation mask
            ground_truth_mask: Ground truth segmentation mask
            class_of_interest: Class to focus evaluation on (1 = oil spill)
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        try:
            # Flatten masks for sklearn metrics
            y_true = ground_truth_mask.flatten()
            y_pred = predicted_mask.flatten()
            
            # Binary masks for class of interest
            y_true_binary = (y_true == class_of_interest).astype(int)
            y_pred_binary = (y_pred == class_of_interest).astype(int)
            
            metrics = {}
            
            # Standard segmentation metrics
            metrics['jaccard_index'] = self._calculate_jaccard_index(y_pred_binary, y_true_binary)
            metrics['dice_coefficient'] = self._calculate_dice_coefficient(y_pred_binary, y_true_binary)
            metrics['pixel_accuracy'] = self._calculate_pixel_accuracy(y_pred, y_true)
            
            # Precision, Recall, F1-Score
            metrics['precision'] = self._calculate_precision(y_pred_binary, y_true_binary)
            metrics['recall'] = self._calculate_recall(y_pred_binary, y_true_binary)
            metrics['f1_score'] = self._calculate_f1_score(y_pred_binary, y_true_binary)
            
            # Boundary-based metrics
            metrics['boundary_f1'] = self._calculate_boundary_f1(
                predicted_mask == class_of_interest,
                ground_truth_mask == class_of_interest
            )
            
            # Object-level metrics
            object_metrics = self._calculate_object_level_metrics(
                predicted_mask == class_of_interest,
                ground_truth_mask == class_of_interest
            )
            metrics.update(object_metrics)
            
            # SAR-specific metrics
            sar_metrics = self._calculate_sar_specific_metrics(
                predicted_mask, ground_truth_mask, class_of_interest
            )
            metrics.update(sar_metrics)
            
            self.evaluation_history.append(metrics)
            logger.info(f"Calculated {len(metrics)} evaluation metrics")
            
            return metrics
            
        except Exception as error:
            logger.error(f"Error calculating metrics: {error}")
            return {}
    
    def _calculate_jaccard_index(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Jaccard Index (IoU)."""
        intersection = np.logical_and(y_pred, y_true).sum()
        union = np.logical_or(y_pred, y_true).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float(intersection) / float(union)
    
    def _calculate_dice_coefficient(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Dice Coefficient (F1 for segmentation)."""
        intersection = np.logical_and(y_pred, y_true).sum()
        total = y_pred.sum() + y_true.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * float(intersection) / float(total)
    
    def _calculate_pixel_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate overall pixel accuracy."""
        return float(np.sum(y_pred == y_true)) / float(len(y_true))
    
    def _calculate_precision(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate precision for binary classification."""
        true_positives = np.logical_and(y_pred == 1, y_true == 1).sum()
        predicted_positives = (y_pred == 1).sum()
        
        if predicted_positives == 0:
            return 0.0
        
        return float(true_positives) / float(predicted_positives)
    
    def _calculate_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate recall (sensitivity) for binary classification."""
        true_positives = np.logical_and(y_pred == 1, y_true == 1).sum()
        actual_positives = (y_true == 1).sum()
        
        if actual_positives == 0:
            return 0.0
        
        return float(true_positives) / float(actual_positives)
    
    def _calculate_f1_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate F1 score."""
        precision = self._calculate_precision(y_pred, y_true)
        recall = self._calculate_recall(y_pred, y_true)
        
        if precision + recall == 0:
            return 0.0
        
        return 2.0 * (precision * recall) / (precision + recall)
    
    def _calculate_boundary_f1(self, pred_mask: np.ndarray, 
                              true_mask: np.ndarray, tolerance: int = 2) -> float:
        """Calculate boundary F1 score for edge accuracy."""
        try:
            # Extract boundaries
            pred_boundary = self._extract_boundary(pred_mask)
            true_boundary = self._extract_boundary(true_mask)
            
            # Calculate distance-tolerant matches
            pred_matched = self._match_boundaries(pred_boundary, true_boundary, tolerance)
            true_matched = self._match_boundaries(true_boundary, pred_boundary, tolerance)
            
            # Calculate precision and recall
            boundary_precision = np.sum(pred_matched) / (np.sum(pred_boundary) + 1e-8)
            boundary_recall = np.sum(true_matched) / (np.sum(true_boundary) + 1e-8)
            
            # Calculate F1
            if boundary_precision + boundary_recall == 0:
                return 0.0
            
            return 2.0 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
            
        except Exception as error:
            logger.warning(f"Error calculating boundary F1: {error}")
            return 0.0
    
    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary pixels from binary mask."""
        from skimage import morphology
        eroded = morphology.erosion(mask, morphology.disk(1))
        boundary = mask & (~eroded)
        return boundary
    
    def _match_boundaries(self, boundary1: np.ndarray, boundary2: np.ndarray, 
                         tolerance: int) -> np.ndarray:
        """Find boundary pixels within tolerance distance."""
        # Create distance map
        distance_map = ndimage.distance_transform_edt(~boundary2)
        
        # Find matches within tolerance
        matches = (distance_map <= tolerance) & boundary1
        return matches
    
    def _calculate_object_level_metrics(self, pred_mask: np.ndarray, 
                                       true_mask: np.ndarray) -> Dict[str, float]:
        """Calculate object-level detection metrics."""
        try:
            # Label connected components
            pred_labeled = label(pred_mask)
            true_labeled = label(true_mask)
            
            pred_objects = regionprops(pred_labeled)
            true_objects = regionprops(true_labeled)
            
            # Object detection metrics
            true_detections = 0
            false_positives = len(pred_objects)
            
            for true_obj in true_objects:
                true_coords = true_obj.coords
                
                # Check if any predicted object overlaps significantly
                for pred_obj in pred_objects:
                    pred_coords = pred_obj.coords
                    
                    # Calculate overlap
                    overlap = self._calculate_object_overlap(
                        true_coords, pred_coords, pred_mask.shape
                    )
                    
                    if overlap > 0.5:  # 50% overlap threshold
                        true_detections += 1
                        false_positives -= 1
                        break
            
            # Calculate object-level metrics
            total_true_objects = len(true_objects)
            object_precision = true_detections / (true_detections + false_positives) if (true_detections + false_positives) > 0 else 0.0
            object_recall = true_detections / total_true_objects if total_true_objects > 0 else 0.0
            object_f1 = 2.0 * (object_precision * object_recall) / (object_precision + object_recall) if (object_precision + object_recall) > 0 else 0.0
            
            return {
                'object_precision': object_precision,
                'object_recall': object_recall,
                'object_f1': object_f1,
                'detected_objects': true_detections,
                'false_positive_objects': false_positives,
                'total_true_objects': total_true_objects
            }
            
        except Exception as error:
            logger.warning(f"Error calculating object-level metrics: {error}")
            return {}
    
    def _calculate_object_overlap(self, coords1: np.ndarray, coords2: np.ndarray, 
                                 shape: Tuple[int, int]) -> float:
        """Calculate overlap between two sets of coordinates."""
        # Create masks from coordinates
        mask1 = np.zeros(shape, dtype=bool)
        mask2 = np.zeros(shape, dtype=bool)
        
        mask1[coords1[:, 0], coords1[:, 1]] = True
        mask2[coords2[:, 0], coords2[:, 1]] = True
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_sar_specific_metrics(self, pred_mask: np.ndarray, 
                                       true_mask: np.ndarray, 
                                       oil_class: int) -> Dict[str, float]:
        """Calculate SAR-specific evaluation metrics."""
        try:
            # Oil spill area estimation accuracy
            pred_oil_area = np.sum(pred_mask == oil_class)
            true_oil_area = np.sum(true_mask == oil_class)
            
            area_estimation_error = abs(pred_oil_area - true_oil_area) / (true_oil_area + 1e-8)
            
            # Shape similarity (using Hu moments)
            shape_similarity = self._calculate_shape_similarity(
                pred_mask == oil_class, true_mask == oil_class
            )
            
            return {
                'area_estimation_error': area_estimation_error,
                'shape_similarity': shape_similarity,
                'predicted_oil_area': pred_oil_area,
                'true_oil_area': true_oil_area
            }
            
        except Exception as error:
            logger.warning(f"Error calculating SAR-specific metrics: {error}")
            return {}
    
    def _calculate_shape_similarity(self, pred_mask: np.ndarray, 
                                   true_mask: np.ndarray) -> float:
        """Calculate shape similarity using Hu moments."""
        try:
            import cv2
            
            # Calculate Hu moments
            pred_moments = cv2.HuMoments(cv2.moments(pred_mask.astype(np.uint8))).flatten()
            true_moments = cv2.HuMoments(cv2.moments(true_mask.astype(np.uint8))).flatten()
            
            # Calculate similarity (inverse of distance)
            distance = np.linalg.norm(pred_moments - true_moments)
            similarity = 1.0 / (1.0 + distance)
            
            return similarity
            
        except Exception as error:
            logger.warning(f"Error calculating shape similarity: {error}")
            return 0.0
    
    def generate_confusion_matrix(self, y_pred: np.ndarray, y_true: np.ndarray, 
                                normalize: str = 'true') -> np.ndarray:
        """Generate and visualize confusion matrix."""
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), normalize=normalize)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        return cm
    
    def generate_evaluation_report(self, metrics: Dict[str, float]) -> str:
        """Generate comprehensive evaluation report."""
        report = "\n" + "=" * 60 + "\n"
        report += "           SAR OIL SPILL SEGMENTATION EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Core segmentation metrics
        report += "CORE SEGMENTATION METRICS:\n"
        report += "-" * 30 + "\n"
        report += f"Jaccard Index (IoU):      {metrics.get('jaccard_index', 0.0):.4f}\n"
        report += f"Dice Coefficient:         {metrics.get('dice_coefficient', 0.0):.4f}\n"
        report += f"Pixel Accuracy:           {metrics.get('pixel_accuracy', 0.0):.4f}\n"
        report += f"Precision:                {metrics.get('precision', 0.0):.4f}\n"
        report += f"Recall:                   {metrics.get('recall', 0.0):.4f}\n"
        report += f"F1 Score:                 {metrics.get('f1_score', 0.0):.4f}\n\n"
        
        # Boundary accuracy
        report += "BOUNDARY ACCURACY:\n"
        report += "-" * 20 + "\n"
        report += f"Boundary F1 Score:        {metrics.get('boundary_f1', 0.0):.4f}\n\n"
        
        # Object-level detection
        if 'object_precision' in metrics:
            report += "OBJECT-LEVEL DETECTION:\n"
            report += "-" * 25 + "\n"
            report += f"Object Precision:         {metrics.get('object_precision', 0.0):.4f}\n"
            report += f"Object Recall:            {metrics.get('object_recall', 0.0):.4f}\n"
            report += f"Object F1 Score:          {metrics.get('object_f1', 0.0):.4f}\n"
            report += f"Detected Objects:         {metrics.get('detected_objects', 0)}\n"
            report += f"False Positives:          {metrics.get('false_positive_objects', 0)}\n"
            report += f"Total True Objects:       {metrics.get('total_true_objects', 0)}\n\n"
        
        # SAR-specific metrics
        if 'area_estimation_error' in metrics:
            report += "SAR-SPECIFIC METRICS:\n"
            report += "-" * 22 + "\n"
            report += f"Area Estimation Error:    {metrics.get('area_estimation_error', 0.0):.4f}\n"
            report += f"Shape Similarity:         {metrics.get('shape_similarity', 0.0):.4f}\n"
            report += f"Predicted Oil Area:       {metrics.get('predicted_oil_area', 0)} pixels\n"
            report += f"True Oil Area:            {metrics.get('true_oil_area', 0)} pixels\n\n"
        
        report += "=" * 60 + "\n"
        
        return report
    
    def save_evaluation_results(self, metrics: Dict[str, float], 
                              output_path: str, include_plots: bool = True):
        """Save evaluation results to file."""
        try:
            from pathlib import Path
            import json
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics as JSON
            with open(output_path / 'evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save evaluation report
            report = self.generate_evaluation_report(metrics)
            with open(output_path / 'evaluation_report.txt', 'w') as f:
                f.write(report)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as error:
            logger.error(f"Error saving evaluation results: {error}")
    
    def get_evaluation_history(self) -> List[Dict[str, float]]:
        """Get history of all evaluations performed."""
        return self.evaluation_history.copy()
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]], 
                      metrics_to_compare: List[str] = None) -> None:
        """Compare multiple model results."""
        if metrics_to_compare is None:
            metrics_to_compare = ['jaccard_index', 'dice_coefficient', 'f1_score']
        
        # Create comparison plot
        fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(15, 5))
        if len(metrics_to_compare) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_compare):
            model_names = list(model_results.keys())
            values = [model_results[model][metric] for model in model_names]
            
            axes[idx].bar(model_names, values)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Model comparison completed for {len(model_results)} models")
