import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

class Visualization:
    @staticmethod
    def plot_detection_result(
        original: np.ndarray,
        mask: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Visualize detection results"""
        # Placeholder: show using plt.show()
        plt.imshow(original, cmap='gray')
        plt.imshow(mask, cmap='jet', alpha=0.5)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    @staticmethod
    def plot_metrics(metrics: dict, save_path: Optional[str] = None):
        """Visualize evaluation metrics"""
        # Placeholder
        print(metrics)
