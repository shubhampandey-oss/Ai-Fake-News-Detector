"""
Visualization Module for Fake News Detection

Creates plots and charts for model evaluation:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Model comparison charts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAVED_MODELS_DIR


class Visualizer:
    """
    Creates visualizations for model evaluation results.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
        """
        self.output_dir = output_dir or (SAVED_MODELS_DIR / "plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-whitegrid')
        
        # Color palette
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#4CAF50',
            'danger': '#F44336',
            'warning': '#FF9800',
            'info': '#00BCD4'
        }
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = ['FAKE', 'REAL'],
        title: str = 'Confusion Matrix',
        save_name: Optional[str] = None
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            class_names: Names for classes
            title: Plot title
            save_name: Filename to save (without extension)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {path}")
        
        plt.close()
        return fig
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        model_name: str = 'Model',
        save_name: Optional[str] = None
    ):
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: Area under curve
            model_name: Name of the model
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(
            fpr, tpr,
            color=self.colors['primary'],
            linewidth=2,
            label=f'{model_name} (AUC = {auc_score:.3f})'
        )
        
        # Diagonal reference line
        ax.plot(
            [0, 1], [0, 1],
            color='gray',
            linestyle='--',
            label='Random Classifier'
        )
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved ROC curve to {path}")
        
        plt.close()
        return fig
    
    def plot_roc_curves_comparison(
        self,
        curves_data: Dict[str, Dict],
        save_name: Optional[str] = None
    ):
        """
        Plot multiple ROC curves for model comparison.
        
        Args:
            curves_data: Dict mapping model names to {fpr, tpr, auc} dicts
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10.colors
        
        for i, (model_name, data) in enumerate(curves_data.items()):
            ax.plot(
                data['fpr'],
                data['tpr'],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{model_name} (AUC = {data['auc']:.3f})"
            )
        
        # Diagonal reference
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved ROC comparison to {path}")
        
        plt.close()
        return fig
    
    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        ap_score: float,
        model_name: str = 'Model',
        save_name: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            ap_score: Average precision score
            model_name: Name of the model
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(
            recall, precision,
            color=self.colors['secondary'],
            linewidth=2,
            label=f'{model_name} (AP = {ap_score:.3f})'
        )
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved PR curve to {path}")
        
        plt.close()
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_data: Dict[str, Dict[str, float]],
        metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
        save_name: Optional[str] = None
    ):
        """
        Bar chart comparing metrics across models.
        
        Args:
            metrics_data: Dict mapping model names to metric dicts
            metrics_to_plot: Which metrics to include
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(metrics_data.keys())
        n_models = len(models)
        n_metrics = len(metrics_to_plot)
        
        bar_width = 0.8 / n_metrics
        x = np.arange(n_models)
        
        colors = [
            self.colors['primary'],
            self.colors['secondary'],
            self.colors['warning'],
            self.colors['info'],
            self.colors['danger']
        ]
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_data[m].get(metric, 0) for m in models]
            offset = (i - n_metrics/2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                values,
                bar_width,
                label=metric.capitalize(),
                color=colors[i % len(colors)]
            )
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved metrics comparison to {path}")
        
        plt.close()
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ):
        """
        Plot training/validation loss and accuracy over epochs.
        
        Args:
            history: Dict with 'train_loss', 'val_loss', 'val_accuracy' keys
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # Loss plot
        ax1 = axes[0]
        if 'train_loss' in history:
            ax1.plot(epochs, history['train_loss'], label='Train Loss', 
                    color=self.colors['primary'], linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], label='Val Loss',
                    color=self.colors['danger'], linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2 = axes[1]
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], label='Val Accuracy',
                    color=self.colors['secondary'], linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved training history to {path}")
        
        plt.close()
        return fig
    
    def generate_all_plots(
        self,
        evaluator,
        prefix: str = 'evaluation'
    ):
        """
        Generate all plots for evaluation results.
        
        Args:
            evaluator: ModelEvaluator with computed results
            prefix: Filename prefix
        """
        print("Generating evaluation plots...")
        
        # Metrics comparison
        if evaluator.results:
            metrics_data = {
                name: results['metrics']
                for name, results in evaluator.results.items()
            }
            self.plot_metrics_comparison(
                metrics_data,
                save_name=f"{prefix}_metrics_comparison"
            )
        
        print(f"Plots saved to {self.output_dir}")


# Testing
if __name__ == "__main__":
    print("Testing Visualization Module")
    print("=" * 50)
    
    viz = Visualizer()
    
    # Sample data
    cm = np.array([[85, 15], [10, 90]])
    fpr = np.array([0, 0.1, 0.2, 0.4, 1.0])
    tpr = np.array([0, 0.6, 0.8, 0.9, 1.0])
    
    print("\n1. Creating confusion matrix...")
    viz.plot_confusion_matrix(cm, save_name='test_confusion_matrix')
    
    print("\n2. Creating ROC curve...")
    viz.plot_roc_curve(fpr, tpr, 0.85, 'Test Model', save_name='test_roc')
    
    print("\n3. Creating metrics comparison...")
    metrics = {
        'Model A': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85},
        'Model B': {'accuracy': 0.88, 'precision': 0.90, 'recall': 0.85, 'f1': 0.87},
        'Model C': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84, 'f1': 0.82}
    }
    viz.plot_metrics_comparison(metrics, save_name='test_comparison')
    
    print("\nAll visualization tests passed!")
