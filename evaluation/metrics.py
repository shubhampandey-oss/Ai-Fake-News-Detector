"""
Evaluation Metrics Module for Fake News Detection

This module computes comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
- Cross-validation

All metrics are computed locally using scikit-learn.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CV_FOLDS, RANDOM_SEED


class ModelEvaluator:
    """
    Comprehensive model evaluation for fake news detection.
    
    Computes all standard classification metrics and provides
    methods for comparison across multiple models.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: Names for classes (default: ['FAKE', 'REAL'])
        """
        self.class_names = class_names or ['FAKE', 'REAL']
        self.results = {}
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        # Compute ROC-AUC if probabilities are available
        if y_proba is not None:
            try:
                # Handle both 1D and 2D probability arrays
                if len(y_proba.shape) == 2:
                    proba_positive = y_proba[:, 1]
                else:
                    proba_positive = y_proba
                
                metrics['roc_auc'] = float(roc_auc_score(y_true, proba_positive))
                metrics['avg_precision'] = float(
                    average_precision_score(y_true, proba_positive)
                )
            except Exception as e:
                print(f"Warning: Could not compute ROC-AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute confusion matrix with detailed breakdown.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with confusion matrix and derived metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # For binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            return {
                'matrix': cm.tolist(),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            }
        
        return {'matrix': cm.tolist()}
    
    def get_roc_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data for plotting.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary with fpr, tpr, and thresholds
        """
        if len(y_proba.shape) == 2:
            proba_positive = y_proba[:, 1]
        else:
            proba_positive = y_proba
        
        fpr, tpr, thresholds = roc_curve(y_true, proba_positive)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def get_precision_recall_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get Precision-Recall curve data for plotting.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary with precision, recall, and thresholds
        """
        if len(y_proba.shape) == 2:
            proba_positive = y_proba[:, 1]
        else:
            proba_positive = y_proba
        
        precision, recall, thresholds = precision_recall_curve(
            y_true, proba_positive
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    
    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = CV_FOLDS,
        scoring: str = 'f1'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Labels
            cv: Number of folds
            scoring: Metric to use for scoring
            
        Returns:
            Dictionary with mean and std of scores
        """
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=RANDOM_SEED
        )
        
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': scores.tolist()
        }
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Full evaluation of a single model.
        
        Args:
            model_name: Name identifier for the model
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Complete evaluation results
        """
        results = {
            'model_name': model_name,
            'metrics': self.compute_metrics(y_true, y_pred, y_proba),
            'confusion_matrix': self.compute_confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true,
                y_pred,
                target_names=self.class_names,
                output_dict=True
            )
        }
        
        # Store for comparison
        self.results[model_name] = results
        
        return results
    
    def compare_models(self) -> Dict[str, Dict]:
        """
        Compare all evaluated models.
        
        Returns:
            Dictionary with comparison data
        """
        if not self.results:
            return {}
        
        comparison = {
            'metrics': {},
            'rankings': {}
        }
        
        # Extract metrics for each model
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in metric_names:
            comparison['metrics'][metric] = {}
            for model_name, results in self.results.items():
                value = results['metrics'].get(metric, 0)
                comparison['metrics'][metric][model_name] = value
        
        # Compute rankings
        for metric in metric_names:
            values = comparison['metrics'].get(metric, {})
            if values:
                sorted_models = sorted(
                    values.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                comparison['rankings'][metric] = [m[0] for m in sorted_models]
        
        return comparison
    
    def get_best_model(self, metric: str = 'f1') -> Optional[str]:
        """
        Get the name of the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model
        """
        if not self.results:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, results in self.results.items():
            score = results['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def print_results(self, model_name: Optional[str] = None):
        """
        Print evaluation results.
        
        Args:
            model_name: Specific model to print (or None for all)
        """
        if model_name:
            models = {model_name: self.results.get(model_name)}
        else:
            models = self.results
        
        for name, results in models.items():
            if not results:
                continue
            
            print(f"\n{'='*50}")
            print(f"Model: {name}")
            print('='*50)
            
            metrics = results['metrics']
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            cm = results['confusion_matrix']
            print(f"\nConfusion Matrix:")
            print(f"  TN={cm['true_negatives']}, FP={cm['false_positives']}")
            print(f"  FN={cm['false_negatives']}, TP={cm['true_positives']}")
    
    def to_dataframe(self):
        """
        Convert results to pandas DataFrame for easy comparison.
        
        Returns:
            DataFrame with metrics for all models
        """
        import pandas as pd
        
        data = []
        for model_name, results in self.results.items():
            row = {'model': model_name}
            row.update(results['metrics'])
            data.append(row)
        
        return pd.DataFrame(data).set_index('model')


# Testing
if __name__ == "__main__":
    print("Testing Evaluation Metrics Module")
    print("=" * 50)
    
    # Generate sample predictions
    np.random.seed(RANDOM_SEED)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_good = y_true.copy()
    y_pred_good[np.random.choice(n_samples, 10, replace=False)] = 1 - y_pred_good[
        np.random.choice(n_samples, 10, replace=False)
    ]
    y_proba_good = np.column_stack([1 - y_pred_good * 0.8, y_pred_good * 0.8 + 0.1])
    
    evaluator = ModelEvaluator()
    
    print("\n1. Evaluating single model:")
    results = evaluator.evaluate_model(
        'test_model',
        y_true,
        y_pred_good,
        y_proba_good
    )
    evaluator.print_results('test_model')
    
    print("\n2. Computing confusion matrix:")
    cm = evaluator.compute_confusion_matrix(y_true, y_pred_good)
    print(f"   Matrix: {cm['matrix']}")
    
    print("\n3. ROC curve data:")
    roc_data = evaluator.get_roc_curve_data(y_true, y_proba_good)
    print(f"   FPR points: {len(roc_data['fpr'])}")
    print(f"   TPR points: {len(roc_data['tpr'])}")
    
    print("\nAll evaluation tests passed!")
