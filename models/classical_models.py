"""
Classical Machine Learning Models for Fake News Detection

This module implements traditional ML classifiers:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

All models are trained locally from scratch using scikit-learn.
No pretrained models or external APIs are used.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LR_MAX_ITER, LR_C,
    SVM_KERNEL, SVM_C, SVM_GAMMA,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT,
    RANDOM_SEED, SAVED_MODELS_DIR
)


class FakeNewsClassifier:
    """
    Base class for fake news classifiers.
    
    Provides common interface for training, prediction, and saving models.
    """
    
    def __init__(self, name: str):
        """
        Initialize classifier.
        
        Args:
            name: Name identifier for the model
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.classes_ = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None
    ) -> 'FakeNewsClassifier':
        """
        Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            class_weight: Optional class weights for imbalance
            
        Returns:
            self
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction confidence scores.
        
        Returns the probability of the predicted class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Confidence scores (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)
    
    def save(self, path: Optional[str] = None):
        """
        Save model to disk.
        
        Args:
            path: Save path (default: saved_models/{name}.joblib)
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if path is None:
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVED_MODELS_DIR / f"{self.name}.joblib"
        
        joblib.dump({
            'model': self.model,
            'classes_': self.classes_
        }, path)
        print(f"Saved {self.name} model to {path}")
    
    def load(self, path: Optional[str] = None):
        """
        Load model from disk.
        
        Args:
            path: Load path
        """
        if path is None:
            path = SAVED_MODELS_DIR / f"{self.name}.joblib"
        
        data = joblib.load(path)
        self.model = data['model']
        self.classes_ = data['classes_']
        self.is_trained = True
        print(f"Loaded {self.name} model from {path}")
    
    def predict_deterministic(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels with guaranteed deterministic behavior.
        
        Same input will ALWAYS produce the same output.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        # Import here to avoid circular imports
        from utils.deterministic import set_all_seeds
        set_all_seeds(42)
        return self.predict(X)
    
    def predict_proba_deterministic(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with guaranteed deterministic behavior.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        from utils.deterministic import set_all_seeds
        set_all_seeds(42)
        return self.predict_proba(X)


class LogisticRegressionClassifier(FakeNewsClassifier):
    """
    Logistic Regression classifier for fake news detection.
    
    Why Logistic Regression:
    - Simple, interpretable baseline
    - Provides probability estimates
    - Works well with sparse text features (TF-IDF)
    - Fast to train and predict
    - Coefficients show feature importance
    """
    
    def __init__(
        self,
        C: float = LR_C,
        max_iter: int = LR_MAX_ITER,
        penalty: str = 'l2',
        solver: str = 'lbfgs'
    ):
        """
        Initialize Logistic Regression classifier.
        
        Args:
            C: Regularization strength (smaller = stronger)
            max_iter: Maximum iterations for convergence
            penalty: Regularization type ('l1', 'l2')
            solver: Optimization algorithm
        """
        super().__init__("logistic_regression")
        self.C = C
        self.max_iter = max_iter
        self.penalty = penalty
        self.solver = solver
        
        # Use saga solver for l1 regularization
        if penalty == 'l1' and solver == 'lbfgs':
            self.solver = 'saga'
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None
    ) -> 'LogisticRegressionClassifier':
        """
        Train the Logistic Regression model.
        
        Args:
            X: Feature matrix
            y: Labels
            class_weight: Optional class weights
            
        Returns:
            self
        """
        print(f"Training Logistic Regression (C={self.C})...")
        
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            penalty=self.penalty,
            solver=self.solver,
            class_weight=class_weight or 'balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        print(f"Training complete. Classes: {self.classes_}")
        return self
    
    def get_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 20
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Get most important features for each class.
        
        Returns features with highest positive and negative coefficients.
        
        Args:
            feature_names: Names of features
            top_n: Number of top features per class
            
        Returns:
            Tuple of (fake_features, real_features)
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        coefficients = self.model.coef_[0]
        
        # Get indices sorted by coefficient value
        sorted_indices = np.argsort(coefficients)
        
        # Negative coefficients -> FAKE (class 0)
        fake_indices = sorted_indices[:top_n]
        fake_features = [
            (feature_names[i], coefficients[i])
            for i in fake_indices
        ]
        
        # Positive coefficients -> REAL (class 1)
        real_indices = sorted_indices[-top_n:][::-1]
        real_features = [
            (feature_names[i], coefficients[i])
            for i in real_indices
        ]
        
        return fake_features, real_features


class SVMClassifier(FakeNewsClassifier):
    """
    Support Vector Machine classifier for fake news detection.
    
    Why SVM:
    - Effective in high-dimensional spaces (text features)
    - Works well with clear margin of separation
    - Robust to overfitting in high dimensions
    - Kernel trick for non-linear boundaries
    """
    
    def __init__(
        self,
        kernel: str = SVM_KERNEL,
        C: float = SVM_C,
        gamma: str = SVM_GAMMA,
        probability: bool = True  # Enable probability estimates
    ):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            probability: Whether to enable probability estimates
        """
        super().__init__("svm")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None
    ) -> 'SVMClassifier':
        """
        Train the SVM model.
        
        Uses LinearSVC for linear kernel (much faster on large datasets).
        Falls back to SVC for other kernels.
        
        Args:
            X: Feature matrix
            y: Labels
            class_weight: Optional class weights
            
        Returns:
            self
        """
        print(f"Training SVM (kernel={self.kernel}, C={self.C})...")
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        
        if self.kernel == 'linear':
            # Use LinearSVC for linear kernel - MUCH faster on large datasets
            print(f"  Using LinearSVC for fast linear SVM training...")
            
            base_svm = LinearSVC(
                C=self.C,
                class_weight=class_weight or 'balanced',
                random_state=RANDOM_SEED,
                max_iter=5000,  # Increase max iterations for convergence
                dual='auto'  # Let sklearn choose optimal formulation
            )
            
            if self.probability:
                # Wrap with CalibratedClassifierCV for probability support
                self.model = CalibratedClassifierCV(
                    base_svm,
                    cv=3,  # 3-fold cross-validation for calibration
                    method='sigmoid'
                )
            else:
                self.model = base_svm
        else:
            # Use standard SVC for non-linear kernels
            if n_samples > 10000:
                print(f"  WARNING: Large dataset ({n_samples} samples) with {self.kernel} kernel.")
                print(f"  Consider using kernel='linear' for faster training.")
            
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=self.probability,
                class_weight=class_weight or 'balanced',
                random_state=RANDOM_SEED,
                cache_size=500  # MB, increase for speed
            )
        
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        if self.kernel == 'linear':
            print(f"Training complete. LinearSVC fitted successfully.")
        else:
            print(f"Training complete. Support vectors: {len(self.model.support_)}")
        return self


class RandomForestClassifierWrapper(FakeNewsClassifier):
    """
    Random Forest classifier for fake news detection.
    
    Why Random Forest:
    - Ensemble of decision trees (reduces overfitting)
    - Handles feature interactions well
    - Provides feature importance rankings
    - Robust to noise and outliers
    - No feature scaling required
    """
    
    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: Optional[int] = RF_MAX_DEPTH,
        min_samples_split: int = RF_MIN_SAMPLES_SPLIT,
        max_features: str = 'sqrt'
    ):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_split: Minimum samples to split node
            max_features: Features to consider at each split
        """
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None
    ) -> 'RandomForestClassifierWrapper':
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Labels
            class_weight: Optional class weights
            
        Returns:
            self
        """
        print(f"Training Random Forest (n_estimators={self.n_estimators})...")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            class_weight=class_weight or 'balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        print(f"Training complete. Trees: {len(self.model.estimators_)}")
        return self
    
    def get_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get most important features.
        
        Uses Gini importance (mean decrease in impurity).
        
        Args:
            feature_names: Names of features
            top_n: Number of top features
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        
        return [
            (feature_names[i], importances[i])
            for i in sorted_indices
        ]


class ModelEnsemble:
    """
    Ensemble of multiple classifiers.
    
    Combines predictions from multiple models for more robust results.
    Uses soft voting (average probabilities) by default.
    """
    
    def __init__(
        self,
        models: Optional[List[FakeNewsClassifier]] = None,
        voting: str = 'soft'
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of classifiers
            voting: 'soft' (probability average) or 'hard' (majority vote)
        """
        self.models = models or []
        self.voting = voting
        self.is_trained = False
    
    def add_model(self, model: FakeNewsClassifier):
        """Add a model to the ensemble."""
        self.models.append(model)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None
    ) -> 'ModelEnsemble':
        """
        Train all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Labels
            class_weight: Optional class weights
            
        Returns:
            self
        """
        for model in self.models:
            model.fit(X, y, class_weight)
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.voting == 'soft':
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # Hard voting
            predictions = np.array([
                model.predict(X) for model in self.models
            ])
            # Majority vote
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=predictions
            )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble probability predictions (soft voting).
        
        Args:
            X: Feature matrix
            
        Returns:
            Averaged class probabilities
        """
        probas = [model.predict_proba(X) for model in self.models]
        return np.mean(probas, axis=0)
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence scores."""
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)


def create_all_models() -> Dict[str, FakeNewsClassifier]:
    """
    Create instances of all classical models.
    
    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        'logistic_regression': LogisticRegressionClassifier(),
        'svm': SVMClassifier(),
        'random_forest': RandomForestClassifierWrapper()
    }


# Testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("Testing Classical Models Module")
    print("=" * 50)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_classes=2,
        random_state=RANDOM_SEED
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Test each model
    models = create_all_models()
    
    for name, model in models.items():
        print(f"\n2. Testing {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        confidence = model.get_confidence(X_test)
        
        # Accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Avg confidence: {np.mean(confidence):.4f}")
    
    # Test ensemble
    print("\n3. Testing Model Ensemble...")
    ensemble = ModelEnsemble(list(models.values()))
    
    y_pred = ensemble.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Ensemble accuracy: {accuracy:.4f}")
    
    # Test feature importance (LR)
    print("\n4. Testing feature importance...")
    lr = models['logistic_regression']
    fake_features, real_features = lr.get_feature_importance(
        [f"feature_{i}" for i in range(100)],
        top_n=5
    )
    print("   Top features for FAKE detection:")
    for feat, coef in fake_features[:3]:
        print(f"      {feat}: {coef:.4f}")
    
    print("\nAll classical model tests passed!")
