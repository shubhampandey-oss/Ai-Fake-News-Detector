"""
Training Script for Classical ML Models

This script trains Logistic Regression, SVM, and Random Forest models
for fake news detection using TF-IDF features.

All training is done locally from scratch.
No pretrained models or external APIs.

Usage:
    python training/train_classical.py --evaluate
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SAVED_MODELS_DIR, RANDOM_SEED
from preprocessing.data_loader import FakeNewsDataLoader, load_data
from features.textual_features import TfidfExtractor, CombinedTextualFeatures
from features.linguistic_features import LinguisticFeatureExtractor
from models.classical_models import (
    LogisticRegressionClassifier,
    SVMClassifier,
    RandomForestClassifierWrapper,
    create_all_models
)
from evaluation.metrics import ModelEvaluator
from evaluation.visualizations import Visualizer


def prepare_features(data: dict):
    """
    Prepare feature extractors and transform data.
    
    Args:
        data: Dictionary from data loader
        
    Returns:
        Tuple of (X_train, X_val, X_test, feature_extractor)
    """
    print("\n" + "="*50)
    print("FEATURE EXTRACTION")
    print("="*50)
    
    # Initialize extractors
    tfidf = TfidfExtractor()
    linguistic = LinguisticFeatureExtractor()
    
    # Fit on training data
    print("\nFitting TF-IDF on training data...")
    X_train_tfidf = tfidf.fit_transform(data['train_texts'].tolist())
    
    print("Extracting linguistic features...")
    X_train_ling = linguistic.extract_batch(
        data['train_texts'].tolist(),
        show_progress=True
    )
    
    # Transform validation and test
    print("\nTransforming validation data...")
    X_val_tfidf = tfidf.transform(data['val_texts'].tolist())
    X_val_ling = linguistic.extract_batch(
        data['val_texts'].tolist(),
        show_progress=False
    )
    
    print("Transforming test data...")
    X_test_tfidf = tfidf.transform(data['test_texts'].tolist())
    X_test_ling = linguistic.extract_batch(
        data['test_texts'].tolist(),
        show_progress=False
    )
    
    # Combine features
    from scipy.sparse import hstack, csr_matrix
    from sklearn.preprocessing import MaxAbsScaler
    
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_ling)])
    X_val = hstack([X_val_tfidf, csr_matrix(X_val_ling)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_ling)])
    
    # Scale features to improve convergence (especially for Logistic Regression and SVM)
    # Using MaxAbsScaler which works well with sparse matrices
    print("\nScaling features...")
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save the scaler for inference
    scaler_path = SAVED_MODELS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")
    
    print(f"\nFeature dimensions:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Save extractors
    tfidf.save()
    
    return X_train, X_val, X_test, tfidf, linguistic


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: dict = None
):
    """
    Train all classical models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weights: Optional class weights
        
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    models = create_all_models()
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train, class_weights)
        model.save()
    
    return models


def evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Evaluate all models on test set.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        ModelEvaluator with results
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        evaluator.evaluate_model(name, y_test, y_pred, y_proba)
        evaluator.print_results(name)
    
    return evaluator


def main(args):
    """Main training pipeline."""
    
    print("="*60)
    print("FAKE NEWS DETECTION - CLASSICAL MODEL TRAINING")
    print("="*60)
    print("\nThis trains models locally from scratch.")
    print("No pretrained models or external APIs are used.\n")
    
    # Load data
    print("Loading data...")
    data = load_data(clean_text=True)
    
    if data is None:
        print("\nERROR: Could not load data.")
        print("Please download the Kaggle Fake News dataset and place it in:")
        print(f"  {Path(__file__).parent.parent / 'data' / 'raw'}")
        print("\nRequired files: Fake.csv, True.csv")
        print("Dataset URL: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        return
    
    # Prepare features
    X_train, X_val, X_test, tfidf, linguistic = prepare_features(data)
    
    # Train models
    models = train_models(
        X_train,
        data['train_labels'],
        data.get('class_weights')
    )
    
    # Evaluate if requested
    if args.evaluate:
        evaluator = evaluate_models(
            models,
            X_test,
            data['test_labels']
        )
        
        # Compare models
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison = evaluator.compare_models()
        best_model = evaluator.get_best_model('f1')
        print(f"\nBest model (by F1): {best_model}")
        
        # Generate plots
        if args.plot:
            print("\nGenerating evaluation plots...")
            viz = Visualizer()
            viz.generate_all_plots(evaluator, 'classical_models')
        
        # Save results
        results_df = evaluator.to_dataframe()
        results_path = SAVED_MODELS_DIR / "classical_results.csv"
        results_df.to_csv(results_path)
        print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nModels saved to: {SAVED_MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train classical ML models for fake news detection"
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate models after training'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate evaluation plots'
    )
    
    args = parser.parse_args()
    main(args)
