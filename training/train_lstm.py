"""
Training Script for LSTM Model

This script trains a Bi-LSTM model for fake news detection.
Optimized for CPU training with smaller architecture.

All training is done locally from scratch.
No pretrained embeddings or external APIs.

Usage:
    python training/train_lstm.py --epochs 10 --evaluate
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SAVED_MODELS_DIR,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE,
    RANDOM_SEED
)
from preprocessing.data_loader import load_data
from models.lstm_model import Vocabulary, LSTMTrainer
from evaluation.metrics import ModelEvaluator
from evaluation.visualizations import Visualizer


def main(args):
    """Main LSTM training pipeline."""
    
    print("="*60)
    print("FAKE NEWS DETECTION - LSTM MODEL TRAINING")
    print("="*60)
    print("\nThis trains an LSTM model locally from scratch.")
    print("Embeddings are learned during training (not pretrained).")
    print("Optimized for CPU training.\n")
    
    # Load data
    print("Loading data...")
    data = load_data(clean_text=True)
    
    if data is None:
        print("\nERROR: Could not load data.")
        print("Please download the Kaggle Fake News dataset first.")
        return
    
    # Build vocabulary
    print("\n" + "="*50)
    print("BUILDING VOCABULARY")
    print("="*50)
    
    vocab = Vocabulary(max_vocab_size=args.vocab_size)
    vocab.build_vocab(data['train_texts'].tolist())
    
    # Initialize trainer
    print("\n" + "="*50)
    print("INITIALIZING MODEL")
    print("="*50)
    
    trainer = LSTMTrainer(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device='cpu'  # Force CPU
    )
    
    print(f"\nModel architecture:")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Train
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    history = trainer.fit(
        train_texts=data['train_texts'].tolist(),
        train_labels=data['train_labels'],
        val_texts=data['val_texts'].tolist(),
        val_labels=data['val_labels'],
        vocab=vocab
    )
    
    # Save model and vocabulary
    trainer.save()
    
    # Save vocabulary
    import joblib
    vocab_path = SAVED_MODELS_DIR / "lstm_vocabulary.joblib"
    joblib.dump(vocab, vocab_path)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Evaluate
    if args.evaluate:
        print("\n" + "="*50)
        print("EVALUATION")
        print("="*50)
        
        # Get predictions on test set
        y_pred = trainer.predict(
            data['test_texts'].tolist(),
            vocab
        )
        y_proba = trainer.predict_proba(
            data['test_texts'].tolist(),
            vocab
        )
        
        # Compute metrics
        evaluator = ModelEvaluator()
        evaluator.evaluate_model(
            'lstm',
            data['test_labels'],
            y_pred,
            y_proba
        )
        evaluator.print_results('lstm')
        
        # Generate plots
        if args.plot:
            print("\nGenerating plots...")
            viz = Visualizer()
            viz.plot_training_history(history, save_name='lstm_training_history')
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(data['test_labels'], y_pred)
            viz.plot_confusion_matrix(cm, save_name='lstm_confusion_matrix')
            
            # ROC curve
            roc_data = evaluator.get_roc_curve_data(
                data['test_labels'],
                y_proba
            )
            viz.plot_roc_curve(
                roc_data['fpr'],
                roc_data['tpr'],
                evaluator.results['lstm']['metrics'].get('roc_auc', 0),
                'LSTM',
                save_name='lstm_roc_curve'
            )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nModel saved to: {SAVED_MODELS_DIR}")
    
    if history:
        final_acc = history.get('val_accuracy', [0])[-1]
        print(f"Final validation accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM model for fake news detection"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=LSTM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=LSTM_BATCH_SIZE,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=LSTM_LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=64,
        help='Embedding dimension (smaller for CPU)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='LSTM hidden dimension (smaller for CPU)'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=30000,
        help='Maximum vocabulary size'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model after training'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate training plots'
    )
    
    args = parser.parse_args()
    main(args)
