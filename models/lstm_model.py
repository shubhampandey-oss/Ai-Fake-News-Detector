"""
LSTM/Bi-LSTM Model for Fake News Detection

This module implements a deep learning approach using LSTM networks.
Optimized for CPU training with smaller architecture.

Key features:
- Embedding layer (trained from scratch, NOT pretrained)
- Bidirectional LSTM for capturing context in both directions
- Dropout for regularization
- Early stopping to prevent overfitting

All training happens locally - no external APIs or pretrained models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LSTM_EMBEDDING_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    LSTM_BIDIRECTIONAL,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE,
    LSTM_MAX_SEQ_LENGTH,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    SAVED_MODELS_DIR
)

# Set seeds for reproducibility (CRITICAL for deterministic predictions)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Enable deterministic algorithms for reproducible predictions
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Try to enable fully deterministic mode (PyTorch 1.8+)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except (TypeError, AttributeError):
    pass  # Older PyTorch versions


def set_deterministic_mode():
    """
    Enable full deterministic mode for inference.
    
    Call this before any prediction to ensure reproducibility.
    Same input will ALWAYS produce the same output.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Vocabulary:
    """
    Vocabulary builder for text tokenization.
    
    Maps words to integer indices for embedding lookup.
    """
    
    def __init__(self, max_vocab_size: int = 50000):
        """
        Initialize vocabulary.
        
        Args:
            max_vocab_size: Maximum vocabulary size
        """
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1
        self.idx2word[0] = self.pad_token
        self.idx2word[1] = self.unk_token
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text documents
        """
        # Count word frequencies
        for text in texts:
            for word in text.lower().split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(
            self.word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_vocab_size - 2]  # -2 for PAD and UNK
        
        # Assign indices
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def encode(self, text: str, max_length: int) -> List[int]:
        """
        Encode text to sequence of indices.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of word indices (padded/truncated to max_length)
        """
        words = text.lower().split()
        
        # Convert words to indices
        indices = [
            self.word2idx.get(word, self.word2idx[self.unk_token])
            for word in words[:max_length]
        ]
        
        # Pad if necessary
        if len(indices) < max_length:
            indices += [self.word2idx[self.pad_token]] * (max_length - len(indices))
        
        return indices
    
    def __len__(self):
        return len(self.word2idx)


class NewsDataset(Dataset):
    """
    PyTorch Dataset for news articles.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        vocab: Vocabulary,
        max_length: int = LSTM_MAX_SEQ_LENGTH
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text documents
            labels: Array of labels (0=FAKE, 1=REAL)
            vocab: Vocabulary object
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.vocab.encode(text, self.max_length)
        
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for fake news detection.
    
    Architecture (CPU-optimized):
    - Embedding layer (64 dim, trained from scratch)
    - Bidirectional LSTM (64 hidden, 1 layer)
    - Dropout (0.3)
    - Fully connected output layer
    - Sigmoid activation for binary classification
    
    Why LSTM:
    - Captures sequential dependencies in text
    - Handles variable-length inputs
    - Learns long-range dependencies
    - Bidirectional version sees context from both directions
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = LSTM_EMBEDDING_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = LSTM_BIDIRECTIONAL,
        pad_idx: int = 0
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            pad_idx: Padding token index
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer (trained from scratch)
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(output_dim, 1)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output probabilities of shape (batch_size,)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch, seq_len, embed_dim) -> (batch, seq_len, hidden_dim * num_directions)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate final hidden states from both directions
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Output layer
        output = self.fc(hidden)
        output = self.sigmoid(output)
        
        return output.squeeze()


class LSTMTrainer:
    """
    Trainer class for LSTM model.
    
    Handles training loop, validation, and early stopping.
    Optimized for CPU training.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = LSTM_EMBEDDING_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        learning_rate: float = LSTM_LEARNING_RATE,
        batch_size: int = LSTM_BATCH_SIZE,
        epochs: int = LSTM_EPOCHS,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum epochs
            device: 'cpu' or 'cuda'
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)
        
        # Initialize model
        self.model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            texts, labels = batch
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                texts, labels = batch
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        vocab: Vocabulary
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            vocab: Vocabulary object
            
        Returns:
            Training history dictionary
        """
        # Create datasets and dataloaders
        train_dataset = NewsDataset(train_texts, train_labels, vocab)
        val_dataset = NewsDataset(val_texts, val_labels, vocab)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training LSTM on {self.device}...")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f"  Epoch {epoch+1}/{self.epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.load_checkpoint('best_model.pt')
        
        return self.history
    
    def predict(self, texts: List[str], vocab: Vocabulary) -> np.ndarray:
        """
        Get predictions for texts.
        
        Args:
            texts: Input texts
            vocab: Vocabulary object
            
        Returns:
            Predictions (0=FAKE, 1=REAL)
        """
        self.model.eval()
        
        # Create dataset
        dummy_labels = np.zeros(len(texts))
        dataset = NewsDataset(texts, dummy_labels, vocab)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in loader:
                texts_batch, _ = batch
                texts_batch = texts_batch.to(self.device)
                
                outputs = self.model(texts_batch)
                predictions = (outputs >= 0.5).int().cpu().numpy()
                all_predictions.extend(predictions)
        
        return np.array(all_predictions)
    
    def predict_proba(self, texts: List[str], vocab: Vocabulary) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: Input texts
            vocab: Vocabulary object
            
        Returns:
            Probabilities of shape (n_samples, 2)
        """
        self.model.eval()
        
        dummy_labels = np.zeros(len(texts))
        dataset = NewsDataset(texts, dummy_labels, vocab)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                texts_batch, _ = batch
                texts_batch = texts_batch.to(self.device)
                
                outputs = self.model(texts_batch).cpu().numpy()
                # Convert to 2-class probabilities
                probs = np.column_stack([1 - outputs, outputs])
                all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = SAVED_MODELS_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = SAVED_MODELS_DIR / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
    
    def save(self, path: Optional[str] = None):
        """Save trained model."""
        if path is None:
            path = SAVED_MODELS_DIR / "lstm_model.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'history': self.history
        }, path)
        print(f"Saved LSTM model to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load trained model."""
        if path is None:
            path = SAVED_MODELS_DIR / "lstm_model.pt"
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model with saved parameters
        self.model = LSTMClassifier(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        
        print(f"Loaded LSTM model from {path}")


# Testing
if __name__ == "__main__":
    print("Testing LSTM Model Module")
    print("=" * 50)
    
    # Sample data
    sample_texts = [
        "breaking news major scandal rocks the nation",
        "scientists discover new treatment for disease",
        "shocking revelation you wont believe what happened",
        "stock market shows steady growth this quarter",
        "unbelievable secret government conspiracy exposed",
        "research study shows positive health outcomes",
        "celebrity caught in shocking scandal photos",
        "new economic policy announced by government",
    ] * 10  # Repeat for more data
    
    sample_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 10)
    
    # Split
    n_train = int(len(sample_texts) * 0.8)
    train_texts = sample_texts[:n_train]
    train_labels = sample_labels[:n_train]
    val_texts = sample_texts[n_train:]
    val_labels = sample_labels[n_train:]
    
    print(f"\n1. Creating vocabulary...")
    vocab = Vocabulary(max_vocab_size=1000)
    vocab.build_vocab(train_texts)
    
    print(f"\n2. Training LSTM (5 epochs for testing)...")
    trainer = LSTMTrainer(
        vocab_size=len(vocab),
        embedding_dim=32,  # Small for testing
        hidden_dim=32,
        epochs=5,
        batch_size=8
    )
    
    history = trainer.fit(
        train_texts, train_labels,
        val_texts, val_labels,
        vocab
    )
    
    print(f"\n3. Testing predictions...")
    predictions = trainer.predict(val_texts[:5], vocab)
    probs = trainer.predict_proba(val_texts[:5], vocab)
    
    print(f"   Predictions: {predictions}")
    print(f"   Probabilities (REAL class): {probs[:, 1]}")
    
    print(f"\n4. Saving model...")
    trainer.save()
    
    print("\nAll LSTM tests passed!")
