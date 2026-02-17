"""
Word2Vec Training Module for Fake News Detection

This module trains Word2Vec embeddings LOCALLY from scratch using Gensim.
No pretrained embeddings or external APIs are used.

Word2Vec creates dense vector representations of words that capture
semantic relationships (e.g., king - man + woman ≈ queen).

For fake news detection, we use these embeddings to:
1. Create document embeddings (average of word vectors)
2. Capture semantic content beyond bag-of-words
3. Measure semantic similarity between articles
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import joblib
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    WORD2VEC_VECTOR_SIZE,
    WORD2VEC_WINDOW,
    WORD2VEC_MIN_COUNT,
    WORD2VEC_WORKERS,
    WORD2VEC_EPOCHS,
    EMBEDDINGS_DIR,
    SAVED_MODELS_DIR
)


class EpochLogger(CallbackAny2Vec):
    """Callback to log training progress."""
    
    def __init__(self):
        self.epoch = 0
        self.losses = []
    
    def on_epoch_end(self, model):
        self.epoch += 1
        loss = model.get_latest_training_loss()
        if self.losses:
            epoch_loss = loss - self.losses[-1]
        else:
            epoch_loss = loss
        self.losses.append(loss)
        print(f"  Epoch {self.epoch}: Loss = {epoch_loss:.4f}")


class Word2VecTrainer:
    """
    Trains Word2Vec embeddings locally from a text corpus.
    
    This class provides:
    - Local Word2Vec training (no external APIs)
    - Document embedding generation (average pooling)
    - Semantic similarity computation
    
    Why train locally:
    - Full control over training data
    - No dependency on external services
    - Embeddings tuned to our specific domain (news articles)
    """
    
    def __init__(
        self,
        vector_size: int = WORD2VEC_VECTOR_SIZE,
        window: int = WORD2VEC_WINDOW,
        min_count: int = WORD2VEC_MIN_COUNT,
        workers: int = WORD2VEC_WORKERS,
        epochs: int = WORD2VEC_EPOCHS,
        sg: int = 1  # 1 = Skip-gram, 0 = CBOW
    ):
        """
        Initialize the Word2Vec trainer.
        
        Args:
            vector_size: Dimensionality of word vectors (100 for CPU efficiency)
            window: Context window size
            min_count: Minimum word frequency to include
            workers: Number of training threads
            epochs: Number of training epochs
            sg: Training algorithm (1=Skip-gram, 0=CBOW)
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        
        self.model = None
        self.is_trained = False
    
    def _tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Convert texts to list of token lists for Word2Vec.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of tokenized documents
        """
        tokenized = []
        for text in texts:
            if isinstance(text, str):
                tokens = text.lower().split()
                if tokens:
                    tokenized.append(tokens)
        return tokenized
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> 'Word2VecTrainer':
        """
        Train Word2Vec model on the provided texts.
        
        This trains embeddings from scratch using only the provided corpus.
        No pretrained weights or external resources are used.
        
        Args:
            texts: List of text documents for training
            show_progress: Whether to show training progress
            
        Returns:
            self
        """
        print("Training Word2Vec embeddings locally...")
        print(f"  Vector size: {self.vector_size}")
        print(f"  Window: {self.window}")
        print(f"  Min count: {self.min_count}")
        print(f"  Epochs: {self.epochs}")
        
        # Tokenize texts
        print("Tokenizing texts...")
        sentences = self._tokenize_texts(texts)
        print(f"  Total sentences: {len(sentences)}")
        
        # Initialize and train model
        callbacks = [EpochLogger()] if show_progress else []
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
            callbacks=callbacks,
            compute_loss=True
        )
        
        self.is_trained = True
        print(f"\nTraining complete!")
        print(f"  Vocabulary size: {len(self.model.wv)}")
        
        return self
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the vector representation of a word.
        
        Args:
            word: Word to get vector for
            
        Returns:
            Word vector or None if word not in vocabulary
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        word = word.lower()
        if word in self.model.wv:
            return self.model.wv[word]
        return None
    
    def get_document_vector(
        self,
        text: str,
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Get vector representation of a document.
        
        Uses averaging (mean pooling) of word vectors.
        This is a simple but effective approach for document embedding.
        
        Args:
            text: Input text
            pooling: Pooling method ('mean' or 'max')
            
        Returns:
            Document vector of shape (vector_size,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        words = text.lower().split()
        
        # Get vectors for words in vocabulary
        vectors = []
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if not vectors:
            # Return zero vector if no words found
            return np.zeros(self.vector_size)
        
        vectors = np.array(vectors)
        
        if pooling == 'mean':
            return np.mean(vectors, axis=0)
        elif pooling == 'max':
            return np.max(vectors, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
    
    def get_document_vectors(
        self,
        texts: List[str],
        pooling: str = 'mean',
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Get vector representations for multiple documents.
        
        Args:
            texts: List of text documents
            pooling: Pooling method
            show_progress: Whether to show progress bar
            
        Returns:
            Document vectors of shape (n_documents, vector_size)
        """
        if show_progress:
            texts = tqdm(texts, desc="Creating document embeddings")
        
        vectors = [self.get_document_vector(text, pooling) for text in texts]
        return np.array(vectors, dtype=np.float32)
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two documents.
        
        Useful for cross-source comparison in credibility assessment.
        
        Args:
            text1: First document
            text2: Second document
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = self.get_document_vector(text1)
        vec2 = self.get_document_vector(text2)
        
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def find_similar_words(
        self,
        word: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find words most similar to the given word.
        
        Args:
            word: Query word
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        word = word.lower()
        if word not in self.model.wv:
            return []
        
        return self.model.wv.most_similar(word, topn=top_n)
    
    def save(self, path: Optional[str] = None):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save model (default: embeddings/word2vec.model)
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if path is None:
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            path = EMBEDDINGS_DIR / "word2vec.model"
        
        self.model.save(str(path))
        print(f"Saved Word2Vec model to {path}")
    
    def load(self, path: Optional[str] = None):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load model from
        """
        if path is None:
            path = EMBEDDINGS_DIR / "word2vec.model"
        
        self.model = Word2Vec.load(str(path))
        self.is_trained = True
        print(f"Loaded Word2Vec model from {path}")
        print(f"  Vocabulary size: {len(self.model.wv)}")


class DocumentEmbedder:
    """
    High-level interface for document embedding.
    
    Wraps Word2VecTrainer with convenient fit/transform interface
    for integration with sklearn pipelines.
    """
    
    def __init__(
        self,
        vector_size: int = WORD2VEC_VECTOR_SIZE,
        pooling: str = 'mean'
    ):
        """
        Initialize the document embedder.
        
        Args:
            vector_size: Dimensionality of embeddings
            pooling: Pooling method for document vectors
        """
        self.vector_size = vector_size
        self.pooling = pooling
        self.trainer = Word2VecTrainer(vector_size=vector_size)
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'DocumentEmbedder':
        """
        Train the embedding model on texts.
        
        Args:
            texts: Training texts
            
        Returns:
            self
        """
        self.trainer.train(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to document vectors.
        
        Args:
            texts: Texts to transform
            
        Returns:
            Document vector matrix
        """
        if not self.is_fitted:
            raise ValueError("Embedder not fitted. Call fit() first.")
        return self.trainer.get_document_vectors(texts, self.pooling)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, path: Optional[str] = None):
        """Save the trained model."""
        self.trainer.save(path)
    
    def load(self, path: Optional[str] = None):
        """Load a trained model."""
        self.trainer.load(path)
        self.is_fitted = True


# Testing
if __name__ == "__main__":
    print("Testing Word2Vec Training Module")
    print("=" * 50)
    
    # Sample corpus for testing
    sample_corpus = [
        "The stock market showed gains today after strong earnings reports",
        "Scientists announce breakthrough in cancer research treatment",
        "Election results show close race between candidates",
        "New study reveals benefits of healthy eating habits",
        "Tech companies report record quarterly profits",
        "Climate change effects seen in rising sea levels globally",
        "Sports team wins championship game in overtime thriller",
        "Government announces new economic stimulus package",
        "Medical researchers develop promising vaccine candidate",
        "International trade tensions escalate between nations",
    ] * 10  # Repeat for more training data
    
    # Train Word2Vec
    print("\n1. Training Word2Vec model:")
    trainer = Word2VecTrainer(
        vector_size=50,  # Small for testing
        window=3,
        min_count=1,
        epochs=5
    )
    trainer.train(sample_corpus, show_progress=True)
    
    # Test word vectors
    print("\n2. Testing word vectors:")
    test_words = ['market', 'research', 'election']
    for word in test_words:
        vec = trainer.get_word_vector(word)
        if vec is not None:
            print(f"   '{word}' vector shape: {vec.shape}")
    
    # Test document vectors
    print("\n3. Testing document vectors:")
    doc = "Stock market reports strong gains after positive earnings"
    doc_vec = trainer.get_document_vector(doc)
    print(f"   Document vector shape: {doc_vec.shape}")
    
    # Test similarity
    print("\n4. Testing similarity:")
    doc1 = "The economy shows strong growth this quarter"
    doc2 = "Economic indicators suggest positive trends"
    doc3 = "Sports team wins championship game"
    
    sim_12 = trainer.compute_similarity(doc1, doc2)
    sim_13 = trainer.compute_similarity(doc1, doc3)
    
    print(f"   Similarity (economy texts): {sim_12:.4f}")
    print(f"   Similarity (economy vs sports): {sim_13:.4f}")
    
    # Test similar words
    print("\n5. Testing similar words:")
    similar = trainer.find_similar_words('market', top_n=5)
    print(f"   Words similar to 'market': {similar}")
    
    # Test DocumentEmbedder
    print("\n6. Testing DocumentEmbedder:")
    embedder = DocumentEmbedder(vector_size=50)
    vectors = embedder.fit_transform(sample_corpus[:5])
    print(f"   Embedded 5 documents to shape: {vectors.shape}")
    
    print("\nAll Word2Vec tests passed!")
