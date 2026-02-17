"""
Textual Feature Extraction Module for Fake News Detection

This module implements traditional text representation features:
- Bag of Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-grams (unigrams, bigrams, trigrams)

These features capture word frequency patterns that often differ
between fake and real news articles.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    SAVED_MODELS_DIR
)


class BagOfWordsExtractor:
    """
    Bag of Words feature extractor.
    
    BoW represents text as a sparse vector where each dimension
    corresponds to a word in the vocabulary, and the value is
    the word count in the document.
    
    Why use BoW for fake news detection:
    - Simple baseline representation
    - Captures word frequency patterns
    - Fake news often uses specific vocabulary patterns
    """
    
    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int = TFIDF_MIN_DF,
        max_df: float = TFIDF_MAX_DF
    ):
        """
        Initialize BoW extractor.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: Range of n-grams (min_n, max_n)
            min_df: Minimum document frequency for words
            max_df: Maximum document frequency for words
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'BagOfWordsExtractor':
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"BoW vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to BoW features.
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of shape (n_samples, n_features)
        """
        self.is_fitted = True
        result = self.vectorizer.fit_transform(texts)
        print(f"BoW vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get the vocabulary (feature names)."""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, path: Optional[str] = None):
        """Save vectorizer to disk."""
        if path is None:
            path = SAVED_MODELS_DIR / "bow_vectorizer.joblib"
        joblib.dump(self.vectorizer, path)
        print(f"Saved BoW vectorizer to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load vectorizer from disk."""
        if path is None:
            path = SAVED_MODELS_DIR / "bow_vectorizer.joblib"
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        print(f"Loaded BoW vectorizer from {path}")


class TfidfExtractor:
    """
    TF-IDF feature extractor.
    
    TF-IDF weighs words by their importance:
    - TF (Term Frequency): How often a word appears in a document
    - IDF (Inverse Document Frequency): How rare a word is across documents
    
    Why use TF-IDF for fake news detection:
    - Downweights common words that appear everywhere
    - Upweights distinctive words that characterize content
    - Better discrimination than raw word counts
    - Captures stylistic differences between fake and real news
    """
    
    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
        min_df: int = TFIDF_MIN_DF,
        max_df: float = TFIDF_MAX_DF,
        use_idf: bool = True,
        sublinear_tf: bool = True
    ):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: Range of n-grams (min_n, max_n)
            min_df: Minimum document frequency
            max_df: Maximum document frequency (proportion)
            use_idf: Whether to use IDF weighting
            sublinear_tf: Apply sublinear TF scaling (1 + log(tf))
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
            lowercase=True,
            norm='l2'  # L2 normalization
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'TfidfExtractor':
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of shape (n_samples, n_features)
        """
        self.is_fitted = True
        result = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get the vocabulary (feature names)."""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a single text.
        
        Useful for explaining predictions.
        
        Args:
            text: Input text
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, tfidf_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
        
        vector = self.vectorizer.transform([text])
        feature_names = self.get_feature_names()
        
        # Get indices of non-zero values sorted by value
        indices = vector.toarray()[0].argsort()[::-1][:top_n]
        scores = vector.toarray()[0][indices]
        
        return [
            (feature_names[idx], scores[i])
            for i, idx in enumerate(indices)
            if scores[i] > 0
        ]
    
    def save(self, path: Optional[str] = None):
        """Save vectorizer to disk."""
        if path is None:
            path = SAVED_MODELS_DIR / "tfidf_vectorizer.joblib"
        joblib.dump(self.vectorizer, path)
        print(f"Saved TF-IDF vectorizer to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load vectorizer from disk."""
        if path is None:
            path = SAVED_MODELS_DIR / "tfidf_vectorizer.joblib"
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        print(f"Loaded TF-IDF vectorizer from {path}")


class NgramExtractor:
    """
    N-gram feature extractor supporting multiple n-gram types.
    
    N-grams capture word sequences:
    - Unigrams: Single words
    - Bigrams: Two-word sequences
    - Trigrams: Three-word sequences
    
    Why use N-grams for fake news detection:
    - Capture phrasal patterns (e.g., "breaking news", "you won't believe")
    - Fake news often uses specific sensational phrases
    - Better context than individual words
    """
    
    def __init__(
        self,
        ngram_types: List[str] = ['unigram', 'bigram', 'trigram'],
        max_features_per_type: int = 5000,
        min_df: int = TFIDF_MIN_DF,
        max_df: float = TFIDF_MAX_DF
    ):
        """
        Initialize N-gram extractor.
        
        Args:
            ngram_types: List of n-gram types to extract
            max_features_per_type: Max features per n-gram type
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.ngram_types = ngram_types
        self.vectorizers = {}
        
        ngram_ranges = {
            'unigram': (1, 1),
            'bigram': (2, 2),
            'trigram': (3, 3)
        }
        
        for ngram_type in ngram_types:
            if ngram_type in ngram_ranges:
                self.vectorizers[ngram_type] = TfidfVectorizer(
                    max_features=max_features_per_type,
                    ngram_range=ngram_ranges[ngram_type],
                    min_df=min_df,
                    max_df=max_df,
                    lowercase=True
                )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'NgramExtractor':
        """Fit all vectorizers on training texts."""
        for name, vectorizer in self.vectorizers.items():
            vectorizer.fit(texts)
            print(f"{name} vocabulary size: {len(vectorizer.vocabulary_)}")
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to combined n-gram features.
        
        Returns concatenated features from all n-gram types.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizers not fitted.")
        
        from scipy.sparse import hstack
        
        features = []
        for vectorizer in self.vectorizers.values():
            features.append(vectorizer.transform(texts))
        
        return hstack(features)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    def get_all_feature_names(self) -> List[str]:
        """Get all feature names across all n-gram types."""
        all_names = []
        for name, vectorizer in self.vectorizers.items():
            names = vectorizer.get_feature_names_out().tolist()
            all_names.extend([f"{name}_{n}" for n in names])
        return all_names
    
    def save(self, path: Optional[str] = None):
        """Save all vectorizers."""
        if path is None:
            path = SAVED_MODELS_DIR / "ngram_vectorizers.joblib"
        joblib.dump(self.vectorizers, path)
        print(f"Saved N-gram vectorizers to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load all vectorizers."""
        if path is None:
            path = SAVED_MODELS_DIR / "ngram_vectorizers.joblib"
        self.vectorizers = joblib.load(path)
        self.is_fitted = True
        print(f"Loaded N-gram vectorizers from {path}")


class CombinedTextualFeatures:
    """
    Combined textual feature extractor.
    
    Combines BoW, TF-IDF, and N-gram features into a single
    feature matrix for model training.
    """
    
    def __init__(
        self,
        use_bow: bool = False,  # Disabled by default (TF-IDF is better)
        use_tfidf: bool = True,
        use_ngrams: bool = True,
        tfidf_max_features: int = TFIDF_MAX_FEATURES,
        ngram_max_features: int = 3000
    ):
        """
        Initialize combined extractor.
        
        Args:
            use_bow: Whether to include BoW features
            use_tfidf: Whether to include TF-IDF features
            use_ngrams: Whether to include N-gram features
            tfidf_max_features: Max features for TF-IDF
            ngram_max_features: Max features per N-gram type
        """
        self.extractors = {}
        
        if use_bow:
            self.extractors['bow'] = BagOfWordsExtractor(
                max_features=tfidf_max_features
            )
        
        if use_tfidf:
            self.extractors['tfidf'] = TfidfExtractor(
                max_features=tfidf_max_features
            )
        
        if use_ngrams:
            self.extractors['ngrams'] = NgramExtractor(
                ngram_types=['bigram', 'trigram'],
                max_features_per_type=ngram_max_features
            )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'CombinedTextualFeatures':
        """Fit all extractors."""
        for name, extractor in self.extractors.items():
            print(f"Fitting {name}...")
            extractor.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using all extractors."""
        if not self.is_fitted:
            raise ValueError("Extractors not fitted.")
        
        from scipy.sparse import hstack
        
        features = []
        for extractor in self.extractors.values():
            features.append(extractor.transform(texts))
        
        return hstack(features)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform."""
        self.fit(texts)
        return self.transform(texts)
    
    def get_tfidf_top_features(self, text: str, top_n: int = 10):
        """Get top TF-IDF features for explanation."""
        if 'tfidf' in self.extractors:
            return self.extractors['tfidf'].get_top_features(text, top_n)
        return []
    
    def save(self, directory: Optional[str] = None):
        """Save all extractors."""
        if directory is None:
            directory = SAVED_MODELS_DIR
        
        for name, extractor in self.extractors.items():
            extractor.save(Path(directory) / f"{name}_extractor.joblib")
    
    def load(self, directory: Optional[str] = None):
        """Load all extractors."""
        if directory is None:
            directory = SAVED_MODELS_DIR
        
        for name, extractor in self.extractors.items():
            extractor.load(Path(directory) / f"{name}_extractor.joblib")
        self.is_fitted = True


# Testing
if __name__ == "__main__":
    print("Testing Textual Features Module")
    print("=" * 50)
    
    # Sample texts for testing
    sample_texts = [
        "This is breaking news about the election results",
        "Scientists discover new treatment for disease",
        "You won't believe what happened next shocking revelation",
        "Stock market shows steady growth this quarter",
        "Celebrity scandal rocks Hollywood industry",
    ]
    
    # Test TF-IDF
    print("\n1. TF-IDF Extraction:")
    tfidf = TfidfExtractor(max_features=100, ngram_range=(1, 2))
    features = tfidf.fit_transform(sample_texts)
    print(f"   Feature matrix shape: {features.shape}")
    
    # Show top features for first text
    print(f"\n   Top features for: '{sample_texts[0]}'")
    top = tfidf.get_top_features(sample_texts[0], top_n=5)
    for word, score in top:
        print(f"   - {word}: {score:.4f}")
    
    # Test N-grams
    print("\n2. N-gram Extraction:")
    ngrams = NgramExtractor(ngram_types=['bigram', 'trigram'], max_features_per_type=50)
    ngram_features = ngrams.fit_transform(sample_texts)
    print(f"   N-gram feature matrix shape: {ngram_features.shape}")
    
    # Test combined features
    print("\n3. Combined Features:")
    combined = CombinedTextualFeatures(
        use_bow=False,
        use_tfidf=True,
        use_ngrams=True,
        tfidf_max_features=100,
        ngram_max_features=50
    )
    combined_features = combined.fit_transform(sample_texts)
    print(f"   Combined feature matrix shape: {combined_features.shape}")
    
    print("\nAll textual feature tests passed!")
