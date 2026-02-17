"""
Data Loader Module for Fake News Detection System

This module handles loading, preprocessing, and splitting datasets for
training fake news detection models. Supports multiple dataset formats
and implements class imbalance handling.

Supported datasets:
- Kaggle Fake News Dataset (primary)
- LIAR Dataset (optional extension)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, USE_SMOTE, USE_CLASS_WEIGHTS,
    MIN_DOC_LENGTH, MAX_DOC_LENGTH
)
from preprocessing.text_cleaner import TextCleaner


class FakeNewsDataLoader:
    """
    Data loader for fake news detection datasets.
    
    This class handles loading raw data, preprocessing, splitting,
    and balancing classes for model training.
    
    Attributes:
        cleaner: TextCleaner instance for preprocessing
        data: Loaded and preprocessed DataFrame
    """
    
    def __init__(self, clean_text: bool = True):
        """
        Initialize the data loader.
        
        Args:
            clean_text: If True, apply text cleaning during loading
        """
        self.cleaner = TextCleaner() if clean_text else None
        self.data = None
        self._label_mapping = {"FAKE": 0, "REAL": 1}
        self._inverse_label_mapping = {0: "FAKE", 1: "REAL"}
    
    def load_kaggle_dataset(
        self,
        fake_path: Optional[str] = None,
        real_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load the Kaggle Fake News dataset.
        
        The Kaggle dataset typically comes as two CSV files:
        - Fake.csv: Contains fake news articles
        - True.csv: Contains real news articles
        
        Each file contains columns: title, text, subject, date
        
        Args:
            fake_path: Path to Fake.csv (or None for default)
            real_path: Path to True.csv (or None for default)
            
        Returns:
            Combined DataFrame with fake and real news
        """
        # Default paths
        if fake_path is None:
            fake_path = RAW_DATA_DIR / "Fake.csv"
        if real_path is None:
            real_path = RAW_DATA_DIR / "True.csv"
        
        # Check if files exist
        if not os.path.exists(fake_path) or not os.path.exists(real_path):
            print(f"Dataset files not found at:")
            print(f"  - {fake_path}")
            print(f"  - {real_path}")
            print("\nPlease download the Kaggle Fake News dataset and place files in:")
            print(f"  {RAW_DATA_DIR}")
            print("\nDataset URL: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
            return None
        
        # Load datasets
        print("Loading Kaggle Fake News dataset...")
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Add labels
        fake_df['label'] = 0  # FAKE
        real_df['label'] = 1  # REAL
        
        # Add source information based on subject
        fake_df['source_type'] = 'unreliable'
        real_df['source_type'] = 'reliable'
        
        # Combine datasets
        data = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Combine title and text for full content
        data['full_text'] = data['title'].fillna('') + ' ' + data['text'].fillna('')
        
        print(f"Loaded {len(fake_df)} fake and {len(real_df)} real articles")
        print(f"Total: {len(data)} articles")
        
        return data
    
    def load_liar_dataset(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load the LIAR dataset.
        
        The LIAR dataset contains short statements with labels:
        pants-fire, false, barely-true, half-true, mostly-true, true
        
        For binary classification, we map:
        - FAKE: pants-fire, false, barely-true
        - REAL: half-true, mostly-true, true
        
        Args:
            train_path: Path to train.tsv
            val_path: Path to valid.tsv  
            test_path: Path to test.tsv
            
        Returns:
            Combined DataFrame
        """
        # Default paths
        if train_path is None:
            train_path = RAW_DATA_DIR / "liar" / "train.tsv"
        if val_path is None:
            val_path = RAW_DATA_DIR / "liar" / "valid.tsv"
        if test_path is None:
            test_path = RAW_DATA_DIR / "liar" / "test.tsv"
        
        # LIAR dataset columns
        columns = [
            'id', 'label', 'statement', 'subject', 'speaker',
            'speaker_job', 'state_info', 'party', 'barely_true_count',
            'false_count', 'half_true_count', 'mostly_true_count',
            'pants_on_fire_count', 'context'
        ]
        
        dfs = []
        for path, name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
            if os.path.exists(path):
                df = pd.read_csv(path, sep='\t', header=None, names=columns)
                df['split'] = name
                dfs.append(df)
                print(f"Loaded {len(df)} samples from {name}")
        
        if not dfs:
            print("LIAR dataset not found.")
            return None
        
        data = pd.concat(dfs, ignore_index=True)
        
        # Map labels to binary
        fake_labels = ['pants-fire', 'false', 'barely-true']
        real_labels = ['half-true', 'mostly-true', 'true']
        
        data['binary_label'] = data['label'].apply(
            lambda x: 0 if x in fake_labels else (1 if x in real_labels else None)
        )
        
        # Remove samples with unknown labels
        data = data.dropna(subset=['binary_label'])
        data['label'] = data['binary_label'].astype(int)
        
        # Use statement as full_text
        data['full_text'] = data['statement']
        
        print(f"Total: {len(data)} statements")
        
        return data
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        text_column: str = 'full_text'
    ) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Steps:
        1. Remove duplicates
        2. Remove empty texts
        3. Filter by document length
        4. Clean text (if cleaner is initialized)
        
        Args:
            data: Raw DataFrame
            text_column: Column containing text to process
            
        Returns:
            Preprocessed DataFrame
        """
        print("\nPreprocessing data...")
        original_len = len(data)
        
        # Remove duplicates
        data = data.drop_duplicates(subset=[text_column])
        print(f"After removing duplicates: {len(data)} (removed {original_len - len(data)})")
        
        # Remove empty texts
        data = data.dropna(subset=[text_column])
        data = data[data[text_column].str.strip().str.len() > 0]
        print(f"After removing empty texts: {len(data)}")
        
        # Filter by document length
        data['word_count'] = data[text_column].apply(lambda x: len(str(x).split()))
        data = data[
            (data['word_count'] >= MIN_DOC_LENGTH) & 
            (data['word_count'] <= MAX_DOC_LENGTH)
        ]
        print(f"After filtering by length ({MIN_DOC_LENGTH}-{MAX_DOC_LENGTH} words): {len(data)}")
        
        # Clean text
        if self.cleaner:
            print("Cleaning texts...")
            data['cleaned_text'] = self.cleaner.clean_batch(
                data[text_column].tolist(),
                return_tokens=False,
                show_progress=True
            )
        else:
            data['cleaned_text'] = data[text_column]
        
        # Remove samples that became too short after cleaning
        data = data[data['cleaned_text'].str.split().str.len() >= MIN_DOC_LENGTH // 2]
        
        print(f"Final dataset size: {len(data)}")
        
        return data
    
    def get_class_distribution(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Args:
            data: DataFrame with 'label' column
            
        Returns:
            Dictionary with class counts
        """
        distribution = data['label'].value_counts().to_dict()
        named_distribution = {
            self._inverse_label_mapping.get(k, k): v 
            for k, v in distribution.items()
        }
        return named_distribution
    
    def split_data(
        self,
        data: pd.DataFrame,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Uses stratified splitting to maintain class distribution
        across all splits.
        
        Args:
            data: Full DataFrame
            stratify: If True, maintain class distribution in splits
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\nSplitting data...")
        
        stratify_col = data['label'] if stratify else None
        
        # First split: separate test set
        train_val, test = train_test_split(
            data,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=stratify_col
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        stratify_col = train_val['label'] if stratify else None
        
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=RANDOM_SEED,
            stratify=stratify_col
        )
        
        print(f"Train: {len(train)} samples")
        print(f"Validation: {len(val)} samples")
        print(f"Test: {len(test)} samples")
        
        # Show class distribution in each split
        for name, split in [('Train', train), ('Val', val), ('Test', test)]:
            dist = self.get_class_distribution(split)
            print(f"  {name} distribution: {dist}")
        
        return train, val, test
    
    def compute_class_weights(
        self,
        labels: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute class weights for handling imbalance.
        
        Class weights are inversely proportional to class frequencies,
        helping the model pay more attention to minority classes.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary mapping class index to weight
        """
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return dict(enumerate(weights))
    
    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique).
        
        SMOTE generates synthetic samples for the minority class by
        interpolating between existing minority samples, helping
        balance the dataset.
        
        Note: Only apply to training data, never to validation/test.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        try:
            from imblearn.over_sampling import SMOTE
            
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=RANDOM_SEED)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"Before SMOTE: {len(y)} samples")
            print(f"After SMOTE: {len(y_resampled)} samples")
            
            return X_resampled, y_resampled
            
        except ImportError:
            print("Warning: imbalanced-learn not installed. Skipping SMOTE.")
            return X, y
    
    def save_processed_data(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        prefix: str = "kaggle"
    ):
        """
        Save processed datasets to disk.
        
        Args:
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame
            prefix: Filename prefix
        """
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        train.to_csv(PROCESSED_DATA_DIR / f"{prefix}_train.csv", index=False)
        val.to_csv(PROCESSED_DATA_DIR / f"{prefix}_val.csv", index=False)
        test.to_csv(PROCESSED_DATA_DIR / f"{prefix}_test.csv", index=False)
        
        print(f"\nSaved processed data to {PROCESSED_DATA_DIR}")
    
    def load_processed_data(
        self,
        prefix: str = "kaggle"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously processed datasets.
        
        Args:
            prefix: Filename prefix
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        train = pd.read_csv(PROCESSED_DATA_DIR / f"{prefix}_train.csv")
        val = pd.read_csv(PROCESSED_DATA_DIR / f"{prefix}_val.csv")
        test = pd.read_csv(PROCESSED_DATA_DIR / f"{prefix}_test.csv")
        
        return train, val, test
    
    def prepare_for_training(
        self,
        text_column: str = 'cleaned_text',
        label_column: str = 'label'
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Full pipeline: load, preprocess, split, and prepare data.
        
        Returns:
            Dictionary containing train/val/test data and metadata
        """
        # Try to load processed data first
        try:
            train, val, test = self.load_processed_data()
            print("Loaded preprocessed data from disk.")
        except FileNotFoundError:
            # Load and process from scratch
            data = self.load_kaggle_dataset()
            if data is None:
                return None
            
            data = self.preprocess_data(data)
            train, val, test = self.split_data(data)
            self.save_processed_data(train, val, test)
        
        # Prepare output
        result = {
            'train_texts': train[text_column].values,
            'train_labels': train[label_column].values,
            'val_texts': val[text_column].values,
            'val_labels': val[label_column].values,
            'test_texts': test[text_column].values,
            'test_labels': test[label_column].values,
            'train_df': train,
            'val_df': val,
            'test_df': test,
        }
        
        # Compute class weights if needed
        if USE_CLASS_WEIGHTS:
            result['class_weights'] = self.compute_class_weights(
                result['train_labels']
            )
            print(f"Class weights: {result['class_weights']}")
        
        return result


# Convenience function
def load_data(clean_text: bool = True) -> Dict:
    """
    Convenience function to load and prepare data.
    
    Args:
        clean_text: If True, apply text cleaning
        
    Returns:
        Dictionary with prepared data
    """
    loader = FakeNewsDataLoader(clean_text=clean_text)
    return loader.prepare_for_training()


# Testing
if __name__ == "__main__":
    print("Testing Data Loader...")
    print("=" * 50)
    
    # Test with sample data
    loader = FakeNewsDataLoader(clean_text=True)
    
    # Try to load Kaggle dataset
    data = loader.load_kaggle_dataset()
    
    if data is not None:
        # Show sample
        print("\nSample data:")
        print(data[['title', 'label']].head())
        
        # Preprocess
        data = loader.preprocess_data(data)
        
        # Split
        train, val, test = loader.split_data(data)
        
        # Save
        loader.save_processed_data(train, val, test)
    else:
        print("\nTo test this module, please download the Kaggle dataset.")
        print("Creating sample data for demonstration...")
        
        # Create minimal sample data for testing
        sample_data = pd.DataFrame({
            'title': ['Breaking News: Major Event'] * 5 + ['Normal News Story'] * 5,
            'text': ['This is fake content with sensational claims!'] * 5 + 
                   ['This is factual reporting from reliable sources.'] * 5,
            'label': [0] * 5 + [1] * 5,
            'full_text': ['Fake news content'] * 5 + ['Real news content'] * 5
        })
        
        print("\nSample data created:")
        print(sample_data[['title', 'label']].head(10))
        print("\nClass distribution:", loader.get_class_distribution(sample_data))
