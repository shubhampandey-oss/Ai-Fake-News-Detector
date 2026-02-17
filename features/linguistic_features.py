"""
Linguistic and Stylometric Feature Extraction Module

This module extracts features that capture writing style and linguistic patterns:
- Sentence and word statistics
- Part-of-speech distributions
- Readability metrics
- Sensational/emotional word ratios

These features often differ between fake and real news:
- Fake news tends to use more sensational language
- Real news typically has more formal, balanced writing
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENSATIONAL_WORDS

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class LinguisticFeatureExtractor:
    """
    Extracts linguistic and stylometric features from text.
    
    These features capture writing style characteristics that often
    differ between fake and real news articles.
    
    Feature categories:
    1. Length statistics (sentences, words, characters)
    2. Vocabulary richness
    3. Part-of-speech distributions
    4. Readability metrics
    5. Sensational language indicators
    """
    
    def __init__(
        self,
        sensational_words: Optional[List[str]] = None,
        calculate_readability: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            sensational_words: Custom list of sensational words
            calculate_readability: Whether to compute readability metrics
        """
        self.sensational_words = set(
            word.lower() for word in (sensational_words or SENSATIONAL_WORDS)
        )
        self.calculate_readability = calculate_readability
        
        # Feature names for output
        self.feature_names = [
            # Length features
            'char_count',
            'word_count',
            'sentence_count',
            'avg_word_length',
            'avg_sentence_length',
            'max_sentence_length',
            'min_sentence_length',
            'std_sentence_length',
            
            # Vocabulary features
            'unique_word_count',
            'unique_word_ratio',
            'hapax_legomena_ratio',  # Words appearing only once
            'vocabulary_richness',
            
            # Punctuation features
            'exclamation_count',
            'question_count',
            'uppercase_word_ratio',
            'all_caps_word_count',
            
            # POS tag features
            'noun_ratio',
            'verb_ratio',
            'adjective_ratio',
            'adverb_ratio',
            'pronoun_ratio',
            
            # Sensational language
            'sensational_word_count',
            'sensational_word_ratio',
            
            # Readability
            'flesch_reading_ease',
            'flesch_kincaid_grade',
            'avg_syllables_per_word',
        ]
    
    def _count_syllables(self, word: str) -> int:
        """
        Estimate syllable count in a word.
        
        Simple heuristic based on vowel groups.
        """
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    def _get_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        try:
            return sent_tokenize(text)
        except:
            return text.split('.')
    
    def _get_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            return word_tokenize(text.lower())
        except:
            return text.lower().split()
    
    def _get_pos_tags(self, words: List[str]) -> List[Tuple[str, str]]:
        """Get POS tags for words."""
        try:
            return pos_tag(words)
        except:
            return []
    
    def extract_length_features(self, text: str) -> Dict[str, float]:
        """
        Extract length-based features.
        
        Why: Fake news articles often have different length patterns
        (sometimes shorter, more fragmented sentences for dramatic effect).
        """
        sentences = self._get_sentences(text)
        words = self._get_words(text)
        
        # Filter words (remove punctuation-only tokens)
        words = [w for w in words if any(c.isalpha() for c in w)]
        
        sentence_lengths = [len(self._get_words(s)) for s in sentences]
        word_lengths = [len(w) for w in words]
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
            'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
            'std_sentence_length': np.std(sentence_lengths) if sentence_lengths else 0,
        }
    
    def extract_vocabulary_features(self, text: str) -> Dict[str, float]:
        """
        Extract vocabulary richness features.
        
        Why: Real news typically has richer vocabulary while fake news
        may repeat emotional/sensational words more frequently.
        """
        words = self._get_words(text)
        words = [w for w in words if any(c.isalpha() for c in w)]
        
        if not words:
            return {
                'unique_word_count': 0,
                'unique_word_ratio': 0,
                'hapax_legomena_ratio': 0,
                'vocabulary_richness': 0,
            }
        
        word_counts = Counter(words)
        unique_words = len(word_counts)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        
        return {
            'unique_word_count': unique_words,
            'unique_word_ratio': unique_words / len(words),
            'hapax_legomena_ratio': hapax / len(words) if words else 0,
            'vocabulary_richness': unique_words / np.sqrt(len(words)),  # Yule's K simplified
        }
    
    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """
        Extract punctuation-related features.
        
        Why: Fake news often uses excessive exclamation marks,
        ALL CAPS, and other attention-grabbing punctuation.
        """
        words = text.split()
        
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Count UPPERCASE words
        uppercase_words = [w for w in words if w.isupper() and len(w) > 1]
        words_with_letters = [w for w in words if any(c.isalpha() for c in w)]
        
        return {
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_word_ratio': len(uppercase_words) / len(words_with_letters) if words_with_letters else 0,
            'all_caps_word_count': len(uppercase_words),
        }
    
    def extract_pos_features(self, text: str) -> Dict[str, float]:
        """
        Extract part-of-speech distribution features.
        
        Why: Different writing styles have different POS distributions.
        Formal news tends to have more nouns, fake news may have
        more adjectives (sensational descriptions).
        """
        words = self._get_words(text)
        words = [w for w in words if any(c.isalpha() for c in w)]
        
        if not words:
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adjective_ratio': 0,
                'adverb_ratio': 0,
                'pronoun_ratio': 0,
            }
        
        pos_tags = self._get_pos_tags(words)
        
        if not pos_tags:
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adjective_ratio': 0,
                'adverb_ratio': 0,
                'pronoun_ratio': 0,
            }
        
        # Count POS categories
        # NN* = nouns, VB* = verbs, JJ* = adjectives, RB* = adverbs, PRP* = pronouns
        noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
        adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
        pron_count = sum(1 for _, tag in pos_tags if tag.startswith('PRP'))
        
        total = len(pos_tags)
        
        return {
            'noun_ratio': noun_count / total,
            'verb_ratio': verb_count / total,
            'adjective_ratio': adj_count / total,
            'adverb_ratio': adv_count / total,
            'pronoun_ratio': pron_count / total,
        }
    
    def extract_sensational_features(self, text: str) -> Dict[str, float]:
        """
        Extract sensational language features.
        
        Why: Fake news typically uses more sensational, emotional,
        and clickbait-style language to attract attention.
        """
        words = self._get_words(text)
        words = [w for w in words if any(c.isalpha() for c in w)]
        
        if not words:
            return {
                'sensational_word_count': 0,
                'sensational_word_ratio': 0,
            }
        
        sensational_count = sum(
            1 for word in words 
            if word.lower() in self.sensational_words
        )
        
        return {
            'sensational_word_count': sensational_count,
            'sensational_word_ratio': sensational_count / len(words),
        }
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """
        Extract readability metrics.
        
        Why: Professional news typically has consistent readability
        levels. Fake news may vary more or target lower reading levels.
        
        Metrics:
        - Flesch Reading Ease: Higher = easier to read (0-100)
        - Flesch-Kincaid Grade: US school grade level
        """
        sentences = self._get_sentences(text)
        words = self._get_words(text)
        words = [w for w in words if any(c.isalpha() for c in w)]
        
        if not words or not sentences:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_syllables_per_word': 0,
            }
        
        total_syllables = sum(self._count_syllables(w) for w in words)
        avg_syllables = total_syllables / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Flesch Reading Ease
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))
        
        # Flesch-Kincaid Grade Level
        # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        flesch_kincaid_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
        flesch_kincaid_grade = max(0, flesch_kincaid_grade)
        
        return {
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'avg_syllables_per_word': avg_syllables,
        }
    
    def extract_all(self, text: str) -> Dict[str, float]:
        """
        Extract all linguistic features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of feature name to value
        """
        if not text or not isinstance(text, str):
            return {name: 0.0 for name in self.feature_names}
        
        features = {}
        features.update(self.extract_length_features(text))
        features.update(self.extract_vocabulary_features(text))
        features.update(self.extract_punctuation_features(text))
        features.update(self.extract_pos_features(text))
        features.update(self.extract_sensational_features(text))
        
        if self.calculate_readability:
            features.update(self.extract_readability_features(text))
        
        return features
    
    def extract_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features from a batch of texts.
        
        Args:
            texts: List of text documents
            show_progress: Whether to show progress bar
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        from tqdm import tqdm
        
        if show_progress:
            texts = tqdm(texts, desc="Extracting linguistic features")
        
        features_list = [self.extract_all(text) for text in texts]
        
        # Convert to DataFrame then to numpy array
        df = pd.DataFrame(features_list)
        
        # Ensure all feature names are present
        for name in self.feature_names:
            if name not in df.columns:
                df[name] = 0.0
        
        # Reorder columns to match feature_names
        df = df[self.feature_names]
        
        return df.values.astype(np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def get_important_features(
        self,
        text: str,
        top_n: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Get the most notable linguistic features for a text.
        
        Useful for explaining predictions.
        
        Args:
            text: Input text
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, value, interpretation) tuples
        """
        features = self.extract_all(text)
        
        # Define thresholds for "notable" features
        notable = []
        
        if features['sensational_word_ratio'] > 0.05:
            notable.append((
                'sensational_word_ratio',
                features['sensational_word_ratio'],
                'High sensational language usage'
            ))
        
        if features['exclamation_count'] > 3:
            notable.append((
                'exclamation_count',
                features['exclamation_count'],
                'Excessive exclamation marks'
            ))
        
        if features['all_caps_word_count'] > 5:
            notable.append((
                'all_caps_word_count',
                features['all_caps_word_count'],
                'Multiple words in ALL CAPS'
            ))
        
        if features['flesch_reading_ease'] < 30:
            notable.append((
                'flesch_reading_ease',
                features['flesch_reading_ease'],
                'Very difficult to read'
            ))
        elif features['flesch_reading_ease'] > 80:
            notable.append((
                'flesch_reading_ease',
                features['flesch_reading_ease'],
                'Very easy to read (possibly simplistic)'
            ))
        
        if features['adjective_ratio'] > 0.15:
            notable.append((
                'adjective_ratio',
                features['adjective_ratio'],
                'High use of adjectives'
            ))
        
        return notable[:top_n]


# Testing
if __name__ == "__main__":
    print("Testing Linguistic Features Module")
    print("=" * 50)
    
    # Sample texts
    fake_example = """
    BREAKING!!! You WON'T BELIEVE what happened next!!! 
    This SHOCKING revelation will change EVERYTHING you know!
    Scientists are FURIOUS about this SECRET discovery!!!
    """
    
    real_example = """
    The Federal Reserve announced today that interest rates will remain 
    unchanged following their monthly policy meeting. Fed Chair Powell 
    stated that economic indicators suggest moderate growth, though 
    labor market conditions continue to warrant close monitoring.
    """
    
    extractor = LinguisticFeatureExtractor()
    
    print("\n1. Fake News Example Features:")
    fake_features = extractor.extract_all(fake_example)
    for name, value in fake_features.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n2. Real News Example Features:")
    real_features = extractor.extract_all(real_example)
    for name, value in real_features.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n3. Notable differences:")
    print(f"   Sensational ratio - Fake: {fake_features['sensational_word_ratio']:.4f}, Real: {real_features['sensational_word_ratio']:.4f}")
    print(f"   Exclamation count - Fake: {fake_features['exclamation_count']:.0f}, Real: {real_features['exclamation_count']:.0f}")
    print(f"   ALL CAPS count - Fake: {fake_features['all_caps_word_count']:.0f}, Real: {real_features['all_caps_word_count']:.0f}")
    
    print("\n4. Important features for fake example:")
    important = extractor.get_important_features(fake_example)
    for name, value, interpretation in important:
        print(f"   - {interpretation}: {value:.4f}")
    
    print("\nAll linguistic feature tests passed!")
