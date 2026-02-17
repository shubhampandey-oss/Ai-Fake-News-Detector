"""
Text Cleaning Module for Fake News Detection System

This module provides comprehensive text preprocessing functionality including:
- Lowercasing
- Punctuation and special character removal
- URL and email removal
- Stopword removal
- Lemmatization

Each preprocessing step is designed to normalize text while preserving
semantic meaning for credibility assessment.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import contractions

# Download required NLTK data
def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

# Initialize resources
download_nltk_resources()


class TextCleaner:
    """
    Comprehensive text cleaning class for preprocessing news articles.
    
    This cleaner is designed for fake news detection and maintains
    important stylistic features while removing noise.
    
    Attributes:
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to apply lemmatization
        min_word_length (int): Minimum word length to keep
    """
    
    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_length: int = 2,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the TextCleaner.
        
        Args:
            remove_stopwords: If True, removes common English stopwords
            lemmatize: If True, reduces words to their base form
            min_word_length: Minimum character length for words to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|bit\.ly/\S+'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.html_pattern = re.compile(r'<[^>]+>')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text (e.g., "don't" -> "do not").
        
        Why: Contractions can affect tokenization and feature extraction.
        Expanding them ensures consistent representation.
        
        Args:
            text: Input text with potential contractions
            
        Returns:
            Text with contractions expanded
        """
        try:
            return contractions.fix(text)
        except Exception:
            # Fallback if contractions library fails
            return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Why: URLs are not useful for linguistic analysis and can create
        noise in the feature space.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        return self.url_pattern.sub(' ', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Why: Emails are typically not informative for fake news detection
        and can create unnecessary unique tokens.
        
        Args:
            text: Input text
            
        Returns:
            Text with emails removed
        """
        return self.email_pattern.sub(' ', text)
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Why: HTML tags are formatting artifacts that don't carry
        semantic meaning for news classification.
        
        Args:
            text: Input text
            
        Returns:
            Text with HTML tags removed
        """
        return self.html_pattern.sub(' ', text)
    
    def remove_mentions_hashtags(self, text: str) -> str:
        """
        Remove social media mentions and hashtags.
        
        Why: While these could be informative, they create sparse features
        that may not generalize well across different news sources.
        
        Args:
            text: Input text
            
        Returns:
            Text with mentions and hashtags removed
        """
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        return text
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove standalone numbers from text.
        
        Why: Numbers without context (e.g., years, statistics) may not
        be consistently useful and can create noise.
        
        Args:
            text: Input text
            
        Returns:
            Text with numbers removed
        """
        return self.number_pattern.sub(' ', text)
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Why: Punctuation doesn't carry semantic meaning for bag-of-words
        or embedding-based features.
        
        Args:
            text: Input text
            
        Returns:
            Text with punctuation removed
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters while keeping letters, numbers, and spaces.
        
        Why: Special characters are noise that don't contribute to
        content-based fake news detection.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        return self.special_chars_pattern.sub(' ', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize multiple whitespaces to single space.
        
        Why: Consistent spacing ensures proper tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def to_lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Why: Reduces vocabulary size and ensures "Fake" and "fake"
        are treated as the same word.
        
        Args:
            text: Input text
            
        Returns:
            Lowercased text
        """
        return text.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Why: Tokenization is essential for all downstream NLP tasks
        including feature extraction and model training.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        try:
            return word_tokenize(text)
        except Exception:
            # Fallback to simple split if NLTK tokenizer fails
            return text.split()
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Why: Stopwords (the, is, at, which) don't carry discriminative
        information for classification and inflate feature space.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list without stopwords
        """
        return [
            token for token in tokens 
            if token.lower() not in self.stop_words
        ]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Why: Reduces vocabulary by mapping inflected forms to their
        base form (e.g., "running", "runs", "ran" -> "run").
        This improves generalization.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.
        
        Why: Very short tokens (single characters) are often noise
        or artifacts of preprocessing.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list with tokens meeting minimum length
        """
        return [
            token for token in tokens 
            if len(token) >= self.min_word_length
        ]
    
    def clean(self, text: str, return_tokens: bool = False):
        """
        Apply full cleaning pipeline to text.
        
        Pipeline order:
        1. Expand contractions
        2. Remove URLs
        3. Remove emails
        4. Remove HTML tags
        5. Remove mentions/hashtags
        6. Lowercase
        7. Remove punctuation
        8. Remove special characters
        9. Normalize whitespace
        10. Tokenize
        11. Remove stopwords (optional)
        12. Lemmatize (optional)
        13. Filter by length
        
        Args:
            text: Input text to clean
            return_tokens: If True, return list of tokens; else return string
            
        Returns:
            Cleaned text as string or list of tokens
        """
        if not text or not isinstance(text, str):
            return [] if return_tokens else ""
        
        # Apply cleaning steps
        text = self.expand_contractions(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_html_tags(text)
        text = self.remove_mentions_hashtags(text)
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Apply token-level processing
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        tokens = self.filter_by_length(tokens)
        
        if return_tokens:
            return tokens
        
        return ' '.join(tokens)
    
    def clean_batch(
        self, 
        texts: List[str], 
        return_tokens: bool = False,
        show_progress: bool = True
    ) -> List:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of texts to clean
            return_tokens: If True, return lists of tokens
            show_progress: If True, show progress bar
            
        Returns:
            List of cleaned texts or token lists
        """
        from tqdm import tqdm
        
        if show_progress:
            texts = tqdm(texts, desc="Cleaning texts")
        
        return [self.clean(text, return_tokens) for text in texts]


def get_sentence_count(text: str) -> int:
    """
    Count sentences in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of sentences
    """
    if not text:
        return 0
    try:
        return len(sent_tokenize(text))
    except Exception:
        # Fallback: count by common sentence endings
        return len(re.findall(r'[.!?]+', text))


def get_word_count(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of words
    """
    if not text:
        return 0
    return len(text.split())


# Example usage and testing
if __name__ == "__main__":
    # Test the cleaner
    sample_text = """
    BREAKING: You won't BELIEVE what happened!!! 
    Check out https://example.com for more info. 
    Contact us at news@example.com.
    <p>This is a <b>shocking</b> story!</p>
    @user mentioned #FakeNews in their post.
    The event happened in 2023 with 500 people attending.
    """
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean(sample_text)
    tokens = cleaner.clean(sample_text, return_tokens=True)
    
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(cleaned)
    print("\nTokens:")
    print(tokens)
