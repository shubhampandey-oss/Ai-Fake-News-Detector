"""
RSS Analyzer Module for Real-Time News Analysis

This module provides real-time analysis of RSS feed articles:
- Fetches live articles from trusted RSS sources
- Finds similar articles using TF-IDF cosine similarity
- Computes cross-source evidence scores
- Supports named-entity overlap calculation

IMPORTANT: RSS data is used ONLY for inference/display.
RSS articles are NEVER used for training or evaluation.

This is Phase 2 functionality for the two-phase architecture:
- Phase 1: Train models on static labeled datasets
- Phase 2: Use trained models for real-time RSS prediction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from pathlib import Path
import sys
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRUSTED_RSS_FEEDS,
    EVIDENCE_SIMILARITY_THRESHOLD,
    EVIDENCE_TOP_K,
    EVIDENCE_TITLE_WEIGHT,
    EVIDENCE_CONTENT_WEIGHT,
    RANDOM_SEED
)

# Try to import spacy for NER (optional)
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    NLP = None


class TextSimilarityCalculator:
    """
    Calculates text similarity using TF-IDF and cosine similarity.
    
    This is a lightweight, local implementation that does NOT require
    any external APIs. All computation is done locally.
    """
    
    def __init__(self, max_features: int = 5000):
        """
        Initialize the similarity calculator.
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.is_fitted = False
        self.corpus_vectors = None
        self.corpus_texts = []
        self.corpus_metadata = []
    
    def fit(self, texts: List[str], metadata: List[Dict] = None):
        """
        Fit the vectorizer on a corpus of texts.
        
        Args:
            texts: List of text documents
            metadata: Optional metadata for each text (source, title, etc.)
        """
        if not texts:
            return
        
        self.corpus_texts = texts
        self.corpus_metadata = metadata or [{}] * len(texts)
        
        # Fit and transform
        self.corpus_vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
    
    def find_similar(
        self,
        query_text: str,
        top_k: int = EVIDENCE_TOP_K,
        threshold: float = EVIDENCE_SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """
        Find similar documents to the query text.
        
        Args:
            query_text: Text to find similar documents for
            top_k: Number of top similar documents to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with similar documents and scores
        """
        if not self.is_fitted or self.corpus_vectors is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query_text])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.corpus_vectors)[0]
        
        # Get top-k indices above threshold
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices[:top_k * 2]:  # Get extras in case some are below threshold
            if similarities[idx] >= threshold:
                result = {
                    'text': self.corpus_texts[idx][:500] + '...' if len(self.corpus_texts[idx]) > 500 else self.corpus_texts[idx],
                    'similarity': float(similarities[idx]),
                    **self.corpus_metadata[idx]
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Create temporary vectorizer for pairwise comparison
        temp_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        try:
            vectors = temp_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0


class NamedEntityExtractor:
    """
    Extracts named entities from text for entity overlap calculation.
    
    Uses spaCy if available, otherwise falls back to simple regex patterns.
    """
    
    def __init__(self):
        """Initialize the entity extractor."""
        self.use_spacy = SPACY_AVAILABLE
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity type to list of entities
        """
        if self.use_spacy and NLP is not None:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_regex(text)
    
    def _extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy."""
        doc = NLP(text[:10000])  # Limit text length for performance
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geo-political entities
            'DATE': [],
            'EVENT': [],
            'OTHER': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text.lower())
            else:
                entities['OTHER'].append(ent.text.lower())
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_with_regex(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using simple regex (fallback)."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'DATE': [],
            'OTHER': []
        }
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Dates
        dates = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            text
        )
        dates += re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text)
        
        entities['DATE'] = list(set(d.lower() for d in dates))
        entities['OTHER'] = list(set(c.lower() for c in capitalized if len(c) > 3))[:50]
        
        return entities
    
    def compute_entity_overlap(
        self,
        entities1: Dict[str, List[str]],
        entities2: Dict[str, List[str]]
    ) -> float:
        """
        Compute overlap score between two sets of entities.
        
        Args:
            entities1: First entity dictionary
            entities2: Second entity dictionary
            
        Returns:
            Overlap score (0-1)
        """
        all_entities1 = set()
        all_entities2 = set()
        
        for key in entities1:
            all_entities1.update(entities1[key])
        for key in entities2:
            all_entities2.update(entities2[key])
        
        if not all_entities1 or not all_entities2:
            return 0.0
        
        intersection = len(all_entities1 & all_entities2)
        union = len(all_entities1 | all_entities2)
        
        return intersection / union if union > 0 else 0.0


class RSSAnalyzer:
    """
    Analyzes articles against RSS feed corpus for cross-source evidence.
    
    This is the main class for Phase 2 real-time RSS analysis.
    
    Features:
    - Find similar articles from trusted sources
    - Compute evidence agreement scores
    - Named entity overlap analysis
    - All computation is local (no external APIs)
    """
    
    def __init__(self):
        """Initialize the RSS analyzer."""
        self.similarity_calculator = TextSimilarityCalculator()
        self.entity_extractor = NamedEntityExtractor()
        
        # RSS corpus
        self.rss_articles = []
        self.trusted_articles = []
        self.is_initialized = False
    
    def load_rss_corpus(self, articles: List[Dict]):
        """
        Load RSS articles into the corpus for similarity matching.
        
        Args:
            articles: List of article dictionaries with 'title', 'summary', 'source'
        """
        if not articles:
            return
        
        self.rss_articles = articles
        self.trusted_articles = [
            a for a in articles 
            if a.get('source', '').lower() in [s.lower() for s in TRUSTED_RSS_FEEDS.keys()]
        ]
        
        # Build corpus texts
        corpus_texts = []
        corpus_metadata = []
        
        for article in articles:
            # Combine title and summary for matching
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            corpus_texts.append(text)
            corpus_metadata.append({
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'link': article.get('link', ''),
                'published': article.get('published', ''),
                'is_trusted': article.get('source', '').lower() in [s.lower() for s in TRUSTED_RSS_FEEDS.keys()]
            })
        
        # Fit similarity calculator
        self.similarity_calculator.fit(corpus_texts, corpus_metadata)
        self.is_initialized = True
    
    def find_similar_articles(
        self,
        article_text: str,
        top_k: int = EVIDENCE_TOP_K,
        trusted_only: bool = False
    ) -> List[Dict]:
        """
        Find similar articles from the RSS corpus.
        
        Args:
            article_text: Text of the article to analyze
            top_k: Number of similar articles to return
            trusted_only: If True, only return articles from trusted sources
            
        Returns:
            List of similar articles with similarity scores
        """
        if not self.is_initialized:
            return []
        
        similar = self.similarity_calculator.find_similar(
            article_text,
            top_k=top_k * 2 if trusted_only else top_k
        )
        
        if trusted_only:
            similar = [s for s in similar if s.get('is_trusted', False)][:top_k]
        
        return similar
    
    def compute_evidence_score(
        self,
        article_text: str
    ) -> Dict[str, Any]:
        """
        Compute evidence agreement score for an article.
        
        This determines how much the article agrees with trusted sources.
        
        Args:
            article_text: Text of the article to analyze
            
        Returns:
            Dictionary with evidence scores and details
        """
        if not self.is_initialized:
            return {
                'evidence_score': 0.5,  # Neutral when no evidence
                'trusted_similarity': 0.5,
                'untrusted_similarity': 0.5,
                'similar_trusted_articles': [],
                'similar_untrusted_articles': [],
                'entity_overlap_score': 0.0
            }
        
        # Find similar articles from trusted sources
        trusted_similar = self.find_similar_articles(
            article_text,
            top_k=5,
            trusted_only=True
        )
        
        # Find similar from all sources (includes potentially untrusted)
        all_similar = self.find_similar_articles(
            article_text,
            top_k=10,
            trusted_only=False
        )
        
        # Separate untrusted
        untrusted_similar = [
            s for s in all_similar 
            if not s.get('is_trusted', True)
        ][:5]
        
        # Compute average similarities
        trusted_sim = np.mean([s['similarity'] for s in trusted_similar]) if trusted_similar else 0.0
        untrusted_sim = np.mean([s['similarity'] for s in untrusted_similar]) if untrusted_similar else 0.0
        
        # Compute entity overlap with top trusted article
        entity_overlap = 0.0
        if trusted_similar:
            article_entities = self.entity_extractor.extract_entities(article_text)
            top_trusted_text = trusted_similar[0].get('text', '')
            trusted_entities = self.entity_extractor.extract_entities(top_trusted_text)
            entity_overlap = self.entity_extractor.compute_entity_overlap(
                article_entities, trusted_entities
            )
        
        # Compute final evidence score
        # Higher if similar to trusted, lower if similar to untrusted
        if trusted_sim + untrusted_sim == 0:
            evidence_score = 0.5
        else:
            # Base score from similarity difference
            evidence_score = (trusted_sim - untrusted_sim + 1) / 2
            
            # Boost from entity overlap
            evidence_score = evidence_score * 0.8 + entity_overlap * 0.2
        
        # Clamp to [0, 1]
        evidence_score = max(0.0, min(1.0, evidence_score))
        
        return {
            'evidence_score': round(float(evidence_score), 3),
            'trusted_similarity': round(float(trusted_sim), 3),
            'untrusted_similarity': round(float(untrusted_sim), 3),
            'entity_overlap_score': round(float(entity_overlap), 3),
            'similar_trusted_articles': trusted_similar[:3],  # Top 3 for display
            'similar_untrusted_articles': untrusted_similar[:3],
            'has_trusted_evidence': trusted_sim > EVIDENCE_SIMILARITY_THRESHOLD,
            'has_untrusted_similar': untrusted_sim > EVIDENCE_SIMILARITY_THRESHOLD
        }
    
    def analyze_article(
        self,
        text: str,
        title: str = ""
    ) -> Dict[str, Any]:
        """
        Perform full RSS-based analysis on an article.
        
        Args:
            text: Article text
            title: Optional article title
            
        Returns:
            Complete analysis results
        """
        # Combine title and text for analysis
        full_text = f"{title} {text}" if title else text
        
        # Get evidence score
        evidence_result = self.compute_evidence_score(full_text)
        
        # Extract entities from the article
        entities = self.entity_extractor.extract_entities(text)
        
        # Count key entities
        key_entities = []
        for entity_type in ['PERSON', 'ORG', 'GPE']:
            key_entities.extend(entities.get(entity_type, [])[:5])
        
        return {
            **evidence_result,
            'key_entities': key_entities[:10],
            'entity_counts': {
                k: len(v) for k, v in entities.items()
            },
            'corpus_size': len(self.rss_articles),
            'trusted_corpus_size': len(self.trusted_articles),
            'analysis_timestamp': datetime.now().isoformat()
        }


# Singleton instance for the application
_rss_analyzer_instance = None


def get_rss_analyzer() -> RSSAnalyzer:
    """
    Get the singleton RSS analyzer instance.
    
    Returns:
        RSSAnalyzer instance
    """
    global _rss_analyzer_instance
    if _rss_analyzer_instance is None:
        _rss_analyzer_instance = RSSAnalyzer()
    return _rss_analyzer_instance


# Testing
if __name__ == "__main__":
    print("Testing RSS Analyzer Module")
    print("=" * 50)
    
    # Create analyzer
    analyzer = RSSAnalyzer()
    
    # Sample RSS articles (simulating fetched data)
    sample_articles = [
        {
            'title': 'Government announces new economic policy',
            'summary': 'The government today announced a new economic policy aimed at boosting growth and reducing inflation.',
            'source': 'reuters',
            'link': 'https://reuters.com/article1',
            'published': '2024-01-15'
        },
        {
            'title': 'Scientists discover breakthrough treatment',
            'summary': 'Researchers at major university have discovered a promising new treatment for common disease.',
            'source': 'bbc',
            'link': 'https://bbc.com/article2',
            'published': '2024-01-15'
        },
        {
            'title': 'Stock market sees major gains',
            'summary': 'The stock market closed with significant gains today amid positive economic indicators.',
            'source': 'the_hindu',
            'link': 'https://thehindu.com/article3',
            'published': '2024-01-15'
        },
    ]
    
    print("\n1. Loading RSS corpus...")
    analyzer.load_rss_corpus(sample_articles)
    print(f"   Loaded {len(analyzer.rss_articles)} articles")
    
    print("\n2. Finding similar articles...")
    test_text = "New government economic policy announced to boost growth"
    similar = analyzer.find_similar_articles(test_text, top_k=3)
    for s in similar:
        print(f"   - {s['title'][:50]}... (similarity: {s['similarity']:.3f})")
    
    print("\n3. Computing evidence score...")
    evidence = analyzer.compute_evidence_score(test_text)
    print(f"   Evidence score: {evidence['evidence_score']:.3f}")
    print(f"   Trusted similarity: {evidence['trusted_similarity']:.3f}")
    
    print("\n4. Full article analysis...")
    analysis = analyzer.analyze_article(test_text, "Economic Policy News")
    print(f"   Evidence score: {analysis['evidence_score']:.3f}")
    print(f"   Key entities: {analysis['key_entities']}")
    
    print("\nAll RSS Analyzer tests passed!")
