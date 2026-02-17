"""
Evidence Aggregation Module

This module aggregates evidence from multiple sources:
- ML model predictions
- Source credibility scores
- Cross-source similarity analysis

Produces final credibility assessment with explanations.

IMPORTANT: This performs probabilistic assessment, NOT absolute
truth verification. All outputs should be interpreted as
"likelihood of credibility" based on available evidence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    WEIGHT_MODEL_PREDICTION,
    WEIGHT_SOURCE_CREDIBILITY,
    WEIGHT_EVIDENCE_AGREEMENT,
    SIMILARITY_THRESHOLD
)


class EvidenceAggregator:
    """
    Aggregates multiple evidence sources for credibility assessment.
    
    Evidence sources:
    1. ML Model Prediction: Binary classification with confidence
    2. Source Credibility: Pre-compiled source reputation scores
    3. Cross-Source Agreement: Similarity with trusted/untrusted sources
    
    The final score is a weighted combination of these factors.
    """
    
    def __init__(
        self,
        weight_model: float = WEIGHT_MODEL_PREDICTION,
        weight_source: float = WEIGHT_SOURCE_CREDIBILITY,
        weight_evidence: float = WEIGHT_EVIDENCE_AGREEMENT
    ):
        """
        Initialize aggregator.
        
        Args:
            weight_model: Weight for ML model prediction (default: 0.50)
            weight_source: Weight for source credibility (default: 0.30)
            weight_evidence: Weight for evidence agreement (default: 0.20)
        """
        self.weight_model = weight_model
        self.weight_source = weight_source
        self.weight_evidence = weight_evidence
        
        # Normalize weights
        total = weight_model + weight_source + weight_evidence
        self.weight_model /= total
        self.weight_source /= total
        self.weight_evidence /= total
    
    def compute_cross_source_similarity(
        self,
        article_embedding: np.ndarray,
        trusted_embeddings: np.ndarray,
        untrusted_embeddings: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute similarity with trusted and untrusted source embeddings.
        
        Args:
            article_embedding: Embedding of the article to assess
            trusted_embeddings: Embeddings from trusted sources
            untrusted_embeddings: Embeddings from untrusted sources
            
        Returns:
            Tuple of (avg_similarity_to_trusted, avg_similarity_to_untrusted)
        """
        if len(article_embedding.shape) == 1:
            article_embedding = article_embedding.reshape(1, -1)
        
        # Compute similarities
        trusted_sim = 0.0
        untrusted_sim = 0.0
        
        if len(trusted_embeddings) > 0:
            similarities = cosine_similarity(article_embedding, trusted_embeddings)
            trusted_sim = float(np.mean(np.max(similarities, axis=1)))
        
        if len(untrusted_embeddings) > 0:
            similarities = cosine_similarity(article_embedding, untrusted_embeddings)
            untrusted_sim = float(np.mean(np.max(similarities, axis=1)))
        
        return trusted_sim, untrusted_sim
    
    def compute_evidence_score(
        self,
        trusted_similarity: float,
        untrusted_similarity: float
    ) -> float:
        """
        Compute evidence score based on cross-source similarities.
        
        Higher score means more agreement with trusted sources.
        
        Args:
            trusted_similarity: Similarity to trusted sources (0-1)
            untrusted_similarity: Similarity to untrusted sources (0-1)
            
        Returns:
            Evidence score (0-1)
        """
        # If similar to trusted and different from untrusted: high score
        # If similar to untrusted and different from trusted: low score
        
        if trusted_similarity + untrusted_similarity == 0:
            return 0.5  # No evidence
        
        # Normalized difference
        score = (trusted_similarity - untrusted_similarity + 1) / 2
        
        # Adjust by absolute similarity values
        if trusted_similarity > SIMILARITY_THRESHOLD:
            score = min(1.0, score + 0.1)  # Bonus for strong trusted match
        if untrusted_similarity > SIMILARITY_THRESHOLD:
            score = max(0.0, score - 0.1)  # Penalty for strong untrusted match
        
        return float(score)
    
    def aggregate(
        self,
        model_prediction: int,
        model_confidence: float,
        source_credibility: float,
        trusted_similarity: float = 0.5,
        untrusted_similarity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Aggregate all evidence into final credibility assessment.
        
        Args:
            model_prediction: Model prediction (0=FAKE, 1=REAL)
            model_confidence: Model confidence (0-1)
            source_credibility: Source credibility score (0-1)
            trusted_similarity: Similarity to trusted sources (0-1)
            untrusted_similarity: Similarity to untrusted sources (0-1)
            
        Returns:
            Dictionary with credibility assessment
        """
        # Convert model prediction to credibility scale
        if model_prediction == 1:  # REAL
            model_score = model_confidence
        else:  # FAKE
            model_score = 1 - model_confidence
        
        # Compute evidence score
        evidence_score = self.compute_evidence_score(
            trusted_similarity,
            untrusted_similarity
        )
        
        # Weighted combination
        final_score = (
            self.weight_model * model_score +
            self.weight_source * source_credibility +
            self.weight_evidence * evidence_score
        )
        
        # Determine final prediction
        if final_score >= 0.6:
            prediction = "LIKELY REAL"
            prediction_code = 1
        elif final_score <= 0.4:
            prediction = "LIKELY FAKE"
            prediction_code = 0
        else:
            prediction = "UNCERTAIN"
            prediction_code = -1
        
        # Generate explanation
        explanation = self._generate_explanation(
            model_prediction,
            model_confidence,
            source_credibility,
            evidence_score,
            final_score
        )
        
        return {
            'prediction': prediction,
            'prediction_code': prediction_code,
            'credibility_score': round(final_score, 3),
            'confidence_percentage': round(final_score * 100, 1),
            'components': {
                'model_score': round(model_score, 3),
                'source_score': round(source_credibility, 3),
                'evidence_score': round(evidence_score, 3)
            },
            'weights': {
                'model': round(self.weight_model, 2),
                'source': round(self.weight_source, 2),
                'evidence': round(self.weight_evidence, 2)
            },
            'explanation': explanation,
            'disclaimer': (
                "This is a probabilistic credibility assessment, not absolute "
                "truth verification. Results should be interpreted as 'likelihood "
                "of credibility' based on available evidence."
            )
        }
    
    def _generate_explanation(
        self,
        model_prediction: int,
        model_confidence: float,
        source_credibility: float,
        evidence_score: float,
        final_score: float
    ) -> List[str]:
        """
        Generate human-readable explanation of the assessment.
        
        Args:
            Various component scores
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Model explanation
        model_pred_text = "REAL" if model_prediction == 1 else "FAKE"
        if model_confidence >= 0.8:
            explanations.append(
                f"ML model strongly predicts {model_pred_text} "
                f"(confidence: {model_confidence:.0%})"
            )
        elif model_confidence >= 0.6:
            explanations.append(
                f"ML model predicts {model_pred_text} "
                f"(moderate confidence: {model_confidence:.0%})"
            )
        else:
            explanations.append(
                f"ML model prediction is uncertain "
                f"(low confidence: {model_confidence:.0%})"
            )
        
        # Source explanation
        if source_credibility >= 0.8:
            explanations.append(
                "Source has high credibility reputation"
            )
        elif source_credibility >= 0.5:
            explanations.append(
                "Source has moderate credibility reputation"
            )
        elif source_credibility >= 0.3:
            explanations.append(
                "Source has limited credibility reputation"
            )
        else:
            explanations.append(
                "Source is associated with unreliable content"
            )
        
        # Evidence explanation
        if evidence_score >= 0.7:
            explanations.append(
                "Content shows agreement with trusted news sources"
            )
        elif evidence_score <= 0.3:
            explanations.append(
                "Content shows patterns similar to unreliable sources"
            )
        
        return explanations


class CredibilityAssessor:
    """
    High-level interface for credibility assessment.
    
    Combines all components:
    - Feature extraction
    - Model prediction
    - Source scoring
    - Evidence aggregation
    """
    
    def __init__(
        self,
        model=None,
        feature_extractor=None,
        source_scorer=None,
        document_embedder=None
    ):
        """
        Initialize assessor with components.
        
        Args:
            model: Trained classification model
            feature_extractor: Feature extraction pipeline
            source_scorer: Source credibility scorer
            document_embedder: Document embedding model
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.source_scorer = source_scorer
        self.document_embedder = document_embedder
        self.aggregator = EvidenceAggregator()
        
        # Reference embeddings (populated during setup)
        self.trusted_embeddings = np.array([])
        self.untrusted_embeddings = np.array([])
    
    def set_reference_embeddings(
        self,
        trusted_texts: List[str],
        untrusted_texts: List[str]
    ):
        """
        Set reference embeddings for cross-source comparison.
        
        Args:
            trusted_texts: Texts from trusted sources
            untrusted_texts: Texts from untrusted sources
        """
        if self.document_embedder is not None:
            if trusted_texts:
                self.trusted_embeddings = self.document_embedder.transform(
                    trusted_texts
                )
            if untrusted_texts:
                self.untrusted_embeddings = self.document_embedder.transform(
                    untrusted_texts
                )
    
    def assess(
        self,
        text: str,
        source: Optional[str] = None,
        include_features: bool = True
    ) -> Dict[str, Any]:
        """
        Perform full credibility assessment on an article.
        
        Args:
            text: Article text
            source: Optional source URL/domain
            include_features: Whether to include feature importance
            
        Returns:
            Complete assessment dictionary
        """
        result = {
            'text_length': len(text.split()),
            'source': source or 'Unknown'
        }
        
        # Get model prediction
        if self.model is not None and self.feature_extractor is not None:
            features = self.feature_extractor.transform([text])
            prediction = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            confidence = float(proba[prediction])
            
            result['model_prediction'] = 'REAL' if prediction == 1 else 'FAKE'
            result['model_confidence'] = round(confidence, 3)
        else:
            prediction = 0
            confidence = 0.5
        
        # Get source credibility
        if self.source_scorer is not None and source:
            source_score, source_explanation = self.source_scorer.get_score(source)
            result['source_credibility'] = round(source_score, 3)
            result['source_explanation'] = source_explanation
        else:
            source_score = 0.5
        
        # Get cross-source similarity
        trusted_sim = 0.5
        untrusted_sim = 0.5
        
        if self.document_embedder is not None:
            article_embedding = self.document_embedder.transform([text])
            
            if len(self.trusted_embeddings) > 0:
                trusted_sim, _ = self.aggregator.compute_cross_source_similarity(
                    article_embedding,
                    self.trusted_embeddings,
                    np.array([])
                )
            
            if len(self.untrusted_embeddings) > 0:
                _, untrusted_sim = self.aggregator.compute_cross_source_similarity(
                    article_embedding,
                    np.array([]),
                    self.untrusted_embeddings
                )
        
        # Aggregate evidence
        assessment = self.aggregator.aggregate(
            model_prediction=prediction,
            model_confidence=confidence,
            source_credibility=source_score,
            trusted_similarity=trusted_sim,
            untrusted_similarity=untrusted_sim
        )
        
        result.update(assessment)
        
        return result


# Testing
if __name__ == "__main__":
    print("Testing Evidence Aggregation Module")
    print("=" * 50)
    
    aggregator = EvidenceAggregator()
    
    # Test case 1: High credibility
    print("\n1. High credibility scenario:")
    result = aggregator.aggregate(
        model_prediction=1,  # REAL
        model_confidence=0.85,
        source_credibility=0.90,
        trusted_similarity=0.8,
        untrusted_similarity=0.2
    )
    print(f"   Prediction: {result['prediction']}")
    print(f"   Score: {result['credibility_score']:.2f}")
    print(f"   Explanation: {result['explanation']}")
    
    # Test case 2: Low credibility
    print("\n2. Low credibility scenario:")
    result = aggregator.aggregate(
        model_prediction=0,  # FAKE
        model_confidence=0.90,
        source_credibility=0.15,
        trusted_similarity=0.2,
        untrusted_similarity=0.8
    )
    print(f"   Prediction: {result['prediction']}")
    print(f"   Score: {result['credibility_score']:.2f}")
    print(f"   Explanation: {result['explanation']}")
    
    # Test case 3: Mixed signals
    print("\n3. Mixed signals scenario:")
    result = aggregator.aggregate(
        model_prediction=1,
        model_confidence=0.55,
        source_credibility=0.50,
        trusted_similarity=0.5,
        untrusted_similarity=0.5
    )
    print(f"   Prediction: {result['prediction']}")
    print(f"   Score: {result['credibility_score']:.2f}")
    print(f"   Explanation: {result['explanation']}")
    print(f"\n   Disclaimer: {result['disclaimer']}")
    
    print("\nAll evidence aggregation tests passed!")
