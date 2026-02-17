"""
Claim Verifier Module for Evidence-Based Credibility Assessment

Matches user claims against RSS corpus articles and aggregates
evidence to compute credibility scores.

This module:
- Searches RSS corpus for relevant articles
- Scores entity and keyword overlap
- Detects confirmation vs contradiction
- Aggregates multi-source evidence

IMPORTANT: This is probabilistic assessment, NOT absolute truth verification.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from credibility.claim_extractor import ClaimExtractor, get_claim_extractor
from credibility.source_scorer import SourceCredibilityScorer


# Contradiction indicators
CONTRADICTION_PHRASES = [
    'denies', 'denied', 'false', 'fake', 'hoax', 'rumor', 'rumour',
    'not true', 'no truth', 'misinformation', 'disinformation',
    'debunked', 'fact check', 'fact-check', 'clarifies', 'clarified',
    'rubbishes', 'dismisses', 'dismissed', 'refutes', 'refuted',
    'baseless', 'unfounded', 'unverified', 'is alive', 'safe and sound',
    'no evidence', 'did not happen', 'never happened',
]

# Confirmation indicators (removed generic phrases that appear in all news)
CONFIRMATION_PHRASES = [
    'confirms', 'confirmed', 'officially confirmed',
    'has died', 'passed away', 'was arrested', 'resigned',
    'earthquake hit', 'earthquake struck', 'tremors felt',
    'magnitude', 'seismic activity', 'richter scale',
]

# Trusted Indian sources (prioritized for Indian claims)
TRUSTED_INDIAN_SOURCES = [
    'pti', 'ani', 'thehindu', 'indianexpress', 'ndtv', 'hindustantimes',
    'timesofindia', 'indiatoday', 'livemint', 'economictimes',
    'theprint', 'thewire', 'scroll', 'firstpost', 'deccanherald',
]

# Trusted international sources
TRUSTED_INTERNATIONAL_SOURCES = [
    'reuters', 'apnews', 'bbc', 'cnn', 'nytimes', 'washingtonpost',
    'theguardian', 'npr', 'aljazeera', 'dw',
]


class EvidenceMatch:
    """Represents a matched article as evidence."""
    
    def __init__(
        self,
        article: Dict,
        entity_score: float,
        keyword_score: float,
        is_confirmation: bool,
        is_contradiction: bool,
        source_credibility: float
    ):
        self.article = article
        self.entity_score = entity_score
        self.keyword_score = keyword_score
        self.is_confirmation = is_confirmation
        self.is_contradiction = is_contradiction
        self.source_credibility = source_credibility
        
        # STRICTER relevance: require BOTH entity AND keyword match
        # Only consider valid if both scores are significant
        if entity_score >= 0.3 and keyword_score >= 0.3:
            self.relevance = (entity_score * 0.5 + keyword_score * 0.5)
        else:
            # Weak match - only one signal
            self.relevance = max(entity_score, keyword_score) * 0.3
        
        # STRICTER evidence strength thresholds
        if is_contradiction and keyword_score >= 0.4:
            self.evidence_type = 'CONTRADICTION'
        elif is_confirmation and entity_score >= 0.3 and keyword_score >= 0.5:
            # Must have good keyword match for confirmation
            self.evidence_type = 'CONFIRMATION'
        elif entity_score >= 0.5 and keyword_score >= 0.4:
            self.evidence_type = 'PARTIAL_MENTION'
        else:
            self.evidence_type = 'WEAK_MATCH'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON response."""
        return {
            'title': self.article.get('title', ''),
            'source': self.article.get('source', ''),
            'link': self.article.get('link', ''),
            'published': self.article.get('published', ''),
            'summary': self.article.get('summary', '')[:200] + '...' if len(self.article.get('summary', '')) > 200 else self.article.get('summary', ''),
            'entity_score': round(self.entity_score, 3),
            'keyword_score': round(self.keyword_score, 3),
            'relevance': round(self.relevance, 3),
            'evidence_type': self.evidence_type,
            'is_confirmation': self.is_confirmation,
            'is_contradiction': self.is_contradiction,
            'source_credibility': round(self.source_credibility, 3),
        }


class ClaimVerifier:
    """
    Verifies claims by matching against RSS corpus and aggregating evidence.
    
    Uses multi-signal approach:
    - Evidence agreement (50%): Confirmation/contradiction from sources
    - Source credibility (30%): Trustworthiness of sources
    - Content plausibility (20%): Linguistic analysis
    """
    
    def __init__(self, rss_corpus: List[Dict] = None):
        """
        Initialize verifier.
        
        Args:
            rss_corpus: List of RSS articles to search
        """
        self.rss_corpus = rss_corpus or []
        self.claim_extractor = get_claim_extractor()
        self.source_scorer = SourceCredibilityScorer()
    
    def update_corpus(self, articles: List[Dict]):
        """Update the RSS corpus with new articles."""
        self.rss_corpus = articles
    
    def verify_claim(self, claim: str) -> Dict:
        """
        Verify a claim against RSS evidence.
        
        Args:
            claim: User's claim or question
            
        Returns:
            Verification result with verdict, confidence, evidence
        """
        # Extract claim information
        claim_info = self.claim_extractor.extract_claim_info(claim)
        
        # Search for relevant articles
        matches = self._search_evidence(claim_info)
        
        # Aggregate evidence
        result = self._aggregate_evidence(claim_info, matches)
        
        return result
    
    def _search_evidence(self, claim_info: Dict) -> List[EvidenceMatch]:
        """
        Search RSS corpus for relevant articles.
        
        Args:
            claim_info: Extracted claim information
            
        Returns:
            List of EvidenceMatch objects
        """
        matches = []
        
        entities = claim_info.get('entities', {})
        keywords = claim_info.get('keywords', [])
        is_indian = claim_info.get('is_indian_context', False)
        
        # All entity names to match
        entity_names = []
        for category in ['people', 'places', 'orgs', 'other']:
            entity_names.extend(entities.get(category, []))
        
        for article in self.rss_corpus:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            content = article.get('content', '').lower()
            full_text = f"{title} {summary} {content}"
            
            # Calculate entity score
            entity_score = self._calculate_entity_score(entity_names, full_text)
            
            # Calculate keyword score (excluding entity words for independence)
            entity_words = set()
            for name in entity_names:
                entity_words.update(name.lower().split())
            non_entity_keywords = [k for k in keywords if k.lower() not in entity_words]
            keyword_score = self._calculate_keyword_score(non_entity_keywords, full_text)
            
            # STRICTER: Skip if not BOTH entity AND keyword match
            # This prevents matching unrelated articles just because location is mentioned
            if entity_score < 0.3 or keyword_score < 0.25:
                continue
            
            # Check for confirmation/contradiction
            is_confirmation = self._check_confirmation(full_text)
            is_contradiction = self._check_contradiction(full_text)
            
            # Get source credibility
            source = article.get('source', '')
            source_cred = self._get_source_credibility(source, is_indian)
            
            match = EvidenceMatch(
                article=article,
                entity_score=entity_score,
                keyword_score=keyword_score,
                is_confirmation=is_confirmation,
                is_contradiction=is_contradiction,
                source_credibility=source_cred
            )
            
            matches.append(match)
        
        # Sort by relevance
        matches.sort(key=lambda m: m.relevance, reverse=True)
        
        return matches[:20]  # Top 20 matches
    
    def _calculate_entity_score(self, entities: List[str], text: str) -> float:
        """Calculate entity overlap score."""
        if not entities:
            return 0.0
        
        matches = 0
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in text:
                matches += 1
            else:
                # Check partial match (e.g., last name only)
                parts = entity_lower.split()
                for part in parts:
                    if len(part) > 3 and part in text:
                        matches += 0.5
                        break
        
        return min(1.0, matches / len(entities))
    
    def _calculate_keyword_score(self, keywords: List[str], text: str) -> float:
        """Calculate keyword overlap score."""
        if not keywords:
            return 0.0
        
        # Weight earlier keywords higher (more important)
        total_weight = 0
        matched_weight = 0
        
        for i, keyword in enumerate(keywords[:10]):  # Top 10 keywords
            weight = 1.0 / (i + 1)  # Decreasing weight
            total_weight += weight
            
            if keyword.lower() in text:
                matched_weight += weight
        
        return matched_weight / total_weight if total_weight > 0 else 0.0
    
    def _check_confirmation(self, text: str) -> bool:
        """Check if text contains confirmation indicators."""
        for phrase in CONFIRMATION_PHRASES:
            if phrase in text:
                return True
        return False
    
    def _check_contradiction(self, text: str) -> bool:
        """Check if text contains contradiction indicators."""
        for phrase in CONTRADICTION_PHRASES:
            if phrase in text:
                return True
        return False
    
    def _get_source_credibility(self, source: str, is_indian_context: bool) -> float:
        """Get source credibility, prioritizing Indian sources for Indian claims."""
        source_lower = source.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        # Check Indian sources
        for indian_src in TRUSTED_INDIAN_SOURCES:
            if indian_src in source_lower:
                return 0.9 if is_indian_context else 0.8
        
        # Check international sources
        for intl_src in TRUSTED_INTERNATIONAL_SOURCES:
            if intl_src in source_lower:
                return 0.85
        
        # Use source scorer for unknown sources
        score, _ = self.source_scorer.get_score(source)
        return score
    
    def _aggregate_evidence(
        self, 
        claim_info: Dict, 
        matches: List[EvidenceMatch]
    ) -> Dict:
        """
        Aggregate evidence to compute final credibility.
        
        Returns verdict with explanation.
        """
        is_indian = claim_info.get('is_indian_context', False)
        
        # Count evidence types
        confirmations = [m for m in matches if m.evidence_type == 'CONFIRMATION']
        contradictions = [m for m in matches if m.evidence_type == 'CONTRADICTION']
        partial = [m for m in matches if m.evidence_type == 'PARTIAL_MENTION']
        
        # Calculate evidence score (50% weight)
        evidence_score = self._calculate_evidence_score(
            confirmations, contradictions, partial, is_indian
        )
        
        # Calculate source credibility score (30% weight)
        source_score = self._calculate_source_score(matches, is_indian)
        
        # Calculate plausibility score (20% weight)
        plausibility = self._calculate_plausibility(claim_info)
        
        # Weighted aggregation
        final_score = (
            evidence_score * 0.50 +
            source_score * 0.30 +
            plausibility * 0.20
        )
        
        # Determine verdict
        verdict, confidence = self._determine_verdict(
            final_score, confirmations, contradictions, matches
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            verdict, confirmations, contradictions, partial, is_indian
        )
        
        return {
            'claim': claim_info['original_claim'],
            'verdict': verdict,
            'confidence': round(confidence, 3),
            'credibility_score': round(final_score, 3),
            'explanation': explanation,
            'is_indian_context': is_indian,
            'evidence': {
                'total_matches': len(matches),
                'confirmations': len(confirmations),
                'contradictions': len(contradictions),
                'partial_mentions': len(partial),
            },
            'matching_articles': [m.to_dict() for m in matches[:10]],
            'components': {
                'evidence_score': round(evidence_score, 3),
                'source_score': round(source_score, 3),
                'plausibility_score': round(plausibility, 3),
            },
            'disclaimer': (
                "This is a probabilistic assessment based on available RSS news sources. "
                "The system does not claim absolute truth or fact-checking. "
                "Always verify important claims through multiple authoritative sources."
            )
        }
    
    def _calculate_evidence_score(
        self,
        confirmations: List[EvidenceMatch],
        contradictions: List[EvidenceMatch],
        partial: List[EvidenceMatch],
        is_indian: bool
    ) -> float:
        """Calculate evidence agreement score."""
        # No evidence = neutral (0.5)
        if not confirmations and not contradictions and not partial:
            return 0.3  # Low score for no evidence
        
        # Strong contradictions reduce score significantly
        if contradictions:
            # Check if contradictions are from trusted sources
            trusted_contradictions = sum(
                1 for c in contradictions if c.source_credibility > 0.7
            )
            if trusted_contradictions > 0:
                return 0.15  # Very low - trusted sources contradict
        
        # Confirmations increase score
        if confirmations:
            # Check trusted confirmations
            trusted_confirmations = sum(
                1 for c in confirmations if c.source_credibility > 0.7
            )
            if trusted_confirmations >= 2:
                return 0.85  # High - multiple trusted confirmations
            elif trusted_confirmations == 1:
                return 0.70  # Moderate - single trusted confirmation
            else:
                return 0.55  # Weak confirmations
        
        # Only partial mentions
        if partial:
            return 0.45  # Slightly below neutral - mentioned but not confirmed
        
        return 0.35
    
    def _calculate_source_score(
        self, 
        matches: List[EvidenceMatch], 
        is_indian: bool
    ) -> float:
        """Calculate source credibility score."""
        if not matches:
            return 0.3  # No sources = low score
        
        # Check for trusted Indian sources for Indian claims
        if is_indian:
            indian_trusted = sum(
                1 for m in matches 
                if any(src in m.article.get('source', '').lower() 
                       for src in TRUSTED_INDIAN_SOURCES)
            )
            if indian_trusted == 0:
                return 0.35  # No Indian trusted sources
        
        # Average source credibility of top matches
        top_matches = matches[:5]
        avg_cred = sum(m.source_credibility for m in top_matches) / len(top_matches)
        
        return avg_cred
    
    def _calculate_plausibility(self, claim_info: Dict) -> float:
        """Calculate content plausibility based on claim characteristics."""
        claim = claim_info.get('original_claim', '').lower()
        
        # Sensational indicators reduce plausibility
        sensational_words = [
            'shocking', 'unbelievable', 'you wont believe', 'breaking',
            'urgent', 'just in', 'viral', 'exposed', 'leaked'
        ]
        
        sensational_count = sum(1 for w in sensational_words if w in claim)
        
        # Exclamation marks
        exclamation_count = claim.count('!')
        
        # Calculate score
        score = 0.6  # Base score
        score -= sensational_count * 0.1
        score -= exclamation_count * 0.05
        
        return max(0.2, min(0.8, score))
    
    def _determine_verdict(
        self,
        score: float,
        confirmations: List[EvidenceMatch],
        contradictions: List[EvidenceMatch],
        all_matches: List[EvidenceMatch]
    ) -> Tuple[str, float]:
        """Determine verdict and confidence."""
        # No evidence
        if not all_matches:
            return 'UNVERIFIED', 0.2
        
        # Strong contradiction
        if contradictions and not confirmations:
            if any(c.source_credibility > 0.7 for c in contradictions):
                return 'LIKELY FALSE', 0.75
            return 'LIKELY FALSE', 0.55
        
        # Strong confirmation
        if confirmations and not contradictions:
            trusted = [c for c in confirmations if c.source_credibility > 0.7]
            if len(trusted) >= 2:
                return 'LIKELY TRUE', 0.80
            elif len(trusted) == 1:
                return 'LIKELY TRUE', 0.65
            elif confirmations:
                return 'POSSIBLY TRUE', 0.50
        
        # Mixed signals
        if confirmations and contradictions:
            return 'DISPUTED', 0.40
        
        # Only partial mentions
        if score > 0.6:
            return 'POSSIBLY TRUE', 0.50
        elif score > 0.4:
            return 'UNVERIFIED', 0.35
        else:
            return 'INSUFFICIENT EVIDENCE', 0.25
    
    def _generate_explanation(
        self,
        verdict: str,
        confirmations: List[EvidenceMatch],
        contradictions: List[EvidenceMatch],
        partial: List[EvidenceMatch],
        is_indian: bool
    ) -> List[str]:
        """Generate human-readable explanation."""
        explanations = []
        
        if verdict == 'LIKELY TRUE':
            if confirmations:
                sources = [c.article.get('source', 'Unknown') for c in confirmations[:3]]
                explanations.append(
                    f"Multiple news sources report this event: {', '.join(sources)}."
                )
        
        elif verdict == 'LIKELY FALSE':
            if contradictions:
                sources = [c.article.get('source', 'Unknown') for c in contradictions[:3]]
                explanations.append(
                    f"Trusted sources contradict or deny this claim: {', '.join(sources)}."
                )
            if not confirmations:
                explanations.append("No credible news sources confirm this claim.")
        
        elif verdict == 'UNVERIFIED' or verdict == 'INSUFFICIENT EVIDENCE':
            explanations.append("No reliable news sources were found reporting this event.")
            if is_indian:
                explanations.append(
                    "Major Indian news outlets (PTI, ANI, The Hindu, NDTV) have not reported this."
                )
        
        elif verdict == 'DISPUTED':
            explanations.append("News sources show conflicting information about this claim.")
        
        elif verdict == 'POSSIBLY TRUE':
            if partial:
                explanations.append(
                    f"Related articles found but no direct confirmation. "
                    f"Found {len(partial)} partial mentions."
                )
        
        # Always add disclaimer
        explanations.append(
            "This assessment is based on available RSS news sources only."
        )
        
        return explanations


# Singleton instance
_verifier_instance = None


def get_claim_verifier() -> ClaimVerifier:
    """Get singleton ClaimVerifier instance."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = ClaimVerifier()
    return _verifier_instance


# Testing
if __name__ == "__main__":
    print("Testing Claim Verifier")
    print("=" * 60)
    
    # Create sample corpus
    sample_corpus = [
        {
            'title': 'PM Modi addresses nation on Independence Day',
            'summary': 'Prime Minister Narendra Modi addressed the nation from Red Fort on Independence Day.',
            'source': 'the_hindu',
            'link': 'https://example.com/article1',
            'published': '2024-08-15'
        },
        {
            'title': 'Central government denies reports of PM resignation',
            'summary': 'The PMO has dismissed as false and baseless rumors about PM Modi resignation.',
            'source': 'pti',
            'link': 'https://example.com/article2',
            'published': '2024-08-16'
        },
        {
            'title': 'No earthquake reported in Delhi, says IMD',
            'summary': 'India Meteorological Department clarifies no seismic activity in Delhi.',
            'source': 'ndtv',
            'link': 'https://example.com/article3',
            'published': '2024-08-16'
        }
    ]
    
    verifier = ClaimVerifier(rss_corpus=sample_corpus)
    
    test_claims = [
        "Did Narendra Modi resign as PM?",
        "Earthquake in Delhi today?",
    ]
    
    for claim in test_claims:
        print(f"\nClaim: {claim}")
        print("-" * 40)
        result = verifier.verify_claim(claim)
        print(f"  Verdict: {result['verdict']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Explanation: {result['explanation'][0]}")
        print(f"  Matches: {result['evidence']['total_matches']}")
    
    print("\n" + "=" * 60)
    print("Claim Verifier module loaded successfully!")
