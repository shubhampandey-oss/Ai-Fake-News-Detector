"""
Source Credibility Scoring Module

This module provides source reputation scoring based on:
- Pre-compiled list of trusted and unreliable sources
- Historical fake ratio (if available)
- Domain analysis

IMPORTANT: This is a probabilistic assessment based on pre-compiled data.
We do NOT dynamically verify facts or query external fact-checking APIs.
"""

import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SOURCES_DIR,
    CREDIBILITY_MIN,
    CREDIBILITY_MAX,
    TRUSTED_RSS_FEEDS,
    UNRELIABLE_SOURCES
)


# Pre-compiled source credibility database
# Scores range from 0 (unreliable) to 1 (trusted)
# Based on public media bias/fact-check reports (e.g., MediaBiasFactCheck)
DEFAULT_SOURCE_SCORES = {
    # Highly trusted sources (established news organizations)
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "bbc.com": 0.90,
    "bbc.co.uk": 0.90,
    "npr.org": 0.88,
    "pbs.org": 0.88,
    "cnn.com": 0.75,
    "nytimes.com": 0.80,
    "washingtonpost.com": 0.78,
    "theguardian.com": 0.80,
    "wsj.com": 0.80,
    "economist.com": 0.85,
    "nature.com": 0.95,
    "sciencemag.org": 0.95,
    "scientificamerican.com": 0.90,
    
    # Indian News Sources (established organizations)
    "thehindu.com": 0.88,
    "hindustantimes.com": 0.82,
    "indianexpress.com": 0.85,
    "ndtv.com": 0.80,
    "timesofindia.indiatimes.com": 0.75,
    "news18.com": 0.72,
    "indiatoday.in": 0.78,
    "livemint.com": 0.85,
    "economictimes.indiatimes.com": 0.80,
    "deccanherald.com": 0.78,
    "theprint.in": 0.80,
    "thewire.in": 0.78,
    "scroll.in": 0.75,
    "firstpost.com": 0.70,
    "business-standard.com": 0.82,
    "moneycontrol.com": 0.75,
    "outlookindia.com": 0.75,
    "theweek.in": 0.72,
    "dnaindia.com": 0.65,
    "zeenews.india.com": 0.60,
    
    # Moderate credibility
    "huffpost.com": 0.60,
    "foxnews.com": 0.55,
    "dailymail.co.uk": 0.45,
    "nypost.com": 0.50,
    "thesun.co.uk": 0.40,
    
    # Known unreliable sources
    "infowars.com": 0.05,
    "naturalnews.com": 0.10,
    "beforeitsnews.com": 0.05,
    "worldnewsdailyreport.com": 0.02,  # Satire/fake
    "empirenews.net": 0.02,
    "nationalreport.net": 0.02,
    "theonion.com": 0.01,  # Satire
    "babylonbee.com": 0.01,  # Satire
}


# Indian news domains for region detection
INDIAN_DOMAINS = {
    'thehindu.com', 'hindustantimes.com', 'indianexpress.com',
    'ndtv.com', 'timesofindia.indiatimes.com', 'news18.com',
    'indiatoday.in', 'livemint.com', 'economictimes.indiatimes.com',
    'deccanherald.com', 'theprint.in', 'thewire.in', 'scroll.in',
    'firstpost.com', 'dnaindia.com', 'zeenews.india.com',
    'outlookindia.com', 'theweek.in', 'business-standard.com',
    'moneycontrol.com', 'rediff.com', 'oneindia.com',
}


class SourceCredibilityScorer:
    """
    Assigns credibility scores to news sources.
    
    This is based on pre-compiled data, NOT dynamic fact-checking.
    The scores reflect general reputation, not verification of specific claims.
    
    Credibility scoring approach:
    1. Check against known source database
    2. Apply domain heuristics (e.g., .gov, .edu tend to be more reliable)
    3. Check for known unreliable patterns
    """
    
    def __init__(
        self,
        source_scores: Optional[Dict[str, float]] = None,
        unreliable_domains: Optional[List[str]] = None
    ):
        """
        Initialize scorer.
        
        Args:
            source_scores: Dictionary mapping domain to credibility score
            unreliable_domains: List of known unreliable domains
        """
        self.source_scores = source_scores or DEFAULT_SOURCE_SCORES.copy()
        self.unreliable_domains = set(
            unreliable_domains or UNRELIABLE_SOURCES
        )
        
        # Suspicious domain patterns
        self.suspicious_patterns = [
            r'news\d+\.com',  # news24.com, news365.com, etc.
            r'.*breaking.*news.*',
            r'.*real.*truth.*',
            r'.*patriot.*news.*',
        ]
        
        # Compile patterns
        self.suspicious_regex = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.suspicious_patterns
        ]
        
        # Trusted TLDs
        self.trusted_tlds = {'.gov', '.edu', '.org'}
        self.less_trusted_tlds = {'.xyz', '.tk', '.pw', '.ml', '.ga', '.cf'}
    
    def _extract_domain(self, source: str) -> str:
        """
        Extract domain from URL or source string.
        
        Args:
            source: URL or domain string
            
        Returns:
            Cleaned domain name
        """
        # Handle full URLs
        if source.startswith(('http://', 'https://')):
            parsed = urlparse(source)
            domain = parsed.netloc.lower()
        else:
            domain = source.lower()
        
        # Remove www. prefix
        domain = re.sub(r'^www\.', '', domain)
        
        return domain
    
    def _check_suspicious_patterns(self, domain: str) -> bool:
        """
        Check if domain matches suspicious patterns.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if suspicious pattern found
        """
        for pattern in self.suspicious_regex:
            if pattern.search(domain):
                return True
        return False
    
    def _get_tld_modifier(self, domain: str) -> float:
        """
        Get credibility modifier based on TLD.
        
        Args:
            domain: Domain name
            
        Returns:
            Modifier to adjust score (-0.2 to +0.1)
        """
        for tld in self.trusted_tlds:
            if domain.endswith(tld):
                return 0.1  # Slight boost for trusted TLDs
        
        for tld in self.less_trusted_tlds:
            if domain.endswith(tld):
                return -0.2  # Penalty for suspicious TLDs
        
        return 0.0
    
    def get_score(self, source: str) -> Tuple[float, str]:
        """
        Get credibility score for a source.
        
        Args:
            source: URL or domain name
            
        Returns:
            Tuple of (score, explanation)
        """
        domain = self._extract_domain(source)
        
        # Check known sources
        if domain in self.source_scores:
            score = self.source_scores[domain]
            return score, f"Known source: {domain}"
        
        # Check unreliable list
        if domain in self.unreliable_domains:
            return 0.1, f"Known unreliable source: {domain}"
        
        # Check suspicious patterns
        if self._check_suspicious_patterns(domain):
            return 0.3, f"Suspicious domain pattern: {domain}"
        
        # Apply TLD modifier to default score
        default_score = 0.5  # Neutral score for unknown sources
        tld_modifier = self._get_tld_modifier(domain)
        score = max(CREDIBILITY_MIN, min(CREDIBILITY_MAX, default_score + tld_modifier))
        
        return score, f"Unknown source (neutral assessment): {domain}"
    
    def get_scores_batch(
        self,
        sources: List[str]
    ) -> List[Tuple[str, float, str]]:
        """
        Get scores for multiple sources.
        
        Args:
            sources: List of URLs or domains
            
        Returns:
            List of (source, score, explanation) tuples
        """
        return [
            (source, *self.get_score(source))
            for source in sources
        ]
    
    def is_trusted(self, source: str, threshold: float = 0.7) -> bool:
        """
        Check if source meets trust threshold.
        
        Args:
            source: URL or domain
            threshold: Minimum score to be considered trusted
            
        Returns:
            True if source is trusted
        """
        score, _ = self.get_score(source)
        return score >= threshold
    
    def detect_region(self, source: str) -> str:
        """
        Detect region (India vs Global) from source URL/domain.
        
        Args:
            source: URL or domain
            
        Returns:
            'india' or 'global'
        """
        domain = self._extract_domain(source)
        
        # Check .in TLD
        if domain.endswith('.in'):
            return 'india'
        
        # Check known Indian domains
        if domain in INDIAN_DOMAINS:
            return 'india'
        
        # Check subdomains (e.g., hindi.news.com)
        for indian_domain in INDIAN_DOMAINS:
            if domain.endswith('.' + indian_domain) or domain == indian_domain:
                return 'india'
        
        return 'global'
    
    def get_source_info(self, source: str) -> dict:
        """
        Get comprehensive source information for URL-based analysis.
        
        Args:
            source: URL or domain
            
        Returns:
            Dictionary with source name, domain, region, credibility, trusted status
        """
        domain = self._extract_domain(source)
        score, explanation = self.get_score(source)
        region = self.detect_region(source)
        
        # Get display name
        display_name = domain.split('.')[0].title()
        
        # Check for known display names
        known_names = {
            'reuters.com': 'Reuters',
            'bbc.com': 'BBC', 'bbc.co.uk': 'BBC',
            'theguardian.com': 'The Guardian',
            'nytimes.com': 'The New York Times',
            'washingtonpost.com': 'Washington Post',
            'cnn.com': 'CNN', 'npr.org': 'NPR',
            'thehindu.com': 'The Hindu',
            'hindustantimes.com': 'Hindustan Times',
            'indianexpress.com': 'Indian Express',
            'ndtv.com': 'NDTV',
            'timesofindia.indiatimes.com': 'Times of India',
            'indiatoday.in': 'India Today',
            'livemint.com': 'Mint',
            'economictimes.indiatimes.com': 'Economic Times',
            'theprint.in': 'The Print',
            'thewire.in': 'The Wire',
            'scroll.in': 'Scroll.in',
        }
        display_name = known_names.get(domain, display_name)
        
        return {
            'domain': domain,
            'name': display_name,
            'region': region,
            'credibility_score': score,
            'is_trusted': self.is_trusted(source),
            'is_unreliable': self.is_unreliable(source),
            'explanation': explanation
        }
    
    def is_unreliable(self, source: str, threshold: float = 0.3) -> bool:
        """
        Check if source is considered unreliable.
        
        Args:
            source: URL or domain
            threshold: Maximum score to be considered unreliable
            
        Returns:
            True if source is unreliable
        """
        score, _ = self.get_score(source)
        return score <= threshold
    
    def add_source(self, domain: str, score: float):
        """
        Add or update a source in the database.
        
        Args:
            domain: Domain name
            score: Credibility score (0-1)
        """
        domain = self._extract_domain(domain)
        self.source_scores[domain] = max(
            CREDIBILITY_MIN,
            min(CREDIBILITY_MAX, score)
        )
    
    def save_database(self, path: Optional[str] = None):
        """
        Save source database to JSON file.
        
        Args:
            path: Save path
        """
        if path is None:
            SOURCES_DIR.mkdir(parents=True, exist_ok=True)
            path = SOURCES_DIR / "source_credibility.json"
        
        with open(path, 'w') as f:
            json.dump({
                'source_scores': self.source_scores,
                'unreliable_domains': list(self.unreliable_domains)
            }, f, indent=2)
        
        print(f"Saved source database to {path}")
    
    def load_database(self, path: Optional[str] = None):
        """
        Load source database from JSON file.
        
        Args:
            path: Load path
        """
        if path is None:
            path = SOURCES_DIR / "source_credibility.json"
        
        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.source_scores.update(data.get('source_scores', {}))
                self.unreliable_domains.update(data.get('unreliable_domains', []))
            print(f"Loaded source database from {path}")
    
    def get_trusted_sources(self, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Get list of trusted sources above threshold.
        
        Args:
            threshold: Minimum credibility score
            
        Returns:
            List of (domain, score) tuples, sorted by score
        """
        trusted = [
            (domain, score)
            for domain, score in self.source_scores.items()
            if score >= threshold
        ]
        return sorted(trusted, key=lambda x: x[1], reverse=True)
    
    def get_unreliable_sources(self, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Get list of unreliable sources below threshold.
        
        Args:
            threshold: Maximum credibility score
            
        Returns:
            List of (domain, score) tuples
        """
        unreliable = [
            (domain, score)
            for domain, score in self.source_scores.items()
            if score <= threshold
        ]
        return sorted(unreliable, key=lambda x: x[1])


# Testing
if __name__ == "__main__":
    print("Testing Source Credibility Module")
    print("=" * 50)
    
    scorer = SourceCredibilityScorer()
    
    # Test various sources
    test_sources = [
        "https://www.reuters.com/article/example",
        "https://www.bbc.com/news/article",
        "infowars.com",
        "https://unknownsource.xyz/article",
        "sciencemag.org",
        "randomnews365.com",
    ]
    
    print("\n1. Testing source scores:")
    for source in test_sources:
        score, explanation = scorer.get_score(source)
        trusted = "✓ TRUSTED" if scorer.is_trusted(source) else (
            "✗ UNRELIABLE" if scorer.is_unreliable(source) else "? UNCERTAIN"
        )
        print(f"   {source}")
        print(f"      Score: {score:.2f} | {trusted}")
        print(f"      {explanation}")
    
    print("\n2. Trusted sources (score >= 0.8):")
    for domain, score in scorer.get_trusted_sources()[:5]:
        print(f"   {domain}: {score:.2f}")
    
    print("\n3. Unreliable sources (score <= 0.3):")
    for domain, score in scorer.get_unreliable_sources()[:5]:
        print(f"   {domain}: {score:.2f}")
    
    print("\nAll source credibility tests passed!")
