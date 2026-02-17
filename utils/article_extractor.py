"""
Article Extraction Module for URL-Based News Analysis

Extracts article content from news website URLs using:
- requests for HTTP fetching
- BeautifulSoup for HTML parsing
- newspaper3k (optional) for enhanced extraction

This module is for INFERENCE ONLY - extracted content is never used for training.

Key features:
- Extracts headline, body text, source, publish date
- Respects robots.txt
- Local caching for deterministic results
- Graceful fallback on extraction failure
"""

import re
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Try to import newspaper3k (optional, for enhanced extraction)
try:
    from newspaper import Article as NewspaperArticle
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

from config import PROJECT_ROOT


# Cache directory
CACHE_DIR = PROJECT_ROOT / "data" / "article_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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

# Known source display names
SOURCE_DISPLAY_NAMES = {
    'bbc.com': 'BBC',
    'bbc.co.uk': 'BBC',
    'reuters.com': 'Reuters',
    'theguardian.com': 'The Guardian',
    'nytimes.com': 'The New York Times',
    'washingtonpost.com': 'Washington Post',
    'cnn.com': 'CNN',
    'npr.org': 'NPR',
    'apnews.com': 'AP News',
    'thehindu.com': 'The Hindu',
    'hindustantimes.com': 'Hindustan Times',
    'indianexpress.com': 'Indian Express',
    'ndtv.com': 'NDTV',
    'timesofindia.indiatimes.com': 'Times of India',
    'news18.com': 'News18',
    'indiatoday.in': 'India Today',
    'livemint.com': 'Mint',
    'economictimes.indiatimes.com': 'Economic Times',
    'theprint.in': 'The Print',
    'thewire.in': 'The Wire',
    'scroll.in': 'Scroll.in',
    'firstpost.com': 'Firstpost',
}


class ExtractionResult:
    """Container for article extraction results."""
    
    def __init__(
        self,
        success: bool,
        url: str,
        title: str = "",
        text: str = "",
        source_domain: str = "",
        source_name: str = "",
        publish_date: str = "",
        region: str = "global",
        error: str = "",
        extraction_method: str = ""
    ):
        self.success = success
        self.url = url
        self.title = title
        self.text = text
        self.source_domain = source_domain
        self.source_name = source_name
        self.publish_date = publish_date
        self.region = region
        self.error = error
        self.extraction_method = extraction_method
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'url': self.url,
            'title': self.title,
            'text': self.text,
            'text_length': len(self.text.split()),
            'source_domain': self.source_domain,
            'source_name': self.source_name,
            'publish_date': self.publish_date,
            'region': self.region,
            'error': self.error,
            'extraction_method': self.extraction_method,
            'timestamp': self.timestamp
        }


class ArticleExtractor:
    """
    Extracts article content from news URLs.
    
    Extraction priority:
    1. newspaper3k (if available) - best quality
    2. BeautifulSoup - reliable fallback
    
    Features:
    - Caches extracted articles for determinism
    - Detects Indian vs Global sources
    - Respects robots.txt
    """
    
    def __init__(self, use_cache: bool = True, timeout: int = 15):
        """
        Initialize extractor.
        
        Args:
            use_cache: Whether to cache extracted articles
            timeout: HTTP request timeout in seconds
        """
        self.use_cache = use_cache
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self._robot_parsers = {}
    
    def extract_from_url(self, url: str) -> ExtractionResult:
        """
        Extract article content from a URL.
        
        Args:
            url: News article URL
            
        Returns:
            ExtractionResult with article content or error
        """
        # Validate URL
        if not url or not url.startswith(('http://', 'https://')):
            return ExtractionResult(
                success=False,
                url=url,
                error="Invalid URL. Please provide a valid HTTP/HTTPS URL."
            )
        
        # Check cache first (for determinism)
        if self.use_cache:
            cached = self._get_cached(url)
            if cached:
                return cached
        
        # Parse URL for source info
        source_domain, source_name, region = self._parse_source(url)
        
        # Check robots.txt
        if not self._is_allowed(url):
            return ExtractionResult(
                success=False,
                url=url,
                source_domain=source_domain,
                source_name=source_name,
                region=region,
                error="This website does not allow automated content access."
            )
        
        # Try extraction methods
        result = None
        
        # Method 1: newspaper3k (best quality)
        if NEWSPAPER_AVAILABLE:
            result = self._extract_with_newspaper(url, source_domain, source_name, region)
            if result.success:
                if self.use_cache:
                    self._save_cache(url, result)
                return result
        
        # Method 2: BeautifulSoup (fallback)
        if BS4_AVAILABLE and REQUESTS_AVAILABLE:
            result = self._extract_with_beautifulsoup(url, source_domain, source_name, region)
            if result.success:
                if self.use_cache:
                    self._save_cache(url, result)
                return result
        
        # All methods failed
        if result and not result.success:
            return result
        
        return ExtractionResult(
            success=False,
            url=url,
            source_domain=source_domain,
            source_name=source_name,
            region=region,
            error="Could not extract article. Required libraries (requests, beautifulsoup4) may not be installed."
        )
    
    def _extract_with_newspaper(
        self, url: str, source_domain: str, source_name: str, region: str
    ) -> ExtractionResult:
        """Extract using newspaper3k library."""
        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()
            
            title = article.title or ""
            text = article.text or ""
            
            # Get publish date
            publish_date = ""
            if article.publish_date:
                publish_date = article.publish_date.isoformat()
            
            if not text or len(text.split()) < 20:
                return ExtractionResult(
                    success=False,
                    url=url,
                    source_domain=source_domain,
                    source_name=source_name,
                    region=region,
                    error="Article content too short or could not be extracted."
                )
            
            return ExtractionResult(
                success=True,
                url=url,
                title=title,
                text=text,
                source_domain=source_domain,
                source_name=source_name,
                publish_date=publish_date,
                region=region,
                extraction_method="newspaper3k"
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                url=url,
                source_domain=source_domain,
                source_name=source_name,
                region=region,
                error=f"newspaper3k extraction failed: {str(e)}"
            )
    
    def _extract_with_beautifulsoup(
        self, url: str, source_domain: str, source_name: str, region: str
    ) -> ExtractionResult:
        """Extract using requests + BeautifulSoup."""
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            text = self._extract_body(soup)
            
            # Extract publish date
            publish_date = self._extract_date(soup)
            
            if not text or len(text.split()) < 20:
                return ExtractionResult(
                    success=False,
                    url=url,
                    source_domain=source_domain,
                    source_name=source_name,
                    region=region,
                    error="Could not extract sufficient article content."
                )
            
            return ExtractionResult(
                success=True,
                url=url,
                title=title,
                text=text,
                source_domain=source_domain,
                source_name=source_name,
                publish_date=publish_date,
                region=region,
                extraction_method="beautifulsoup"
            )
            
        except requests.exceptions.Timeout:
            return ExtractionResult(
                success=False,
                url=url,
                source_domain=source_domain,
                source_name=source_name,
                region=region,
                error="Request timed out. The website may be slow or unavailable."
            )
        except requests.exceptions.RequestException as e:
            return ExtractionResult(
                success=False,
                url=url,
                source_domain=source_domain,
                source_name=source_name,
                region=region,
                error=f"Could not fetch URL: {str(e)}"
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                url=url,
                source_domain=source_domain,
                source_name=source_name,
                region=region,
                error=f"Extraction error: {str(e)}"
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title from HTML."""
        # Priority order for title extraction
        
        # 1. Open Graph title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        # 2. H1 tag
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        # 3. Title tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        return ""
    
    def _extract_body(self, soup: BeautifulSoup) -> str:
        """Extract main article body from HTML."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                            'aside', 'form', 'iframe', 'noscript']):
            element.decompose()
        
        # Remove common ad/navigation classes
        for cls in ['ad', 'advertisement', 'sidebar', 'nav', 'menu', 
                   'comment', 'related', 'social', 'share']:
            for elem in soup.find_all(class_=re.compile(cls, re.I)):
                elem.decompose()
        
        # Priority order for content extraction
        content = None
        
        # 1. Article tag
        article = soup.find('article')
        if article:
            content = article
        
        # 2. Main tag
        if not content:
            main = soup.find('main')
            if main:
                content = main
        
        # 3. Common content class names
        if not content:
            for cls in ['article-body', 'article-content', 'story-body', 
                       'post-content', 'entry-content', 'content-body']:
                elem = soup.find(class_=re.compile(cls, re.I))
                if elem:
                    content = elem
                    break
        
        # 4. Div with most paragraphs
        if not content:
            max_p = 0
            for div in soup.find_all('div'):
                p_count = len(div.find_all('p'))
                if p_count > max_p:
                    max_p = p_count
                    content = div
        
        if content:
            # Extract text from paragraphs
            paragraphs = content.find_all('p')
            text_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 30:  # Skip very short paragraphs
                    text_parts.append(text)
            return '\n\n'.join(text_parts)
        
        return ""
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publish date from HTML."""
        # Try various date sources
        
        # 1. Time tag with datetime
        time_tag = soup.find('time', datetime=True)
        if time_tag:
            return time_tag['datetime']
        
        # 2. Meta tags
        for prop in ['article:published_time', 'datePublished', 'pubdate']:
            meta = soup.find('meta', property=prop) or soup.find('meta', attrs={'name': prop})
            if meta and meta.get('content'):
                return meta['content']
        
        # 3. JSON-LD
        script = soup.find('script', type='application/ld+json')
        if script:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and 'datePublished' in data:
                    return data['datePublished']
            except:
                pass
        
        return ""
    
    def _parse_source(self, url: str) -> Tuple[str, str, str]:
        """
        Parse URL to get source information.
        
        Returns:
            Tuple of (domain, display_name, region)
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Get display name
            display_name = SOURCE_DISPLAY_NAMES.get(domain, domain.split('.')[0].title())
            
            # Detect region
            region = 'global'
            if domain.endswith('.in') or domain in INDIAN_DOMAINS:
                region = 'india'
            
            # Check subdomains for Indian sites (e.g., hindi.news18.com)
            for indian_domain in INDIAN_DOMAINS:
                if domain.endswith(indian_domain):
                    region = 'india'
                    break
            
            return domain, display_name, region
            
        except Exception:
            return "", "Unknown", "global"
    
    def _is_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            if robots_url not in self._robot_parsers:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                except:
                    # If we can't read robots.txt, assume allowed
                    return True
                self._robot_parsers[robots_url] = rp
            
            return self._robot_parsers[robots_url].can_fetch(self.user_agent, url)
        except:
            return True
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached(self, url: str) -> Optional[ExtractionResult]:
        """Get cached extraction result."""
        cache_key = self._get_cache_key(url)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ExtractionResult(
                    success=data['success'],
                    url=data['url'],
                    title=data.get('title', ''),
                    text=data.get('text', ''),
                    source_domain=data.get('source_domain', ''),
                    source_name=data.get('source_name', ''),
                    publish_date=data.get('publish_date', ''),
                    region=data.get('region', 'global'),
                    error=data.get('error', ''),
                    extraction_method=data.get('extraction_method', 'cached')
                )
            except:
                pass
        return None
    
    def _save_cache(self, url: str, result: ExtractionResult):
        """Save extraction result to cache."""
        cache_key = self._get_cache_key(url)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def clear_cache(self):
        """Clear all cached articles."""
        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                cache_file.unlink()
            except:
                pass


# Singleton instance
_extractor_instance = None


def get_article_extractor() -> ArticleExtractor:
    """Get singleton ArticleExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ArticleExtractor()
    return _extractor_instance


# Testing
if __name__ == "__main__":
    print("Testing Article Extractor")
    print("=" * 50)
    
    extractor = ArticleExtractor(use_cache=False)
    
    # Test URL parsing
    print("\n1. Testing URL parsing...")
    test_urls = [
        "https://www.thehindu.com/news/national/article123.html",
        "https://www.bbc.com/news/world-123456",
        "https://timesofindia.indiatimes.com/news/story.cms",
    ]
    
    for url in test_urls:
        domain, name, region = extractor._parse_source(url)
        print(f"   {url[:40]}... -> {name} ({region})")
    
    print("\n2. Extraction capabilities:")
    print(f"   newspaper3k available: {NEWSPAPER_AVAILABLE}")
    print(f"   BeautifulSoup available: {BS4_AVAILABLE}")
    print(f"   requests available: {REQUESTS_AVAILABLE}")
    
    print("\nArticle Extractor module loaded successfully!")
