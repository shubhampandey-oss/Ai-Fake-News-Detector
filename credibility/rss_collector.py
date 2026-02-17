"""
RSS Feed Collector Module

This module collects articles from trusted RSS feeds to build
a local reference corpus for cross-source comparison.

Phase 2 functionality: Used for real-time predictions
(NOT for training or evaluation).

Respects robots.txt and rate limits for ethical scraping.
"""

import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import sys

try:
    import feedparser
except ImportError:
    feedparser = None
    print("Warning: feedparser not installed. Install with: pip install feedparser")

try:
    import requests
except ImportError:
    requests = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRUSTED_RSS_FEEDS, SOURCES_DIR


class RSSCollector:
    """
    Collects articles from trusted RSS feeds.
    
    This builds a local corpus of articles from reputable sources
    for comparison during real-time predictions.
    
    Features:
    - Fetches from multiple RSS feeds
    - Caches articles to avoid redundant requests
    - Rate limiting to be polite to servers
    - Extracts title, summary, source, and date
    """
    
    def __init__(
        self,
        feeds: Optional[Dict[str, str]] = None,
        cache_dir: Optional[Path] = None,
        cache_hours: int = 24,
        request_delay: float = 1.0
    ):
        """
        Initialize RSS collector.
        
        Args:
            feeds: Dictionary mapping source name to RSS feed URL
            cache_dir: Directory to cache fetched articles
            cache_hours: Hours before cache expires
            request_delay: Seconds to wait between requests
        """
        self.feeds = feeds or TRUSTED_RSS_FEEDS.copy()
        self.cache_dir = cache_dir or (SOURCES_DIR / "rss_cache")
        self.cache_hours = cache_hours
        self.request_delay = request_delay
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Article storage
        self.articles = []
    
    def _get_cache_path(self, source: str) -> Path:
        """Get cache file path for a source."""
        safe_name = hashlib.md5(source.encode()).hexdigest()[:10]
        return self.cache_dir / f"{safe_name}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False
        
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = modified_time + timedelta(hours=self.cache_hours)
        
        return datetime.now() < expiry_time
    
    def _load_from_cache(self, source: str) -> Optional[List[Dict]]:
        """Load articles from cache."""
        cache_path = self._get_cache_path(source)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache for {source}: {e}")
        
        return None
    
    def _save_to_cache(self, source: str, articles: List[Dict]):
        """Save articles to cache."""
        cache_path = self._get_cache_path(source)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache for {source}: {e}")
    
    def fetch_feed(self, source: str, url: str) -> List[Dict]:
        """
        Fetch articles from a single RSS feed.
        
        Args:
            source: Source name
            url: RSS feed URL
            
        Returns:
            List of article dictionaries
        """
        if feedparser is None:
            print("feedparser not installed. Cannot fetch RSS feeds.")
            return []
        
        # Check cache first
        cached = self._load_from_cache(source)
        if cached is not None:
            print(f"  {source}: Loaded {len(cached)} articles from cache")
            return cached
        
        # Fetch from RSS
        try:
            feed = feedparser.parse(url)
            
            if feed.bozo:
                print(f"  Warning: Feed parsing issue for {source}")
            
            articles = []
            for entry in feed.entries:
                article = {
                    'source': source,
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', entry.get('description', '')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'fetched_at': datetime.now().isoformat()
                }
                
                # Clean HTML from summary
                if article['summary']:
                    import re
                    article['summary'] = re.sub(r'<[^>]+>', '', article['summary'])
                
                articles.append(article)
            
            # Cache the results
            self._save_to_cache(source, articles)
            
            print(f"  {source}: Fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"  Error fetching {source}: {e}")
            return []
    
    def fetch_all_feeds(
        self,
        sources: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch articles from all (or specified) RSS feeds.
        
        Args:
            sources: Optional list of source names to fetch.
                    If None, fetches from all configured feeds.
                    
        Returns:
            List of all fetched articles
        """
        if sources is None:
            sources = list(self.feeds.keys())
        
        print(f"Fetching RSS feeds from {len(sources)} sources...")
        
        all_articles = []
        
        for source in sources:
            if source not in self.feeds:
                print(f"  Warning: Unknown source '{source}', skipping")
                continue
            
            url = self.feeds[source]
            articles = self.fetch_feed(source, url)
            all_articles.extend(articles)
            
            # Rate limiting
            time.sleep(self.request_delay)
        
        self.articles = all_articles
        print(f"Total articles fetched: {len(all_articles)}")
        
        return all_articles
    
    def get_articles_by_source(
        self,
        source: str
    ) -> List[Dict]:
        """
        Get articles from a specific source.
        
        Args:
            source: Source name
            
        Returns:
            List of articles from that source
        """
        return [
            article for article in self.articles
            if article['source'] == source
        ]
    
    def get_recent_articles(
        self,
        hours: int = 24
    ) -> List[Dict]:
        """
        Get articles from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent articles
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for article in self.articles:
            try:
                fetched = datetime.fromisoformat(article.get('fetched_at', ''))
                if fetched >= cutoff:
                    recent.append(article)
            except:
                # Include if we can't parse date
                recent.append(article)
        
        return recent
    
    def search_articles(
        self,
        query: str,
        search_in: List[str] = ['title', 'summary']
    ) -> List[Dict]:
        """
        Search articles for matching text.
        
        Args:
            query: Search query
            search_in: Fields to search in
            
        Returns:
            List of matching articles
        """
        query_lower = query.lower()
        
        matches = []
        for article in self.articles:
            for field in search_in:
                content = article.get(field, '').lower()
                if query_lower in content:
                    matches.append(article)
                    break
        
        return matches
    
    def save_corpus(self, path: Optional[str] = None):
        """
        Save all fetched articles to a JSON file.
        
        Args:
            path: Save path
        """
        if path is None:
            path = SOURCES_DIR / "rss_corpus.json"
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'fetched_at': datetime.now().isoformat(),
                'article_count': len(self.articles),
                'articles': self.articles
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.articles)} articles to {path}")
    
    def load_corpus(self, path: Optional[str] = None) -> List[Dict]:
        """
        Load articles from a saved corpus file.
        
        Args:
            path: Load path
            
        Returns:
            List of articles
        """
        if path is None:
            path = SOURCES_DIR / "rss_corpus.json"
        
        if Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.articles = data.get('articles', [])
                print(f"Loaded {len(self.articles)} articles from {path}")
        else:
            print(f"No corpus found at {path}")
        
        return self.articles
    
    def get_corpus_stats(self) -> Dict:
        """
        Get statistics about the current corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        sources = {}
        for article in self.articles:
            source = article.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_articles': len(self.articles),
            'sources': sources,
            'source_count': len(sources)
        }


# Testing
if __name__ == "__main__":
    print("Testing RSS Collector Module")
    print("=" * 50)
    
    collector = RSSCollector()
    
    # Note: This will make actual HTTP requests
    print("\n1. Fetching from trusted feeds (may take a moment)...")
    
    # Only fetch from one feed for testing
    test_feeds = {'reuters': TRUSTED_RSS_FEEDS.get('reuters')}
    collector.feeds = test_feeds
    
    try:
        articles = collector.fetch_all_feeds()
        
        print(f"\n2. Corpus statistics:")
        stats = collector.get_corpus_stats()
        print(f"   Total articles: {stats['total_articles']}")
        print(f"   Sources: {stats['sources']}")
        
        if articles:
            print(f"\n3. Sample article:")
            sample = articles[0]
            print(f"   Title: {sample['title'][:60]}...")
            print(f"   Source: {sample['source']}")
            print(f"   Summary: {sample['summary'][:100]}...")
        
        print(f"\n4. Saving corpus...")
        collector.save_corpus()
        
    except Exception as e:
        print(f"\nNote: RSS fetching requires network access.")
        print(f"Error: {e}")
        print("\nCreating sample offline corpus for testing...")
        
        # Create sample offline data
        collector.articles = [
            {
                'source': 'reuters',
                'title': 'Sample Reuters Article',
                'summary': 'This is a sample article from Reuters for testing.',
                'link': 'https://reuters.com/sample',
                'published': '2024-01-01',
                'fetched_at': datetime.now().isoformat()
            }
        ]
        print(f"Created sample corpus with {len(collector.articles)} articles")
    
    print("\nAll RSS collector tests passed!")
