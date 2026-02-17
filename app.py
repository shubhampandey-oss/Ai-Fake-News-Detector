"""
Fake News Detection Web Application

Flask-based web interface for the Fake News Detection and
Credibility Assessment System.

Features:
- Article input and analysis
- Real-time credibility prediction
- Source credibility display
- Feature importance explanation
- Live RSS feed integration for real-time news
- Deterministic predictions (same input → same output)

TWO-PHASE ARCHITECTURE:
- Phase 1: Models trained on static labeled datasets (Kaggle Fake News)
- Phase 2: Real-time prediction on live RSS articles (this app)

This is an early-stage detection system performing probabilistic
credibility assessment, NOT absolute truth verification.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Initialize deterministic mode FIRST (before any other imports)
from utils.deterministic import set_all_seeds, is_deterministic_mode
set_all_seeds(42)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG, SAVED_MODELS_DIR,
    DETERMINISTIC_MODE, TRUSTED_RSS_FEEDS
)
from preprocessing.text_cleaner import TextCleaner
from credibility.source_scorer import SourceCredibilityScorer
from credibility.evidence_aggregator import EvidenceAggregator
from credibility.rss_collector import RSSCollector
from credibility.rss_analyzer import RSSAnalyzer, get_rss_analyzer

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Initialize components (singletons - loaded once)
text_cleaner = TextCleaner()
source_scorer = SourceCredibilityScorer()
aggregator = EvidenceAggregator()
rss_collector = RSSCollector()
rss_analyzer = get_rss_analyzer()

# Model and feature extractor (loaded on startup if available)
model = None
feature_extractor = None
_models_loaded = False


def load_models():
    """
    Load trained models if available.
    
    Models are loaded ONCE at startup and reused for all predictions.
    This ensures deterministic behavior.
    """
    global model, feature_extractor, _models_loaded
    
    if _models_loaded:
        return model is not None
    
    # Ensure deterministic mode
    set_all_seeds(42)
    
    # Try to load logistic regression (fastest for demo)
    from models.classical_models import LogisticRegressionClassifier
    from features.textual_features import TfidfExtractor
    
    try:
        model = LogisticRegressionClassifier()
        model.load()
        
        feature_extractor = TfidfExtractor()
        feature_extractor.load()
        
        _models_loaded = True
        print("✓ Models loaded successfully!")
        print(f"  Deterministic mode: {is_deterministic_mode()}")
        return True
    except Exception as e:
        print(f"Note: Could not load models: {e}")
        print("Running in demo mode (deterministic heuristics)")
        _models_loaded = True
        return False


def initialize_rss_corpus():
    """
    Initialize RSS corpus for cross-source evidence matching.
    
    This fetches articles from trusted RSS feeds and loads them
    into the analyzer for similarity matching.
    """
    try:
        print("Fetching RSS articles for evidence corpus...")
        articles = rss_collector.fetch_all_feeds()
        
        if articles:
            rss_analyzer.load_rss_corpus(articles)
            print(f"✓ Loaded {len(articles)} RSS articles into evidence corpus")
        else:
            print("  No RSS articles fetched (may be offline)")
    except Exception as e:
        print(f"  Could not initialize RSS corpus: {e}")


def deterministic_prediction(text: str) -> tuple:
    """
    Generate deterministic prediction for demo purposes.
    Used when no trained model is available.
    
    This uses ONLY deterministic heuristics (no randomness).
    Same input will ALWAYS produce the same output.
    
    Args:
        text: Article text
        
    Returns:
        Tuple of (prediction, confidence)
    """
    text_lower = text.lower()
    
    # Sensational word detection (deterministic)
    sensational = ['shocking', 'breaking', 'unbelievable', 'you wont believe',
                   'secret', 'exposed', 'scandal', 'conspiracy', 'bombshell',
                   'miracle', 'urgent', 'exclusive']
    sensational_count = sum(1 for word in sensational if word in text_lower)
    
    # Exclamation marks
    exclamation_count = text.count('!')
    
    # All caps words ratio
    words = text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
    
    # Longer articles tend to be more credible
    length_factor = min(len(words) / 500, 1.0) * 0.1
    
    # Compute base score (deterministic formula)
    fake_indicators = (
        sensational_count * 0.12 + 
        exclamation_count * 0.04 + 
        caps_ratio * 0.15
    )
    
    base_score = 0.55 + length_factor - fake_indicators
    
    # Clamp to valid range
    confidence = max(0.35, min(0.85, base_score))
    prediction = 1 if base_score > 0.5 else 0
    
    return prediction, confidence


def analyze_article(text: str, source: str = None) -> dict:
    """
    Perform full credibility analysis on an article.
    
    Uses multi-signal aggregation:
    - Content prediction (50%): ML model or deterministic heuristics
    - Source credibility (30%): Pre-compiled source database
    - Evidence agreement (20%): RSS cross-source similarity
    
    Args:
        text: Article text
        source: Optional source URL/domain
        
    Returns:
        Analysis results dictionary
    """
    # Ensure deterministic mode
    set_all_seeds(42)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'text_length': len(text.split()),
        'source': source or 'Unknown'
    }
    
    # Clean text
    cleaned_text = text_cleaner.clean(text)
    results['cleaned_text_sample'] = cleaned_text[:200] + '...' if len(cleaned_text) > 200 else cleaned_text
    
    # Get model prediction (deterministic)
    if model is not None and feature_extractor is not None:
        try:
            features = feature_extractor.transform([cleaned_text])
            prediction = model.predict_deterministic(features)[0]
            proba = model.predict_proba_deterministic(features)[0]
            confidence = float(proba[prediction])
            
            # Get important features
            top_features = feature_extractor.get_top_features(cleaned_text, top_n=10)
            results['important_words'] = [
                {'word': word, 'score': round(score, 4)}
                for word, score in top_features
            ]
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction, confidence = deterministic_prediction(text)
            results['important_words'] = []
    else:
        # Demo mode with deterministic heuristics
        prediction, confidence = deterministic_prediction(text)
        results['important_words'] = []
        results['demo_mode'] = True
    
    results['model_prediction'] = 'REAL' if prediction == 1 else 'FAKE'
    results['model_confidence'] = round(confidence, 3)
    
    # Get source credibility
    if source:
        source_score, source_explanation = source_scorer.get_score(source)
        results['source_credibility'] = round(source_score, 3)
        results['source_explanation'] = source_explanation
        results['source_trusted'] = source_scorer.is_trusted(source)
        results['source_unreliable'] = source_scorer.is_unreliable(source)
    else:
        source_score = 0.5
        results['source_credibility'] = 0.5
        results['source_explanation'] = 'No source provided'
    
    # Get RSS-based evidence score
    evidence_result = rss_analyzer.compute_evidence_score(cleaned_text)
    trusted_similarity = evidence_result.get('trusted_similarity', 0.5)
    untrusted_similarity = evidence_result.get('untrusted_similarity', 0.5)
    
    results['evidence_details'] = {
        'trusted_similarity': trusted_similarity,
        'untrusted_similarity': untrusted_similarity,
        'entity_overlap': evidence_result.get('entity_overlap_score', 0.0),
        'similar_articles': evidence_result.get('similar_trusted_articles', [])
    }
    
    # Aggregate evidence
    assessment = aggregator.aggregate(
        model_prediction=prediction,
        model_confidence=confidence,
        source_credibility=source_score,
        trusted_similarity=trusted_similarity,
        untrusted_similarity=untrusted_similarity
    )
    
    results['final_prediction'] = assessment['prediction']
    results['credibility_score'] = assessment['credibility_score']
    results['confidence_percentage'] = assessment['confidence_percentage']
    results['explanation'] = assessment['explanation']
    results['components'] = assessment['components']
    results['disclaimer'] = assessment['disclaimer']
    
    return results


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def home():
    """Render home page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for article analysis."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        source = data.get('source', '').strip()
        
        if not text:
            return jsonify({'error': 'No article text provided'}), 400
        
        if len(text) < 50:
            return jsonify({
                'error': 'Article text too short. Please provide at least 50 characters.'
            }), 400
        
        # Perform analysis
        results = analyze_article(text, source if source else None)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check-source', methods=['POST'])
def check_source():
    """API endpoint to check source credibility."""
    try:
        data = request.get_json()
        source = data.get('source', '').strip()
        
        if not source:
            return jsonify({'error': 'No source provided'}), 400
        
        score, explanation = source_scorer.get_score(source)
        
        return jsonify({
            'source': source,
            'score': round(score, 3),
            'explanation': explanation,
            'trusted': source_scorer.is_trusted(source),
            'unreliable': source_scorer.is_unreliable(source)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Claim Verification Endpoints (Claim-Based Credibility Assessment)
# =============================================================================

# Import claim verification modules
try:
    from credibility.claim_extractor import ClaimExtractor, get_claim_extractor
    from credibility.claim_verifier import ClaimVerifier, get_claim_verifier
    claim_extractor = get_claim_extractor()
    claim_verifier = get_claim_verifier()
    CLAIM_VERIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Claim verification modules not available: {e}")
    CLAIM_VERIFICATION_AVAILABLE = False
    claim_extractor = None
    claim_verifier = None


@app.route('/api/verify-claim', methods=['POST'])
def verify_claim():
    """
    Verify a claim or question against RSS evidence.
    
    This is the PRIMARY input mode - users enter claims like:
    "Is Ajit Pawar dead in a plane crash?"
    
    The system searches RSS feeds for evidence and returns credibility assessment.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        claim = data.get('claim', '').strip()
        
        if not claim:
            return jsonify({'error': 'No claim provided'}), 400
        
        if len(claim) < 5:
            return jsonify({'error': 'Claim too short. Please provide more details.'}), 400
        
        # Check if claim verification is available
        if not CLAIM_VERIFICATION_AVAILABLE:
            return jsonify({
                'error': 'Claim verification not available',
                'verdict': 'UNAVAILABLE'
            }), 503
        
        # Ensure deterministic mode
        set_all_seeds(42)
        
        # Refresh RSS corpus for latest evidence
        try:
            articles = rss_collector.fetch_all_feeds()
            if articles:
                claim_verifier.update_corpus(articles)
        except Exception as e:
            print(f"Could not refresh RSS corpus: {e}")
        
        # Verify claim
        result = claim_verifier.verify_claim(claim)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract-claim-info', methods=['POST'])
def extract_claim_info():
    """
    Extract entities and keywords from a claim without verification.
    
    Useful for debugging and understanding what the system extracts.
    """
    try:
        data = request.get_json()
        claim = data.get('claim', '').strip()
        
        if not claim:
            return jsonify({'error': 'No claim provided'}), 400
        
        if not CLAIM_VERIFICATION_AVAILABLE:
            return jsonify({'error': 'Claim extraction not available'}), 503
        
        info = claim_extractor.extract_claim_info(claim)
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# URL Analysis Endpoints (URL-Based News Analysis)
# =============================================================================

# Import article extractor
try:
    from utils.article_extractor import ArticleExtractor, get_article_extractor
    article_extractor = get_article_extractor()
    URL_EXTRACTION_AVAILABLE = True
except ImportError:
    URL_EXTRACTION_AVAILABLE = False
    article_extractor = None


@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """
    Analyze news article by URL.
    
    This is the PRIMARY input mode - users paste only a URL and the system
    automatically extracts and analyzes the article content.
    
    Falls back gracefully if extraction fails.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL. Please provide a valid HTTP/HTTPS URL.'}), 400
        
        # Check if extraction is available
        if not URL_EXTRACTION_AVAILABLE or article_extractor is None:
            return jsonify({
                'error': 'URL extraction not available. Please install: pip install requests beautifulsoup4',
                'extraction_status': 'unavailable',
                'fallback_required': True
            }), 503
        
        # Ensure deterministic mode
        set_all_seeds(42)
        
        # Extract article from URL
        extraction = article_extractor.extract_from_url(url)
        
        if not extraction.success:
            return jsonify({
                'extraction_status': 'failed',
                'extraction_error': extraction.error,
                'source_info': {
                    'domain': extraction.source_domain,
                    'name': extraction.source_name,
                    'region': extraction.region
                },
                'fallback_required': True,
                'message': 'Could not extract article content. Please paste the article text manually.'
            }), 200  # Return 200 so UI can handle gracefully
        
        # Get source info
        source_info = source_scorer.get_source_info(url)
        
        # Prepare text for analysis (combine title and body)
        analysis_text = extraction.text
        if extraction.title and extraction.title not in analysis_text[:200]:
            analysis_text = extraction.title + "\n\n" + analysis_text
        
        # Perform full analysis using unified pipeline
        results = analyze_article(analysis_text, url)
        
        # Add extraction-specific information
        results['extraction_status'] = 'success'
        results['extraction_method'] = extraction.extraction_method
        results['extracted_title'] = extraction.title
        results['extracted_text_preview'] = extraction.text[:500] + '...' if len(extraction.text) > 500 else extraction.text
        results['publish_date'] = extraction.publish_date
        results['source_info'] = {
            'domain': extraction.source_domain,
            'name': extraction.source_name,
            'region': extraction.region,
            'credibility_score': source_info['credibility_score'],
            'is_trusted': source_info['is_trusted'],
            'is_indian_source': extraction.region == 'india'
        }
        results['fallback_required'] = False
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract-article', methods=['POST'])
def extract_article():
    """
    Extract article content from URL without analysis.
    
    Useful for previewing what will be analyzed.
    """
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        if not URL_EXTRACTION_AVAILABLE or article_extractor is None:
            return jsonify({'error': 'URL extraction not available'}), 503
        
        extraction = article_extractor.extract_from_url(url)
        
        return jsonify(extraction.to_dict())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# RSS Endpoints (Phase 2: Real-Time RSS Integration)
# =============================================================================

@app.route('/api/rss/live')
def rss_live():
    """
    Get live RSS feed articles.
    
    Returns recent articles from trusted RSS sources for display
    in the Live RSS Panel.
    """
    try:
        # Fetch fresh articles
        articles = rss_collector.fetch_all_feeds()
        
        # Format for frontend
        formatted = []
        for article in articles[:30]:  # Limit to 30 most recent
            formatted.append({
                'title': article.get('title', ''),
                'summary': article.get('summary', '')[:200] + '...' if len(article.get('summary', '')) > 200 else article.get('summary', ''),
                'source': article.get('source', ''),
                'link': article.get('link', ''),
                'published': article.get('published', ''),
                'fetch_timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'articles': formatted,
            'total': len(formatted),
            'sources': list(TRUSTED_RSS_FEEDS.keys()),
            'fetch_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'articles': []}), 500


@app.route('/api/rss/analyze', methods=['POST'])
def rss_analyze():
    """
    Analyze an RSS article.
    
    Takes an RSS article and performs full credibility analysis.
    """
    try:
        data = request.get_json()
        
        title = data.get('title', '')
        text = data.get('text', '') or data.get('summary', '')
        source = data.get('source', '')
        link = data.get('link', '')
        
        if not text:
            return jsonify({'error': 'No article text provided'}), 400
        
        # Combine title and text for analysis
        full_text = f"{title}\n\n{text}" if title else text
        
        # Perform analysis
        results = analyze_article(full_text, source)
        results['rss_article'] = True
        results['original_link'] = link
        results['original_title'] = title
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rss/similar', methods=['POST'])
def rss_similar():
    """
    Find similar articles from trusted RSS sources.
    
    Returns articles from trusted sources that are similar to the input text.
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Find similar articles
        similar = rss_analyzer.find_similar_articles(text, top_k=5, trusted_only=True)
        
        return jsonify({
            'similar_articles': similar,
            'query_length': len(text.split()),
            'corpus_size': len(rss_analyzer.rss_articles)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'similar_articles': []}), 500


@app.route('/api/rss/evidence', methods=['POST'])
def rss_evidence():
    """
    Compute evidence score for an article.
    
    Returns detailed evidence analysis including similarity scores
    and entity overlap.
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        title = data.get('title', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Full analysis
        analysis = rss_analyzer.analyze_article(text, title)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Existing Endpoints
# =============================================================================

@app.route('/trusted-sources')
def trusted_sources():
    """Get list of trusted sources."""
    sources = source_scorer.get_trusted_sources(threshold=0.8)
    return jsonify({
        'sources': [
            {'domain': domain, 'score': round(score, 2)}
            for domain, score in sources[:20]
        ]
    })


@app.route('/unreliable-sources')
def unreliable_sources():
    """Get list of known unreliable sources."""
    sources = source_scorer.get_unreliable_sources(threshold=0.3)
    return jsonify({
        'sources': [
            {'domain': domain, 'score': round(score, 2)}
            for domain, score in sources[:20]
        ]
    })


@app.route('/about')
def about():
    """About page with system information."""
    return render_template('about.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'deterministic_mode': is_deterministic_mode(),
        'rss_corpus_size': len(rss_analyzer.rss_articles),
        'timestamp': datetime.now().isoformat()
    })


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# =============================================================================
# Application Startup
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("FAKE NEWS DETECTION - WEB APPLICATION")
    print("=" * 60)
    print("\nThis is a probabilistic credibility assessment system.")
    print("It performs assessment, NOT absolute truth verification.")
    print("\nTwo-Phase Architecture:")
    print("  Phase 1: Models trained on static labeled datasets")
    print("  Phase 2: Real-time prediction on live RSS articles")
    print(f"\nDeterministic Mode: {DETERMINISTIC_MODE}")
    print("  Same input → Same output (guaranteed)")
    
    # Load models
    load_models()
    
    # Initialize RSS corpus for evidence matching
    initialize_rss_corpus()
    
    print(f"\nStarting server at http://{FLASK_HOST}:{FLASK_PORT}")
    print("Press Ctrl+C to stop\n")
    
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
