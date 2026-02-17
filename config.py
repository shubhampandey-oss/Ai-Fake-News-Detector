"""
Fake News Detection & Credibility Assessment System
Configuration Settings

This is an early-stage fake news detection system using linguistic analysis,
semantic modeling, source reputation scoring, and multi-source evidence aggregation.
It performs probabilistic credibility assessment, NOT absolute truth verification.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SOURCES_DIR = DATA_DIR / "sources"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SOURCES_DIR, SAVED_MODELS_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA SETTINGS
# =============================================================================
# Train/Validation/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Class imbalance handling
USE_SMOTE = True
USE_CLASS_WEIGHTS = True

# =============================================================================
# TEXT PREPROCESSING SETTINGS
# =============================================================================
# Minimum and maximum document length (in words)
MIN_DOC_LENGTH = 10
MAX_DOC_LENGTH = 5000

# Stopwords language
STOPWORDS_LANGUAGE = "english"

# =============================================================================
# FEATURE ENGINEERING SETTINGS
# =============================================================================
# TF-IDF settings
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.95

# Word2Vec settings (trained locally)
WORD2VEC_VECTOR_SIZE = 100  # Smaller for CPU efficiency
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 5
WORD2VEC_WORKERS = 4
WORD2VEC_EPOCHS = 10

# =============================================================================
# MODEL SETTINGS (CPU-OPTIMIZED)
# =============================================================================
# Logistic Regression
LR_MAX_ITER = 1000
LR_C = 1.0

# SVM (using linear kernel for faster training on large datasets)
SVM_KERNEL = "linear"
SVM_C = 1.0
SVM_GAMMA = "scale"  # Only used for rbf kernel

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5

# LSTM (CPU-optimized - smaller architecture)
LSTM_EMBEDDING_DIM = 64  # Reduced for CPU
LSTM_HIDDEN_DIM = 64     # Reduced for CPU
LSTM_NUM_LAYERS = 1      # Single layer for speed
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 10
LSTM_LEARNING_RATE = 0.001
LSTM_MAX_SEQ_LENGTH = 256  # Truncate long sequences

# Early stopping
EARLY_STOPPING_PATIENCE = 3

# =============================================================================
# SOURCE CREDIBILITY SETTINGS
# =============================================================================
# Credibility score range
CREDIBILITY_MIN = 0.0
CREDIBILITY_MAX = 1.0

# Weight factors for final credibility aggregation
WEIGHT_MODEL_PREDICTION = 0.50
WEIGHT_SOURCE_CREDIBILITY = 0.30
WEIGHT_EVIDENCE_AGREEMENT = 0.20

# Similarity threshold for cross-source matching
SIMILARITY_THRESHOLD = 0.7

# =============================================================================
# DETERMINISTIC MODE SETTINGS
# =============================================================================
# Enable deterministic mode for reproducible predictions
DETERMINISTIC_MODE = True

# Global seed for all random operations
GLOBAL_SEED = 42

# =============================================================================
# RSS FEED SOURCES (TRUSTED) - For Phase 2 Real-Time Inference
# =============================================================================
# NOTE: RSS data is used ONLY for inference, NEVER for training or evaluation

TRUSTED_RSS_FEEDS = {
    # International Sources
    "reuters": "https://feeds.reuters.com/reuters/topNews",
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "ap_news": "https://rsshub.app/apnews/topics/apf-topnews",
    "npr": "https://feeds.npr.org/1001/rss.xml",
    "the_guardian": "https://www.theguardian.com/world/rss",
    "nytimes": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "washington_post": "https://feeds.washingtonpost.com/rss/world",
    
    # Indian Sources
    "the_hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "indian_express": "https://indianexpress.com/feed/",
    "ndtv": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "times_of_india": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "hindustan_times": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
}

# RSS Cache Settings
RSS_CACHE_HOURS = 1  # How long to cache RSS articles (short for real-time)
RSS_FETCH_TIMEOUT = 10  # Seconds to wait for RSS feed response
RSS_MAX_ARTICLES_PER_SOURCE = 20  # Maximum articles to fetch per source

# Known unreliable sources (for demonstration - curated from public reports)
UNRELIABLE_SOURCES = [
    "infowars.com",
    "naturalnews.com",
    "beforeitsnews.com",
    "worldnewsdailyreport.com",
    "empirenews.net",
    "nationalreport.net",
    "theonion.com",  # Satire
    "babylonbee.com",  # Satire
]

# =============================================================================
# FLASK SETTINGS
# =============================================================================
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================
# Metrics to compute
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Cross-validation folds
CV_FOLDS = 5

# =============================================================================
# SENSATIONAL / EMOTIONAL WORDS (For linguistic features)
# =============================================================================
SENSATIONAL_WORDS = [
    "shocking", "unbelievable", "breaking", "urgent", "exclusive",
    "bombshell", "scandal", "explosive", "stunning", "horrifying",
    "terrifying", "incredible", "outrageous", "disgusting", "amazing",
    "miracle", "secret", "conspiracy", "exposed", "revealed",
    "devastating", "catastrophic", "unprecedented", "shocking truth",
    "you won't believe", "mind-blowing", "jaw-dropping"
]

# =============================================================================
# EVIDENCE MATCHING SETTINGS (For RSS Cross-Source Comparison)
# =============================================================================
# Minimum similarity score to consider articles as related
EVIDENCE_SIMILARITY_THRESHOLD = 0.3

# Number of similar articles to return
EVIDENCE_TOP_K = 5

# Weights for different similarity components
EVIDENCE_TITLE_WEIGHT = 0.6
EVIDENCE_CONTENT_WEIGHT = 0.4

