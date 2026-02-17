# Fake News Detection & Credibility Assessment System

An early-stage fake news detection and credibility assessment system using linguistic analysis, semantic modeling, source reputation scoring, and multi-source evidence aggregation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)

## ⚠️ Important Disclaimer

**This system performs probabilistic credibility assessment, NOT absolute truth verification.** Results should be interpreted as "likelihood of credibility" based on available evidence. This is a research project, not a production-ready fact-checking service.

## 🌟 Features

- **ML-Based Classification**: Multiple models (Logistic Regression, SVM, Random Forest, Bi-LSTM) trained locally
- **Linguistic Analysis**: Text style, readability metrics, sensational language detection
- **Source Credibility**: Pre-compiled database of source reputation scores
- **Evidence Aggregation**: Weighted combination of multiple credibility signals
- **Live RSS Integration**: Real-time news feeds from trusted sources (new!)
- **Deterministic Predictions**: Same input → Same output, guaranteed reproducibility
- **Web Interface**: Modern Bootstrap UI with real-time analysis
- **Fully Offline**: All models trained from scratch, no external APIs

## 🏗️ Two-Phase Architecture

This system implements a strict two-phase architecture:

### Phase 1: Offline Training
- Uses **static labeled datasets** (Kaggle Fake News, LIAR)
- Models trained locally: Logistic Regression, Linear SVM, Random Forest, LSTM
- Evaluation on held-out test sets
- Models saved for inference

### Phase 2: Real-Time Inference
- **Live RSS feeds** from trusted sources (Reuters, BBC, The Hindu, etc.)
- Trained models used for prediction (never retrained on live data)
- Multi-signal credibility aggregation
- Cross-source evidence matching

> **⚠️ Critical Rule**: RSS data is used ONLY for inference and display, NEVER for training or evaluation.

## 🎯 Credibility Assessment

The system uses a weighted multi-signal approach:

| Signal | Weight | Description |
|--------|--------|-------------|
| **Content Analysis** | 50% | ML model prediction probability |
| **Source Credibility** | 30% | Pre-compiled source reputation database |
| **Evidence Agreement** | 20% | Cross-source similarity matching |

### Prediction Categories
- **LIKELY REAL** (≥60% credibility): Content shows characteristics of credible news
- **LIKELY FAKE** (≤40% credibility): Content shows characteristics of unreliable news
- **UNCERTAIN** (40-60%): Insufficient evidence for confident assessment

## 📡 Live RSS Integration

The system fetches real-time articles from trusted news sources:

**International Sources:**
- Reuters, BBC, AP News, NPR, The Guardian, NY Times, Washington Post

**Indian Sources:**
- The Hindu, Indian Express, NDTV, Times of India, Hindustan Times

## 🔗 URL-Based News Analysis (NEW)

The system supports URL-based news analysis by automatically extracting article content from news websites. Manual article input is optional and used only as a fallback.

**Two Input Modes:**
1. **Analyze by URL** (Primary/Default): Paste a news article link → automatic extraction and analysis
2. **Paste Text** (Fallback): Manual text input when URL extraction fails

**Features:**
- Automatic headline, body, source extraction
- Region detection (India vs Global)
- Respects robots.txt
- Article caching for deterministic results
- Graceful fallback to manual mode

## 🔍 Claim-Based Verification (NEW - PRIMARY)

Users can now enter claims or questions directly, and the system verifies credibility by searching RSS feeds for evidence.

**Example Claims:**
- "Is Ajit Pawar dead in a plane crash?"
- "Did PM Modi resign?"
- "Earthquake in Delhi today?"

**Verdicts:**
- **LIKELY TRUE** - Multiple trusted sources confirm
- **LIKELY FALSE** - Trusted sources contradict or deny
- **UNVERIFIED** - No evidence found in RSS feeds
- **DISPUTED** - Conflicting reports

**Evidence Aggregation:**
| Signal | Weight | Description |
|--------|--------|-------------|
| Evidence Agreement | 50% | Confirmations vs contradictions |
| Source Credibility | 30% | Trusted Indian sources prioritized |
| Plausibility | 20% | Sensational language detection |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/verify-claim` | POST | **Verify Claim** - Primary input mode |
| `/api/analyze-url` | POST | Analyze by URL |
| `/api/extract-article` | POST | Extract article without analysis |
| `/api/rss/live` | GET | Get live RSS feed articles |
| `/analyze` | POST | Analyze by pasted text (fallback) |
| `/check-source` | POST | Check source credibility |
| `/api/health` | GET | System health check |

## 🔒 Deterministic Predictions

All predictions are guaranteed to be deterministic:
- Same article text → Same credibility score (every time)
- Seeds set for: numpy, random, torch, sklearn
- Models loaded once at startup (singleton pattern)
- No random elements in inference pipeline

## 📁 Project Structure

```
AI Fake News Detector/
├── app.py                    # Flask web application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
│
├── preprocessing/            # Data preprocessing
│   ├── text_cleaner.py       # Text cleaning pipeline
│   └── data_loader.py        # Dataset loading utilities
│
├── features/                 # Feature engineering
│   ├── textual_features.py   # TF-IDF, BoW, N-grams
│   ├── linguistic_features.py # Stylometric features
│   └── word2vec_trainer.py   # Local Word2Vec embeddings
│
├── models/                   # ML/DL models
│   ├── classical_models.py   # LR, SVM, Random Forest
│   └── lstm_model.py         # Bi-LSTM classifier
│
├── training/                 # Training scripts
│   ├── train_classical.py    # Train classical models
│   └── train_lstm.py         # Train LSTM model
│
├── evaluation/               # Model evaluation
│   ├── metrics.py            # Evaluation metrics
│   └── visualizations.py     # Plots and charts
│
├── credibility/              # Credibility assessment
│   ├── source_scorer.py      # Source reputation scoring
│   ├── rss_collector.py      # RSS feed collection
│   └── evidence_aggregator.py # Evidence combination
│
├── templates/                # HTML templates
│   ├── index.html            # Main analysis page
│   └── about.html            # System information
│
└── data/                     # Data directories
    ├── raw/                  # Original datasets
    ├── processed/            # Processed data
    └── sources/              # Source credibility data
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/create the project
cd "AI Fake News Detector"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Get the Dataset

Download the [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset):

1. Download `Fake.csv` and `True.csv`
2. Place them in `data/raw/`

### 3. Train Models

```bash
# Train classical ML models
python training/train_classical.py --evaluate --plot

# Train LSTM model (CPU-optimized)
python training/train_lstm.py --epochs 10 --evaluate --plot
```

### 4. Run Web Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 🔬 Technical Details

### Feature Engineering

| Feature Type | Description |
|--------------|-------------|
| **TF-IDF** | Term frequency-inverse document frequency vectors |
| **N-grams** | Bigram and trigram patterns |
| **Linguistic** | Readability, vocabulary richness, POS distributions |
| **Semantic** | Word2Vec document embeddings (locally trained) |

### Model Architecture

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier |
| **SVM** | Linear SVM with class weighting |
| **Random Forest** | 100 trees, CPU-optimized |
| **Bi-LSTM** | Bidirectional LSTM (small architecture for CPU) |

### Credibility Scoring

Final score combines:
- **Model Prediction (50%)**: ML classifier confidence
- **Source Credibility (30%)**: Pre-compiled reputation score
- **Evidence Agreement (20%)**: Cross-source similarity

## 📊 Expected Performance

On the Kaggle Fake News dataset (train on 70%, test on 15%):

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~93% | ~0.93 |
| SVM | ~94% | ~0.94 |
| Random Forest | ~92% | ~0.92 |
| LSTM | ~90% | ~0.90 |

*Note: Actual results may vary based on preprocessing and hyperparameters.*

## 🔒 Constraints & Limitations

- **No External APIs**: All processing is local
- **CPU Training**: Models optimized for CPU (no GPU required)
- **Static Source Database**: Source credibility is pre-compiled
- **Dataset Dependent**: Performance depends on training data
- **Not Real-Time Fact-Checking**: Cannot verify claims against the internet

## 🎓 Research Context

This system is designed as a **B.Tech final-year / IEEE-style research project** demonstrating:

1. Natural Language Processing for text classification
2. Multi-signal credibility assessment
3. Web application development
4. Machine learning model evaluation

## 📚 References

- Kaggle Fake News Dataset
- TextBlob for readability metrics
- Gensim for Word2Vec embeddings
- scikit-learn for classical ML
- PyTorch for deep learning

## 📄 License

This project is for educational and research purposes.

---

**Remember**: This is a probabilistic assessment tool. Always verify important information through multiple reputable sources.

---

## 🐍 Alternative Installation (Conda)

> [!TIP]
> **Use this method if gensim installation fails** with MSVC or C++ compiler errors. Conda handles C++ dependencies automatically.

### Step 0: Open Anaconda Prompt

> [!IMPORTANT]
> Use **Anaconda Prompt**, NOT regular CMD or PowerShell.

`Start Menu → Anaconda Prompt`

### Step 1: Create Conda Environment

```bash
conda create -n fake_news python=3.10
conda activate fake_news
```

You should see `(fake_news)` in your prompt.

### Step 2: Navigate to Project

```bash
cd "AI Fake News Detector"
```

### Step 3: Install Gensim via Conda

```bash
conda install -c conda-forge gensim
```

> [!NOTE]
> This bypasses all MSVC/C++ compilation issues.

### Step 4: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

*(Mixing conda + pip this way is safe)*

### Step 5: Download NLTK Resources

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### Step 6: Add Kaggle Dataset

Place the files in:
- `data/raw/Fake.csv`
- `data/raw/True.csv`

### Step 7: Train Models

```bash
python training/train_classical.py --evaluate --plot
python training/train_lstm.py --epochs 10 --evaluate --plot
```

### Step 8: Run Flask App

```bash
python app.py
```

Then open: http://localhost:5000



