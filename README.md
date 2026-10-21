# HinglishID-Router

**Word-Level Language Identification with Character N-Grams and Naive Bayes to Improve Intent Classification of Romanized Hindi-English Customer Support Queries**

---

## Project Overview

Indian users write support queries mixing Hindi and English (Hinglish):
> *"Mera order abhi tak nahi aaya, please refund karo"*

Standard English NLP pipelines treat Hindi words as noise. This project:
1. **Tags** each word as `HI` / `EN` / `OTHER` using character n-gram Naive Bayes
2. **Validates utility** by comparing 3 intent classifier setups (A/B/C)

## Setup

```bash
# 1. Clone and enter
git clone <repo-url>
cd hinglish_id_router

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Download LinCE dataset
# Register at: https://ritual.uh.edu/lince/
# Place files in data/raw/
```

## Data

- **LinCE Hindi-English LID** — primary training data (not committed; see above)
- `data/intent_raw.csv` — team-written Hinglish customer support queries
- `data/ambiguous_words.txt` — 30-40 ambiguous words (frozen before test)

## Run

```bash
# Run all tests
pytest tests/

# Run corpus stats notebook
jupyter notebook notebooks/01_corpus_stats.ipynb

# Run intent experiments
python src/run_intent_experiments.py

# Launch Streamlit demo
streamlit run app/demo.py
```

## Results

See `results/lid_dev.csv` and `results/intent_scores.csv`.
Generate figures: `python make_figures.py`

## Team

| Role | Modules |
|---|---|
| Member A — Tagger Lead | M1.2, M1.4, M1.5, M1.6, M2.1, M2.2, M2.3 |
| Member B — Data & Intent Lead | M1.4(verify), M2.4, M2.5, M3.1, M3.2 |
| Member C — Pipeline & Demo Lead | M1.3, M2.6, M2.7, M3.3, M3.4, M3.6 |

## Citation

```
Aguilar et al. (2020). LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation. LREC.
Patwa et al. (2020). SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text. SemEval.
```
