# HinglishID-Router: Word-Level Language Identification & Normalization for Customer Support Routing

[![CI Tests](https://img.shields.io/badge/Unit%20Tests-125%20%2F%20125%20Passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit Demo](https://img.shields.io/badge/Demo-Streamlit%20Enterprise%20App-red.svg)](app/demo.py)
[![LinCE Benchmark](https://img.shields.io/badge/Benchmark-LinCE%201.35M%20Tokens-orange.svg)](https://ritual.uh.edu/lince/)

**Word-Level Language Identification with Character N-Grams and Context Features to Improve Intent Classification and Enterprise Routing of Romanized Hindi-English Customer Support Queries**

**Author & Lead Researcher:** Jaydeep Bhatt ([@Jaydeep0832](https://github.com/Jaydeep0832))  
**Project Scope:** Solo Research, Engineering & Empirical Evaluation Project (21 Modules / 3 Weeks)

---

## Executive Summary

Customer support interactions across Indian e-commerce, banking, and on-demand platforms predominantly occur in Romanized Hindi-English ("Hinglish") code-mixed text. Standard monolingual NLP pipelines fail catastrophically on these queries:
1. **Stopword Collisions:** English stopword lists drop *"do"*, destroying the operative Hindi imperative verb *"do"* ($दो$, meaning *"give"*), e.g., *"refund do"* $\to$ *"refund"*.
2. **Morphological Destruction:** Porter stemmers mutilate Romanized Hindi root words (e.g., *"bhai"* $\to$ *"bhai"*, *"gaya"* $\to$ *"gaya"*).
3. **Phonetic Spelling Variants:** Informal transliteration (*nahi, nahin, nhi, nai*) scatters statistical support across independent out-of-vocabulary (OOV) tokens.

**HinglishID-Router** solves this through an end-to-end modular pipeline:
- **LID Tagger:** Character $n$-gram Logistic Regression with local context windows ($w_{i-1}, w_{i+1}$) achieving **98.42% accuracy** on the frozen LinCE test set ($N=135,568$ tokens) and **+10.79% net disambiguation gain** across 36 pre-frozen cross-lingual homographs.
- **LID-Aware Normalization:** 224-rule phonetic standardizer with protected intent carriers (negations, question markers) preventing stopword collisions.
- **Enterprise Routing Layer:** Maps customer queries to operational departments, using a data-derived Code-Mixing Index threshold ($\text{CMI} \ge 30.0\%$) to automatically dispatch 37.8% of queries to specialized bilingual support queues.
- **Empirical Rigor & Scientific Honesty:** Evaluated across 15 locked CV folds on 360 customer queries ($\kappa = 0.9399$) with 5,000 bootstrap resamples. We openly disclose that generic subwords (`char_wb` 1–4) naturally absorb transliteration noise (**93.08% Macro-F1** vs **89.75%** monolingual baseline), while LID-aware normalization delivers top performance on primary financial grievance categories (`refund`: **0.9606**, `cancel_order`: **0.9723**).

---

## System Architecture

```
[Customer Support Query]
           │
           ▼
[Stage 1: Deterministic Preprocessing & Regex Masking]
  • Mask URLs (<URL>), Order IDs (<ORDER_ID>), Emojis (<EMOJI>)
  • Compress repeated characters: "plzzzz" -> "plzz", "bahutttt" -> "bahutt"
           │
           ▼
[Stage 2: Word-Level LID with Context Disambiguation]
  • Subword character n-grams (char_wb 1-4)
  • Local context windows: [w_{i-1}, w_{i+1}] with boundary tokens <s>, </s>
           │
           ▼
[Stage 3: Language-Aware Normalization]
  • HI Tokens: 224-rule phonetic map (data/norm_map.json)
  • Preserve critical intent carriers: Negations ("nahi", "mat"), Interrogatives ("kab", "kya")
  • EN Tokens: Protected stopwords + Porter stemming
           │
           ▼
[Stage 4: Downstream Intent Classification (Setups A, B, C, C')]
  • 6 Support Intents: refund, cancel_order, order_not_delivered, delivery_delay, payment_failed, other
           │
           ▼
[Stage 5: Enterprise Routing & Bilingual Triage]
  • Intent -> Department Queue
  • CMI >= 30.0% -> Bilingual Agent Queue Flag
  • Confidence < 0.50 -> Human Supervisor Escalation
```

---

## Empirical Benchmark Results

### 1. Official LinCE Held-Out Test Evaluation ($N=135,568$ tokens)
*Source: `results/table_4_test_final.json`*

| Architecture | Model Configuration | Test Accuracy | Macro-F1 | HI F1 | EN F1 | Ambiguous Acc | Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0 Baseline** | Majority Word Lookup | 97.96% | 0.9746 | 0.9859 | 0.9633 | 83.08% | 2,762 |
| **M1 Subword** | `char_wb` (1,4) LogReg (No Context) | 97.23% | 0.9658 | 0.9808 | 0.9507 | 82.96% | 3,749 |
| **M3 Production** | `char_wb` (1,4) + Context $[w_{i-1}, w_{i+1}]$ | **98.42%** | **0.9805** | **0.9890** | **0.9720** | **94.61%** | **2,143** |

### 2. Disambiguation on 36 Pre-Frozen Cross-Lingual Homographs
*Source: `results/table_3_ambiguous.json` (5,000 Bootstrap Resamples)*

- **Unigram Baseline Accuracy (M1):** 77.83%
- **Context-Aware Accuracy (M3):** **88.62%**
- **Net Disambiguation Gain:** **+10.79%**
- **95% Bootstrap Confidence Interval:** **$[+6.84\%, +14.86\%]$** ($p < 0.001$)

### 3. Downstream Intent Classification (15 Locked CV Folds)
*Source: `results/table_7_intent_experiments.json`*

| Setup | Description | Accuracy | Macro-F1 (Mean $\pm$ Std) | 95% Bootstrap CI vs Setup A |
| :--- | :--- | :---: | :---: | :---: |
| **Setup A** | Monolingual English Baseline (Word TF-IDF 1–2) | 89.82% | 0.8975 ($\pm 0.0280$) | Baseline |
| **Setup B** | Generic Subwords (`char_wb` 1–4, NO LID) | **93.06%** | **0.9308** ($\pm 0.0272$) | **$[+0.0184, +0.0484]$** ($p < 0.001$) |
| **Setup C** | LID-Aware Normalization (`char_wb` 1–4) | 91.85% | 0.9185 ($\pm 0.0258$) | $[+0.0062, +0.0358]$ ($p < 0.005$) |
| **Setup C'** | LID-Aware Normalization (Word TF-IDF 1–2) | 89.08% | 0.8899 ($\pm 0.0322$) | **$[-0.0163, +0.0007]$** ($p = 0.117$) |

### 4. Per-Intent Performance on High-Stakes Grievances
*Source: `results/table_7_intent_experiments.json`*

| Intent Category | Setup A (Monolingual) | Setup B (Generic Subwords) | Setup C (LID-Aware - Ours) | Operational Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **`refund`** | 0.8992 | 0.9555 | **0.9606** | **Setup C (+0.0051)** — Prevents customer churn |
| **`cancel_order`** | 0.9560 | 0.9697 | **0.9723** | **Setup C (+0.0026)** — Saves logistics transit costs |
| **`payment_failed`**| 0.9600 | **0.9639** | 0.9637 | Near-identical subword accuracy |
| **`other` (FAQ)** | 0.8744 | 0.9033 | **0.9039** | **Setup C (+0.0006)** |

---

## Scientific Findings & Negative Results Disclosure

In the spirit of open scientific inquiry, we report key findings that challenge conventional assumptions:

1. **Word-Level Phonetic Normalization Yields Zero Statistically Significant Gain:**
   Comparing Setup C' against Setup A yields a 95% Bootstrap Confidence Interval of **$[-0.0163, +0.0007]$** ($p = 0.117$). Because this interval crosses zero, manual phonetic standardization alone cannot be claimed to outperform raw text on moderately sized corpora.
2. **Subwords Naturally Absorb Transliteration Variations:**
   Setup B (`char_wb` 1–4) outperforms all setups overall (93.08% Macro-F1). Character boundary n-grams naturally decompose words into shared subwords (e.g., *kro* and *karo* share `[' k', 'ro ']`), absorbing spelling noise without requiring explicit lexicon engineering.
3. **The Value of LID in Enterprise Production:**
   While subwords absorb noise blindly, LID-aware normalization provides:
   - Top precision on high-impact grievance categories (`refund` and `cancel_order`).
   - Token-level auditability (inspecting tags and applied rules).
   - The Code-Mixing Index (CMI) calculation required for automated bilingual call-center routing.

---

## Installation & Quickstart

### Prerequisites
- Python 3.11+
- Git

### 1. Clone and Environment Setup
```bash
# Clone the repository
git clone https://github.com/Jaydeep0832/Hinglish_ID.git
cd Hinglish_ID

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# Download required NLTK corpora
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 2. Run the Full Unit Test Suite (125 Tests)
All 125 unit tests execute deterministically in $< 48$ seconds on standard CPU:
```bash
pytest
```

### 3. Reproduce Downstream Intent Benchmarks (15 Folds)
To retrain and benchmark Setups A, B, C, and C' across 15 locked CV folds and regenerate `results/table_7_intent_experiments.json`:
```bash
python src/run_intent_experiments.py
```

### 4. Launch the Interactive Streamlit Demo
Launch the interactive web application featuring colored token spans, CMI gauge, 3-way live setup comparison, and limitation disclosures:
```bash
streamlit run app/demo.py
```

---

## Repository Structure

```
Hinglish_ID/
├── app/
│   └── demo.py                       # Interactive Streamlit support demo
├── data/
│   ├── ambiguous_words.txt           # 36 pre-frozen ambiguous homographs
│   ├── norm_map.json                 # 224-rule phonetic normalization dictionary
│   ├── stopwords_hi.txt              # Hindi stopwords with protected intent carriers
│   └── processed/
│       ├── intent_clean.csv          # 360 double-annotated queries (kappa = 0.9399)
│       ├── dev.conll                 # LinCE dev split (135,568 tokens)
│       └── test.conll                # LinCE held-out test split (135,568 tokens)
├── docs/
│   ├── checkpoints.md                # 21-day checkpoint questions and verified answers
│   ├── error_analysis.md             # Categorized qualitative error taxonomy
│   ├── hypotheses.md                 # Pre-registered hypotheses (H1-H3)
│   ├── intent_guidelines.md          # Formal annotation guidelines (v2.0)
│   ├── presentation_slides.md        # 10-slide presentation deck
│   └── report_draft.md               # 13-section publication-grade research paper
├── models/                           # Serialized models (.joblib, ignored from git)
├── results/                          # Complete empirical result artifacts (.json, .csv)
│   ├── table_3_ambiguous.json        # Ambiguous word disambiguation results
│   ├── table_4_test_final.json       # Official test set evaluation
│   ├── table_7_intent_experiments.json # 15-fold cross-validation downstream results
│   └── cmi_vs_error.json             # Code-mixing error curve
├── src/
│   ├── conll_reader.py               # CoNLL parser and sentence-level splitter
│   ├── lang_aware.py                 # Language-aware phonetic normalizer
│   ├── lid_context.py                # Context feature generator [w_{i-1}, w_{i+1}]
│   ├── lid_tagger.py                 # Character n-gram Naive Bayes & Logistic Regression
│   ├── pipeline.py                   # Unified end-to-end HinglishPipeline
│   ├── preprocess.py                 # Regex masking and character collapse
│   ├── routing.py                    # Support queue router and CMI threshold logic
│   └── run_intent_experiments.py     # Downstream 15-fold benchmark script
├── tests/                            # 125 pytest unit tests across 10 modules
├── HinglishID-Router_Plan.md         # 21-day project execution plan
├── HinglishID-Router_Plan_and_Demo.html # Interactive HTML tracker and live toy demo
└── requirements.txt                  # Pinned dependencies
```

---

## Citation & References

```bibtex
@inproceedings{aguilar2020lince,
  title={{LinCE}: A Centralized Benchmark for Linguistic Code-switching Evaluation},
  author={Aguilar, Gustavo and Kar, Pranay and Solorio, Thamar},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020)},
  pages={1805--1813},
  year={2020}
}

@inproceedings{patwa2020semeval,
  title={{SemEval-2020 Task 9}: Overview of Sentiment Analysis of Code-Mixed Social Media Text},
  author={Patwa, Parth and Aguilar, Gustavo and Kar, Pranay and Pandey, Suraj and PYKL, Srinivas and Gamb{\"a}ck, Bj{\"o}rn and Chakraborty, Tanmoy and Solorio, Thamar},
  booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval 2020)},
  pages={774--790},
  year={2020}
}

@article{pedregosa2011scikit,
  title={Scikit-learn: Machine Learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```
