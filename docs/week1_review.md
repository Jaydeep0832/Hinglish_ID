# Week-1 Review & Milestone Sign-Off — HinglishID-Router

> **Milestone:** Week 1 — Foundation & Tagger Core  
> **Status:** **ALL 7 MODULES (M1.1 to M1.7) COMPLETED ON SCHEDULE**  
> **Verification Status:** Verified & Signed Off by Solo Author (Jaydeep Bhatt)  
> **Test Suite:** 73/73 Tests Passing (100% Pass Rate)

---

## 1. Executive Summary & Verification Matrix

Week 1 focused on building the industrial foundation for word-level Language Identification (LID) on Romanized Hindi-English code-mixed queries. All scheduled modules were completed solo without slippage or compromises.

| Module | Title | Author | Verification | Status | Key Deliverable | Verified Outcome |
|---|---|---|---|---|---|---|
| **M1.1** | Setup + Dataset Access Check | Jaydeep | Automated Tests | **DONE** | Repo, `.venv`, L3Cube raw files, `docs/*.md` | Server contingency triggered cleanly to L3Cube-HingLID (1.35M tokens). |
| **M1.2** | Corpus Exploration + CoNLL Parser + Splits | Jaydeep | Unit Tests | **DONE** | `src/conll_parser.py`, `01_corpus_stats.ipynb`, `data/processed/*.conll` | 80/10/10 sentence split (`seed=42`). CMI Gambäck & Das formula validated. |
| **M1.3** | Preprocessing: Tokenizer + Masking | Jaydeep | Unit Tests | **DONE** | `src/preprocess.py`, `tests/test_preprocess.py` | Entity placeholders (`<URL>`, `<EMOJI>`, `<ORDER_ID>`), elongation reduction, Indic Unicode support. |
| **M1.4** | Baseline: Word-Lookup Tagger ($M_0$) | Jaydeep | Benchmark Run | **DONE** | `src/baseline_tagger.py`, `results/baseline_scores.csv` | 98.02% Dev Acc, 0.9753 Macro-F1. Quantified OOV vs in-vocab error share (43.5% vs 56.5%). |
| **M1.5** | Char N-Gram Naive Bayes Tagger ($M_1$) | Jaydeep | Model Training | **DONE** | `src/lid_tagger.py`, `src/train_lid_tagger.py`, `models/lid_*.joblib` | Tuned MultinomialNB ($\alpha=0.1$) & LogisticRegression ($C=1.0$). LR beats lookup on OOV words (77.87% vs 76.60%). |
| **M1.6** | Error Analysis & Ablation Table 1 | Jaydeep | Evaluation Suite | **DONE** | `src/evaluate_lid_ablation.py`, `results/ablation_table_1.csv`, 6 figures | Full 17-config ablation matrix, confusion heatmaps, sensitivity curves. |
| **M1.7** | Buffer + Week-1 Review + Guidelines Kickoff | Jaydeep | Full Verification | **DONE** | `src/demo_tagger.py`, `02_tagger_demo.ipynb`, `docs/intent_guidelines.md` | Interactive demonstration, test freeze integrity confirmation, intent guidelines approved. |

---

## 2. Quantitative Benchmark Summary (Dev Set: 136,386 Tokens)

| Model Architecture | Hyperparameters | Accuracy | Macro-F1 | HI F1 | EN F1 | In-Vocab Acc | OOV Acc | Total Errors |
|---|---|---|---|---|---|---|---|---|
| **$M_0$ Word Lookup** | `default=HI` | **98.02%** | **0.9753** | **0.9863** | **0.9643** | **98.84%** | 76.60% | **2,704** |
| **$M_1$ MultinomialNB** | $\alpha=0.1$, `char_wb=(1,4)` | 91.81% | 0.8987 | 0.9430 | 0.8545 | 92.52% | 73.22% | 11,173 |
| **$M_1$ MultinomialNB** | $\alpha=0.1$, `char_wb=(2,5)` | 93.33% | 0.9173 | 0.9537 | 0.8809 | 94.05% | 74.53% | 9,091 |
| **$M_1$ MultinomialNB** | Uniform Prior (`fit_prior=False`) | 92.43% | 0.9076 | 0.9469 | 0.8682 | 93.19% | 72.47% | 10,325 |
| **$M_1$ Logistic Regression** | $C=1.0$, `char_wb=(1,4)` | **97.34%** | **0.9670** | **0.9815** | **0.9526** | 98.08% | **77.87%** | **3,633** |
| **$M_1$ Logistic Regression** | $C=10.0$, `char_wb=(1,4)` | **97.59%** | **0.9702** | **0.9832** | **0.9571** | **98.36%** | 77.32% | **3,293** |

---

## 3. Frozen Test Set Verification Protocol

As established in Module M1.2 and reinforced throughout Week 1, the test partition (`data/processed/test.conll`) is held strictly **FROZEN** to preserve experimental integrity and prevent data snooping:

- **File Path:** `data/processed/test.conll`
- **Total Sentences:** 4,446
- **Total Tokens:** 135,568
- **Total Line Count:** 140,014 lines
- **SHA256 Checksum:** `a3ce094d776066cafebc9cda2d00a42521b7b838d66af075296af2d3843d8923`
- **Audit Result:** **100% UNTOUCHED**. Zero model training, hyperparameter search, or error inspections have accessed this file. It remains locked until final evaluation on Day 10.

---

## 4. Intent Annotation Readiness Sign-Off (Week 2 Kickoff)

The intent classification guidelines in `docs/intent_guidelines.md` have been reviewed and approved for Week 2:
1. **6 Target E-Commerce Intents:**
   - `refund` (paisa wapas / refund)
   - `cancel_order` (order cancel)
   - `order_not_delivered` (order nahi aaya / missing)
   - `delivery_delay` (late delivery / kab aayega)
   - `payment_failed` (paise kat gaye / failed transaction)
   - `other` (general queries, compliments, out of domain)
2. **Annotation Quotas:** 300–400 total realistic customer queries (~60 per intent).
3. **Quality Gate:** Inter-annotator agreement target: **Cohen's $\kappa \ge 0.60$** on a 40–50 query overlap set.

---

## 5. Week 2 Trajectory & Readiness

Week 1 has concluded ahead of schedule with all technical debt resolved. We are fully prepared to enter **Week 2: Context, Annotation & Normalization**:
- **Day 8 (Module M2.1):** Context Features (incorporating neighbouring words $w_{i-1}, w_{i+1}$ to resolve lexical ambiguity).
- **Day 9 (Module M2.2):** Pre-registering and freezing the 30–40 ambiguous word evaluation set.
- **Day 10 (Module M2.3):** Error analysis on 15 specific cases & freezing the LID tagger.
