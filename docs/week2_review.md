# Week 2 Review: Context Features, Intent Data & Normalization

**Milestone Period:** Days 8 to 14 (Modules M2.1 to M2.7)  
**Project:** HinglishID-Router  
**Status:** **100% ON TRACK — ZERO TECHNICAL DEBT**  
**Test Suite Health:** **112 / 112 Unit Tests Passing (100% Pass Rate)**  

---

## 1. Executive Summary

Week 2 successfully bridged the project from word-level Language Identification (LID) into **Part 2: Downstream Customer Support Intent Routing**. Over the course of 7 modules (M2.1 through M2.7), the author (Jaydeep):
1. Enhanced the subword sequence tagger with neighboring context features ($w_{i-1}, w_{i+1}$), officially proving Hypothesis H2 on unseen test tokens with statistical significance (+11.53% gain on cross-lingual homographs).
2. Deep-dived into 2,149 dev set errors across 7 linguistic categories, cataloging 15 concrete case studies in [`docs/error_analysis.md`](file:///c:/Users/jayde/OneDrive/Attachments/Documents/Desktop/Projects/Hinglish_ID/docs/error_analysis.md).
3. Locked and frozen the production LID model ([`models/lid_final.joblib`](file:///c:/Users/jayde/OneDrive/Attachments/Documents/Desktop/Projects/Hinglish_ID/models/lid_final.joblib)) with SHA-256 verification and completed the single official test set evaluation.
4. Curated and validated the 360-query Hinglish Customer Support Intent Dataset across 6 core intents, achieving an inter-annotator agreement of **Cohen's Kappa = 0.9399** and **Fleiss' Kappa = 0.9398** on a 40-query blind overlap set.
5. Fixed and locked 15 deterministic stratified cross-validation splits (5 folds $\times$ 3 repeats) for rigorous downstream intent model benchmarking.
6. Benchmarked in-domain support queries (`data/lid_indomain.conll`), proving the context tagger cuts domain shift errors by 16.7% over word lookup.
7. Built an empirical 224-rule normalization lexicon (`data/norm_map.json`) and an intent-preserving stopword filter (`src/normalize.py`) with LID-aware homograph collision resolution (`do`).
8. Integrated all components into a single callable pipeline ([`src/pipeline.py`](file:///c:/Users/jayde/OneDrive/Attachments/Documents/Desktop/Projects/Hinglish_ID/src/pipeline.py)) with an average query latency of ~15ms.

---

## 2. Module-by-Module Progress

| Module | Day | Description | Key Deliverables | Verification Status |
|---|---|---|---|---|
| **M2.1** | Day 8 | Context Features ($w_{i-1}, w_{i+1}$) | `src/lid_context.py`, `table_2_context.csv` | **DONE** (Macro-F1 0.9806, beats lookup baseline) |
| **M2.2** | Day 9 | Ambiguous Word Mining & Evaluation | `data/ambiguous_words.txt`, `table_3_ambiguous.csv` | **DONE** (37 frozen homographs, +10.79% dev gain) |
| **M2.3** | Day 10 | Error Analysis & Single Test Evaluation | `docs/error_analysis.md`, `models/lid_final.joblib`, `table_4_test_final.csv` | **DONE** (98.42% test acc, 94.61% ambiguous acc) |
| **M2.4** | Day 11 | Intent Dataset Authoring (360 queries) | `data/intent_dataset.csv`, `data/intent_overlap_40.csv` | **DONE** (60/intent, 3 style sessions, 3 mix styles) |
| **M2.5** | Day 12 | Label QA, Kappa & CV Folds Protocol | `results/intent_kappa_qa.json`, `data/intent_splits_5fold.json`, `lid_indomain.conll` | **DONE** (Kappa: 0.9399, 15 locked CV folds) |
| **M2.6** | Day 13 | Normalization Lexicon & Stopwords | `data/norm_map.json`, `src/normalize.py` | **DONE** (224 rules, protected negations) |
| **M2.7** | Day 14 | Pipeline Integration & Week-2 Review | `src/pipeline.py`, `table_6_pipeline_samples.csv`, `docs/week2_review.md` | **DONE** (Single call `analyze_query()`, sub-15ms) |

---

## 3. Quantitative Milestones Summary

### Table W2.1: Final Official Test Set Benchmark (135,568 Tokens)
| Architecture | Overall Test Acc | Macro-F1 | Ambiguous Acc (H2) | Total Errors |
|---|---|---|---|---|
| **$M_0$ Word Lookup (Baseline)** | 97.96% | 0.9746 | 83.08% | 2,762 |
| **$M_1$ Char-Only LogReg** | 97.23% | 0.9658 | 82.96% | 3,749 |
| **$M_3$ Final Context Tagger (Ours)** | **98.42%** | **0.9805** | **94.61% (+11.53%)** | **2,143 (-22.4%)** |

### Table W2.2: Inter-Annotator Agreement QA (40 Overlap Queries)
| Evaluated Pair / Metric | Score | Landis & Koch Interpretation |
|---|---|---|
| Annotator A vs Annotator B | **0.9398** | Almost Perfect Agreement (0.81–1.00) |
| Annotator A vs Annotator C | **0.9699** | Almost Perfect Agreement (0.81–1.00) |
| Annotator B vs Annotator C | **0.9098** | Almost Perfect Agreement (0.81–1.00) |
| **Mean Pairwise Cohen's Kappa** | **0.9399** | Almost Perfect Agreement (0.81–1.00) |
| **Multi-Rater Fleiss' Kappa** | **0.9398** | Almost Perfect Agreement (0.81–1.00) |
| **Consensus Agreement Rate** | **92.5%** | 37 of 40 queries unanimous 3-way agreement |

### Table W2.3: In-Domain Customer Support LID Benchmark (45 Sentences, 389 Tokens)
| Model | Support Accuracy | Macro-F1 | HI-F1 | EN-F1 | Domain Shift Errors |
|---|---|---|---|---|---|
| $M_0$ Word Lookup Baseline | 87.66% | 0.6132 | 0.8708 | 0.9688 | 48 |
| **$M_3$ Final Context Tagger** | **89.72% (+2.06%)** | **0.6332** | **0.9936** | **0.9061** | **40 (-16.7%)** |

---

## 4. Architectural Integrity & Anti-Leakage Verification

1. **Zero Circular Dependency Guaranteed:**
   - The production language identification model ([`models/lid_final.joblib`](file:///c:/Users/jayde/OneDrive/Attachments/Documents/Desktop/Projects/Hinglish_ID/models/lid_final.joblib)) was trained strictly on `data/processed/train.conll` (35,564 sentences, 1,083,543 tokens from the L3Cube corpus).
   - The tagger was **never exposed to or trained on** the customer intent dataset ([`data/intent_dataset.csv`](file:///c:/Users/jayde/OneDrive/Attachments/Documents/Desktop/Projects/Hinglish_ID/data/intent_dataset.csv)).
   - This ensures complete methodological integrity for downstream intent classification in Week 3.
2. **Deterministic Evaluation Protocol:**
   - The 15 cross-validation folds (`data/intent_splits_5fold.json`) are mathematically locked.
   - Setups A, B, and C in Week 3 will be trained and validated on identical indices, ensuring that performance differences reflect true linguistic enhancements rather than fold sampling variance.

---

## 5. Week 3 Readiness (Downstream Comparison & Demo)

Week 2 deliverables are 100% complete and validated. The repository is ready to execute Week 3:
- **Day 15 (Module M3.1):** Implement Setups A, B, and C for fair downstream intent classification comparison across all 15 locked CV folds.
- **Day 16 (Module M3.2):** Statistical Significance Analysis (paired t-test, Wilcoxon signed-rank test, 95% bootstrap confidence intervals, McNemar's test).
- **Day 17 (Module M3.3):** CMI Routing Threshold Module (smart hybrid routing based on code-mixing intensity).
- **Day 18 (Module M3.4):** Interactive Streamlit Web Application Demo (`demo/app.py`).
- **Day 19 (Module M3.5):** Academic Final Report Drafting (`docs/final_report.md`).
- **Day 20 (Module M3.6):** Repository Polish & Reproducibility Harness (`Makefile`, `run_all.sh`).
- **Day 21 (Module M3.7):** Final Submission & Viva/Presentation Deck Preparation.
