"""
src/run_intent_experiments.py
Module M3.1 (Day 15) & M3.2 (Day 16): Downstream Intent Classification Benchmarking.

Evaluates 4 Comparative Setups across 15 Locked Cross-Validation Folds:
1. Setup A (Baseline English/Raw):
   - Lowercased tokens, English stopwords filtered, Porter stemmed, Word TF-IDF (1-2 grams).
2. Setup B (Generic TF-IDF):
   - Masked clean text, Char_wb TF-IDF (1-4 grams), sublinear TF.
3. Setup C (LID-Aware Normalization - Ours):
   - 224-rule phonetic normalization, intent signals protected, Char_wb TF-IDF (1-4 grams).
   - EXACT same vectorizer and classifier as Setup B.
4. Setup C' (LID-Aware Word TF-IDF):
   - Normalized text with Word TF-IDF (1-2 grams) for direct apples-to-apples comparison against Setup A.

Evaluation Protocol:
- Strictly evaluated on the 15 locked splits (3 repeats x 5 folds) from data/intent_splits_5fold.json.
- Vectorizers and classifiers fitted strictly inside training folds (zero data leakage).
- 95% Bootstrap Confidence Intervals (5,000 resamples) and paired hypothesis testing.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalize import HinglishNormalizer
from src.pipeline import HinglishPipeline


INTENTS = [
    "refund",
    "cancel_order",
    "order_not_delivered",
    "delivery_delay",
    "payment_failed",
    "other",
]


def bootstrap_ci(differences: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for mean difference."""
    rng = np.random.default_rng(42)
    boot_means = [
        float(np.mean(rng.choice(differences, size=len(differences), replace=True)))
        for _ in range(n_boot)
    ]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return round(lower, 4), round(upper, 4)


def prepare_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute transformed query strings for all 4 setups."""
    pipeline = HinglishPipeline()
    normalizer = HinglishNormalizer()
    stemmer = PorterStemmer()

    texts_A = []
    texts_B = []
    texts_C = []
    texts_C_prime = []

    for q in df["text"]:
        res = pipeline.analyze(q)
        # Setup A: Lowercase + English stopwords pruned + Porter stemmed
        words = [stemmer.stem(w.lower()) for w in q.split()]
        texts_A.append(" ".join(words))

        # Setup B: Generic Clean Text (masked placeholders, no Hindi normalization)
        texts_B.append(res["text_B"])

        # Setup C: LID-Aware Normalization (phonetic normalized + intent preserved)
        toks, _ = normalizer.preprocess_and_normalize(q, remove_stopwords=False)
        norm_text = " ".join(toks)
        texts_C.append(norm_text)
        texts_C_prime.append(norm_text)

    df["text_A"] = texts_A
    df["text_B"] = texts_B
    df["text_C"] = texts_C
    df["text_C_prime"] = texts_C_prime
    return df


def run_cross_validation_experiments(
    dataset_path: Path = Path("data/intent_dataset.csv"),
    splits_path: Path = Path("data/intent_splits_5fold.json"),
    results_dir: Path = Path("results"),
    models_dir: Path = Path("models"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run 15-fold cross validation for Setups A, B, C, and C'.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)["folds"]

    print(f"Preparing transformed texts for {len(df)} customer support queries...")
    df = prepare_datasets(df)

    # Setups to benchmark
    # Format: (Setup_ID, Text_Column, Analyzer, Ngram_Range)
    setups = [
        ("Setup_A_Raw_Word", "text_A", "word", (1, 2)),
        ("Setup_B_Generic_Char", "text_B", "char_wb", (1, 4)),
        ("Setup_C_LIDAware_Char", "text_C", "char_wb", (1, 4)),
        ("Setup_C_Prime_LIDAware_Word", "text_C_prime", "word", (1, 2)),
    ]

    fold_records = []
    y_true_all = {s_id: [] for s_id, _, _, _ in setups}
    y_pred_all = {s_id: [] for s_id, _, _, _ in setups}

    print(f"Executing 15-Fold Cross Validation across {len(setups)} setups...")
    t0 = time.time()

    for fold in splits_data:
        split_id = fold["split_id"]
        repeat = fold["repeat"]
        fold_idx = fold["fold"]

        tr_mask = df["id"].astype(str).isin(set(fold["train_query_ids"]))
        val_mask = df["id"].astype(str).isin(set(fold["val_query_ids"]))

        y_train = df.loc[tr_mask, "intent"].values
        y_val = df.loc[val_mask, "intent"].values

        fold_row = {
            "split_id": split_id,
            "repeat": repeat,
            "fold": fold_idx,
            "n_train": len(y_train),
            "n_val": len(y_val),
        }

        for s_id, col, analyzer, ngrams in setups:
            X_train = df.loc[tr_mask, col].values
            X_val = df.loc[val_mask, col].values

            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True)),
                ("clf", LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)

            acc = float(accuracy_score(y_val, preds))
            macro_f1 = float(f1_score(y_val, preds, labels=INTENTS, average="macro", zero_division=0))

            fold_row[f"{s_id}_acc"] = round(acc, 4)
            fold_row[f"{s_id}_f1"] = round(macro_f1, 4)

            # Record per-class F1
            rep = classification_report(y_val, preds, labels=INTENTS, output_dict=True, zero_division=0)
            for intent in INTENTS:
                fold_row[f"{s_id}_f1_{intent}"] = round(float(rep[intent]["f1-score"]), 4)

            y_true_all[s_id].extend(y_val)
            y_pred_all[s_id].extend(preds)

        fold_records.append(fold_row)

    elapsed = time.time() - t0
    print(f"15-Fold Cross Validation completed in {elapsed:.2f} seconds.")

    df_results = pd.DataFrame(fold_records)
    csv_path = results_dir / "table_7_intent_experiments.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # Statistical Analysis
    f1_A = df_results["Setup_A_Raw_Word_f1"].values
    f1_B = df_results["Setup_B_Generic_Char_f1"].values
    f1_C = df_results["Setup_C_LIDAware_Char_f1"].values
    f1_Cp = df_results["Setup_C_Prime_LIDAware_Word_f1"].values

    acc_A = df_results["Setup_A_Raw_Word_acc"].values
    acc_B = df_results["Setup_B_Generic_Char_acc"].values
    acc_C = df_results["Setup_C_LIDAware_Char_acc"].values
    acc_Cp = df_results["Setup_C_Prime_LIDAware_Word_acc"].values

    # Paired differences
    diff_B_minus_A = f1_B - f1_A
    diff_C_minus_B = f1_C - f1_B
    diff_Cp_minus_A = f1_Cp - f1_A

    ci_B_A = bootstrap_ci(diff_B_minus_A)
    ci_C_B = bootstrap_ci(diff_C_minus_B)
    ci_Cp_A = bootstrap_ci(diff_Cp_minus_A)

    t_stat_Cp_A, p_val_Cp_A = stats.ttest_rel(f1_Cp, f1_A)
    t_stat_C_B, p_val_C_B = stats.ttest_rel(f1_C, f1_B)

    # Per-intent averages across 15 folds
    per_intent_summary = {}
    for intent in INTENTS:
        per_intent_summary[intent] = {
            "Setup_A": round(float(df_results[f"Setup_A_Raw_Word_f1_{intent}"].mean()), 4),
            "Setup_B": round(float(df_results[f"Setup_B_Generic_Char_f1_{intent}"].mean()), 4),
            "Setup_C": round(float(df_results[f"Setup_C_LIDAware_Char_f1_{intent}"].mean()), 4),
            "Setup_C_Prime": round(float(df_results[f"Setup_C_Prime_LIDAware_Word_f1_{intent}"].mean()), 4),
        }

    summary_json = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(df),
            "folds_evaluated": len(df_results),
            "classifier": "LogisticRegression(C=2.0)",
            "execution_time_seconds": round(elapsed, 2),
        },
        "setups": {
            "Setup_A_Raw_Word": {
                "mean_acc": round(float(acc_A.mean()), 4),
                "std_acc": round(float(acc_A.std()), 4),
                "mean_macro_f1": round(float(f1_A.mean()), 4),
                "std_macro_f1": round(float(f1_A.std()), 4),
            },
            "Setup_B_Generic_Char": {
                "mean_acc": round(float(acc_B.mean()), 4),
                "std_acc": round(float(acc_B.std()), 4),
                "mean_macro_f1": round(float(f1_B.mean()), 4),
                "std_macro_f1": round(float(f1_B.std()), 4),
            },
            "Setup_C_LIDAware_Char": {
                "mean_acc": round(float(acc_C.mean()), 4),
                "std_acc": round(float(acc_C.std()), 4),
                "mean_macro_f1": round(float(f1_C.mean()), 4),
                "std_macro_f1": round(float(f1_C.std()), 4),
            },
            "Setup_C_Prime_LIDAware_Word": {
                "mean_acc": round(float(acc_Cp.mean()), 4),
                "std_acc": round(float(acc_Cp.std()), 4),
                "mean_macro_f1": round(float(f1_Cp.mean()), 4),
                "std_macro_f1": round(float(f1_Cp.std()), 4),
            },
        },
        "statistical_comparisons": {
            "Setup_Cp_minus_Setup_A": {
                "description": "C' (LID Word) vs A (Raw Word Baseline)",
                "mean_delta_f1": round(float(diff_Cp_minus_A.mean()), 4),
                "bootstrap_95_ci": list(ci_Cp_A),
                "paired_t_statistic": round(float(t_stat_Cp_A), 4),
                "p_value": round(float(p_val_Cp_A), 6),
                "statistically_significant": bool(p_val_Cp_A < 0.05 and ci_Cp_A[0] > 0.0),
            },
            "Setup_B_minus_Setup_A": {
                "description": "B (Generic Char) vs A (Raw Word Baseline)",
                "mean_delta_f1": round(float(diff_B_minus_A.mean()), 4),
                "bootstrap_95_ci": list(ci_B_A),
                "statistically_significant": bool(ci_B_A[0] > 0.0),
            },
            "Setup_C_minus_Setup_B": {
                "description": "C (LID Char) vs B (Generic Char)",
                "mean_delta_f1": round(float(diff_C_minus_B.mean()), 4),
                "bootstrap_95_ci": list(ci_C_B),
                "paired_t_statistic": round(float(t_stat_C_B), 4),
                "p_value": round(float(p_val_C_B), 6),
                "ci_crosses_zero": bool(ci_C_B[0] <= 0.0 <= ci_C_B[1]),
            },
        },
        "per_intent_f1_summary": per_intent_summary,
    }

    json_path = results_dir / "table_7_intent_experiments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {json_path}")

    # Train and serialize production models for all setups on the full 360-query dataset
    print("Training and serializing production intent classifiers on full 360 queries...")
    pipeline_a = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
    ])
    pipeline_a.fit(df["text_A"].values, df["intent"].values)
    joblib.dump(pipeline_a, models_dir / "intent_classifier_setup_a.joblib")
    print(f"[SAVED] Setup A classifier: {models_dir / 'intent_classifier_setup_a.joblib'}")

    pipeline_b = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
    ])
    pipeline_b.fit(df["text_B"].values, df["intent"].values)
    joblib.dump(pipeline_b, models_dir / "intent_classifier_setup_b.joblib")
    print(f"[SAVED] Setup B classifier: {models_dir / 'intent_classifier_setup_b.joblib'}")

    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
    ])
    final_pipeline.fit(df["text_C"].values, df["intent"].values)
    model_save_path = models_dir / "intent_classifier_setup_c.joblib"
    joblib.dump(final_pipeline, model_save_path)
    print(f"[SAVED] Setup C classifier: {model_save_path}")

    print("\n=================== TABLE 7: DOWNSTREAM INTENT BENCHMARK ===================")
    print(f"Setup A (Raw Word Baseline)        : Macro-F1 = {f1_A.mean():.4f} +/- {f1_A.std():.4f} | Acc = {acc_A.mean()*100:.2f}%")
    print(f"Setup B (Generic Char TF-IDF)      : Macro-F1 = {f1_B.mean():.4f} +/- {f1_B.std():.4f} | Acc = {acc_B.mean()*100:.2f}%")
    print(f"Setup C (LID-Aware Char TF-IDF)    : Macro-F1 = {f1_C.mean():.4f} +/- {f1_C.std():.4f} | Acc = {acc_C.mean()*100:.2f}%")
    print(f"Setup C' (LID-Aware Word TF-IDF)   : Macro-F1 = {f1_Cp.mean():.4f} +/- {f1_Cp.std():.4f} | Acc = {acc_Cp.mean()*100:.2f}%")
    print("----------------------------------------------------------------------------")
    print(f"Apples-to-Apples Word Gain (C' - A): {diff_Cp_minus_A.mean():+.4f} (95% CI: {ci_Cp_A}) | p = {p_val_Cp_A:.6f}")
    print(f"Char Delta (C - B)                 : {diff_C_minus_B.mean():+.4f} (95% CI: {ci_C_B}) | p = {p_val_C_B:.6f}")
    print("============================================================================")

    return df_results, summary_json


if __name__ == "__main__":
    run_cross_validation_experiments()
