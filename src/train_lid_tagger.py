"""
src/train_lid_tagger.py
Training, hyperparameter tuning, and evaluation script for Module M1.5:
- Trains MultinomialNB (alpha in [0.1, 0.5, 1.0])
- Trains LogisticRegression (C=1.0)
- Benchmarks against M0 Word-Lookup Baseline
- Saves comparison table to results/lid_dev.csv
- Saves serialized models to models/lid_nb.joblib and models/lid_lr.joblib
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from src.baseline_tagger import WordLookupTagger
from src.conll_parser import read_conll
from src.lid_tagger import CharNgramLIDTagger


def run_experiments(
    train_path: str = "data/processed/train.conll",
    dev_path: str = "data/processed/dev.conll",
    results_dir: str = "results",
    models_dir: str = "models",
) -> List[Dict[str, Any]]:
    """
    Execute Day 5 comparative experiments between Word Lookup (M0),
    MultinomialNB (M1 at alpha=0.1, 0.5, 1.0), and Logistic Regression.
    """
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HinglishID-Router: Module M1.5 Char N-Gram Tagger & Baseline Comparison")
    print("=" * 70)

    print(f"Loading training set from {train_path}...")
    t0 = time.time()
    train_sents = read_conll(train_path)
    print(f"  Loaded {len(train_sents):,} sentences in {time.time()-t0:.2f}s.")

    print(f"Loading dev set from {dev_path}...")
    t1 = time.time()
    dev_sents = read_conll(dev_path)
    print(f"  Loaded {len(dev_sents):,} dev sentences in {time.time()-t1:.2f}s.")

    experiment_rows = []

    # 1. Baseline: M0 Word Lookup
    print("\n[1/5] Evaluating M0: Word-Lookup Baseline...")
    m0_tagger = WordLookupTagger()
    m0_tagger.fit(train_sents)
    m0_metrics = m0_tagger.evaluate(dev_sents)
    print(f"  -> M0 Lookup: Acc = {m0_metrics['accuracy']*100:.2f}%, Macro-F1 = {m0_metrics['macro_f1']:.4f}, In-Vocab Acc = {m0_metrics['in_vocab_accuracy']*100:.2f}%, OOV Acc = {m0_metrics['oov_accuracy']*100:.2f}%")

    experiment_rows.append({
        "model_id": "M0_word_lookup",
        "model_name": "Word-Lookup Baseline",
        "hyperparams": "default=HI",
        "accuracy": m0_metrics["accuracy"],
        "macro_f1": m0_metrics["macro_f1"],
        "weighted_f1": m0_metrics["weighted_f1"],
        "hi_f1": m0_metrics["per_class"]["HI"]["f1"],
        "en_f1": m0_metrics["per_class"]["EN"]["f1"],
        "in_vocab_accuracy": m0_metrics["in_vocab_accuracy"],
        "oov_accuracy": m0_metrics["oov_accuracy"],
        "total_errors": m0_metrics["total_errors"],
        "in_vocab_errors": m0_metrics["in_vocab_errors"],
        "oov_errors": m0_metrics["oov_errors"],
    })

    # 2. MultinomialNB with alpha tuning: [0.1, 0.5, 1.0]
    best_nb_tagger = None
    best_nb_f1 = -1.0

    for idx, alpha in enumerate([0.1, 0.5, 1.0], 2):
        print(f"\n[{idx}/5] Training M1: MultinomialNB (alpha={alpha})...")
        t_fit = time.time()
        nb_tagger = CharNgramLIDTagger(
            model_type="multinomial_nb",
            ngram_range=(1, 4),
            alpha=alpha,
            min_df=2,
        )
        nb_tagger.fit(train_sents)
        metrics = nb_tagger.evaluate(dev_sents)
        print(f"  -> M1 NB (alpha={alpha}): Acc = {metrics['accuracy']*100:.2f}%, Macro-F1 = {metrics['macro_f1']:.4f}, In-Vocab Acc = {metrics['in_vocab_accuracy']*100:.2f}%, OOV Acc = {metrics['oov_accuracy']*100:.2f}% (Fit+Eval: {time.time()-t_fit:.2f}s)")

        if metrics["macro_f1"] > best_nb_f1:
            best_nb_f1 = metrics["macro_f1"]
            best_nb_tagger = nb_tagger

        experiment_rows.append({
            "model_id": f"M1_multinomial_nb_alpha_{alpha}",
            "model_name": f"MultinomialNB (alpha={alpha})",
            "hyperparams": f"alpha={alpha}, char_wb=(1,4)",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "hi_f1": metrics["per_class"]["HI"]["f1"],
            "en_f1": metrics["per_class"]["EN"]["f1"],
            "in_vocab_accuracy": metrics["in_vocab_accuracy"],
            "oov_accuracy": metrics["oov_accuracy"],
            "total_errors": metrics["total_errors"],
            "in_vocab_errors": metrics["in_vocab_errors"],
            "oov_errors": metrics["oov_errors"],
        })

    # 3. Logistic Regression
    print("\n[5/5] Training M1: LogisticRegression (C=1.0)...")
    t_fit_lr = time.time()
    lr_tagger = CharNgramLIDTagger(
        model_type="logistic_regression",
        ngram_range=(1, 4),
        c_param=1.0,
        min_df=2,
    )
    lr_tagger.fit(train_sents)
    lr_metrics = lr_tagger.evaluate(dev_sents)
    print(f"  -> M1 LogReg (C=1.0): Acc = {lr_metrics['accuracy']*100:.2f}%, Macro-F1 = {lr_metrics['macro_f1']:.4f}, In-Vocab Acc = {lr_metrics['in_vocab_accuracy']*100:.2f}%, OOV Acc = {lr_metrics['oov_accuracy']*100:.2f}% (Fit+Eval: {time.time()-t_fit_lr:.2f}s)")

    experiment_rows.append({
        "model_id": "M1_logistic_regression_C_1.0",
        "model_name": "LogisticRegression (C=1.0)",
        "hyperparams": "C=1.0, char_wb=(1,4)",
        "accuracy": lr_metrics["accuracy"],
        "macro_f1": lr_metrics["macro_f1"],
        "weighted_f1": lr_metrics["weighted_f1"],
        "hi_f1": lr_metrics["per_class"]["HI"]["f1"],
        "en_f1": lr_metrics["per_class"]["EN"]["f1"],
        "in_vocab_accuracy": lr_metrics["in_vocab_accuracy"],
        "oov_accuracy": lr_metrics["oov_accuracy"],
        "total_errors": lr_metrics["total_errors"],
        "in_vocab_errors": lr_metrics["in_vocab_errors"],
        "oov_errors": lr_metrics["oov_errors"],
    })

    # 4. Save results to results/lid_dev.csv
    csv_path = results_dir / "lid_dev.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_id", "model_name", "hyperparams", "accuracy", "macro_f1",
            "weighted_f1", "hi_f1", "en_f1", "in_vocab_accuracy", "oov_accuracy",
            "total_errors", "in_vocab_errors", "oov_errors"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in experiment_rows:
            writer.writerow(row)
    print(f"\nSaved dev benchmark comparison to: {csv_path}")

    # 5. Save best models
    if best_nb_tagger:
        best_nb_tagger.save(models_dir / "lid_nb.joblib")
    lr_tagger.save(models_dir / "lid_lr.joblib")

    # 6. Print Summary Table
    print("\n" + "=" * 80)
    print("DAY 5 COMPARATIVE EVALUATION TABLE (Dev Set)")
    print("=" * 80)
    print(f"{'Model':<28} | {'Acc':<7} | {'Macro-F1':<9} | {'In-Vocab':<9} | {'OOV Acc':<8} | {'Errors':<7}")
    print("-" * 80)
    for r in experiment_rows:
        print(f"{r['model_name']:<28} | {r['accuracy']*100:>5.2f}% | {r['macro_f1']:>9.4f} | {r['in_vocab_accuracy']*100:>7.2f}% | {r['oov_accuracy']*100:>6.2f}% | {r['total_errors']:>6,}")
    print("=" * 80)

    return experiment_rows


if __name__ == "__main__":
    run_experiments()
