"""
src/evaluate_lid_ablation.py
Module M1.6: Week-1 Evaluation & Ablation Table 1 for HinglishID-Router.

Conducts systematic ablation experiments on dev.conll (keeping test.conll frozen):
1. M0 Word-Lookup Baseline
2. Subword Character N-Gram Range Ablation ((1,2), (1,3), (1,4), (2,5), (3,5))
3. Additive Smoothing Alpha Tuning (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
4. Class Prior Ablation (fit_prior=True vs fit_prior=False)
5. Discriminative Linear Classifier (LogisticRegression C in [0.1, 1.0, 10.0])

Generates:
- results/ablation_table_1.csv & results/ablation_table_1.json
- results/figures/confusion_matrix_m0_lookup.png
- results/figures/confusion_matrix_m1_nb.png
- results/figures/confusion_matrix_m1_lr.png
- results/figures/confusion_matrices_comparison.png
- results/figures/oov_vs_invocab_accuracy.png
- results/figures/ablation_ngram_alpha.png
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.baseline_tagger import WordLookupTagger
from src.conll_parser import read_conll
from src.lid_tagger import CharNgramLIDTagger


def plot_single_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot and save a standalone styled confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix: {output_path}")


def plot_comparison_confusion_matrices(
    models_data: List[Tuple[str, List[List[int]], List[str]]],
    output_path: Path,
) -> None:
    """Plot side-by-side comparison of confusion matrices for M0, M1-NB, and M1-LR."""
    fig, axes = plt.subplots(1, len(models_data), figsize=(16, 5), dpi=300)
    
    for ax, (name, cm, labels) in zip(axes, models_data):
        cm_arr = np.array(cm)
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(name, fontsize=12, weight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    fig.suptitle("Hinglish LID: Confusion Matrix Comparison on Dev Set (136,386 Tokens)", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison confusion matrix: {output_path}")


def plot_oov_vs_invocab_accuracy(
    models: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot comparative In-Vocabulary vs OOV Accuracy across architectures."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    names = [m["short_name"] for m in models]
    in_vocab_accs = [m["in_vocab_accuracy"] * 100 for m in models]
    oov_accs = [m["oov_accuracy"] * 100 for m in models]
    
    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width/2, in_vocab_accs, width, label="In-Vocabulary Accuracy", color="#2b5c8f", edgecolor="black")
    rects2 = ax.bar(x + width/2, oov_accs, width, label="OOV (Unseen) Accuracy", color="#e27c38", edgecolor="black")

    ax.set_ylabel("Accuracy (%)", fontsize=12, weight="bold")
    ax.set_title("In-Vocabulary vs. Out-Of-Vocabulary (OOV) Generalization on Dev Set", fontsize=13, weight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, weight="bold")
    ax.set_ylim(60, 105)
    ax.axhline(76.60, color="gray", linestyle="--", alpha=0.7, label="M0 OOV Baseline (76.60%)")
    ax.legend(frameon=True, facecolor="white", edgecolor="gray", loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, weight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved OOV vs In-Vocab plot: {output_path}")


def plot_ablation_curves(
    ngram_results: List[Dict[str, Any]],
    alpha_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot ablation sensitivity curves for N-Gram Range and Laplace Alpha."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Panel 1: N-Gram Range Ablation
    ng_labels = [r["hyperparams"].split(",")[0].replace("char_wb=", "") for r in ngram_results]
    ng_f1 = [r["macro_f1"] for r in ngram_results]
    ng_acc = [r["accuracy"] * 100 for r in ngram_results]

    ax1.plot(ng_labels, ng_f1, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="Macro-F1")
    ax1.set_title("Subword N-Gram Range Ablation (MultinomialNB)", fontsize=12, weight="bold", pad=10)
    ax1.set_xlabel("Character N-Gram Range", fontsize=11, weight="bold")
    ax1.set_ylabel("Dev Macro-F1", fontsize=11, weight="bold", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(ng_labels, ng_acc, marker="s", color="#2ca02c", linewidth=2, linestyle="--", markersize=7, label="Accuracy (%)")
    ax1_twin.set_ylabel("Dev Accuracy (%)", fontsize=11, weight="bold", color="#2ca02c")
    ax1_twin.tick_params(axis="y", labelcolor="#2ca02c")

    # Panel 2: Alpha Tuning Ablation
    a_vals = [r["alpha"] for r in alpha_results]
    a_f1 = [r["macro_f1"] for r in alpha_results]
    a_acc = [r["accuracy"] * 100 for r in alpha_results]

    ax2.plot(a_vals, a_f1, marker="^", color="#d62728", linewidth=2.5, markersize=8, label="Macro-F1")
    ax2.set_xscale("log")
    ax2.set_title("Laplace Smoothing Alpha Ablation (char_wb 1-4)", fontsize=12, weight="bold", pad=10)
    ax2.set_xlabel("Smoothing Alpha (log scale)", fontsize=11, weight="bold")
    ax2.set_ylabel("Dev Macro-F1", fontsize=11, weight="bold", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(a_vals, a_acc, marker="D", color="#9467bd", linewidth=2, linestyle="--", markersize=7, label="Accuracy (%)")
    ax2_twin.set_ylabel("Dev Accuracy (%)", fontsize=11, weight="bold", color="#9467bd")
    ax2_twin.tick_params(axis="y", labelcolor="#9467bd")

    fig.suptitle("Module M1.6: Subword Feature & Hyperparameter Ablation Dynamics", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation curves: {output_path}")


def run_m1_6_ablation(
    train_path: str = "data/processed/train.conll",
    dev_path: str = "data/processed/dev.conll",
    results_dir: str = "results",
    figures_dir: str = "results/figures",
) -> Dict[str, Any]:
    """Execute complete Module M1.6 ablation suite and generate Table 1 + figures."""
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HinglishID-Router: Module M1.6 Week-1 Evaluation & Ablation (Table 1)")
    print("=" * 80)

    # 1. Load splits (test split strictly kept FROZEN)
    print(f"Loading train split from {train_path}...")
    t0 = time.time()
    train_sents = read_conll(train_path)
    print(f"  Loaded {len(train_sents):,} sentences in {time.time()-t0:.2f}s.")

    print(f"Loading dev split from {dev_path}...")
    t1 = time.time()
    dev_sents = read_conll(dev_path)
    print(f"  Loaded {len(dev_sents):,} sentences in {time.time()-t1:.2f}s.")
    print("  [VERIFICATION] test.conll remains strictly untouched and FROZEN.")

    all_results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Baseline: M0 Word Lookup
    # -------------------------------------------------------------
    print("\n--- Running [Group 1: Baseline] M0 Word Lookup ---")
    m0 = WordLookupTagger()
    m0.fit(train_sents)
    m0_eval = m0.evaluate(dev_sents)
    m0_cm = m0_eval["confusion_matrix"]
    m0_labels = m0_eval["labels"]
    
    m0_row = {
        "group": "1_Baseline",
        "model_id": "M0_Lookup",
        "model_name": "Word-Lookup Baseline",
        "model_type": "lookup",
        "hyperparams": "default=HI",
        "ngram_range": "N/A",
        "alpha": "N/A",
        "fit_prior": True,
        "c_param": "N/A",
        "accuracy": m0_eval["accuracy"],
        "macro_f1": m0_eval["macro_f1"],
        "weighted_f1": m0_eval["weighted_f1"],
        "hi_precision": m0_eval["per_class"]["HI"]["precision"],
        "hi_recall": m0_eval["per_class"]["HI"]["recall"],
        "hi_f1": m0_eval["per_class"]["HI"]["f1"],
        "en_precision": m0_eval["per_class"]["EN"]["precision"],
        "en_recall": m0_eval["per_class"]["EN"]["recall"],
        "en_f1": m0_eval["per_class"]["EN"]["f1"],
        "in_vocab_accuracy": m0_eval["in_vocab_accuracy"],
        "oov_accuracy": m0_eval["oov_accuracy"],
        "total_errors": m0_eval["total_errors"],
        "in_vocab_errors": m0_eval["in_vocab_errors"],
        "oov_errors": m0_eval["oov_errors"],
        "hi_as_hi": m0_cm[1][1],
        "hi_as_en": m0_cm[1][0],
        "en_as_hi": m0_cm[0][1],
        "en_as_en": m0_cm[0][0],
        "short_name": "M0 Lookup",
        "confusion_matrix": m0_cm,
        "labels": m0_labels,
    }
    all_results.append(m0_row)
    print(f"  M0 Lookup -> Acc: {m0_row['accuracy']*100:.2f}%, Macro-F1: {m0_row['macro_f1']:.4f}, In-Vocab: {m0_row['in_vocab_accuracy']*100:.2f}%, OOV: {m0_row['oov_accuracy']*100:.2f}%")

    # -------------------------------------------------------------
    # 2. Subword N-Gram Range Ablation (MultinomialNB, alpha=0.1, fit_prior=True)
    # -------------------------------------------------------------
    print("\n--- Running [Group 2: N-Gram Range Ablation] ---")
    ngram_ranges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5)]
    ngram_rows: List[Dict[str, Any]] = []

    for ng in ngram_ranges:
        print(f"  Training MultinomialNB with char_wb={ng} (alpha=0.1)...")
        tagger = CharNgramLIDTagger(
            model_type="multinomial_nb",
            ngram_range=ng,
            alpha=0.1,
            fit_prior=True,
            min_df=2,
        )
        tagger.fit(train_sents)
        ev = tagger.evaluate(dev_sents)
        cm = ev["confusion_matrix"]
        row = {
            "group": "2_Ngram_Range",
            "model_id": f"M1_NB_ng_{ng[0]}_{ng[1]}",
            "model_name": f"MultinomialNB char_wb={ng}",
            "model_type": "multinomial_nb",
            "hyperparams": f"char_wb={ng}, alpha=0.1",
            "ngram_range": f"{ng[0]}-{ng[1]}",
            "alpha": 0.1,
            "fit_prior": True,
            "c_param": "N/A",
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "weighted_f1": ev["weighted_f1"],
            "hi_precision": ev["per_class"]["HI"]["precision"],
            "hi_recall": ev["per_class"]["HI"]["recall"],
            "hi_f1": ev["per_class"]["HI"]["f1"],
            "en_precision": ev["per_class"]["EN"]["precision"],
            "en_recall": ev["per_class"]["EN"]["recall"],
            "en_f1": ev["per_class"]["EN"]["f1"],
            "in_vocab_accuracy": ev["in_vocab_accuracy"],
            "oov_accuracy": ev["oov_accuracy"],
            "total_errors": ev["total_errors"],
            "in_vocab_errors": ev["in_vocab_errors"],
            "oov_errors": ev["oov_errors"],
            "hi_as_hi": cm[1][1],
            "hi_as_en": cm[1][0],
            "en_as_hi": cm[0][1],
            "en_as_en": cm[0][0],
            "short_name": f"NB ng={ng[0]}-{ng[1]}",
            "confusion_matrix": cm,
            "labels": ev["labels"],
        }
        all_results.append(row)
        ngram_rows.append(row)
        print(f"    -> Acc: {row['accuracy']*100:.2f}%, Macro-F1: {row['macro_f1']:.4f}, In-Vocab: {row['in_vocab_accuracy']*100:.2f}%, OOV: {row['oov_accuracy']*100:.2f}%")

    # -------------------------------------------------------------
    # 3. Additive Smoothing Alpha Ablation (MultinomialNB, char_wb=(1,4), fit_prior=True)
    # -------------------------------------------------------------
    print("\n--- Running [Group 3: Smoothing Alpha Ablation] ---")
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    alpha_rows: List[Dict[str, Any]] = []

    for a in alphas:
        print(f"  Training MultinomialNB with alpha={a} (char_wb=(1,4))...")
        tagger = CharNgramLIDTagger(
            model_type="multinomial_nb",
            ngram_range=(1, 4),
            alpha=a,
            fit_prior=True,
            min_df=2,
        )
        tagger.fit(train_sents)
        ev = tagger.evaluate(dev_sents)
        cm = ev["confusion_matrix"]
        row = {
            "group": "3_Alpha_Tuning",
            "model_id": f"M1_NB_alpha_{a}",
            "model_name": f"MultinomialNB alpha={a}",
            "model_type": "multinomial_nb",
            "hyperparams": f"alpha={a}, char_wb=(1,4)",
            "ngram_range": "1-4",
            "alpha": a,
            "fit_prior": True,
            "c_param": "N/A",
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "weighted_f1": ev["weighted_f1"],
            "hi_precision": ev["per_class"]["HI"]["precision"],
            "hi_recall": ev["per_class"]["HI"]["recall"],
            "hi_f1": ev["per_class"]["HI"]["f1"],
            "en_precision": ev["per_class"]["EN"]["precision"],
            "en_recall": ev["per_class"]["EN"]["recall"],
            "en_f1": ev["per_class"]["EN"]["f1"],
            "in_vocab_accuracy": ev["in_vocab_accuracy"],
            "oov_accuracy": ev["oov_accuracy"],
            "total_errors": ev["total_errors"],
            "in_vocab_errors": ev["in_vocab_errors"],
            "oov_errors": ev["oov_errors"],
            "hi_as_hi": cm[1][1],
            "hi_as_en": cm[1][0],
            "en_as_hi": cm[0][1],
            "en_as_en": cm[0][0],
            "short_name": f"NB alpha={a}",
            "confusion_matrix": cm,
            "labels": ev["labels"],
        }
        all_results.append(row)
        alpha_rows.append(row)
        print(f"    -> Acc: {row['accuracy']*100:.2f}%, Macro-F1: {row['macro_f1']:.4f}, In-Vocab: {row['in_vocab_accuracy']*100:.2f}%, OOV: {row['oov_accuracy']*100:.2f}%")

    # -------------------------------------------------------------
    # 4. Class Prior Ablation (MultinomialNB, char_wb=(1,4), alpha=0.1)
    # -------------------------------------------------------------
    print("\n--- Running [Group 4: Class Prior Ablation] ---")
    for fp in [True, False]:
        prior_str = "Empirical (fit_prior=True)" if fp else "Uniform (fit_prior=False)"
        print(f"  Training MultinomialNB with {prior_str}...")
        tagger = CharNgramLIDTagger(
            model_type="multinomial_nb",
            ngram_range=(1, 4),
            alpha=0.1,
            fit_prior=fp,
            min_df=2,
        )
        tagger.fit(train_sents)
        ev = tagger.evaluate(dev_sents)
        cm = ev["confusion_matrix"]
        row = {
            "group": "4_Class_Prior",
            "model_id": f"M1_NB_prior_{'empirical' if fp else 'uniform'}",
            "model_name": f"MultinomialNB ({'Empirical Prior' if fp else 'Uniform Prior'})",
            "model_type": "multinomial_nb",
            "hyperparams": f"fit_prior={fp}, alpha=0.1, char_wb=(1,4)",
            "ngram_range": "1-4",
            "alpha": 0.1,
            "fit_prior": fp,
            "c_param": "N/A",
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "weighted_f1": ev["weighted_f1"],
            "hi_precision": ev["per_class"]["HI"]["precision"],
            "hi_recall": ev["per_class"]["HI"]["recall"],
            "hi_f1": ev["per_class"]["HI"]["f1"],
            "en_precision": ev["per_class"]["EN"]["precision"],
            "en_recall": ev["per_class"]["EN"]["recall"],
            "en_f1": ev["per_class"]["EN"]["f1"],
            "in_vocab_accuracy": ev["in_vocab_accuracy"],
            "oov_accuracy": ev["oov_accuracy"],
            "total_errors": ev["total_errors"],
            "in_vocab_errors": ev["in_vocab_errors"],
            "oov_errors": ev["oov_errors"],
            "hi_as_hi": cm[1][1],
            "hi_as_en": cm[1][0],
            "en_as_hi": cm[0][1],
            "en_as_en": cm[0][0],
            "short_name": f"NB prior={'emp' if fp else 'unif'}",
            "confusion_matrix": cm,
            "labels": ev["labels"],
        }
        all_results.append(row)
        print(f"    -> Acc: {row['accuracy']*100:.2f}%, Macro-F1: {row['macro_f1']:.4f}, In-Vocab: {row['in_vocab_accuracy']*100:.2f}%, OOV: {row['oov_accuracy']*100:.2f}%")

    # -------------------------------------------------------------
    # 5. Discriminative Classifier: LogisticRegression (C in [0.1, 1.0, 10.0])
    # -------------------------------------------------------------
    print("\n--- Running [Group 5: Logistic Regression Regularization Ablation] ---")
    c_values = [0.1, 1.0, 10.0]
    lr_best_eval = None
    lr_best_cm = None
    lr_best_labels = None

    for c_val in c_values:
        print(f"  Training LogisticRegression with C={c_val} (char_wb=(1,4))...")
        lr_tagger = CharNgramLIDTagger(
            model_type="logistic_regression",
            ngram_range=(1, 4),
            c_param=c_val,
            min_df=2,
        )
        lr_tagger.fit(train_sents)
        ev = lr_tagger.evaluate(dev_sents)
        cm = ev["confusion_matrix"]
        
        if c_val == 1.0:
            lr_best_eval = ev
            lr_best_cm = cm
            lr_best_labels = ev["labels"]

        row = {
            "group": "5_LogReg_C",
            "model_id": f"M1_LogReg_C_{c_val}",
            "model_name": f"LogisticRegression C={c_val}",
            "model_type": "logistic_regression",
            "hyperparams": f"C={c_val}, char_wb=(1,4)",
            "ngram_range": "1-4",
            "alpha": "N/A",
            "fit_prior": "N/A",
            "c_param": c_val,
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "weighted_f1": ev["weighted_f1"],
            "hi_precision": ev["per_class"]["HI"]["precision"],
            "hi_recall": ev["per_class"]["HI"]["recall"],
            "hi_f1": ev["per_class"]["HI"]["f1"],
            "en_precision": ev["per_class"]["EN"]["precision"],
            "en_recall": ev["per_class"]["EN"]["recall"],
            "en_f1": ev["per_class"]["EN"]["f1"],
            "in_vocab_accuracy": ev["in_vocab_accuracy"],
            "oov_accuracy": ev["oov_accuracy"],
            "total_errors": ev["total_errors"],
            "in_vocab_errors": ev["in_vocab_errors"],
            "oov_errors": ev["oov_errors"],
            "hi_as_hi": cm[1][1],
            "hi_as_en": cm[1][0],
            "en_as_hi": cm[0][1],
            "en_as_en": cm[0][0],
            "short_name": f"LogReg C={c_val}",
            "confusion_matrix": cm,
            "labels": ev["labels"],
        }
        all_results.append(row)
        print(f"    -> Acc: {row['accuracy']*100:.2f}%, Macro-F1: {row['macro_f1']:.4f}, In-Vocab: {row['in_vocab_accuracy']*100:.2f}%, OOV: {row['oov_accuracy']*100:.2f}%")

    # -------------------------------------------------------------
    # 6. Save Table 1 to CSV and JSON
    # -------------------------------------------------------------
    csv_fields = [
        "group", "model_id", "model_name", "hyperparams", "ngram_range", "alpha",
        "fit_prior", "c_param", "accuracy", "macro_f1", "weighted_f1",
        "hi_precision", "hi_recall", "hi_f1", "en_precision", "en_recall", "en_f1",
        "in_vocab_accuracy", "oov_accuracy", "total_errors", "in_vocab_errors",
        "oov_errors", "hi_as_hi", "hi_as_en", "en_as_hi", "en_as_en"
    ]
    
    ablation_csv_path = results_path / "ablation_table_1.csv"
    with open(ablation_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            clean_row = {k: r[k] for k in csv_fields}
            writer.writerow(clean_row)
    print(f"\n[Artifact] Saved Ablation Table 1 to: {ablation_csv_path}")

    ablation_json_path = results_path / "ablation_table_1.json"
    serializable_results = [
        {k: v for k, v in r.items() if k not in ["confusion_matrix", "labels"]}
        for r in all_results
    ]
    with open(ablation_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"[Artifact] Saved Ablation JSON to: {ablation_json_path}")

    # Also update lid_dev.csv with top representative models
    primary_models = [
        m0_row,
        next(r for r in all_results if r["model_id"] == "M1_NB_alpha_0.1"),
        next(r for r in all_results if r["model_id"] == "M1_NB_alpha_0.5"),
        next(r for r in all_results if r["model_id"] == "M1_NB_alpha_1.0"),
        next(r for r in all_results if r["model_id"] == "M1_LogReg_C_1.0"),
    ]
    lid_dev_csv = results_path / "lid_dev.csv"
    with open(lid_dev_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_id", "model_name", "hyperparams", "accuracy", "macro_f1",
            "weighted_f1", "hi_f1", "en_f1", "in_vocab_accuracy", "oov_accuracy",
            "total_errors", "in_vocab_errors", "oov_errors"
        ])
        writer.writeheader()
        for r in primary_models:
            writer.writerow({
                "model_id": r["model_id"],
                "model_name": r["model_name"],
                "hyperparams": r["hyperparams"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "hi_f1": r["hi_f1"],
                "en_f1": r["en_f1"],
                "in_vocab_accuracy": r["in_vocab_accuracy"],
                "oov_accuracy": r["oov_accuracy"],
                "total_errors": r["total_errors"],
                "in_vocab_errors": r["in_vocab_errors"],
                "oov_errors": r["oov_errors"],
            })
    print(f"[Artifact] Updated primary benchmark to: {lid_dev_csv}")

    # -------------------------------------------------------------
    # 7. Generate High-Resolution Visualizations
    # -------------------------------------------------------------
    best_nb_row = next(r for r in all_results if r["model_id"] == "M1_NB_alpha_0.1")
    best_lr_row = next(r for r in all_results if r["model_id"] == "M1_LogReg_C_1.0")

    # Standalone Confusion Matrices
    plot_single_confusion_matrix(
        m0_row["confusion_matrix"],
        m0_row["labels"],
        "Confusion Matrix: M0 Word Lookup Baseline",
        figures_path / "confusion_matrix_m0_lookup.png",
    )
    plot_single_confusion_matrix(
        best_nb_row["confusion_matrix"],
        best_nb_row["labels"],
        "Confusion Matrix: M1 MultinomialNB (alpha=0.1)",
        figures_path / "confusion_matrix_m1_nb.png",
    )
    plot_single_confusion_matrix(
        best_lr_row["confusion_matrix"],
        best_lr_row["labels"],
        "Confusion Matrix: M1 LogisticRegression (C=1.0)",
        figures_path / "confusion_matrix_m1_lr.png",
    )

    # 3-Panel Comparison Confusion Matrix
    comparison_data = [
        ("M0 Word Lookup (Baseline)", m0_row["confusion_matrix"], m0_row["labels"]),
        ("M1 MultinomialNB (alpha=0.1)", best_nb_row["confusion_matrix"], best_nb_row["labels"]),
        ("M1 LogisticRegression (C=1.0)", best_lr_row["confusion_matrix"], best_lr_row["labels"]),
    ]
    plot_comparison_confusion_matrices(
        comparison_data,
        figures_path / "confusion_matrices_comparison.png",
    )

    # In-Vocab vs OOV Accuracy Bar Chart
    key_models_for_chart = [
        {"short_name": "M0 Lookup", "in_vocab_accuracy": m0_row["in_vocab_accuracy"], "oov_accuracy": m0_row["oov_accuracy"]},
        {"short_name": "M1 NB (a=0.1)", "in_vocab_accuracy": best_nb_row["in_vocab_accuracy"], "oov_accuracy": best_nb_row["oov_accuracy"]},
        {"short_name": "M1 LogReg (C=1)", "in_vocab_accuracy": best_lr_row["in_vocab_accuracy"], "oov_accuracy": best_lr_row["oov_accuracy"]},
    ]
    plot_oov_vs_invocab_accuracy(
        key_models_for_chart,
        figures_path / "oov_vs_invocab_accuracy.png",
    )

    # Sensitivity Curves
    plot_ablation_curves(
        ngram_rows,
        alpha_rows,
        figures_path / "ablation_ngram_alpha.png",
    )

    # -------------------------------------------------------------
    # 8. Print Executive Table
    # -------------------------------------------------------------
    print("\n" + "=" * 105)
    print("TABLE 1: FULL SYSTEMATIC ABLATION & WEEK-1 EVALUATION (Dev Set: 136,386 Tokens)")
    print("=" * 105)
    print(f"{'Group':<14} | {'Model / Config':<28} | {'Acc':<7} | {'Macro-F1':<8} | {'HI F1':<7} | {'EN F1':<7} | {'In-Voc':<7} | {'OOV Acc':<7} | {'Errors':<7}")
    print("-" * 105)
    for r in all_results:
        grp = r["group"].split("_", 1)[1] if "_" in r["group"] else r["group"]
        print(f"{grp:<14} | {r['model_name']:<28} | {r['accuracy']*100:>5.2f}% | {r['macro_f1']:>8.4f} | {r['hi_f1']:>7.4f} | {r['en_f1']:>7.4f} | {r['in_vocab_accuracy']*100:>5.2f}% | {r['oov_accuracy']*100:>5.2f}% | {r['total_errors']:>7,}")
    print("=" * 105)

    return {
        "all_results": all_results,
        "m0": m0_row,
        "best_nb": best_nb_row,
        "best_lr": best_lr_row,
    }


if __name__ == "__main__":
    run_m1_6_ablation()
