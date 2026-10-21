"""
src/evaluate_final_test.py
Module M2.3 (Day 10): Official Final Test Set Evaluation & Model Freeze.

STRICT PROTOCOL:
- This is the SINGLE and FINAL evaluation on data/processed/test.conll.
- Models evaluated:
    1. M0: Word Lookup Baseline
    2. M1: Char N-Gram Logistic Regression (char_wb 1-4)
    3. M3: Final Context Tagger (models/lid_final.joblib)
- Hyperparameters are FROZEN. No tuning or post-hoc adjustments allowed.
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_tagger import WordLookupTagger
from src.conll_parser import read_conll
from src.lid_context import ContextLIDTagger
from src.lid_tagger import CharNgramLIDTagger

# Register classes for joblib unpickling
import __main__
__main__.WordLookupTagger = WordLookupTagger
__main__.CharNgramLIDTagger = CharNgramLIDTagger
__main__.ContextLIDTagger = ContextLIDTagger


def freeze_production_model(
    source_model: Path = Path("models/lid_context.joblib"),
    target_model: Path = Path("models/lid_final.joblib"),
) -> str:
    """Freeze and serialize the production tagger as lid_final.joblib."""
    target_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, target_model)

    hasher = hashlib.sha256()
    with open(target_model, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    model_hash = hasher.hexdigest()
    print(f"[FREEZE] Serialized and locked production tagger to: {target_model}")
    print(f"[FREEZE] Model SHA-256 Checksum: {model_hash}")
    return model_hash


def run_final_test_evaluation(
    test_path: Path = Path("data/processed/test.conll"),
    train_path: Path = Path("data/processed/train.conll"),
    ambiguous_path: Path = Path("data/ambiguous_words.txt"),
    models_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("results/figures"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the single final evaluation on the test partition.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify test set checksum before evaluation
    test_hasher = hashlib.sha256()
    line_count = 0
    with open(test_path, "rb") as f:
        for line in f:
            test_hasher.update(line)
            line_count += 1
    test_hash = test_hasher.hexdigest()
    expected_hash = "a3ce094d776066cafebc9cda2d00a42521b7b838d66af075296af2d3843d8923"
    assert test_hash == expected_hash, f"Integrity check failed: {test_hash} != {expected_hash}"
    print(f"[VERIFIED] test.conll checksum confirmed ({test_hash[:16]}..., {line_count:,} lines).")

    # 2. Freeze production tagger
    model_sha256 = freeze_production_model()

    # 3. Load ambiguous word list
    with open(ambiguous_path, "r", encoding="utf-8") as f:
        ambiguous_words = set(line.strip().lower() for line in f if line.strip())
    print(f"Loaded {len(ambiguous_words)} frozen ambiguous words.")

    # 4. Load models
    print("Loading tagger models...")
    m0_lookup: WordLookupTagger = joblib.load(models_dir / "baseline_lookup.joblib")
    m1_char: CharNgramLIDTagger = joblib.load(models_dir / "lid_lr.joblib")
    m3_final: ContextLIDTagger = joblib.load(models_dir / "lid_final.joblib")

    # 5. Read test sentences
    print("Reading test sentences...")
    test_sents = read_conll(test_path)
    total_sents = len(test_sents)

    flat_tokens: List[str] = []
    flat_gold: List[str] = []
    for tokens, labels in test_sents:
        flat_tokens.extend(tokens)
        flat_gold.extend(labels)

    total_tokens = len(flat_tokens)
    print(f"Test Partition Size: {total_sents:,} sentences, {total_tokens:,} tokens.")

    # Training vocabulary for OOV calculation
    train_vocab = getattr(m3_final, "train_vocab_", set())
    if not train_vocab:
        train_sents = read_conll(train_path)
        train_vocab = set(t.lower() for s, _ in train_sents for t in s)

    oov_flags = np.array([t.lower() not in train_vocab for t in flat_tokens])
    amb_flags = np.array([t.lower() in ambiguous_words for t in flat_tokens])
    y_true = np.array(flat_gold)

    print(f"Test OOV Tokens: {np.sum(oov_flags):,} ({np.mean(oov_flags)*100:.2f}%)")
    print(f"Test Ambiguous Tokens: {np.sum(amb_flags):,} ({np.mean(amb_flags)*100:.2f}%)")

    # 6. Evaluate all models
    models_to_eval = [
        ("M0_Lookup", "M0: Word Lookup Baseline", m0_lookup),
        ("M1_Char_Only", "M1: Char N-Gram LogReg (No Context)", m1_char),
        ("M3_Final_Context", "M3: Final Context Tagger (Production)", m3_final),
    ]

    table_rows = []
    full_report: Dict[str, Any] = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_file": str(test_path),
            "test_sha256": test_hash,
            "total_sentences": total_sents,
            "total_tokens": total_tokens,
            "oov_tokens": int(np.sum(oov_flags)),
            "oov_rate": round(float(np.mean(oov_flags)), 4),
            "ambiguous_tokens": int(np.sum(amb_flags)),
            "frozen_model_sha256": model_sha256,
        },
        "models": {},
    }

    labels_order = ["HI", "EN"]
    if "OTHER" in set(flat_gold):
        labels_order.append("OTHER")

    predictions_dict = {}

    for mod_id, mod_name, mod_obj in models_to_eval:
        print(f"\nEvaluating {mod_name} on Test Set...")
        t0 = time.time()
        if mod_id == "M0_Lookup":
            preds = [mod_obj.predict_token(t) for t in flat_tokens]
        elif mod_id == "M1_Char_Only":
            preds = mod_obj.predict_sentence(flat_tokens)
        else:
            preds = []
            for s_tokens, _ in test_sents:
                preds.extend(mod_obj.predict_sentence(s_tokens))
        elapsed = time.time() - t0

        y_pred = np.array(preds)
        predictions_dict[mod_id] = y_pred

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        hi_f1 = float(f1_score(y_true, y_pred, labels=["HI"], average="micro", zero_division=0))
        en_f1 = float(f1_score(y_true, y_pred, labels=["EN"], average="micro", zero_division=0))

        # In-vocab vs OOV accuracy
        invoc_acc = float(accuracy_score(y_true[~oov_flags], y_pred[~oov_flags]))
        oov_acc = float(accuracy_score(y_true[oov_flags], y_pred[oov_flags]))

        # Ambiguous words accuracy
        amb_acc = float(accuracy_score(y_true[amb_flags], y_pred[amb_flags]))
        amb_f1 = float(f1_score(y_true[amb_flags], y_pred[amb_flags], average="macro", zero_division=0))

        total_errs = int(np.sum(y_true != y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)

        table_rows.append({
            "model_id": mod_id,
            "model_name": mod_name,
            "test_accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "hi_f1": round(hi_f1, 4),
            "en_f1": round(en_f1, 4),
            "in_vocab_accuracy": round(invoc_acc, 4),
            "oov_accuracy": round(oov_acc, 4),
            "ambiguous_accuracy": round(amb_acc, 4),
            "ambiguous_macro_f1": round(amb_f1, 4),
            "total_errors": total_errs,
            "inference_seconds": round(elapsed, 2),
        })

        full_report["models"][mod_id] = {
            "name": mod_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "hi_f1": round(hi_f1, 4),
            "en_f1": round(en_f1, 4),
            "in_vocab_accuracy": round(invoc_acc, 4),
            "oov_accuracy": round(oov_acc, 4),
            "ambiguous_accuracy": round(amb_acc, 4),
            "ambiguous_macro_f1": round(amb_f1, 4),
            "total_errors": total_errs,
            "confusion_matrix": cm.tolist(),
            "labels": labels_order,
        }

    df_results = pd.DataFrame(table_rows)

    # Save deliverables
    csv_path = results_dir / "table_4_test_final.csv"
    json_path = results_dir / "table_4_test_final.json"
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[SAVED] Final Test Table saved to: {csv_path}")
    print(f"[SAVED] Final Test JSON saved to: {json_path}")

    # Plot final confusion matrix and comparisons
    plot_test_confusion_matrices(full_report, figures_dir / "test_confusion_matrix.png")
    plot_test_model_comparison(df_results, figures_dir / "test_model_comparison.png")

    return df_results, full_report


def plot_test_confusion_matrices(report: Dict[str, Any], out_path: Path):
    """Plot 3-panel normalized confusion matrix comparing M0, M1, and M3 on Test Set."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    model_keys = ["M0_Lookup", "M1_Char_Only", "M3_Final_Context"]
    titles = [
        "M0: Word Lookup Baseline",
        "M1: Char N-Gram (No Context)",
        "M3: Final Context Tagger (Ours)",
    ]

    for ax, mk, title in zip(axes, model_keys, titles):
        cm = np.array(report["models"][mk]["confusion_matrix"])
        labels = report["models"][mk]["labels"]
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
            annot_kws={"size": 11, "fontweight": "bold"},
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=10, fontweight="bold")

    plt.suptitle("Official Final Test Set Confusion Matrices (135,568 Tokens)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated test confusion matrix: {out_path}")


def plot_test_model_comparison(df: pd.DataFrame, out_path: Path):
    """Plot bar chart comparing Test Set Overall Acc, In-Vocab Acc, OOV Acc, and Ambiguous Acc."""
    metrics = ["test_accuracy", "in_vocab_accuracy", "oov_accuracy", "ambiguous_accuracy"]
    metric_labels = ["Overall Acc", "In-Vocab Acc", "OOV Acc", "Ambiguous Acc (H2)"]

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 5))
    colors = ["#E65100", "#1976D2", "#2E7D32"]

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] * 100 for m in metrics]
        plt.bar(x + (i - 1) * width, values, width, label=row["model_name"], color=colors[i], alpha=0.9)

    plt.xticks(x, metric_labels, fontsize=10, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Official Final Test Benchmark Comparison Across Architectures", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(65, 102)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Generated test comparison chart: {out_path}")


if __name__ == "__main__":
    df, rep = run_final_test_evaluation()
    print("\n--- Final Test Evaluation Table ---")
    print(df.to_string(index=False))
