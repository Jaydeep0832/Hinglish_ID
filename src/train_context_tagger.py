"""
src/train_context_tagger.py
Module M2.1: Training, Tuning & Evaluation of Context Features for Hinglish LID.

Evaluates 3 variants on dev.conll (test.conll strictly FROZEN):
1. M1 Char-Only (CountVectorizer char_wb 1-4, LogisticRegression C=1.0)
2. M3 +Context Words (char_wb 1-4 + w_{i-1} + w_{i+1} + word bigrams)
3. M4 +Neighbour Labels (2-pass model conditioning on predicted neighbour tags)

Generates:
- results/table_2_context.csv & results/table_2_context.json
- results/figures/context_vs_char_ambiguous.png
- results/figures/per_word_ambiguity_gain.png
- Serialized model saved to models/lid_context.joblib
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

from src.baseline_tagger import WordLookupTagger
from src.conll_parser import read_conll
from src.lid_context import ContextLIDTagger, DEFAULT_AMBIGUOUS_WORDS


def plot_ambiguous_accuracy_comparison(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot comparative Overall vs Ambiguous Words Accuracy."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    names = [r["model_name"] for r in rows]
    overall_accs = [r["accuracy"] * 100 for r in rows]
    ambig_accs = [r["ambiguous_accuracy"] * 100 for r in rows]

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width/2, overall_accs, width, label="Overall Dev Accuracy", color="#2b5c8f", edgecolor="black")
    rects2 = ax.bar(x + width/2, ambig_accs, width, label="Ambiguous Words Accuracy (Hypothesis H2)", color="#e27c38", edgecolor="black")

    ax.set_ylabel("Accuracy (%)", fontsize=11, weight="bold")
    ax.set_title("Module M2.1: Context Features Impact — Overall vs. Ambiguous Words", fontsize=12, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, weight="bold")
    ax.set_ylim(70, 102)
    ax.legend(frameon=True, facecolor="white", edgecolor="gray", loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, weight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved ambiguous comparison plot: {output_path}")


def plot_per_word_ambiguity_gain(
    target_words: List[str],
    m1_stats: Dict[str, Dict[str, Any]],
    m3_stats: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot per-word accuracy on top ambiguous overlap words (e.g. to, is, the, do, me)."""
    words = [w for w in target_words if w in m1_stats and w in m3_stats]
    m1_acc = [m1_stats[w]["accuracy"] * 100 for w in words]
    m3_acc = [m3_stats[w]["accuracy"] * 100 for w in words]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(words))
    width = 0.35

    rects1 = ax.bar(x - width/2, m1_acc, width, label="M1: Char-Only (No Context)", color="#6c757d", edgecolor="black")
    rects2 = ax.bar(x + width/2, m3_acc, width, label="M3: +Context Words (w_{i-1}, w_{i+1})", color="#28a745", edgecolor="black")

    ax.set_ylabel("Per-Word Accuracy (%)", fontsize=11, weight="bold")
    ax.set_title("Disambiguation Gain on Cross-Lingual Overlap Words (Dev Set)", fontsize=12, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"'{w}'\n(n={m3_stats[w]['total']:,})" for w in words], fontsize=10, weight="bold")
    ax.set_ylim(40, 105)
    ax.legend(frameon=True, facecolor="white", edgecolor="gray", loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, weight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved per-word ambiguity gain plot: {output_path}")


def run_context_experiments(
    train_path: str = "data/processed/train.conll",
    dev_path: str = "data/processed/dev.conll",
    results_dir: str = "results",
    figures_dir: str = "results/figures",
    models_dir: str = "models",
) -> List[Dict[str, Any]]:
    """Run full Module M2.1 context evaluation and generate Table 2."""
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    models_path = Path(models_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HinglishID-Router: Module M2.1 Context Features Evaluation (Table 2)")
    print("=" * 80)

    # Load splits (test.conll strictly FROZEN)
    print(f"Loading train split from {train_path}...")
    t0 = time.time()
    train_sents = read_conll(train_path)
    print(f"  Loaded {len(train_sents):,} sentences in {time.time()-t0:.2f}s.")

    print(f"Loading dev split from {dev_path}...")
    t1 = time.time()
    dev_sents = read_conll(dev_path)
    print(f"  Loaded {len(dev_sents):,} sentences in {time.time()-t1:.2f}s.")
    print("  [VERIFICATION] test.conll remains strictly untouched and FROZEN.")

    table_rows: List[Dict[str, Any]] = []

    # 1. Baseline: M0 Word Lookup
    print("\n[1/4] Evaluating M0 Word-Lookup Baseline...")
    m0 = WordLookupTagger()
    m0.fit(train_sents)
    m0_eval = m0.evaluate(dev_sents)

    # Evaluate M0 on ambiguous subset
    flat_tokens = [t.lower() for toks, _ in dev_sents for t in toks]
    flat_true = [l for _, labs in dev_sents for l in labs]
    m0_preds = [m0.predict_token(t) for t in flat_tokens]

    ambig_true = [true_l for t, true_l in zip(flat_tokens, flat_true) if t in DEFAULT_AMBIGUOUS_WORDS]
    ambig_pred = [pred_l for t, pred_l in zip(flat_tokens, m0_preds) if t in DEFAULT_AMBIGUOUS_WORDS]
    m0_ambig_acc = accuracy_score(ambig_true, ambig_pred)
    m0_ambig_f1 = f1_score(ambig_true, ambig_pred, average="macro", zero_division=0)

    m0_row = {
        "model_id": "M0_Lookup",
        "model_name": "M0: Word Lookup (Baseline)",
        "feature_mode": "lookup",
        "accuracy": m0_eval["accuracy"],
        "macro_f1": m0_eval["macro_f1"],
        "hi_f1": m0_eval["per_class"]["HI"]["f1"],
        "en_f1": m0_eval["per_class"]["EN"]["f1"],
        "in_vocab_accuracy": m0_eval["in_vocab_accuracy"],
        "oov_accuracy": m0_eval["oov_accuracy"],
        "total_errors": m0_eval["total_errors"],
        "ambiguous_tokens": len(ambig_true),
        "ambiguous_accuracy": round(float(m0_ambig_acc), 4),
        "ambiguous_macro_f1": round(float(m0_ambig_f1), 4),
    }
    table_rows.append(m0_row)
    print(f"  M0 Lookup -> Acc: {m0_row['accuracy']*100:.2f}%, Macro-F1: {m0_row['macro_f1']:.4f}, Ambig Acc: {m0_row['ambiguous_accuracy']*100:.2f}%")

    # 2. M1: Char-Only Model (Logistic Regression C=1.0)
    print("\n[2/4] Training M1: Char-Only Model (char_wb 1-4)...")
    m1_tagger = ContextLIDTagger(
        feature_mode="char_only",
        ngram_range=(1, 4),
        c_param=1.0,
        min_df_char=2,
    )
    t_m1 = time.time()
    m1_tagger.fit(train_sents)
    print(f"  M1 trained in {time.time()-t_m1:.2f}s.")
    m1_eval = m1_tagger.evaluate(dev_sents)

    m1_row = {
        "model_id": "M1_Char_Only",
        "model_name": "M1: Char-Only (char_wb 1-4)",
        "feature_mode": "char_only",
        "accuracy": m1_eval["accuracy"],
        "macro_f1": m1_eval["macro_f1"],
        "hi_f1": m1_eval["per_class"]["HI"]["f1"],
        "en_f1": m1_eval["per_class"]["EN"]["f1"],
        "in_vocab_accuracy": m1_eval["in_vocab_accuracy"],
        "oov_accuracy": m1_eval["oov_accuracy"],
        "total_errors": m1_eval["total_errors"],
        "ambiguous_tokens": m1_eval["ambiguous_total"],
        "ambiguous_accuracy": m1_eval["ambiguous_accuracy"],
        "ambiguous_macro_f1": m1_eval["ambiguous_macro_f1"],
    }
    table_rows.append(m1_row)
    print(f"  M1 Char-Only -> Acc: {m1_row['accuracy']*100:.2f}%, Macro-F1: {m1_row['macro_f1']:.4f}, Ambig Acc: {m1_row['ambiguous_accuracy']*100:.2f}%")

    # 3. M3: +Context Words (char_wb 1-4 + w_{i-1} + w_{i+1} + collocations)
    print("\n[3/4] Training M3: +Context Words (w_{i-1}, w_{i+1}, bigrams)...")
    m3_tagger = ContextLIDTagger(
        feature_mode="context_words",
        ngram_range=(1, 4),
        c_param=1.0,
        min_df_char=2,
        min_df_ctx=2,
    )
    t_m3 = time.time()
    m3_tagger.fit(train_sents)
    print(f"  M3 trained in {time.time()-t_m3:.2f}s.")
    m3_eval = m3_tagger.evaluate(dev_sents)

    m3_row = {
        "model_id": "M3_Context_Words",
        "model_name": "M3: +Context Words (w_{i-1}, w_{i+1})",
        "feature_mode": "context_words",
        "accuracy": m3_eval["accuracy"],
        "macro_f1": m3_eval["macro_f1"],
        "hi_f1": m3_eval["per_class"]["HI"]["f1"],
        "en_f1": m3_eval["per_class"]["EN"]["f1"],
        "in_vocab_accuracy": m3_eval["in_vocab_accuracy"],
        "oov_accuracy": m3_eval["oov_accuracy"],
        "total_errors": m3_eval["total_errors"],
        "ambiguous_tokens": m3_eval["ambiguous_total"],
        "ambiguous_accuracy": m3_eval["ambiguous_accuracy"],
        "ambiguous_macro_f1": m3_eval["ambiguous_macro_f1"],
    }
    table_rows.append(m3_row)
    print(f"  M3 +Context Words -> Acc: {m3_row['accuracy']*100:.2f}%, Macro-F1: {m3_row['macro_f1']:.4f}, Ambig Acc: {m3_row['ambiguous_accuracy']*100:.2f}%")

    # Save primary context model to disk
    m3_tagger.save(models_path / "lid_context.joblib")

    # 4. M4: +Neighbour Labels (2-pass model)
    print("\n[4/4] Training M4: 2-Pass (+Neighbour Predicted Labels)...")
    m4_tagger = ContextLIDTagger(
        feature_mode="context_plus_labels",
        ngram_range=(1, 4),
        c_param=1.0,
        min_df_char=2,
        min_df_ctx=2,
    )
    t_m4 = time.time()
    m4_tagger.fit(train_sents)
    print(f"  M4 trained in {time.time()-t_m4:.2f}s.")
    m4_eval = m4_tagger.evaluate(dev_sents)

    m4_row = {
        "model_id": "M4_Context_Labels",
        "model_name": "M4: 2-Pass (+Neighbour Labels)",
        "feature_mode": "context_plus_labels",
        "accuracy": m4_eval["accuracy"],
        "macro_f1": m4_eval["macro_f1"],
        "hi_f1": m4_eval["per_class"]["HI"]["f1"],
        "en_f1": m4_eval["per_class"]["EN"]["f1"],
        "in_vocab_accuracy": m4_eval["in_vocab_accuracy"],
        "oov_accuracy": m4_eval["oov_accuracy"],
        "total_errors": m4_eval["total_errors"],
        "ambiguous_tokens": m4_eval["ambiguous_total"],
        "ambiguous_accuracy": m4_eval["ambiguous_accuracy"],
        "ambiguous_macro_f1": m4_eval["ambiguous_macro_f1"],
    }
    table_rows.append(m4_row)
    print(f"  M4 +Neighbour Labels -> Acc: {m4_row['accuracy']*100:.2f}%, Macro-F1: {m4_row['macro_f1']:.4f}, Ambig Acc: {m4_row['ambiguous_accuracy']*100:.2f}%")

    # 5. Export Table 2 to CSV and JSON
    table2_csv = results_path / "table_2_context.csv"
    csv_fields = [
        "model_id", "model_name", "feature_mode", "accuracy", "macro_f1",
        "hi_f1", "en_f1", "in_vocab_accuracy", "oov_accuracy", "total_errors",
        "ambiguous_tokens", "ambiguous_accuracy", "ambiguous_macro_f1"
    ]
    with open(table2_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in table_rows:
            writer.writerow(r)
    print(f"\n[Artifact] Saved Table 2 to: {table2_csv}")

    table2_json = results_path / "table_2_context.json"
    with open(table2_json, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)
    print(f"[Artifact] Saved Table 2 JSON to: {table2_json}")

    # 6. Generate Figures
    plot_ambiguous_accuracy_comparison(
        table_rows,
        figures_path / "context_vs_char_ambiguous.png",
    )

    key_ambig_words = ["to", "is", "the", "do", "me", "hi", "sir", "us"]
    plot_per_word_ambiguity_gain(
        key_ambig_words,
        m1_eval["ambiguous_word_stats"],
        m3_eval["ambiguous_word_stats"],
        figures_path / "per_word_ambiguity_gain.png",
    )

    # 7. Print Executive Summary Table
    print("\n" + "=" * 100)
    print("TABLE 2: CONTEXT FEATURES IMPACT & AMBIGUOUS WORD DISAMBIGUATION (Dev Set)")
    print("=" * 100)
    print(f"{'Model Architecture':<32} | {'Overall Acc':<11} | {'Macro-F1':<9} | {'Ambig Acc (H2)':<16} | {'Ambig F1':<9} | {'Errors':<7}")
    print("-" * 100)
    for r in table_rows:
        print(f"{r['model_name']:<32} | {r['accuracy']*100:>9.2f}% | {r['macro_f1']:>9.4f} | {r['ambiguous_accuracy']*100:>14.2f}% | {r['ambiguous_macro_f1']:>9.4f} | {r['total_errors']:>7,}")
    print("=" * 100)

    return table_rows


if __name__ == "__main__":
    run_context_experiments()
