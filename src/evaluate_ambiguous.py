"""
src/evaluate_ambiguous.py
Module M2.2: Comprehensive Evaluation of Ambiguous Words across Models
(Majority-per-word, M0 Lookup, M1 Char-Only, M3 Context Words).

Strict Protocol:
- Evaluates on dev.conll (test.conll remains strictly FROZEN).
- Uses frozen ambiguous word list from data/ambiguous_words.txt.
- Computes per-word accuracy, sample support flags (n < 5 flagged as 'Low Support'),
  pooled micro-accuracy, and 95% Bootstrap Confidence Intervals (1,000 resamples).
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_tagger import WordLookupTagger
from src.conll_parser import read_conll
from src.lid_context import ContextLIDTagger
from src.lid_tagger import CharNgramLIDTagger

# Register in __main__ to avoid unpickling issues
import __main__
__main__.WordLookupTagger = WordLookupTagger
__main__.CharNgramLIDTagger = CharNgramLIDTagger
__main__.ContextLIDTagger = ContextLIDTagger


def load_ambiguous_word_list(filepath: Path) -> List[str]:
    """Load frozen ambiguous words from text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return sorted(list(set(words)))


def compute_word_majority_dict(train_path: Path) -> Dict[str, str]:
    """Compute empirical majority label (HI or EN) for each word from training corpus."""
    train_sents = read_conll(train_path)
    counts: Dict[str, Dict[str, int]] = {}
    for tokens, labels in train_sents:
        for w, l in zip(tokens, labels):
            wl = w.lower()
            if wl not in counts:
                counts[wl] = {"HI": 0, "EN": 0, "OTHER": 0}
            if l in counts[wl]:
                counts[wl][l] += 1

    majority_dict: Dict[str, str] = {}
    for wl, c in counts.items():
        # Tie-breaker defaults to HI due to corpus distribution
        majority_dict[wl] = "HI" if c["HI"] >= c["EN"] else "EN"
    return majority_dict


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute mean accuracy and bootstrap confidence interval.
    Returns: (mean_acc, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    accs = []
    correct_array = (y_true == y_pred).astype(float)
    for _ in range(n_resamples):
        sample_indices = rng.randint(0, n, size=n)
        accs.append(np.mean(correct_array[sample_indices]))

    lower_p = (1.0 - ci) / 2.0 * 100
    upper_p = (1.0 + ci) / 2.0 * 100
    lower = float(np.percentile(accs, lower_p))
    upper = float(np.percentile(accs, upper_p))
    mean = float(np.mean(correct_array))
    return mean, lower, upper


def evaluate_ambiguous_suite(
    train_path: Path = Path("data/processed/train.conll"),
    dev_path: Path = Path("data/processed/dev.conll"),
    ambiguous_words_path: Path = Path("data/ambiguous_words.txt"),
    models_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("results/figures"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run full Table 3 evaluation across models on dev set ambiguous tokens.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ambiguous_words = load_ambiguous_word_list(ambiguous_words_path)
    ambiguous_set = set(ambiguous_words)
    print(f"Loaded {len(ambiguous_words)} frozen ambiguous words.")

    print("Computing majority-per-word dictionary from train...")
    majority_dict = compute_word_majority_dict(train_path)

    print("Loading serialized models...")
    m0_lookup: WordLookupTagger = joblib.load(models_dir / "baseline_lookup.joblib")
    m1_char: CharNgramLIDTagger = joblib.load(models_dir / "lid_lr.joblib")
    m3_context: ContextLIDTagger = joblib.load(models_dir / "lid_context.joblib")

    print("Reading dev sentences...")
    dev_sents = read_conll(dev_path)

    # Flatten tokens and prepare sentence context
    gold_tokens: List[str] = []
    gold_labels: List[str] = []
    sent_indices: List[int] = []
    token_indices: List[int] = []

    for s_idx, (tokens, labels) in enumerate(dev_sents):
        for t_idx, (tok, lbl) in enumerate(zip(tokens, labels)):
            gold_tokens.append(tok)
            gold_labels.append(lbl)
            sent_indices.append(s_idx)
            token_indices.append(t_idx)

    # Predict with all models
    print("Generating predictions across models on full dev set...")
    # 1. Majority Word baseline
    preds_maj = [majority_dict.get(t.lower(), "HI") for t in gold_tokens]

    # 2. M0 Lookup
    preds_m0 = [m0_lookup.predict_token(t) for t in gold_tokens]

    # 3. M1 Char-Only
    preds_m1 = m1_char.predict_sentence(gold_tokens)

    # 4. M3 Context Words
    print("  Predicting with M3 Context tagger over dev sentences...")
    preds_m3 = []
    for tokens, _ in dev_sents:
        preds_m3.extend(m3_context.predict_sentence(tokens))

    # Filter to ambiguous words
    ambiguous_indices = [
        i for i, tok in enumerate(gold_tokens) if tok.lower() in ambiguous_set
    ]
    print(f"Total dev tokens matching ambiguous list: {len(ambiguous_indices)}")

    amb_tokens = [gold_tokens[i].lower() for i in ambiguous_indices]
    amb_gold = np.array([gold_labels[i] for i in ambiguous_indices])
    amb_p_maj = np.array([preds_maj[i] for i in ambiguous_indices])
    amb_p_m0 = np.array([preds_m0[i] for i in ambiguous_indices])
    amb_p_m1 = np.array([preds_m1[i] for i in ambiguous_indices])
    amb_p_m3 = np.array([preds_m3[i] for i in ambiguous_indices])

    # Per-word breakdown
    rows = []
    for word in ambiguous_words:
        w_indices = [i for i, w in enumerate(amb_tokens) if w == word]
        n_support = len(w_indices)

        if n_support == 0:
            rows.append({
                "word": word,
                "n_support": 0,
                "support_flag": "Zero Dev",
                "hi_count": 0,
                "en_count": 0,
                "majority_label": majority_dict.get(word, "HI"),
                "acc_majority": 0.0,
                "acc_m0_lookup": 0.0,
                "acc_m1_char": 0.0,
                "acc_m3_context": 0.0,
                "gain_context_vs_maj": 0.0,
                "gain_context_vs_m1": 0.0,
            })
            continue

        w_gold = amb_gold[w_indices]
        w_p_maj = amb_p_maj[w_indices]
        w_p_m0 = amb_p_m0[w_indices]
        w_p_m1 = amb_p_m1[w_indices]
        w_p_m3 = amb_p_m3[w_indices]

        hi_cnt = int(np.sum(w_gold == "HI"))
        en_cnt = int(np.sum(w_gold == "EN"))
        flag = "Low Support" if n_support < 5 else "Robust"

        acc_maj = float(np.mean(w_gold == w_p_maj))
        acc_m0 = float(np.mean(w_gold == w_p_m0))
        acc_m1 = float(np.mean(w_gold == w_p_m1))
        acc_m3 = float(np.mean(w_gold == w_p_m3))

        rows.append({
            "word": word,
            "n_support": n_support,
            "support_flag": flag,
            "hi_count": hi_cnt,
            "en_count": en_cnt,
            "majority_label": majority_dict.get(word, "HI"),
            "acc_majority": round(acc_maj, 4),
            "acc_m0_lookup": round(acc_m0, 4),
            "acc_m1_char": round(acc_m1, 4),
            "acc_m3_context": round(acc_m3, 4),
            "gain_context_vs_maj": round(acc_m3 - acc_maj, 4),
            "gain_context_vs_m1": round(acc_m3 - acc_m1, 4),
        })

    df_table3 = pd.DataFrame(rows)
    df_table3.sort_values(by="n_support", ascending=False, inplace=True)

    # Pooled metrics
    mean_maj, lo_maj, hi_maj = bootstrap_ci(amb_gold, amb_p_maj)
    mean_m0, lo_m0, hi_m0 = bootstrap_ci(amb_gold, amb_p_m0)
    mean_m1, lo_m1, hi_m1 = bootstrap_ci(amb_gold, amb_p_m1)
    mean_m3, lo_m3, hi_m3 = bootstrap_ci(amb_gold, amb_p_m3)

    pooled_summary = {
        "total_ambiguous_tokens": len(ambiguous_indices),
        "total_ambiguous_vocabulary": len(ambiguous_words),
        "robust_support_words_count": int(np.sum(df_table3["support_flag"] == "Robust")),
        "low_support_words_count": int(np.sum(df_table3["support_flag"] == "Low Support")),
        "zero_dev_words_count": int(np.sum(df_table3["support_flag"] == "Zero Dev")),
        "pooled_models": {
            "majority_word_baseline": {
                "accuracy": round(mean_maj, 4),
                "ci_95_lower": round(lo_maj, 4),
                "ci_95_upper": round(hi_maj, 4),
            },
            "m0_word_lookup": {
                "accuracy": round(mean_m0, 4),
                "ci_95_lower": round(lo_m0, 4),
                "ci_95_upper": round(hi_m0, 4),
            },
            "m1_char_only": {
                "accuracy": round(mean_m1, 4),
                "ci_95_lower": round(lo_m1, 4),
                "ci_95_upper": round(hi_m1, 4),
            },
            "m3_context_words": {
                "accuracy": round(mean_m3, 4),
                "ci_95_lower": round(lo_m3, 4),
                "ci_95_upper": round(hi_m3, 4),
            },
        },
        "pooled_gains": {
            "m3_vs_majority": round(mean_m3 - mean_maj, 4),
            "m3_vs_m0_lookup": round(mean_m3 - mean_m0, 4),
            "m3_vs_m1_char": round(mean_m3 - mean_m1, 4),
        },
    }

    # Save deliverables
    csv_path = results_dir / "table_3_ambiguous.csv"
    json_path = results_dir / "table_3_ambiguous.json"

    df_table3.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pooled_summary, f, indent=2)

    print(f"Saved Table 3 to: {csv_path}")
    print(f"Saved Pooled Summary to: {json_path}")

    # Generate Publication Plot
    plot_ambiguous_per_word(df_table3, figures_dir / "table_3_ambiguous_per_word.png")

    return df_table3, pooled_summary


def plot_ambiguous_per_word(df: pd.DataFrame, out_path: Path):
    """
    Create a clean, publication-grade horizontal grouped bar chart
    comparing M0 Lookup vs M3 Context Words across frequent ambiguous words.
    """
    # Filter to words with support >= 15 for readable visual comparison
    df_plot = df[df["n_support"] >= 15].copy()
    df_plot.sort_values(by="n_support", ascending=True, inplace=True)

    words = df_plot["word"].tolist()
    labels_with_n = [f"{w} (n={n})" for w, n in zip(words, df_plot["n_support"])]
    acc_m0 = df_plot["acc_m0_lookup"].values * 100
    acc_m1 = df_plot["acc_m1_char"].values * 100
    acc_m3 = df_plot["acc_m3_context"].values * 100

    y = np.arange(len(words))
    height = 0.28

    plt.figure(figsize=(10, max(8, len(words) * 0.42)))
    plt.barh(y - height, acc_m0, height, label="M0: Word Lookup Baseline", color="#E65100", alpha=0.85)
    plt.barh(y, acc_m1, height, label="M1: Char N-Gram (No Context)", color="#1976D2", alpha=0.85)
    plt.barh(y + height, acc_m3, height, label="M3: +Context Words (Ours)", color="#2E7D32", alpha=0.95)

    plt.yticks(y, labels_with_n, fontsize=10, fontweight="medium")
    plt.xlabel("Word Identification Accuracy (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Ambiguous Word (Dev Support Count)", fontsize=11, fontweight="bold")
    plt.title("Table 3: Disambiguation Gain per Ambiguous Word (M0 vs M1 vs M3)", fontsize=12, fontweight="bold", pad=12)
    plt.xlim(0, 105)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated publication figure: {out_path}")


if __name__ == "__main__":
    df, summary = evaluate_ambiguous_suite()
    print("\n--- Table 3 Pooled Summary ---")
    print(json.dumps(summary, indent=2))
