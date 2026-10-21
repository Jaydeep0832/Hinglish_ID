"""
src/analyze_corpus.py
Execution script for Module M1.2:
- Parses raw L3Cube-HingLID data
- Computes comprehensive corpus statistics (tokens, sentences, CMI, ambiguous words)
- Performs sentence-level 80/10/10 split with seed=42
- Saves processed splits to data/processed/{train,dev,test}.conll
- Generates publication-ready figures in results/figures/
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.conll_parser import (
    compute_corpus_stats,
    read_conll,
    split_sentences,
    write_conll,
)


def run_corpus_analysis(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    results_dir: Path = Path("results"),
    seed: int = 42,
) -> Dict[str, dict]:
    """
    Execute full Day 2 corpus exploration and data processing pipeline.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HinglishID-Router: Module M1.2 Corpus Exploration & Splitting")
    print("=" * 60)

    # 1. Load raw files
    raw_files = [
        ("train_raw", raw_dir / "hinglid_train.txt"),
        ("valid_raw", raw_dir / "hinglid_valid.txt"),
        ("test_raw", raw_dir / "hinglid_test.txt"),
    ]

    all_sentences: List[Tuple[List[str], List[str]]] = []
    split_stats = {}

    for name, filepath in raw_files:
        if not filepath.exists():
            raise FileNotFoundError(f"Missing expected raw file: {filepath}")
        print(f"Loading {name} from {filepath}...")
        sents = read_conll(filepath)
        stats = compute_corpus_stats(sents)
        split_stats[name] = stats
        all_sentences.extend(sents)
        print(
            f"  -> {len(sents):,} sentences, {stats['num_tokens']:,} tokens, "
            f"avg len: {stats['avg_sentence_length']}, % CM: {stats['pct_code_mixed_sentences']}%, "
            f"avg CMI: {stats['avg_cmi']}"
        )

    # 2. Combined Corpus Statistics
    print("\nComputing full corpus statistics...")
    full_stats = compute_corpus_stats(all_sentences)
    split_stats["full_corpus"] = full_stats

    print("-" * 60)
    print(f"Total Sentences: {full_stats['num_sentences']:,}")
    print(f"Total Tokens:    {full_stats['num_tokens']:,}")
    print(f"Avg Sent Length: {full_stats['avg_sentence_length']:.2f} tokens")
    print(f"Label Counts:    {full_stats['label_counts']}")
    print(f"Label Shares:    {full_stats['label_shares']}")
    print(
        f"Code-Mixed Sents:{full_stats['num_code_mixed_sentences']:,} ({full_stats['pct_code_mixed_sentences']}%)"
    )
    print(f"Average CMI:     {full_stats['avg_cmi']:.2f}")
    print("-" * 60)

    # Top Ambiguous Words
    print("\nTop 15 Ambiguous Words (High frequency in both HI and EN):")
    for word, hi_c, en_c in full_stats["ambiguous_tokens"][:15]:
        ratio = hi_c / (hi_c + en_c)
        print(f"  '{word}': HI={hi_c:,}, EN={en_c:,} (HI ratio: {ratio:.1%})")

    # 3. Perform 80/10/10 sentence-level split
    print(f"\nPerforming sentence-level 80/10/10 split with seed={seed}...")
    train_sents, dev_sents, test_sents = split_sentences(
        all_sentences, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=seed
    )

    print(f"  Train sentences: {len(train_sents):,} (80%)")
    print(f"  Dev sentences:   {len(dev_sents):,} (10%)")
    print(f"  Test sentences:  {len(test_sents):,} (10%) [FROZEN until Day 10]")

    # 4. Save to data/processed/{train,dev,test}.conll
    print("\nSaving processed splits...")
    write_conll(train_sents, processed_dir / "train.conll")
    write_conll(dev_sents, processed_dir / "dev.conll")
    write_conll(test_sents, processed_dir / "test.conll")
    print("  Saved data/processed/train.conll")
    print("  Saved data/processed/dev.conll")
    print("  Saved data/processed/test.conll")

    # Compute stats on new splits
    train_stats = compute_corpus_stats(train_sents)
    dev_stats = compute_corpus_stats(dev_sents)
    test_stats = compute_corpus_stats(test_sents)

    split_stats["train_processed"] = train_stats
    split_stats["dev_processed"] = dev_stats
    split_stats["test_processed"] = test_stats

    # 5. Generate Visualizations
    print("\nGenerating visualizations...")
    _plot_label_distribution(full_stats, figures_dir / "label_distribution.png")
    _plot_cmi_histogram(full_stats["cmi_values"], figures_dir / "cmi_distribution.png")

    # 6. Save JSON summary (omit full cmi_values array for brevity)
    summary_to_save = {}
    for key, val in split_stats.items():
        summary_to_save[key] = {k: v for k, v in val.items() if k != "cmi_values"}

    json_path = results_dir / "corpus_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_to_save, f, indent=2)
    print(f"Corpus statistics summary saved to: {json_path}")

    return split_stats


def _plot_label_distribution(stats: dict, output_path: Path) -> None:
    """Generate bar chart for label distribution."""
    labels = list(stats["label_counts"].keys())
    counts = [stats["label_counts"][l] for l in labels]
    total = sum(counts)
    percentages = [(c / total) * 100 for c in counts]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    colors = ["#2b5c8f", "#e67e22", "#7f8c8d"][: len(labels)]
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)

    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("L3Cube-HingLID Label Distribution (Token Level)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Language Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Token Count", fontsize=12, labelpad=10)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def _plot_cmi_histogram(cmi_values: List[float], output_path: Path) -> None:
    """Generate histogram of per-sentence Code-Mixing Index (CMI)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    mean_cmi = np.mean(cmi_values)
    median_cmi = np.median(cmi_values)

    counts, bins, patches = ax.hist(
        cmi_values,
        bins=30,
        color="#3498db",
        edgecolor="#2980b9",
        linewidth=1.0,
        alpha=0.85,
    )

    ax.axvline(
        mean_cmi,
        color="#c0392b",
        linestyle="--",
        linewidth=2,
        label=f"Mean CMI = {mean_cmi:.2f}",
    )
    ax.axvline(
        median_cmi,
        color="#27ae60",
        linestyle=":",
        linewidth=2,
        label=f"Median CMI = {median_cmi:.2f}",
    )

    ax.set_title(
        "Per-Sentence Code-Mixing Index (CMI) Distribution\n(Das & Gambäck Formula)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Code-Mixing Index (CMI: 0 = Monolingual, 50 = Balanced)", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Sentences", fontsize=12, labelpad=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


if __name__ == "__main__":
    run_corpus_analysis()
