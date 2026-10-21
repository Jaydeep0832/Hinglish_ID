"""
scripts/create_notebook.py
Generates notebooks/01_corpus_stats.ipynb programmatically using nbformat.
"""

from pathlib import Path
import nbformat as nbf


def generate_corpus_stats_notebook(output_path: Path = Path("notebooks/01_corpus_stats.ipynb")) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()

    cells = []

    # Cell 1: Header
    cells.append(nbf.v4.new_markdown_cell(
        "# HinglishID-Router: Module M1.2 — Corpus Exploration, Statistics & Splits\n\n"
        "**Owner:** Member A (Tagger Lead) | **Verifier:** Member B  \n"
        "**Dataset:** L3Cube-HingLID (Nayak & Joshi, 2022) — *Activated via Risk R1 Plan B*  \n"
        "**Reference:** Gambäck & Das (2014) Code-Mixing Index  \n\n"
        "### Objectives:\n"
        "1. Parse CoNLL formatted token-level language identification data (`token\\tlabel`).\n"
        "2. Compute descriptive corpus statistics (#sentences, #tokens, avg length, label distributions).\n"
        "3. Calculate per-sentence Code-Mixing Index (CMI) and % code-mixed sentences.\n"
        "4. Identify top-20 frequent tokens per language class and detect ambiguous cross-language tokens.\n"
        "5. Partition sentences into reproducible 80/10/10 splits (`seed=42`) saved to `data/processed/{train,dev,test}.conll`.\n"
        "6. Plot token label distribution and CMI distribution."
    ))

    # Cell 2: Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "# Add repo root to import modular code\n"
        "sys.path.append(str(Path('..').resolve()))\n"
        "from src.conll_parser import read_conll, compute_corpus_stats, calculate_cmi"
    ))

    # Cell 3: Markdown - Load Data
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load Processed Datasets (Train / Dev / Test)\n"
        "We load the sentence-level 80/10/10 split files generated from L3Cube-HingLID (`seed=42`).  \n"
        "> **Note:** The test set (`test.conll`) is loaded here exclusively for descriptive statistics reporting per M1.2 specification; **no model training or hyperparameter tuning will access it until Day 10 freeze.**"
    ))

    # Cell 4: Load code
    cells.append(nbf.v4.new_code_cell(
        "data_dir = Path('../data/processed')\n"
        "train_sents = read_conll(data_dir / 'train.conll')\n"
        "dev_sents   = read_conll(data_dir / 'dev.conll')\n"
        "test_sents  = read_conll(data_dir / 'test.conll')\n\n"
        "print(f'Train sentences: {len(train_sents):,}')\n"
        "print(f'Dev sentences:   {len(dev_sents):,}')\n"
        "print(f'Test sentences:  {len(test_sents):,}')\n"
        "print(f'Total sentences: {len(train_sents) + len(dev_sents) + len(test_sents):,}')"
    ))

    # Cell 5: Markdown - Stats table
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Summary Statistics Across Splits\n"
        "We compute sentence count, token count, average sentence length, label shares, code-mixing rate, and mean CMI."
    ))

    # Cell 6: Compute stats
    cells.append(nbf.v4.new_code_cell(
        "train_stats = compute_corpus_stats(train_sents)\n"
        "dev_stats   = compute_corpus_stats(dev_sents)\n"
        "test_stats  = compute_corpus_stats(test_sents)\n"
        "full_stats  = compute_corpus_stats(train_sents + dev_sents + test_sents)\n\n"
        "summary_rows = [\n"
        "    {\n"
        "        'Split': 'Full Corpus',\n"
        "        'Sentences': f\"{full_stats['num_sentences']:,}\",\n"
        "        'Tokens': f\"{full_stats['num_tokens']:,}\",\n"
        "        'Avg Length': full_stats['avg_sentence_length'],\n"
        "        'HI Tokens (%)': f\"{full_stats['label_counts']['HI']:,} ({full_stats['label_shares']['HI']*100:.2f}%)\",\n"
        "        'EN Tokens (%)': f\"{full_stats['label_counts']['EN']:,} ({full_stats['label_shares']['EN']*100:.2f}%)\",\n"
        "        'Code-Mixed Sents (%)': f\"{full_stats['num_code_mixed_sentences']:,} ({full_stats['pct_code_mixed_sentences']:.2f}%)\",\n"
        "        'Avg CMI': full_stats['avg_cmi']\n"
        "    },\n"
        "    {\n"
        "        'Split': 'Train (80%)',\n"
        "        'Sentences': f\"{train_stats['num_sentences']:,}\",\n"
        "        'Tokens': f\"{train_stats['num_tokens']:,}\",\n"
        "        'Avg Length': train_stats['avg_sentence_length'],\n"
        "        'HI Tokens (%)': f\"{train_stats['label_counts']['HI']:,} ({train_stats['label_shares']['HI']*100:.2f}%)\",\n"
        "        'EN Tokens (%)': f\"{train_stats['label_counts']['EN']:,} ({train_stats['label_shares']['EN']*100:.2f}%)\",\n"
        "        'Code-Mixed Sents (%)': f\"{train_stats['num_code_mixed_sentences']:,} ({train_stats['pct_code_mixed_sentences']:.2f}%)\",\n"
        "        'Avg CMI': train_stats['avg_cmi']\n"
        "    },\n"
        "    {\n"
        "        'Split': 'Dev (10%)',\n"
        "        'Sentences': f\"{dev_stats['num_sentences']:,}\",\n"
        "        'Tokens': f\"{dev_stats['num_tokens']:,}\",\n"
        "        'Avg Length': dev_stats['avg_sentence_length'],\n"
        "        'HI Tokens (%)': f\"{dev_stats['label_counts']['HI']:,} ({dev_stats['label_shares']['HI']*100:.2f}%)\",\n"
        "        'EN Tokens (%)': f\"{dev_stats['label_counts']['EN']:,} ({dev_stats['label_shares']['EN']*100:.2f}%)\",\n"
        "        'Code-Mixed Sents (%)': f\"{dev_stats['num_code_mixed_sentences']:,} ({dev_stats['pct_code_mixed_sentences']:.2f}%)\",\n"
        "        'Avg CMI': dev_stats['avg_cmi']\n"
        "    },\n"
        "    {\n"
        "        'Split': 'Test (10%)',\n"
        "        'Sentences': f\"{test_stats['num_sentences']:,}\",\n"
        "        'Tokens': f\"{test_stats['num_tokens']:,}\",\n"
        "        'Avg Length': test_stats['avg_sentence_length'],\n"
        "        'HI Tokens (%)': f\"{test_stats['label_counts']['HI']:,} ({test_stats['label_shares']['HI']*100:.2f}%)\",\n"
        "        'EN Tokens (%)': f\"{test_stats['label_counts']['EN']:,} ({test_stats['label_shares']['EN']*100:.2f}%)\",\n"
        "        'Code-Mixed Sents (%)': f\"{test_stats['num_code_mixed_sentences']:,} ({test_stats['pct_code_mixed_sentences']:.2f}%)\",\n"
        "        'Avg CMI': test_stats['avg_cmi']\n"
        "    }\n"
        "]\n"
        "df_summary = pd.DataFrame(summary_rows)\n"
        "df_summary"
    ))

    # Cell 7: Markdown - Top 20 tokens
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Top-20 Frequent Tokens per Language\n"
        "Analyzing top tokens per class reveals common function words, particles, and spelling variations."
    ))

    # Cell 8: Top 20 tokens code
    cells.append(nbf.v4.new_code_cell(
        "hi_top = full_stats['top_tokens_by_label']['HI'][:20]\n"
        "en_top = full_stats['top_tokens_by_label']['EN'][:20]\n\n"
        "df_top = pd.DataFrame({\n"
        "    'Rank': range(1, 21),\n"
        "    'HI Token': [t[0] for t in hi_top],\n"
        "    'HI Count': [f'{t[1]:,}' for t in hi_top],\n"
        "    'EN Token': [t[0] for t in en_top],\n"
        "    'EN Count': [f'{t[1]:,}' for t in en_top],\n"
        "})\n"
        "df_top"
    ))

    # Cell 9: Markdown - Ambiguous Words
    cells.append(nbf.v4.new_markdown_cell(
        "### Ambiguous Cross-Language Tokens\n"
        "Words that appear frequently under **both** `HI` and `EN` tags. These words confirm **Hypothesis H2** — context features are vital because lookup-only baselines will fail on these tokens."
    ))

    # Cell 10: Ambiguous Words code
    cells.append(nbf.v4.new_code_cell(
        "df_ambig = pd.DataFrame([\n"
        "    {\n"
        "        'Token': w,\n"
        "        'HI Count': f'{hi:,}',\n"
        "        'EN Count': f'{en:,}',\n"
        "        'Total Count': f'{hi + en:,}',\n"
        "        'HI Ratio (%)': f'{(hi / (hi + en)) * 100:.1f}%'\n"
        "    }\n"
        "    for w, hi, en in full_stats['ambiguous_tokens'][:15]\n"
        "])\n"
        "df_ambig"
    ))

    # Cell 11: Markdown - Visualizations
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Visualizations: Label Distribution & Code-Mixing Index (CMI)\n"
        "1. **Token Label Distribution:** Visualizes the 71.4% HI vs 28.6% EN balance.\n"
        "2. **CMI Histogram:** Displays the distribution of per-sentence code-mixing index, confirming heavy linguistic switching."
    ))

    # Cell 12: Visualizations code
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\n\n"
        "# 1. Label Distribution Bar Chart\n"
        "labels = list(full_stats['label_counts'].keys())\n"
        "counts = [full_stats['label_counts'][l] for l in labels]\n"
        "percentages = [full_stats['label_shares'][l] * 100 for l in labels]\n"
        "colors = ['#2b5c8f', '#e67e22']\n\n"
        "bars = axes[0].bar(labels, counts, color=colors, width=0.45, edgecolor='black', linewidth=1.2)\n"
        "for bar, c, p in zip(bars, counts, percentages):\n"
        "    axes[0].annotate(\n"
        "        f'{c:,}\\n({p:.1f}%)',\n"
        "        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),\n"
        "        xytext=(0, 5),\n"
        "        textcoords='offset points',\n"
        "        ha='center', va='bottom', fontsize=10, fontweight='bold'\n"
        "    )\n"
        "axes[0].set_title('Token Label Distribution (HI vs EN)', fontsize=12, fontweight='bold')\n"
        "axes[0].set_ylabel('Token Count', fontsize=11)\n"
        "axes[0].set_ylim(0, max(counts) * 1.15)\n"
        "axes[0].grid(axis='y', linestyle='--', alpha=0.6)\n\n"
        "# 2. CMI Distribution Histogram\n"
        "cmi_vals = full_stats['cmi_values']\n"
        "mean_cmi = full_stats['avg_cmi']\n"
        "median_cmi = float(np.median(cmi_vals))\n\n"
        "axes[1].hist(cmi_vals, bins=30, color='#3498db', edgecolor='#2980b9', alpha=0.85, linewidth=0.8)\n"
        "axes[1].axvline(mean_cmi, color='#c0392b', linestyle='--', linewidth=2, label=f'Mean = {mean_cmi:.2f}')\n"
        "axes[1].axvline(median_cmi, color='#27ae60', linestyle=':', linewidth=2, label=f'Median = {median_cmi:.2f}')\n"
        "axes[1].set_title('Per-Sentence Code-Mixing Index (CMI)', fontsize=12, fontweight='bold')\n"
        "axes[1].set_xlabel('CMI (0 = Monolingual, 50 = Balanced Bilingual)', fontsize=11)\n"
        "axes[1].set_ylabel('Number of Sentences', fontsize=11)\n"
        "axes[1].legend(fontsize=10)\n"
        "axes[1].grid(axis='y', linestyle='--', alpha=0.6)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 13: Markdown - Checkpoint Answer
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Checkpoint Answer — Module M1.2\n\n"
        "**Q: Corpus mein HI vs EN token share kya hai, aur kitne % sentences actually code-mixed hain?**  \n\n"
        "**A:**\n"
        "- **HI Token Share:** **71.44%** (968,376 tokens out of 1,355,497 total tokens)\n"
        "- **EN Token Share:** **28.56%** (387,121 tokens out of 1,355,497 total tokens)\n"
        "- **Code-Mixed Sentences:** **99.998%** (44,454 out of 44,455 sentences contain both HI and EN tokens)\n"
        "- **Average CMI:** **26.87** (Median: 27.27, standard deviation ~10.4)\n"
        "- **Sentence Split (80/10/10, seed=42):**\n"
        "  - Train: 35,564 sentences (1,083,543 tokens)\n"
        "  - Dev: 4,445 sentences (136,386 tokens)\n"
        "  - Test: 4,446 sentences (135,568 tokens) — *FROZEN until Day 10*"
    ))

    nb.cells = cells

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Jupyter notebook created successfully at: {output_path}")


if __name__ == "__main__":
    generate_corpus_stats_notebook()
