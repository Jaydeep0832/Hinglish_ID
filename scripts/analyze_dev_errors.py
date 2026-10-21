"""
scripts/analyze_dev_errors.py
Collects, categorizes, and profiles all prediction errors made by the context tagger
on dev.conll to support authoring docs/error_analysis.md (Module M2.3 - Day 10).
"""

from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_tagger import WordLookupTagger
from src.conll_parser import read_conll
from src.lid_context import ContextLIDTagger
from src.lid_tagger import CharNgramLIDTagger

# Register classes for unpickling
import __main__
__main__.WordLookupTagger = WordLookupTagger
__main__.CharNgramLIDTagger = CharNgramLIDTagger
__main__.ContextLIDTagger = ContextLIDTagger


def collect_dev_errors(
    model_path: Path = Path("models/lid_context.joblib"),
    dev_path: Path = Path("data/processed/dev.conll"),
    ambiguous_path: Path = Path("data/ambiguous_words.txt"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collect all prediction errors from dev.conll with sentence context."""
    print(f"Loading model: {model_path}")
    tagger: ContextLIDTagger = joblib.load(model_path)

    print(f"Loading ambiguous word list: {ambiguous_path}")
    with open(ambiguous_path, "r", encoding="utf-8") as f:
        ambiguous_words = set(line.strip().lower() for line in f if line.strip())

    print(f"Loading dev sentences: {dev_path}")
    dev_sents = read_conll(dev_path)

    train_vocab = getattr(tagger, "train_vocab_", set())

    errors = []
    total_tokens = 0
    correct_tokens = 0

    for s_idx, (tokens, labels) in enumerate(dev_sents):
        preds = tagger.predict_sentence(tokens)
        probs_list = tagger.predict_proba_sentence(tokens)

        for t_idx, (tok, gold, pred, probs) in enumerate(zip(tokens, labels, preds, probs_list)):
            total_tokens += 1
            if gold == pred:
                correct_tokens += 1
                continue

            tok_lower = tok.lower()
            is_oov = tok_lower not in train_vocab
            is_ambiguous = tok_lower in ambiguous_words

            # Sentence context snippet (+/- 4 words)
            start_ctx = max(0, t_idx - 3)
            end_ctx = min(len(tokens), t_idx + 4)
            ctx_snippet = " ".join(tokens[start_ctx:end_ctx])
            full_sent = " ".join(tokens)

            errors.append({
                "sentence_idx": s_idx,
                "token_idx": t_idx,
                "token": tok,
                "token_lower": tok_lower,
                "gold_label": gold,
                "pred_label": pred,
                "confidence": probs.get(pred, 0.0),
                "gold_prob": probs.get(gold, 0.0),
                "all_probs": probs,
                "is_oov": is_oov,
                "is_ambiguous": is_ambiguous,
                "context_snippet": ctx_snippet,
                "full_sentence": full_sent,
            })

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    summary = {
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "total_errors": len(errors),
        "overall_accuracy": round(accuracy, 4),
        "error_rate": round(len(errors) / total_tokens, 4),
    }

    return errors, summary


def profile_error_categories(errors: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze frequencies and distributions of error types."""
    token_counter = Counter(e["token_lower"] for e in errors)
    transition_counter = Counter(f"{e['gold_label']}->{e['pred_label']}" for e in errors)

    print("\n--- Error Transitions Breakdown ---")
    for trans, count in transition_counter.most_common():
        pct = count / len(errors) * 100
        print(f"  {trans:10s}: {count:5d} ({pct:5.2f}%)")

    print("\n--- Top 30 Most Frequent Error Tokens ---")
    top_err_tokens = token_counter.most_common(30)
    for tok, count in top_err_tokens:
        pct = count / len(errors) * 100
        print(f"  {tok:15s}: {count:4d} ({pct:4.2f}%)")

    oov_count = sum(1 for e in errors if e["is_oov"])
    amb_count = sum(1 for e in errors if e["is_ambiguous"])
    print(f"\nTotal Errors: {len(errors)}")
    print(f"OOV Errors: {oov_count} ({oov_count/len(errors)*100:.2f}%)")
    print(f"Ambiguous Word Errors: {amb_count} ({amb_count/len(errors)*100:.2f}%)")
    print(f"In-Vocab Non-Ambiguous Errors: {len(errors) - oov_count - amb_count} ({(len(errors)-oov_count-amb_count)/len(errors)*100:.2f}%)")

    return pd.DataFrame(errors)


if __name__ == "__main__":
    errs, summ = collect_dev_errors()
    print("\nSummary:", summ)
    df_errs = profile_error_categories(errs)
    df_errs.to_csv("results/dev_errors_raw.csv", index=False)
    print("\nSaved all dev errors to results/dev_errors_raw.csv")
