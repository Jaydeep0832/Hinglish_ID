"""
src/baseline_tagger.py
Word-Lookup Majority-Label Baseline Tagger (M0) for Hinglish LID.

Conforms to HinglishID-Router specification (Module M1.4):
- Memorizes word -> majority label from lowercased training tokens
- Unseen / OOV words fallback to the corpus majority class (HI)
- Evaluates on dev set (accuracy, macro-F1, per-class P/R/F1, OOV rate, error distribution)
- Exports results to results/baseline_scores.csv and saves model to models/baseline_lookup.joblib
"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.conll_parser import read_conll


class WordLookupTagger:
    """
    Baseline word-level Language Identification tagger using dictionary lookup
    with majority-class fallback for Out-Of-Vocabulary (OOV) tokens.
    """

    def __init__(self, default_label: Optional[str] = None):
        """
        Args:
            default_label: Fallback label for unseen OOV tokens. If None,
                           inferred from the training corpus majority class.
        """
        self.lookup_dict: Dict[str, str] = {}
        self.word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.label_prior: Counter = Counter()
        self.default_label: str = default_label or "HI"
        self.vocab_size: int = 0
        self.is_fitted: bool = False

    def fit(
        self,
        sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    ) -> "WordLookupTagger":
        """
        Build word -> majority label dictionary from training sentences.

        Args:
            sentences: Sequence of (tokens, labels) tuples.

        Returns:
            self
        """
        self.word_counts.clear()
        self.label_prior.clear()

        for tokens, labels in sentences:
            for token, label in zip(tokens, labels):
                norm_token = token.lower()
                self.word_counts[norm_token][label] += 1
                self.label_prior[label] += 1

        # Extract majority label per word
        self.lookup_dict = {
            word: counts.most_common(1)[0][0]
            for word, counts in self.word_counts.items()
        }

        # Set default fallback to overall most common label
        if self.label_prior:
            self.default_label = self.label_prior.most_common(1)[0][0]

        self.vocab_size = len(self.lookup_dict)
        self.is_fitted = True
        return self

    def predict_token(self, token: str) -> str:
        """Predict language label for a single token."""
        return self.lookup_dict.get(token.lower(), self.default_label)

    def is_oov(self, token: str) -> bool:
        """Check whether a token is out-of-vocabulary."""
        return token.lower() not in self.lookup_dict

    def predict_sentence(self, tokens: Sequence[str]) -> List[str]:
        """Predict language labels for a sequence of tokens in a sentence."""
        return [self.predict_token(t) for t in tokens]

    def predict(
        self,
        sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    ) -> List[List[str]]:
        """Predict language labels for multiple sentences."""
        return [self.predict_sentence(tokens) for tokens, _ in sentences]

    def evaluate(
        self,
        dev_sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    ) -> Dict[str, Any]:
        """
        Evaluate tagger on validation/dev set.

        Returns comprehensive metrics:
        - total_tokens, seen_tokens, oov_tokens, oov_rate
        - accuracy, macro_f1, weighted_f1
        - per_class: precision, recall, f1, support
        - in_vocab_errors, oov_errors, error_distribution
        - top_confused_tokens (in-vocab and OOV)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        flat_tokens: List[str] = []
        flat_true: List[str] = []
        flat_pred: List[str] = []

        for tokens, labels in dev_sentences:
            preds = self.predict_sentence(tokens)
            flat_tokens.extend([t.lower() for t in tokens])
            flat_true.extend(labels)
            flat_pred.extend(preds)

        total_tokens = len(flat_tokens)
        oov_flags = [t not in self.lookup_dict for t in flat_tokens]
        oov_count = sum(oov_flags)
        oov_rate = (oov_count / total_tokens) if total_tokens else 0.0

        # Overall metrics
        acc = accuracy_score(flat_true, flat_pred)
        macro_f1 = f1_score(flat_true, flat_pred, average="macro", zero_division=0)
        macro_prec = precision_score(flat_true, flat_pred, average="macro", zero_division=0)
        macro_rec = recall_score(flat_true, flat_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(flat_true, flat_pred, average="weighted", zero_division=0)

        # Per-class metrics
        labels_list = sorted(list(set(flat_true) | set(flat_pred)))
        per_class = {}
        for lab in labels_list:
            prec = precision_score(flat_true, flat_pred, labels=[lab], average="micro", zero_division=0)
            rec = recall_score(flat_true, flat_pred, labels=[lab], average="micro", zero_division=0)
            f1 = f1_score(flat_true, flat_pred, labels=[lab], average="micro", zero_division=0)
            sup = sum(1 for l in flat_true if l == lab)
            per_class[lab] = {
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "support": sup,
            }

        # Error breakdown: In-Vocab vs OOV
        in_vocab_errors = 0
        oov_errors = 0
        in_vocab_total = total_tokens - oov_count
        in_vocab_error_words: Counter = Counter()
        oov_error_words: Counter = Counter()

        for tok, true_l, pred_l, is_oov_tok in zip(flat_tokens, flat_true, flat_pred, oov_flags):
            if true_l != pred_l:
                if is_oov_tok:
                    oov_errors += 1
                    oov_error_words[tok] += 1
                else:
                    in_vocab_errors += 1
                    in_vocab_error_words[tok] += 1

        total_errors = in_vocab_errors + oov_errors
        in_vocab_acc = (in_vocab_total - in_vocab_errors) / in_vocab_total if in_vocab_total else 0.0
        oov_acc = (oov_count - oov_errors) / oov_count if oov_count else 0.0

        report_str = classification_report(flat_true, flat_pred, digits=4)

        return {
            "vocab_size": self.vocab_size,
            "default_label": self.default_label,
            "total_tokens": total_tokens,
            "seen_tokens": in_vocab_total,
            "oov_tokens": oov_count,
            "oov_rate": round(oov_rate, 4),
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "total_errors": total_errors,
            "in_vocab_errors": in_vocab_errors,
            "in_vocab_error_share": round(in_vocab_errors / total_errors, 4) if total_errors else 0.0,
            "in_vocab_accuracy": round(in_vocab_acc, 4),
            "oov_errors": oov_errors,
            "oov_error_share": round(oov_errors / total_errors, 4) if total_errors else 0.0,
            "oov_accuracy": round(oov_acc, 4),
            "top_in_vocab_errors": in_vocab_error_words.most_common(15),
            "top_oov_errors": oov_error_words.most_common(15),
            "labels": labels_list,
            "confusion_matrix": confusion_matrix(flat_true, flat_pred, labels=labels_list).tolist(),
            "classification_report": report_str,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model to disk via joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved successfully to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "WordLookupTagger":
        """Deserialize model from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)


def run_baseline_pipeline(
    train_path: Union[str, Path] = "data/processed/train.conll",
    dev_path: Union[str, Path] = "data/processed/dev.conll",
    output_dir: Union[str, Path] = "results",
    model_dir: Union[str, Path] = "models",
) -> Dict[str, Any]:
    """
    Train and evaluate word-lookup baseline tagger, saving artifacts and CSVs.
    """
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HinglishID-Router: Module M1.4 Word-Lookup Baseline (M0)")
    print("=" * 60)

    print(f"Loading training data from {train_path}...")
    train_sents = read_conll(train_path)
    print(f"  Loaded {len(train_sents):,} training sentences.")

    print(f"Loading dev data from {dev_path}...")
    dev_sents = read_conll(dev_path)
    print(f"  Loaded {len(dev_sents):,} dev sentences.")

    print("\nFitting WordLookupTagger...")
    tagger = WordLookupTagger()
    tagger.fit(train_sents)
    print(f"  Vocabulary size: {tagger.vocab_size:,} words")
    print(f"  Corpus default fallback: '{tagger.default_label}'")

    print("\nEvaluating on dev set (dev.conll)...")
    metrics = tagger.evaluate(dev_sents)

    print("-" * 60)
    print(f"Total Dev Tokens:     {metrics['total_tokens']:,}")
    print(f"Seen Tokens:          {metrics['seen_tokens']:,}")
    print(f"OOV Tokens:           {metrics['oov_tokens']:,} (OOV Rate: {metrics['oov_rate']*100:.2f}%)")
    print(f"Token-Level Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro-F1 Score:       {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1 Score:    {metrics['weighted_f1']:.4f}")
    print("-" * 60)

    print("\nPer-Class Breakdown:")
    for lab, stats in metrics["per_class"].items():
        print(f"  [{lab}] P: {stats['precision']:.4f} | R: {stats['recall']:.4f} | F1: {stats['f1']:.4f} | Support: {stats['support']:,}")

    print("\nError Breakdown:")
    print(f"  Total Errors:     {metrics['total_errors']:,}")
    print(f"  In-Vocab Errors:  {metrics['in_vocab_errors']:,} ({metrics['in_vocab_error_share']*100:.1f}% of errors) | Error Rate on In-Vocab: {(1-metrics['in_vocab_accuracy'])*100:.2f}%")
    print(f"  OOV Errors:       {metrics['oov_errors']:,} ({metrics['oov_error_share']*100:.1f}% of errors) | Error Rate on OOV: {(1-metrics['oov_accuracy'])*100:.2f}%")

    print("\nTop 5 In-Vocab Ambiguity Failures:")
    for w, c in metrics["top_in_vocab_errors"][:5]:
        print(f"  '{w}': {c} errors")

    print("\nTop 5 OOV Spelling / Lexical Failures:")
    for w, c in metrics["top_oov_errors"][:5]:
        print(f"  '{w}': {c} errors")

    # 1. Save results to CSV
    csv_path = output_dir / "baseline_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "split", "accuracy", "macro_f1", "weighted_f1", "oov_rate", "hi_f1", "en_f1", "total_errors", "in_vocab_errors", "oov_errors"])
        writer.writerow([
            "M0_word_lookup",
            "dev",
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
            metrics["oov_rate"],
            metrics["per_class"]["HI"]["f1"],
            metrics["per_class"]["EN"]["f1"],
            metrics["total_errors"],
            metrics["in_vocab_errors"],
            metrics["oov_errors"],
        ])
    print(f"\nSaved baseline evaluation CSV to: {csv_path}")

    # 2. Save full JSON summary
    json_path = output_dir / "baseline_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved detailed metrics JSON to: {json_path}")

    # 3. Save model artifact
    model_path = model_dir / "baseline_lookup.joblib"
    tagger.save(model_path)

    return metrics


if __name__ == "__main__":
    run_baseline_pipeline()
