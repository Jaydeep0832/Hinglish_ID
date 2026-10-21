"""
src/lid_tagger.py
Character N-Gram Language Identification Tagger (M1) for Hinglish LID.

Conforms to HinglishID-Router specification (Module M1.5):
- Subword feature extraction: CountVectorizer(analyzer="char_wb", ngram_range=(1, 4))
- Primary Classifier: MultinomialNB with tunable alpha (0.1, 0.5, 1.0)
- Alternative Linear Model: LogisticRegression(max_iter=1000)
- Handles OOV words via subword morphological char n-gram patterns
- Exports dev metrics and compares against M0 baseline
"""

from collections import Counter
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB

from src.conll_parser import read_conll


class CharNgramLIDTagger:
    """
    Subword Character N-Gram Tagger for word-level Language Identification.
    Uses word-boundary character n-grams to generalize across Romanized spelling
    variants and morphological affixes.
    """

    def __init__(
        self,
        model_type: str = "multinomial_nb",
        ngram_range: Tuple[int, int] = (1, 4),
        alpha: float = 0.1,
        fit_prior: bool = True,
        c_param: float = 1.0,
        min_df: int = 1,
    ):
        """
        Args:
            model_type: 'multinomial_nb' or 'logistic_regression'
            ngram_range: Character n-gram range (default 1 to 4)
            alpha: Laplace smoothing parameter for MultinomialNB (default 0.1)
            fit_prior: Whether to learn class prior probabilities or use uniform (default True)
            c_param: Inverse regularization strength for LogisticRegression (default 1.0)
            min_df: Minimum document frequency for n-gram feature pruning (default 1)
        """
        self.model_type = model_type.lower()
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.c_param = c_param
        self.min_df = min_df

        self.vectorizer = CountVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            lowercase=True,
        )

        if self.model_type == "multinomial_nb":
            self.classifier = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
        elif self.model_type == "logistic_regression":
            self.classifier = LogisticRegression(
                C=self.c_param,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'multinomial_nb' or 'logistic_regression'.")

        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False
        self.train_vocab_: set = set()

    def fit(
        self,
        sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    ) -> "CharNgramLIDTagger":
        """
        Train tagger on parsed CoNLL sentences by aggregating word-label token pairs
        with exact sample-weight equivalence for fast, scalable vectorization.

        Args:
            sentences: Sequence of (tokens, labels) tuples.

        Returns:
            self
        """
        pair_counts: Counter = Counter()
        self.train_vocab_ = set()

        for tokens, labels in sentences:
            for token, label in zip(tokens, labels):
                norm_tok = token.lower()
                pair_counts[(norm_tok, label)] += 1
                self.train_vocab_.add(norm_tok)

        words = [pair[0] for pair in pair_counts.keys()]
        labels = [pair[1] for pair in pair_counts.keys()]
        weights = [pair_counts[pair] for pair in pair_counts.keys()]

        # Fit vectorizer on unique word forms and extract subword n-gram features
        X_train = self.vectorizer.fit_transform(words)

        # Train classifier with exact sample weight frequencies
        self.classifier.fit(X_train, labels, sample_weight=weights)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    def predict_token(self, token: str) -> str:
        """Predict language label for a single token."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = self.vectorizer.transform([token.lower()])
        return str(self.classifier.predict(X)[0])

    def predict_proba_token(self, token: str) -> Dict[str, float]:
        """Get class probability distribution for a single token."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = self.vectorizer.transform([token.lower()])
        probs = self.classifier.predict_proba(X)[0]
        return {str(cls): round(float(p), 4) for cls, p in zip(self.classes_, probs)}

    def predict_sentence(self, tokens: Sequence[str]) -> List[str]:
        """Predict language labels for a sequence of tokens in a sentence."""
        if not tokens:
            return []
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = self.vectorizer.transform([t.lower() for t in tokens])
        return [str(p) for p in self.classifier.predict(X)]

    def predict_proba_sentence(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        """Predict class probabilities for each token in a sentence."""
        if not tokens:
            return []
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = self.vectorizer.transform([t.lower() for t in tokens])
        probs = self.classifier.predict_proba(X)
        return [
            {str(cls): round(float(p), 4) for cls, p in zip(self.classes_, row)}
            for row in probs
        ]

    def evaluate(
        self,
        dev_sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    ) -> Dict[str, Any]:
        """
        Evaluate tagger on validation/dev set sentences.
        Computes overall metrics, per-class metrics, and in-vocab vs OOV error breakdown.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        dev_tokens = [t.lower() for tokens, _ in dev_sentences for t in tokens]
        dev_true = [l for _, labels in dev_sentences for l in labels]

        # Fast prediction using vocabulary cache
        unique_words = list(set(dev_tokens))
        X_unique = self.vectorizer.transform(unique_words)
        pred_unique = self.classifier.predict(X_unique)
        word_to_pred = dict(zip(unique_words, pred_unique))

        dev_pred = [word_to_pred[t] for t in dev_tokens]

        total_tokens = len(dev_tokens)
        oov_flags = [t not in self.train_vocab_ for t in dev_tokens]
        oov_count = sum(oov_flags)
        oov_rate = (oov_count / total_tokens) if total_tokens else 0.0

        # Overall scores
        acc = accuracy_score(dev_true, dev_pred)
        macro_f1 = f1_score(dev_true, dev_pred, average="macro", zero_division=0)
        macro_prec = precision_score(dev_true, dev_pred, average="macro", zero_division=0)
        macro_rec = recall_score(dev_true, dev_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(dev_true, dev_pred, average="weighted", zero_division=0)

        # Per-class scores
        labels_list = sorted(list(self.classes_))
        per_class = {}
        for lab in labels_list:
            prec = precision_score(dev_true, dev_pred, labels=[lab], average="micro", zero_division=0)
            rec = recall_score(dev_true, dev_pred, labels=[lab], average="micro", zero_division=0)
            f1 = f1_score(dev_true, dev_pred, labels=[lab], average="micro", zero_division=0)
            sup = sum(1 for l in dev_true if l == lab)
            per_class[lab] = {
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "support": sup,
            }

        # Error breakdown (In-Vocab vs OOV)
        in_vocab_total = total_tokens - oov_count
        in_vocab_errors = 0
        oov_errors = 0
        in_vocab_error_words: Counter = Counter()
        oov_error_words: Counter = Counter()

        for tok, true_l, pred_l, is_oov_tok in zip(dev_tokens, dev_true, dev_pred, oov_flags):
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

        return {
            "model_type": self.model_type,
            "alpha": self.alpha if self.model_type == "multinomial_nb" else None,
            "c_param": self.c_param if self.model_type == "logistic_regression" else None,
            "vocab_feature_count": len(self.vectorizer.vocabulary_),
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
            "labels": labels_list,
            "confusion_matrix": confusion_matrix(dev_true, dev_pred, labels=labels_list).tolist(),
            "classification_report": classification_report(dev_true, dev_pred, digits=4),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CharNgramLIDTagger":
        """Deserialize model from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
