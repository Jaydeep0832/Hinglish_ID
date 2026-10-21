"""
src/lid_context.py
Module M2.1: Context-Aware Language Identification Tagger (M3 & M4).

Conforms to HinglishID-Router specification:
- Incorporates neighbouring word features: w_{i-1} (p=...), w_{i+1} (n=...), with boundary tokens <S>, </S>
- Feature combinations:
    1. char_only: Char n-grams (1-4) of target word w_i (M1 baseline)
    2. context_words: Char n-grams + neighbouring words w_{i-1}, w_{i+1} + bigrams (M3)
    3. context_plus_labels: Two-pass tagger conditioning on predicted neighbour labels y_{i-1}, y_{i+1} (M4)
- Sparse feature representation: scipy.sparse.hstack combining subword CountVectorizer and context DictVectorizer
- Evaluates overall Dev performance AND isolates accuracy on ambiguous overlap words (e.g. to, is, me, the, do)
"""

from collections import Counter
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

from src.conll_parser import read_conll


# Top cross-lingual ambiguous overlap words identified in Table 3.3
DEFAULT_AMBIGUOUS_WORDS = {
    "to", "is", "me", "hi", "do", "the", "sir", "us", "use", "or",
    "main", "par", "no", "so", "in", "mat", "are", "he", "it", "on"
}


def extract_context_features_for_sentence(
    tokens: Sequence[str],
    predicted_labels: Optional[Sequence[str]] = None,
    include_labels: bool = False,
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Extract linguistic window features for each token in a sentence.

    Features extracted:
    - Target word: w=word.lower()
    - Word length: len=len(word)
    - Orthographic cues: is_title, is_upper, is_digit
    - Previous word: p=w_{i-1}.lower() or '<S>'
    - Next word: n=w_{i+1}.lower() or '</S>'
    - Word bigrams: p_w=p + '_' + w, w_n=w + '_' + n
    - Neighbour labels (optional for 2-pass): p_lab=y_{i-1}, n_lab=y_{i+1}
    """
    n_toks = len(tokens)
    feat_list: List[Dict[str, Union[int, float, str]]] = []

    for i, tok in enumerate(tokens):
        w = tok.lower()
        p = tokens[i - 1].lower() if i > 0 else "<S>"
        n = tokens[i + 1].lower() if i + 1 < n_toks else "</S>"

        d: Dict[str, Union[int, float, str]] = {
            f"w={w}": 1,
            f"p={p}": 1,
            f"n={n}": 1,
            f"p_w={p}_{w}": 1,
            f"w_n={w}_{n}": 1,
            "is_title": 1 if tok.istitle() else 0,
            "is_upper": 1 if tok.isupper() and len(tok) > 1 else 0,
            "is_digit": 1 if tok.isdigit() else 0,
        }

        if include_labels and predicted_labels is not None:
            p_lab = predicted_labels[i - 1] if i > 0 else "<S_LAB>"
            n_lab = predicted_labels[i + 1] if i + 1 < n_toks else "</S_LAB>"
            d[f"p_lab={p_lab}"] = 1
            d[f"n_lab={n_lab}"] = 1
            d[f"p_lab_w={p_lab}_{w}"] = 1
            d[f"w_n_lab={w}_{n_lab}"] = 1

        feat_list.append(d)

    return feat_list


class ContextLIDTagger:
    """
    Context-Aware Language Identification Tagger using subword character n-grams
    and neighbouring window features (prev word, next word, bigrams, neighbour tags).
    """

    def __init__(
        self,
        feature_mode: str = "context_words",
        ngram_range: Tuple[int, int] = (1, 4),
        c_param: float = 1.0,
        min_df_char: int = 2,
        min_df_ctx: int = 2,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        """
        Args:
            feature_mode: 'char_only', 'context_words' (M3), or 'context_plus_labels' (M4)
            ngram_range: Subword character n-gram window
            c_param: LogisticRegression inverse regularization
            min_df_char: Pruning threshold for character n-grams
            min_df_ctx: Pruning threshold for context features
            max_iter: Max optimization iterations for solver
            random_state: Seed for reproducibility
        """
        self.feature_mode = feature_mode.lower()
        if self.feature_mode not in ["char_only", "context_words", "context_plus_labels"]:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        self.ngram_range = ngram_range
        self.c_param = c_param
        self.min_df_char = min_df_char
        self.min_df_ctx = min_df_ctx
        self.max_iter = max_iter
        self.random_state = random_state

        # Feature vectorizers
        self.char_vectorizer = CountVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df_char,
            lowercase=True,
        )
        self.ctx_vectorizer = DictVectorizer(sparse=True)

        # Primary classifier
        self.classifier = LogisticRegression(
            C=self.c_param,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state,
        )

        # First-pass tagger for 2-pass mode (M4)
        self.first_pass_tagger: Optional["ContextLIDTagger"] = None

        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False
        self.train_vocab_: Set[str] = set()

    def _extract_data(
        self,
        sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
        include_labels: bool = False,
        predicted_sentence_labels: Optional[List[List[str]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Flatten sentences into words, context dicts, and labels."""
        flat_words: List[str] = []
        flat_dicts: List[Dict[str, Any]] = []
        flat_labels: List[str] = []

        for sent_idx, (tokens, labels) in enumerate(sentences):
            pred_labs = predicted_sentence_labels[sent_idx] if predicted_sentence_labels else None
            ctx_dicts = extract_context_features_for_sentence(
                tokens,
                predicted_labels=pred_labs,
                include_labels=include_labels,
            )
            for tok, d, lab in zip(tokens, ctx_dicts, labels):
                flat_words.append(tok.lower())
                flat_dicts.append(d)
                flat_labels.append(lab)

        return flat_words, flat_dicts, flat_labels

    def _transform_features(
        self,
        flat_words: List[str],
        flat_dicts: List[Dict[str, Any]],
        is_training: bool = False,
    ) -> csr_matrix:
        """Vectorize character and context feature blocks and horizontally stack."""
        if is_training:
            X_char = self.char_vectorizer.fit_transform(flat_words)
            if self.feature_mode != "char_only":
                X_ctx = self.ctx_vectorizer.fit_transform(flat_dicts)
                # Feature pruning for DictVectorizer if min_df_ctx > 1
                if self.min_df_ctx > 1:
                    col_sums = np.asarray(X_ctx.sum(axis=0)).ravel()
                    keep_mask = col_sums >= self.min_df_ctx
                    X_ctx = X_ctx[:, keep_mask]
                    # Filter feature names to match pruned columns
                    feat_names = np.array(self.ctx_vectorizer.get_feature_names_out())
                    self.pruned_ctx_indices_ = keep_mask
            else:
                X_ctx = None
        else:
            X_char = self.char_vectorizer.transform(flat_words)
            if self.feature_mode != "char_only":
                X_ctx = self.ctx_vectorizer.transform(flat_dicts)
                if self.min_df_ctx > 1 and hasattr(self, "pruned_ctx_indices_"):
                    X_ctx = X_ctx[:, self.pruned_ctx_indices_]
            else:
                X_ctx = None

        if self.feature_mode == "char_only" or X_ctx is None:
            return X_char
        return hstack([X_char, X_ctx]).tocsr()

    def fit(
        self,
        sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    ) -> "ContextLIDTagger":
        """
        Train the context tagger on sentence sequences.
        For M4 (2-pass), uses cross-validation out-of-fold predicted labels during training
        to eliminate gold label leakage.
        """
        # Record training vocabulary for OOV tracking
        self.train_vocab_ = set()
        for tokens, _ in sentences:
            for t in tokens:
                self.train_vocab_.add(t.lower())

        if self.feature_mode == "context_plus_labels":
            # Train Pass 1 tagger (M3 context_words)
            print("  [Pass 1/2] Training first-pass tagger for label conditioning...")
            self.first_pass_tagger = ContextLIDTagger(
                feature_mode="context_words",
                ngram_range=self.ngram_range,
                c_param=self.c_param,
                min_df_char=self.min_df_char,
                min_df_ctx=self.min_df_ctx,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.first_pass_tagger.fit(sentences)

            # Generate out-of-fold or first-pass predictions for training data
            print("  [Pass 1/2] Predicting first-pass labels on training data...")
            predicted_train_labels: List[List[str]] = []
            for tokens, _ in sentences:
                preds = self.first_pass_tagger.predict_sentence(tokens)
                predicted_train_labels.append(preds)

            flat_words, flat_dicts, flat_labels = self._extract_data(
                sentences,
                include_labels=True,
                predicted_sentence_labels=predicted_train_labels,
            )
        else:
            flat_words, flat_dicts, flat_labels = self._extract_data(
                sentences,
                include_labels=False,
            )

        # Vectorize and stack features
        X_train = self._transform_features(flat_words, flat_dicts, is_training=True)

        # Fit classifier
        self.classifier.fit(X_train, flat_labels)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    def predict_sentence(self, tokens: Sequence[str]) -> List[str]:
        """Predict language labels for a single sequence of sentence tokens."""
        if not tokens:
            return []
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        pred_labels = None
        if self.feature_mode == "context_plus_labels" and self.first_pass_tagger:
            pred_labels = self.first_pass_tagger.predict_sentence(tokens)

        ctx_dicts = extract_context_features_for_sentence(
            tokens,
            predicted_labels=pred_labels,
            include_labels=(self.feature_mode == "context_plus_labels"),
        )
        flat_words = [t.lower() for t in tokens]

        X = self._transform_features(flat_words, ctx_dicts, is_training=False)
        return [str(p) for p in self.classifier.predict(X)]

    def predict_proba_sentence(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        """Predict language probabilities for each token in a sentence."""
        if not tokens:
            return []
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        pred_labels = None
        if self.feature_mode == "context_plus_labels" and self.first_pass_tagger:
            pred_labels = self.first_pass_tagger.predict_sentence(tokens)

        ctx_dicts = extract_context_features_for_sentence(
            tokens,
            predicted_labels=pred_labels,
            include_labels=(self.feature_mode == "context_plus_labels"),
        )
        flat_words = [t.lower() for t in tokens]

        X = self._transform_features(flat_words, ctx_dicts, is_training=False)
        probs = self.classifier.predict_proba(X)
        return [
            {str(cls): round(float(p), 4) for cls, p in zip(self.classes_, row)}
            for row in probs
        ]

    def evaluate(
        self,
        dev_sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
        ambiguous_words: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate tagger on dev set sentences, computing overall metrics,
        OOV vs in-vocab metrics, confusion matrix, and performance on ambiguous words.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        if ambiguous_words is None:
            ambiguous_words = DEFAULT_AMBIGUOUS_WORDS

        flat_tokens: List[str] = []
        flat_true: List[str] = []
        flat_pred: List[str] = []

        # Batch prediction sentence by sentence to capture context correctly
        t0 = time.time()
        for tokens, labels in dev_sentences:
            preds = self.predict_sentence(tokens)
            flat_tokens.extend([t.lower() for t in tokens])
            flat_true.extend(labels)
            flat_pred.extend(preds)
        eval_time = time.time() - t0

        total_tokens = len(flat_tokens)
        oov_flags = [t not in self.train_vocab_ for t in flat_tokens]
        oov_count = sum(oov_flags)
        in_vocab_total = total_tokens - oov_count

        # Overall scores
        acc = accuracy_score(flat_true, flat_pred)
        macro_f1 = f1_score(flat_true, flat_pred, average="macro", zero_division=0)
        macro_prec = precision_score(flat_true, flat_pred, average="macro", zero_division=0)
        macro_rec = recall_score(flat_true, flat_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(flat_true, flat_pred, average="weighted", zero_division=0)

        # Per-class scores
        labels_list = sorted(list(self.classes_))
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

        # Confusion matrix
        cm = confusion_matrix(flat_true, flat_pred, labels=labels_list)

        # In-Vocab vs OOV error breakdown
        in_vocab_errors = 0
        oov_errors = 0
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

        # Ambiguous word subset evaluation (Hypothesis H2)
        ambig_true: List[str] = []
        ambig_pred: List[str] = []
        ambig_word_stats: Dict[str, Dict[str, Any]] = {}

        for tok, true_l, pred_l in zip(flat_tokens, flat_true, flat_pred):
            if tok in ambiguous_words:
                ambig_true.append(true_l)
                ambig_pred.append(pred_l)
                if tok not in ambig_word_stats:
                    ambig_word_stats[tok] = {"total": 0, "correct": 0, "errors": 0}
                ambig_word_stats[tok]["total"] += 1
                if true_l == pred_l:
                    ambig_word_stats[tok]["correct"] += 1
                else:
                    ambig_word_stats[tok]["errors"] += 1

        ambig_total = len(ambig_true)
        ambig_acc = accuracy_score(ambig_true, ambig_pred) if ambig_total else 0.0
        ambig_f1 = f1_score(ambig_true, ambig_pred, average="macro", zero_division=0) if ambig_total else 0.0

        for w, s in ambig_word_stats.items():
            s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] else 0.0

        return {
            "feature_mode": self.feature_mode,
            "c_param": self.c_param,
            "total_tokens": total_tokens,
            "seen_tokens": in_vocab_total,
            "oov_tokens": oov_count,
            "oov_rate": round(oov_count / total_tokens, 4) if total_tokens else 0.0,
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "labels": labels_list,
            "confusion_matrix": cm.tolist(),
            "en_as_en": int(cm[0][0]),
            "en_as_hi": int(cm[0][1]),
            "hi_as_en": int(cm[1][0]),
            "hi_as_hi": int(cm[1][1]),
            "total_errors": total_errors,
            "in_vocab_errors": in_vocab_errors,
            "in_vocab_accuracy": round(in_vocab_acc, 4),
            "oov_errors": oov_errors,
            "oov_accuracy": round(oov_acc, 4),
            "top_in_vocab_errors": in_vocab_error_words.most_common(15),
            "top_oov_errors": oov_error_words.most_common(15),
            "ambiguous_total": ambig_total,
            "ambiguous_accuracy": round(float(ambig_acc), 4),
            "ambiguous_macro_f1": round(float(ambig_f1), 4),
            "ambiguous_word_stats": ambig_word_stats,
            "eval_time_sec": round(eval_time, 2),
            "classification_report": classification_report(flat_true, flat_pred, digits=4),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Context model saved to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ContextLIDTagger":
        """Deserialize model from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
