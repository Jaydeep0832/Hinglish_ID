"""
src/pipeline.py
Module M2.7 (Day 14): Unified End-to-End Hinglish Preprocessing & Routing Pipeline.

Connects all components into a single callable interface:
1. Regex Masking & Cleaning (URLs, Emojis, Order IDs, Elongations)
2. Word-Level Language Identification (Frozen Production M3 Context Model)
3. Code-Mixing Index (CMI) Computation (Das & Gambäck)
4. Comparative Text Transforms for 3 Downstream Benchmarks:
   - Setup A: Baseline Raw (Minimal lowercased tokens, no LID awareness, standard English stopwords)
   - Setup B: Generic TF-IDF (Regex masking + basic English stopword pruning, no Hindi normalization, no LID)
   - Setup C: LID-Aware Normalization (Regex masking + 224-rule phonetic normalization + LID-aware protected negations & question signals + homograph collision resolution)

Guaranteed Integrity:
- The LID model (models/lid_final.joblib) was trained exclusively on the L3Cube corpus
  and was NEVER trained on customer support intent data (Zero circular dependency).
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.conll_parser import calculate_cmi as cmi
from src.lid_context import ContextLIDTagger
from src.normalize import (
    ENGLISH_GRAMMATICAL_STOPWORDS,
    HINDI_GRAMMATICAL_STOPWORDS,
    PROTECTED_NEGATIONS,
    PROTECTED_QUESTIONS,
    HinglishNormalizer,
)
from src.preprocess import HinglishPreprocessor, tokenize


class HinglishPipeline:
    """
    Unified end-to-end processing pipeline for code-mixed Hinglish customer support text.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        norm_map_path: Optional[Path] = None,
    ):
        if model_path is None:
            model_path = Path("models/lid_final.joblib")
        if norm_map_path is None:
            norm_map_path = Path("data/norm_map.json")

        self.model_path = Path(model_path)
        self.tagger = None
        if self.model_path.exists():
            self.tagger = ContextLIDTagger.load(self.model_path)

        self.preprocessor = HinglishPreprocessor()
        self.normalizer = HinglishNormalizer(norm_map_path)

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Execute end-to-end analysis on a single customer query string.
        Returns:
            Dictionary containing tokens, predicted LID tags, CMI, and transformed
            texts under Setup A, Setup B, and Setup C.
        """
        t0 = time.perf_counter()

        # 1. Cleaning and Masking
        clean_text = self.preprocessor.clean(query)
        tokens = tokenize(query, clean_first=True, lower=True)

        if not tokens:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "raw_query": query,
                "clean_text": "",
                "tokens": [],
                "tags": [],
                "cmi": 0.0,
                "text_A": "",
                "text_B": "",
                "text_C": "",
                "latency_ms": round(elapsed_ms, 2),
            }

        # 2. Predict Token-Level LID Tags (Frozen Model M3)
        if self.tagger is not None:
            tags = self.tagger.predict_sentence(tokens)
        else:
            tags = ["OTHER" if (t.startswith("<") and t.endswith(">")) else "HI" for t in tokens]

        # 3. Compute Code-Mixing Index (CMI)
        query_cmi = round(cmi(tags), 2)

        # 4. Transform A: Baseline Raw (Whitespace lowercased tokens, standard English stopword filter)
        tokens_A = [t.lower() for t in query.split() if t.lower() not in ENGLISH_GRAMMATICAL_STOPWORDS]
        text_A = " ".join(tokens_A)

        # 5. Transform B: Generic Preprocessed TF-IDF (Masked tokens, English stopwords filtered, no LID or Hindi normalization)
        tokens_B = [t for t in tokens if t not in ENGLISH_GRAMMATICAL_STOPWORDS]
        text_B = " ".join(tokens_B)

        # 6. Transform C: LID-Aware Normalization (Ours)
        # - Entity placeholders kept
        # - 224-rule phonetic Hindi normalization
        # - Intent signals strictly protected (nahi, mat, kab, kya, etc.)
        # - LID-aware collision handling ('do' in HI preserved, in EN filtered)
        # - Language-specific grammatical stopwords filtered
        tokens_C = []
        for tok, tag in zip(tokens, tags):
            norm_tok = self.normalizer.normalize_token(tok)
            if self.normalizer.is_stopword(tok, lid_tag=tag):
                continue
            tokens_C.append(norm_tok)
        text_C = " ".join(tokens_C)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "raw_query": query,
            "clean_text": clean_text,
            "tokens": tokens,
            "tags": tags,
            "cmi": query_cmi,
            "text_A": text_A,
            "text_B": text_B,
            "text_C": text_C,
            "latency_ms": round(elapsed_ms, 2),
        }

    def analyze_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Batch analysis over multiple customer queries."""
        return [self.analyze(q) for q in queries]


# Global singleton pipeline instance
_DEFAULT_PIPELINE: Optional[HinglishPipeline] = None


def analyze_query(query: str, pipeline: Optional[HinglishPipeline] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a single Hinglish query in one line of code.
    Example:
        res = analyze_query("Mera order #123456 cancel kro plz")
    """
    global _DEFAULT_PIPELINE
    if pipeline is None:
        if _DEFAULT_PIPELINE is None:
            _DEFAULT_PIPELINE = HinglishPipeline()
        pipeline = _DEFAULT_PIPELINE
    return pipeline.analyze(query)
