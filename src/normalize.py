"""
src/normalize.py
Module M2.6 (Day 13): Hinglish Phonetic Spelling Normalizer & LID-Aware Stopword Filter.

Core Principles:
1. Normalization Lexicon (data/norm_map.json): Maps 200+ Romanized phonetic spelling variants
   back to their canonical forms (e.g., "nhi/nai/nahin" -> "nahi", "kro" -> "karo", "bohot" -> "bahut").
2. Intent-Signal Preservation: Strict preservation of negation words ("nahi", "mat", "not", "no")
   and question words ("kab", "kya", "kyun", "when", "why", "how") which carry vital customer intent signals.
3. LID-Aware Collision Resolution: Cross-lingual homographs (like "do") are disambiguated by LID tag:
   - "do" with LID="EN": auxiliary verb stopword (can be filtered).
   - "do" with LID="HI": imperative content verb ("give/dena", e.g., "refund do") (MUST be preserved).
"""

import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

from src.preprocess import HinglishPreprocessor, tokenize


# Protected tokens that MUST NEVER be pruned as stopwords because they signal customer intent:
PROTECTED_NEGATIONS: Set[str] = {
    "nahi",
    "mat",
    "na",
    "bina",
    "not",
    "no",
    "never",
    "nor",
    "neither",
    "without",
}

PROTECTED_QUESTIONS: Set[str] = {
    "kab",
    "kya",
    "kyun",
    "kahan",
    "kaise",
    "kitna",
    "kitne",
    "when",
    "where",
    "why",
    "how",
    "what",
    "which",
    "who",
}

# Standard non-informative Hindi grammatical function words (excluding protected intent signals):
HINDI_GRAMMATICAL_STOPWORDS: Set[str] = {
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "mein",
    "me",
    "par",
    "pe",
    "aur",
    "bhi",
    "toh",
    "to",
    "tha",
    "thi",
    "the",
    "gaya",
    "gayi",
    "gaye",
    "hua",
    "hui",
    "hue",
    "hai",
    "hain",
    "ho",
    "kar",
    "diya",
    "liye",
    "saath",
    "apne",
    "apna",
    "apni",
    "is",
    "us",
    "un",
    "in",
}

# Standard English functional stopwords (excluding protected intent negations and questions):
ENGLISH_GRAMMATICAL_STOPWORDS: Set[str] = {
    "the",
    "is",
    "am",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "for",
    "to",
    "from",
    "by",
    "with",
    "an",
    "a",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "it",
    "its",
    "of",
    "as",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
}


class HinglishNormalizer:
    """
    Normalizes Romanized Hindi phonetic variations and applies LID-aware stopword filtering.
    """

    def __init__(self, norm_map_path: Optional[Path] = None):
        if norm_map_path is None:
            norm_map_path = Path("data/norm_map.json")
        self.norm_map_path = Path(norm_map_path)
        self.norm_map = self._load_norm_map(self.norm_map_path)
        self.preprocessor = HinglishPreprocessor()

    def _load_norm_map(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def normalize_token(self, token: str) -> str:
        """
        Normalize a single token using the phonetic dictionary.
        Preserves special placeholders like <ORDER_ID>, <URL>, <EMOJI>.
        """
        if token.startswith("<") and token.endswith(">"):
            return token
        lowered = token.lower()
        return self.norm_map.get(lowered, lowered)

    def is_stopword(self, token: str, lid_tag: Optional[str] = None) -> bool:
        """
        Determine if a token is a stopword, applying strict intent protection
        and LID-aware collision resolution.
        """
        lowered = token.lower()
        normalized = self.norm_map.get(lowered, lowered)

        # 1. Protected intent signals are NEVER stopwords
        if normalized in PROTECTED_NEGATIONS or normalized in PROTECTED_QUESTIONS:
            return False
        if lowered in PROTECTED_NEGATIONS or lowered in PROTECTED_QUESTIONS:
            return False

        # 2. LID-Aware Homograph Collision Resolution
        # Case: "do"
        if lowered == "do":
            if lid_tag == "HI":
                # Hindi imperative verb "dena / do" (e.g. "paise wapas do") -> NOT a stopword
                return False
            elif lid_tag == "EN":
                # English auxiliary verb "do" -> IS a stopword
                return True
            else:
                # Default without tag: preserve to avoid losing critical Hindi imperative
                return False

        # Case: "to"
        if lowered == "to":
            if lid_tag == "HI":
                # Hindi discourse particle ("karo to sahi")
                return True
            elif lid_tag == "EN":
                # English preposition ("send to my account")
                return True
            return True

        # Case: "me" / "mein"
        if lowered in {"me", "mein"}:
            return True

        # 3. Check language-specific stopword dictionaries
        if lid_tag == "HI":
            return lowered in HINDI_GRAMMATICAL_STOPWORDS or normalized in HINDI_GRAMMATICAL_STOPWORDS
        elif lid_tag == "EN":
            return lowered in ENGLISH_GRAMMATICAL_STOPWORDS or normalized in ENGLISH_GRAMMATICAL_STOPWORDS
        else:
            # Untagged: check union of grammatical stopwords
            return (
                lowered in HINDI_GRAMMATICAL_STOPWORDS
                or lowered in ENGLISH_GRAMMATICAL_STOPWORDS
                or normalized in HINDI_GRAMMATICAL_STOPWORDS
                or normalized in ENGLISH_GRAMMATICAL_STOPWORDS
            )

    def normalize_tokens(
        self,
        tokens: List[str],
        lid_tags: Optional[List[str]] = None,
        remove_stopwords: bool = True,
    ) -> List[str]:
        """
        Normalize a list of tokens and optionally filter stopwords with LID awareness.
        """
        if lid_tags is None:
            lid_tags = [None] * len(tokens)

        result = []
        for tok, tag in zip(tokens, lid_tags):
            norm_tok = self.normalize_token(tok)
            if remove_stopwords and self.is_stopword(tok, tag):
                continue
            result.append(norm_tok)
        return result

    def preprocess_and_normalize(
        self,
        text: str,
        lid_tagger=None,
        remove_stopwords: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        Full end-to-end normalization pipeline:
        Text -> Preprocess (masking) -> Tokenize -> (Optional LID Tagger) -> Normalize & Filter.
        Returns: (normalized_tokens, corresponding_lid_tags)
        """
        tokens = tokenize(text, lower=True)
        if not tokens:
            return [], []

        if lid_tagger is not None:
            lid_tags = lid_tagger.predict_sentence(tokens)
        else:
            lid_tags = [None] * len(tokens)

        norm_tokens = []
        kept_tags = []
        for tok, tag in zip(tokens, lid_tags):
            norm_tok = self.normalize_token(tok)
            if remove_stopwords and self.is_stopword(tok, tag):
                continue
            norm_tokens.append(norm_tok)
            kept_tags.append(tag if tag is not None else "UNKNOWN")

        return norm_tokens, kept_tags


def normalize_text_string(text: str, normalizer: Optional[HinglishNormalizer] = None) -> str:
    """Convenience function returning a single normalized string."""
    if normalizer is None:
        normalizer = HinglishNormalizer()
    tokens, _ = normalizer.preprocess_and_normalize(text, remove_stopwords=False)
    return " ".join(tokens)
