"""
src/preprocess.py
Modular text preprocessing and tokenization engine for Hinglish NLP.

Conforms to HinglishID-Router specification (Module M1.3):
- URL masking -> <URL>
- Emoji masking -> <EMOJI>
- Order ID masking -> <ORDER_ID> (differentiated from normal quantities)
- Character elongation normalization (e.g., 'plzzzzz' -> 'plzz')
- Tokenizer: preserves placeholders, handles Unicode Latin & Devanagari script,
  isolates punctuation, and supports case retention or lowercase feature extraction.
"""

from typing import List, Optional, Pattern, Sequence, Union
import regex as re

# =====================================================================
# Compiled Regex Patterns
# =====================================================================

# 1. URLs: http://..., https://..., or www....
URL_PATTERN: Pattern = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

# 2. Emojis: Matches single or grouped extended pictographic characters
EMOJI_PATTERN: Pattern = re.compile(
    r"\p{Extended_Pictographic}",
)

# 3. Order IDs:
#    - # followed by 4+ digits (e.g. #45678, #100234)
#    - 1-4 letters + optional [- or _] + 5+ digits (e.g. OD123456789, ORD-987654, inv_12345)
#    - Standalone numbers with 6 or more digits (e.g. 123456, 9876543210)
#    - Normal numbers (e.g., '2 din', '3 items', '500 rs', '2024') are intentionally NOT matched.
ORDER_ID_PATTERN: Pattern = re.compile(
    r"#\d{4,}|\b[A-Za-z]{1,4}[-_]?\d{5,}\b|\b\d{6,}\b",
)

# 4. Character Elongation:
#    Reduces 3 or more consecutive identical characters down to 2 (e.g., 'plzzzzz' -> 'plzz').
#    Preserves standard English/Hindi double letters (e.g., 'good', 'soon', 'bohoot').
ELONGATION_PATTERN: Pattern = re.compile(
    r"([A-Za-z])\1{2,}",
)

# 5. Tokenizer Pattern:
#    - Placeholders: <URL>, <EMOJI>, <ORDER_ID>, etc.
#    - Words: Unicode alphabetic & combining mark sequences (Latin, Devanagari \p{L}\p{M})
#             including internal apostrophes
#    - Numbers: digits including decimals (e.g. 2, 3.5, 500)
#    - Punctuation / symbols: single non-whitespace non-alphanumeric characters
TOKEN_PATTERN: Pattern = re.compile(
    r"<[A-Z_]+>|[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)?|\d+(?:[.,]\d+)?|[^\s\p{L}\p{M}\d]",
)


# =====================================================================
# Core Normalization & Masking Functions
# =====================================================================

def mask_urls(text: str, placeholder: str = "<URL>") -> str:
    """Mask URLs in text with a standard placeholder."""
    if not text:
        return ""
    return URL_PATTERN.sub(f" {placeholder} ", text)


def mask_emojis(text: str, placeholder: str = "<EMOJI>") -> str:
    """Mask emojis in text with a standard placeholder."""
    if not text:
        return ""
    return EMOJI_PATTERN.sub(f" {placeholder} ", text)


def mask_order_ids(text: str, placeholder: str = "<ORDER_ID>") -> str:
    """
    Mask e-commerce order identifiers with a placeholder while leaving
    common quantities, dates, and prices intact.
    """
    if not text:
        return ""
    return ORDER_ID_PATTERN.sub(f" {placeholder} ", text)


def normalize_elongation(text: str, max_repeats: int = 2) -> str:
    """
    Normalize informal character elongations (e.g., 'plzzzzz' -> 'plzz', 'haaaan' -> 'haan').
    Collapses 3 or more repeated letters to `max_repeats` (default 2).
    """
    if not text:
        return ""
    replacement = r"\1" * max_repeats
    return ELONGATION_PATTERN.sub(replacement, text)


def clean(
    text: str,
    mask_url: bool = True,
    mask_emoji: bool = True,
    mask_oid: bool = True,
    normalize_elong: bool = True,
    url_placeholder: str = "<URL>",
    emoji_placeholder: str = "<EMOJI>",
    oid_placeholder: str = "<ORDER_ID>",
) -> str:
    """
    Full text cleaning pipeline for Hinglish text:
    1. Mask URLs
    2. Mask Emojis
    3. Mask Order IDs
    4. Normalize Character Elongation
    5. Clean up redundant whitespace

    Args:
        text: Input raw string.
        mask_url: Whether to mask URLs (default True).
        mask_emoji: Whether to mask emojis (default True).
        mask_oid: Whether to mask order IDs (default True).
        normalize_elong: Whether to normalize character elongations (default True).
        url_placeholder: Placeholder token for URLs.
        emoji_placeholder: Placeholder token for emojis.
        oid_placeholder: Placeholder token for order IDs.

    Returns:
        Cleaned string with placeholders and normalized characters.
    """
    if not text or not text.strip():
        return ""

    t = text
    if mask_url:
        t = mask_urls(t, url_placeholder)
    if mask_emoji:
        t = mask_emojis(t, emoji_placeholder)
    if mask_oid:
        t = mask_order_ids(t, oid_placeholder)
    if normalize_elong:
        t = normalize_elongation(t, max_repeats=2)

    # Collapse excess whitespace while preserving clean token boundaries
    return " ".join(t.split())


def tokenize(
    text: str,
    clean_first: bool = True,
    lower: bool = False,
) -> List[str]:
    """
    Tokenize Hinglish text into words, placeholders, numbers, and punctuation.

    Args:
        text: Input string.
        clean_first: Whether to apply `clean()` before tokenizing (default True).
        lower: Whether to lowercase non-placeholder tokens (default False).
               Original casing is preserved by default to keep Named Entity clues.

    Returns:
        List of string tokens.
    """
    if not text or not text.strip():
        return []

    processed_text = clean(text) if clean_first else text
    tokens = TOKEN_PATTERN.findall(processed_text)

    if lower:
        # Lowercase words while keeping placeholders like <URL>, <ORDER_ID> uppercase
        return [
            tok if (tok.startswith("<") and tok.endswith(">")) else tok.lower()
            for tok in tokens
        ]

    return tokens


def is_special_token(token: str) -> bool:
    """Check if token is an entity placeholder (e.g. <URL>, <EMOJI>, <ORDER_ID>)."""
    return bool(token.startswith("<") and token.endswith(">") and len(token) > 2)


# =====================================================================
# Object-Oriented Preprocessor Interface
# =====================================================================

class HinglishPreprocessor:
    """
    Configurable preprocessor class for batch processing in training and inference pipelines.
    """

    def __init__(
        self,
        mask_url: bool = True,
        mask_emoji: bool = True,
        mask_oid: bool = True,
        normalize_elong: bool = True,
        lower: bool = False,
    ):
        self.mask_url = mask_url
        self.mask_emoji = mask_emoji
        self.mask_oid = mask_oid
        self.normalize_elong = normalize_elong
        self.lower = lower

    def clean_text(self, text: str) -> str:
        """Clean a single string."""
        return clean(
            text,
            mask_url=self.mask_url,
            mask_emoji=self.mask_emoji,
            mask_oid=self.mask_oid,
            normalize_elong=self.normalize_elong,
        )

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize a single string."""
        return tokenize(
            text,
            clean_first=True,
            lower=self.lower,
        )

    # Convenient aliases
    clean = clean_text
    tokenize = tokenize_text


    def transform_batch(self, texts: Sequence[str]) -> List[List[str]]:
        """Transform an iterable of raw texts into token lists."""
        return [self.tokenize_text(t) for t in texts]
