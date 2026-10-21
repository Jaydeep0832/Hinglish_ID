"""
tests/test_normalize.py
Unit tests verifying Day 13 (Module M2.6) Normalization Lexicon,
Phonetic Spelling Standardization, Intent-Signal Preservation, and LID-Aware Stopword Filtering.
"""

from pathlib import Path
import pytest

from src.lid_context import ContextLIDTagger
from src.normalize import (
    HinglishNormalizer,
    PROTECTED_NEGATIONS,
    PROTECTED_QUESTIONS,
    normalize_text_string,
)


@pytest.fixture
def normalizer():
    return HinglishNormalizer(Path("data/norm_map.json"))


def test_norm_map_loaded_and_valid(normalizer):
    """Ensure normalization lexicon exists and contains >= 150 rules."""
    assert len(normalizer.norm_map) >= 150, f"Expected >= 150 rules, found {len(normalizer.norm_map)}"


def test_fifteen_before_after_normalizations(normalizer):
    """Verify >= 15 concrete before/after spelling normalization pairs."""
    test_pairs = [
        ("nahin", "nahi"),
        ("nhi", "nahi"),
        ("nai", "nahi"),
        ("hain", "hai"),
        ("kro", "karo"),
        ("bohot", "bahut"),
        ("bhot", "bahut"),
        ("plzz", "please"),
        ("pls", "please"),
        ("waps", "wapas"),
        ("chaiye", "chahiye"),
        ("jldi", "jaldi"),
        ("mjhe", "mujhe"),
        ("kese", "kaise"),
        ("kyu", "kyun"),
        ("cancle", "cancel"),
        ("refnd", "refund"),
        ("prblm", "problem"),
        ("khrab", "kharab"),
        ("bkar", "bekar"),
    ]

    assert len(test_pairs) >= 15
    for raw_token, expected_canonical in test_pairs:
        actual = normalizer.normalize_token(raw_token)
        assert actual == expected_canonical, f"Failed for '{raw_token}': expected '{expected_canonical}', got '{actual}'"


def test_strict_intent_signal_preservation(normalizer):
    """Verify negation and question words are NEVER treated as stopwords."""
    # Critical Hindi intent signals
    hindi_signals = ["nahi", "mat", "na", "bina", "kab", "kya", "kyun", "kaise", "kahan"]
    for word in hindi_signals:
        assert not normalizer.is_stopword(word, lid_tag="HI"), f"Hindi intent word '{word}' was falsely pruned as stopword!"
        assert not normalizer.is_stopword(word, lid_tag=None), f"Hindi intent word '{word}' was falsely pruned without tag!"

    # Critical English intent signals
    english_signals = ["not", "no", "never", "without", "when", "why", "how", "what"]
    for word in english_signals:
        assert not normalizer.is_stopword(word, lid_tag="EN"), f"English intent word '{word}' was falsely pruned as stopword!"
        assert not normalizer.is_stopword(word, lid_tag=None), f"English intent word '{word}' was falsely pruned without tag!"


def test_lid_aware_collision_do(normalizer):
    """
    Test LID-Aware collision resolution for 'do':
    - In Hindi ('paise do'): 'do' is a content imperative verb -> MUST NOT be stopword.
    - In English ('do you have'): 'do' is an auxiliary function word -> IS a stopword.
    """
    assert not normalizer.is_stopword("do", lid_tag="HI"), "'do' with LID=HI should be retained as imperative verb!"
    assert normalizer.is_stopword("do", lid_tag="EN"), "'do' with LID=EN should be treated as auxiliary stopword!"


def test_standard_grammatical_stopwords_filtered(normalizer):
    """Verify non-informative grammatical words are correctly identified as stopwords."""
    # Hindi grammatical stopwords
    assert normalizer.is_stopword("ka", lid_tag="HI")
    assert normalizer.is_stopword("ke", lid_tag="HI")
    assert normalizer.is_stopword("ko", lid_tag="HI")
    assert normalizer.is_stopword("se", lid_tag="HI")
    assert normalizer.is_stopword("hai", lid_tag="HI")
    assert normalizer.is_stopword("aur", lid_tag="HI")

    # English grammatical stopwords
    assert normalizer.is_stopword("the", lid_tag="EN")
    assert normalizer.is_stopword("is", lid_tag="EN")
    assert normalizer.is_stopword("at", lid_tag="EN")
    assert normalizer.is_stopword("for", lid_tag="EN")


def test_full_pipeline_normalization_with_tagger(normalizer):
    """Test full preprocess_and_normalize with production LID tagger."""
    model_path = Path("models/lid_final.joblib")
    if model_path.exists():
        tagger = ContextLIDTagger.load(model_path)
    else:
        tagger = None

    query = "Plzzzz mera order cancel kro, mujhe refund abhi tak nhi mila!"
    norm_tokens, tags = normalizer.preprocess_and_normalize(query, lid_tagger=tagger, remove_stopwords=True)

    # Check that phonetics are normalized
    assert "please" in norm_tokens
    assert "cancel" in norm_tokens
    assert "karo" in norm_tokens
    assert "refund" in norm_tokens
    assert "nahi" in norm_tokens

    # Check that elongation was reduced
    assert "plzzzz" not in norm_tokens

    # Check that vital negation is retained
    assert "nahi" in norm_tokens


def test_convenience_function():
    """Verify normalize_text_string produces clean normalized text."""
    res = normalize_text_string("Bhot bhot dhanyawad sir ji, mera payment milenga")
    assert "bahut bahut thank you sir" in res or "bahut" in res
    assert "milega" in res
