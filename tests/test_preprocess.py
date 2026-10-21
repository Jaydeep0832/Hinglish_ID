"""
tests/test_preprocess.py
Comprehensive test suite for src/preprocess.py (M1.3)
Verifies 20+ assertions including URL/Emoji/Order-ID masking, elongation normalization,
tokenization, edge cases, and mixed scripts.

Run with: pytest tests/test_preprocess.py -v
"""

import pytest

from src.preprocess import (
    HinglishPreprocessor,
    clean,
    mask_emojis,
    mask_order_ids,
    mask_urls,
    normalize_elongation,
    tokenize,
)


class TestURLMasking:
    """Tests for URL detection and masking."""

    def test_http_url(self):
        text = "Order status yahan dekho: http://tracker.com/status"
        cleaned = clean(text)
        assert "<URL>" in cleaned
        assert "http://tracker.com/status" not in cleaned

    def test_https_url_with_parameters(self):
        text = "Check this https://amazon.in/order?id=123&track=yes now"
        cleaned = clean(text)
        assert "<URL>" in cleaned
        assert "https://" not in cleaned

    def test_www_prefix_url(self):
        text = "Visit www.flipkart.com for tracking"
        cleaned = clean(text)
        assert "<URL>" in cleaned
        assert "www.flipkart.com" not in cleaned


class TestEmojiMasking:
    """Tests for Emoji detection and masking."""

    def test_single_emoji(self):
        text = "Order late hai 😡"
        cleaned = clean(text)
        assert "<EMOJI>" in cleaned
        assert "😡" not in cleaned

    def test_multiple_emojis(self):
        text = "Thank you so much 🙏✨"
        cleaned = clean(text)
        assert cleaned.count("<EMOJI>") == 2

    def test_emoji_only_string(self):
        text = "😡🙏"
        cleaned = clean(text)
        assert cleaned == "<EMOJI> <EMOJI>"
        tokens = tokenize(text)
        assert tokens == ["<EMOJI>", "<EMOJI>"]


class TestOrderIDMasking:
    """Tests for distinguishing Order IDs from normal quantities/prices."""

    def test_order_id_with_hash(self):
        text = "Order #45678 update please"
        cleaned = clean(text)
        assert "<ORDER_ID>" in cleaned
        assert "#45678" not in cleaned

    def test_order_id_with_letter_prefix(self):
        text = "Mera OD123456789 deliver nahi hua"
        cleaned = clean(text)
        assert "<ORDER_ID>" in cleaned
        assert "OD123456789" not in cleaned

    def test_order_id_with_hyphen(self):
        text = "Cancel my order ORD-987654"
        cleaned = clean(text)
        assert "<ORDER_ID>" in cleaned
        assert "ORD-987654" not in cleaned

    def test_order_id_with_underscore(self):
        text = "Tracking code inv_12345"
        cleaned = clean(text)
        assert "<ORDER_ID>" in cleaned
        assert "inv_12345" not in cleaned

    def test_order_id_six_plus_digits(self):
        text = "Order number 123456 kahan hai"
        cleaned = clean(text)
        assert "<ORDER_ID>" in cleaned
        assert "123456" not in cleaned

    def test_preserve_normal_quantities_and_dates(self):
        # 1 to 5 digit standalone numbers should NOT be masked
        text = "2 din me 3 items chahiye the, price 500 rs tha, year 2024"
        cleaned = clean(text)
        assert "<ORDER_ID>" not in cleaned
        assert "2" in cleaned
        assert "3" in cleaned
        assert "500" in cleaned
        assert "2024" in cleaned

    def test_preserve_short_hash_number(self):
        # Hash with fewer than 4 digits is not an order ID (e.g. ranking "#1")
        text = "We are #1 in customer satisfaction"
        cleaned = clean(text)
        assert "<ORDER_ID>" not in cleaned
        assert "#1" in cleaned


class TestElongationNormalization:
    """Tests for character elongation normalization."""

    def test_elongation_plzzzzz(self):
        text = "plzzzzz refund karo"
        cleaned = clean(text)
        assert "plzz" in cleaned
        assert "plzzzzz" not in cleaned

    def test_elongation_kyaaaaa(self):
        text = "kyaaaaa hua"
        cleaned = clean(text)
        assert "kyaa" in cleaned
        assert "kyaaaaa" not in cleaned

    def test_preserve_legitimate_double_letters(self):
        # Double letters in valid English/Hindi words must not be modified
        text = "good book soon gaadi feet"
        cleaned = clean(text)
        assert "good" in cleaned
        assert "book" in cleaned
        assert "soon" in cleaned
        assert "feet" in cleaned

    def test_elongation_uppercase(self):
        text = "NOOOOOOO WAY"
        cleaned = clean(text)
        assert "NOO" in cleaned
        assert "NOOOOOOO" not in cleaned


class TestTokenizer:
    """Tests for Hinglish tokenization engine."""

    def test_tokens_isolate_placeholders_atomically(self):
        text = "Order <ORDER_ID> link <URL> status <EMOJI>"
        tokens = tokenize(text, clean_first=False)
        assert "<ORDER_ID>" in tokens
        assert "<URL>" in tokens
        assert "<EMOJI>" in tokens

    def test_tokens_isolate_punctuation(self):
        text = "Mera refund kahan hai?!?"
        tokens = tokenize(text)
        assert "Mera" in tokens
        assert "refund" in tokens
        assert "?" in tokens
        assert "!" in tokens

    def test_tokens_handle_contractions(self):
        text = "I don't want this order"
        tokens = tokenize(text)
        assert "don't" in tokens or ("don" in tokens and "'" in tokens)

    def test_tokens_preserve_case_by_default(self):
        text = "Amazon Flipkart Delhi Refund"
        tokens = tokenize(text, lower=False)
        assert tokens == ["Amazon", "Flipkart", "Delhi", "Refund"]

    def test_tokens_lowercase_option(self):
        text = "Mera Order #12345 Cancel Karo"
        tokens = tokenize(text, lower=True)
        assert tokens == ["mera", "order", "<ORDER_ID>", "cancel", "karo"]
        # Placeholder should remain uppercase for downstream components
        assert "<ORDER_ID>" in tokens


class TestEdgeCasesAndMixedScripts:
    """Tests for noisy and abnormal real-world inputs."""

    def test_empty_string(self):
        assert clean("") == ""
        assert tokenize("") == []

    def test_whitespace_only_string(self):
        assert clean("   \t\n   ") == ""
        assert tokenize("   \t\n   ") == []

    def test_punctuation_only_string(self):
        text = "???!!!..."
        assert clean(text) == "???!!!..."
        tokens = tokenize(text)
        assert len(tokens) > 0
        assert all(t in "?!." for t in tokens)

    def test_mixed_script_devanagari_and_latin(self):
        text = "मेरा order cancel करो plzzzz"
        tokens = tokenize(text)
        assert "मेरा" in tokens
        assert "order" in tokens
        assert "cancel" in tokens
        assert "करो" in tokens
        assert "plzz" in tokens

    def test_complex_customer_query_pipeline(self):
        raw_query = (
            "Plzzzzz refund karo mera order #987654 nahi aaya 😡 "
            "check https://status.in 2 din ho gaye"
        )
        tokens = tokenize(raw_query, lower=True)
        expected_tokens = [
            "plzz",
            "refund",
            "karo",
            "mera",
            "order",
            "<ORDER_ID>",
            "nahi",
            "aaya",
            "<EMOJI>",
            "check",
            "<URL>",
            "2",
            "din",
            "ho",
            "gaye",
        ]
        assert tokens == expected_tokens

        # Also verify case retention with lower=False
        tokens_raw_case = tokenize(raw_query, lower=False)
        assert tokens_raw_case[0] == "Plzz"


class TestHinglishPreprocessorClass:
    """Tests for object-oriented preprocessor interface."""

    def test_class_interface(self):
        prep = HinglishPreprocessor(lower=True)
        batch = [
            "Order #112233 late hai 😡",
            "Check https://help.com plzzzzz",
        ]
        results = prep.transform_batch(batch)
        assert len(results) == 2
        assert "<ORDER_ID>" in results[0]
        assert "<EMOJI>" in results[0]
        assert "<URL>" in results[1]
        assert "plzz" in results[1]
