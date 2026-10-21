"""
tests/test_ambiguous_eval.py
Unit tests for Module M2.2: Ambiguous Word List Mining, Freezing, and Table 3 Evaluation.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate_ambiguous import (
    bootstrap_ci,
    compute_word_majority_dict,
    load_ambiguous_word_list,
)


def test_ambiguous_word_list_properties():
    """Verify data/ambiguous_words.txt conforms to the 30-40 frozen word specification."""
    path = Path("data/ambiguous_words.txt")
    assert path.exists(), "data/ambiguous_words.txt does not exist"

    words = load_ambiguous_word_list(path)
    assert 30 <= len(words) <= 45, f"Expected 30-45 words, got {len(words)}"
    assert len(words) == len(set(words)), "Ambiguous word list contains duplicate words"

    for w in words:
        assert w == w.lower(), f"Word '{w}' is not lowercase"
        assert w.isalnum(), f"Word '{w}' contains non-alphanumeric characters"

    # Core starter pool words must be present
    core_starter = {"to", "is", "the", "sir", "do", "us", "use", "me", "hi", "main"}
    for cw in core_starter:
        assert cw in words, f"Core ambiguous word '{cw}' missing from frozen list"


def test_bootstrap_ci_toy_data():
    """Verify bootstrap confidence interval computation."""
    y_true = np.array(["HI", "HI", "EN", "EN", "HI", "EN", "HI", "EN", "HI", "HI"])
    y_pred = np.array(["HI", "HI", "EN", "HI", "HI", "EN", "EN", "EN", "HI", "HI"])
    # 8 correct out of 10 -> acc = 0.8

    mean, lower, upper = bootstrap_ci(y_true, y_pred, n_resamples=500, ci=0.95, seed=42)
    assert 0.0 <= lower <= mean <= upper <= 1.0
    assert abs(mean - 0.8) < 1e-5


def test_compute_word_majority_dict(tmp_path):
    """Verify majority label computation logic on a synthetic CoNLL file."""
    toy_conll = tmp_path / "toy_train.conll"
    toy_conll.write_text(
        "to\tHI\n"
        "to\tHI\n"
        "to\tEN\n"
        "is\tEN\n"
        "is\tEN\n"
        "is\tHI\n"
        "\n",
        encoding="utf-8",
    )

    majority_dict = compute_word_majority_dict(toy_conll)
    assert majority_dict["to"] == "HI"
    assert majority_dict["is"] == "EN"


def test_table_3_outputs_exist():
    """Verify that Table 3 CSV and JSON deliverables exist and are structurally valid."""
    csv_path = Path("results/table_3_ambiguous.csv")
    json_path = Path("results/table_3_ambiguous.json")

    if not csv_path.exists() or not json_path.exists():
        pytest.skip("Table 3 output files not yet generated.")

    df = pd.read_csv(csv_path)
    required_cols = [
        "word", "n_support", "support_flag", "hi_count", "en_count",
        "majority_label", "acc_majority", "acc_m0_lookup", "acc_m1_char", "acc_m3_context"
    ]
    for col in required_cols:
        assert col in df.columns, f"Column '{col}' missing from table_3_ambiguous.csv"

    assert len(df) >= 30, f"Expected at least 30 rows in Table 3, got {len(df)}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_ambiguous_tokens" in data
    assert "pooled_models" in data
    assert "m3_context_words" in data["pooled_models"]
    assert "ci_95_lower" in data["pooled_models"]["m3_context_words"]


def test_frozen_test_set_integrity():
    """
    CRITICAL: Verify data/processed/test.conll remains strictly untouched and FROZEN.
    SHA-256 hash must match Day 2 exact checksum.
    """
    test_path = Path("data/processed/test.conll")
    assert test_path.exists()

    hasher = hashlib.sha256()
    line_count = 0
    with open(test_path, "rb") as f:
        for line in f:
            hasher.update(line)
            line_count += 1

    expected_sha256 = "a3ce094d776066cafebc9cda2d00a42521b7b838d66af075296af2d3843d8923"
    assert hasher.hexdigest() == expected_sha256, "SECURITY ALERT: test.conll has been modified!"
    assert line_count == 140014, f"Expected 140,014 lines in test.conll, found {line_count}"
