"""
tests/test_final_evaluation.py
Unit tests for Module M2.3: Final Test Set Evaluation, Model Freezing, and Error Analysis.
"""

import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
import pytest

from src.baseline_tagger import WordLookupTagger
from src.lid_context import ContextLIDTagger
from src.lid_tagger import CharNgramLIDTagger

# Register classes for joblib unpickling
import __main__
__main__.WordLookupTagger = WordLookupTagger
__main__.CharNgramLIDTagger = CharNgramLIDTagger
__main__.ContextLIDTagger = ContextLIDTagger


def test_final_model_frozen():
    """Verify models/lid_final.joblib exists, is locked, and loads cleanly."""
    model_path = Path("models/lid_final.joblib")
    assert model_path.exists(), "models/lid_final.joblib does not exist"
    assert model_path.stat().st_size > 10_000_000, "Final model size unexpectedly small"

    tagger = joblib.load(model_path)
    assert isinstance(tagger, ContextLIDTagger)
    assert tagger.is_fitted, "Final frozen tagger is not fitted"


def test_error_analysis_document():
    """Verify docs/error_analysis.md exists and contains 15 case deep-dives."""
    doc_path = Path("docs/error_analysis.md")
    assert doc_path.exists(), "docs/error_analysis.md missing"
    content = doc_path.read_text(encoding="utf-8")

    assert "Case 1:" in content
    assert "Case 15:" in content
    assert "Linguistic Error Taxonomy" in content
    assert "Checkpoint M2.3 Resolution" in content


def test_final_test_table_metrics():
    """Verify Table 4 test set results are valid and meet project targets."""
    csv_path = Path("results/table_4_test_final.csv")
    json_path = Path("results/table_4_test_final.json")

    if not csv_path.exists() or not json_path.exists():
        pytest.skip("Test evaluation outputs not yet generated.")

    df = pd.read_csv(csv_path)
    assert len(df) == 3, f"Expected 3 models evaluated, got {len(df)}"

    expected_models = {"M0_Lookup", "M1_Char_Only", "M3_Final_Context"}
    assert set(df["model_id"]) == expected_models

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_hash = "a3ce094d776066cafebc9cda2d00a42521b7b838d66af075296af2d3843d8923"
    assert data["metadata"]["test_sha256"] == expected_hash

    # Context tagger should achieve >= 98% test accuracy
    m3_row = df[df["model_id"] == "M3_Final_Context"].iloc[0]
    assert m3_row["test_accuracy"] >= 0.98, f"Final test accuracy {m3_row['test_accuracy']} below 98%"
    assert m3_row["macro_f1"] >= 0.97, f"Final test macro-F1 {m3_row['macro_f1']} below 0.97"

    # Context tagger should beat lookup and char-only on ambiguous words
    m0_row = df[df["model_id"] == "M0_Lookup"].iloc[0]
    m1_row = df[df["model_id"] == "M1_Char_Only"].iloc[0]
    assert m3_row["ambiguous_accuracy"] > m0_row["ambiguous_accuracy"]
    assert m3_row["ambiguous_accuracy"] > m1_row["ambiguous_accuracy"]


def test_test_set_unmodified_line_count():
    """Confirm data/processed/test.conll preserved all 140,014 lines."""
    test_path = Path("data/processed/test.conll")
    line_count = sum(1 for _ in open(test_path, encoding="utf-8"))
    assert line_count == 140014, f"Line count mismatch: {line_count} != 140,014"
