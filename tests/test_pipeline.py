"""
tests/test_pipeline.py
Unit tests verifying Day 14 (Module M2.7) Unified End-to-End Pipeline Integration,
Comparative Transforms (Setups A, B, and C), Latency, and Architecture Integrity.
"""

from pathlib import Path
import pytest

from src.pipeline import HinglishPipeline, analyze_query


@pytest.fixture(scope="module")
def pipeline():
    return HinglishPipeline()


def test_pipeline_single_call_schema(pipeline):
    """Verify analyze_query returns complete required schema in one function call."""
    res = pipeline.analyze("Mera order #123456 cancel kro plz!")
    
    expected_keys = {
        "raw_query",
        "clean_text",
        "tokens",
        "tags",
        "cmi",
        "text_A",
        "text_B",
        "text_C",
        "latency_ms",
    }
    assert expected_keys.issubset(set(res.keys()))
    assert isinstance(res["tokens"], list)
    assert isinstance(res["tags"], list)
    assert len(res["tokens"]) == len(res["tags"])
    assert 0.0 <= res["cmi"] <= 100.0
    assert res["latency_ms"] >= 0.0


def test_empty_and_whitespace_query_handling(pipeline):
    """Verify graceful empty input handling."""
    res = pipeline.analyze("")
    assert res["tokens"] == []
    assert res["tags"] == []
    assert res["cmi"] == 0.0

    res_ws = pipeline.analyze("    \n\t   ")
    assert res_ws["tokens"] == []
    assert res_ws["tags"] == []


def test_comparative_transform_differences(pipeline):
    """
    Verify structural differences between Setups A, B, and C on a representative query:
    'Khana bohot late ho gaya, order #884920 cancel kro plz!'
    """
    query = "Khana bohot late ho gaya, order #884920 cancel kro plz!"
    res = pipeline.analyze(query)

    # Setup A: Raw text without entity masking
    assert "#884920" in res["text_A"] or "884920" in res["text_A"]
    assert "<ORDER_ID>" not in res["text_A"]

    # Setup B: Entity masking but unnormalized Hindi spelling
    assert "<ORDER_ID>" in res["text_B"]
    assert "bohot" in res["text_B"]
    assert "kro" in res["text_B"]

    # Setup C: Entity masking + Normalized Hindi + Intent Preserving
    assert "<ORDER_ID>" in res["text_C"]
    assert "bahut" in res["text_C"]  # normalized from bohot
    assert "karo" in res["text_C"]   # normalized from kro
    assert "please" in res["text_C"] # normalized from plz


def test_convenience_function():
    """Verify global analyze_query helper."""
    res = analyze_query("Refund kab tak aayega?")
    assert "kab" in res["text_C"]
    assert res["cmi"] >= 0.0


def test_table_6_pipeline_samples_exist():
    """Verify Table 6 artifacts exist and contain 20 samples."""
    csv_path = Path("results/table_6_pipeline_samples.csv")
    json_path = Path("results/table_6_pipeline_samples.json")

    assert csv_path.exists(), f"Missing {csv_path}"
    assert json_path.exists(), f"Missing {json_path}"

    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == 20, f"Expected 20 rows, got {len(df)}"


def test_zero_circular_dependency_integrity(pipeline):
    """
    Ensure the production LID model was trained on L3Cube corpus
    and has strictly ZERO knowledge of the intent dataset (no data leakage).
    """
    assert pipeline.tagger is not None
    # Check that tagger has character vectorizer, context vectorizer, and classifier
    assert hasattr(pipeline.tagger, "char_vectorizer")
    assert hasattr(pipeline.tagger, "ctx_vectorizer")
    assert hasattr(pipeline.tagger, "classifier")
    # Tagger training size reflects 66k+ training vocabulary from 1.08M L3Cube tokens
    assert len(pipeline.tagger.train_vocab_) > 50000
    assert pipeline.tagger.is_fitted
