"""
tests/test_demo.py
Unit tests verifying app/demo.py structure, cached loaders, and edge case query handling.
"""

from pathlib import Path
import sys
import pytest

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.demo import (
    load_pipeline,
    load_intent_classifiers,
    SAMPLE_QUERIES,
)
from src.routing import route_intent


def test_demo_resource_loaders():
    """Verify that cached model loaders successfully load all pipelines."""
    pipeline = load_pipeline()
    assert pipeline is not None
    assert pipeline.tagger is not None

    classifiers = load_intent_classifiers()
    assert "Setup_C" in classifiers
    assert "Setup_B" in classifiers
    assert "Setup_A" in classifiers


def test_demo_sample_queries_structure():
    """Verify sample queries are properly structured with non-empty text."""
    assert len(SAMPLE_QUERIES) >= 6
    for sample in SAMPLE_QUERIES:
        assert "label" in sample
        assert "text" in sample
        assert len(sample["text"]) > 5


def test_demo_pipeline_execution():
    """Verify pipeline and classifier inference for sample queries."""
    pipeline = load_pipeline()
    classifiers = load_intent_classifiers()

    query = "Mera order abhi tak nahi aaya, please refund karo"
    analysis = pipeline.analyze(query)

    assert len(analysis["tokens"]) > 0
    assert len(analysis["tags"]) == len(analysis["tokens"])
    assert analysis["cmi"] > 0.0

    # Setup C prediction
    text_c = analysis["text_C"]
    model_c = classifiers["Setup_C"]
    probs = model_c.predict_proba([text_c])[0]
    classes = model_c.classes_

    assert len(probs) == len(classes)
    assert abs(sum(probs) - 1.0) < 1e-4

    top_idx = probs.argmax()
    pred_intent = classes[top_idx]
    pred_conf = float(probs[top_idx])

    # Routing
    routing = route_intent(pred_intent, pred_conf, analysis["cmi"])
    assert "queue" in routing
    assert routing["queue"] in ["Refunds", "Order Ops", "Logistics", "Payments", "General FAQ", "Human agent (low confidence)"]


def test_demo_edge_case_queries():
    """Verify demo pipeline handles edge-case queries without crashing."""
    pipeline = load_pipeline()

    # Empty query
    res_empty = pipeline.analyze("")
    assert res_empty["tokens"] == []
    assert res_empty["cmi"] == 0.0

    # Emojis only query
    res_emoji = pipeline.analyze("😡 📦 🔥")
    assert len(res_emoji["tokens"]) == 3
    assert len(res_emoji["tags"]) == 3

    # English only
    res_en = pipeline.analyze("Where is my order package")
    assert all(t in ["EN", "OTHER"] for t in res_en["tags"])
