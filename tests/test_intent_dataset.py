"""
tests/test_intent_dataset.py
Unit tests verifying the integrity, class quotas, diversity, and annotation validity
of the Hinglish Customer Support Intent Dataset (Day 11 / Module M2.4).
"""

import csv
import json
from pathlib import Path
import pytest
from sklearn.metrics import cohen_kappa_score


DATA_DIR = Path("data")
INTENT_CSV = DATA_DIR / "intent_dataset.csv"
INTENT_JSON = DATA_DIR / "intent_dataset.json"
INTENT_RAW_CSV = DATA_DIR / "intent_dataset_raw.csv"
OVERLAP_CSV = DATA_DIR / "intent_overlap_40.csv"

EXPECTED_INTENTS = {
    "refund",
    "cancel_order",
    "order_not_delivered",
    "delivery_delay",
    "payment_failed",
    "other",
}


def test_intent_files_exist():
    """Ensure all required dataset files were generated."""
    assert INTENT_CSV.exists(), f"Missing {INTENT_CSV}"
    assert INTENT_JSON.exists(), f"Missing {INTENT_JSON}"
    assert INTENT_RAW_CSV.exists(), f"Missing {INTENT_RAW_CSV}"
    assert OVERLAP_CSV.exists(), f"Missing {OVERLAP_CSV}"


def test_intent_dataset_row_counts_and_schema():
    """Verify total size is 360 and schema matches specifications."""
    with open(INTENT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 360, f"Expected 360 rows, found {len(rows)}"
    expected_fields = {"id", "text", "intent", "writer", "mix_style"}
    assert set(rows[0].keys()) == expected_fields, f"Invalid fields: {rows[0].keys()}"

    # Also verify json matches csv
    with open(INTENT_JSON, "r", encoding="utf-8") as f:
        json_rows = json.load(f)
    assert len(json_rows) == 360
    assert [r["text"] for r in rows] == [r["text"] for r in json_rows]


def test_exact_class_balance():
    """Verify each of the 6 intents has exactly 60 queries."""
    with open(INTENT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    counts = {}
    for r in rows:
        intent = r["intent"]
        counts[intent] = counts.get(intent, 0) + 1

    assert set(counts.keys()) == EXPECTED_INTENTS
    for intent, count in counts.items():
        assert count == 60, f"Intent '{intent}' has {count} queries, expected exactly 60"


def test_no_duplicate_queries():
    """Verify all 360 queries are distinct strings."""
    with open(INTENT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        texts = [r["text"].strip().lower() for r in reader]

    assert len(texts) == len(set(texts)), f"Found duplicate queries: {len(texts) - len(set(texts))} duplicates"


def test_writer_and_style_diversity():
    """Verify balanced distribution across authors (A, B, C) and mix styles."""
    with open(INTENT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    writers = {r["writer"] for r in rows}
    assert writers == {"A", "B", "C"}, f"Unexpected writers: {writers}"

    writer_counts = {}
    for r in rows:
        w = r["writer"]
        writer_counts[w] = writer_counts.get(w, 0) + 1

    for w, count in writer_counts.items():
        assert count == 120, f"Writer {w} has {count} queries, expected 120"

    styles = {r["mix_style"] for r in rows}
    assert styles == {"hindi", "english", "balanced"}

    # Each style must have significant representation (>= 60)
    for style in styles:
        cnt = sum(1 for r in rows if r["mix_style"] == style)
        assert cnt >= 60, f"Mix style '{style}' has {cnt} queries, expected >= 60"


def test_query_lengths_are_realistic():
    """Verify queries represent natural conversational lengths."""
    with open(INTENT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        words = r["text"].split()
        assert 3 <= len(words) <= 25, f"Query '{r['text']}' has unrealistic word count: {len(words)}"


def test_overlap_set_for_cohen_kappa():
    """Verify 40 overlap queries and calculate pairwise Cohen's Kappa >= 0.60."""
    with open(OVERLAP_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 40, f"Expected 40 overlap queries, found {len(rows)}"

    ann_A = [r["annotator_A"] for r in rows]
    ann_B = [r["annotator_B"] for r in rows]
    ann_C = [r["annotator_C"] for r in rows]
    gold = [r["gold_intent"] for r in rows]

    # Verify all annotations are valid intents
    for lab in ann_A + ann_B + ann_C + gold:
        assert lab in EXPECTED_INTENTS, f"Invalid intent in overlap: {lab}"

    # Pairwise Cohen's Kappa checks
    kappa_AB = cohen_kappa_score(ann_A, ann_B)
    kappa_AC = cohen_kappa_score(ann_A, ann_C)
    kappa_BC = cohen_kappa_score(ann_B, ann_C)

    assert kappa_AB >= 0.60, f"Kappa A-B too low: {kappa_AB}"
    assert kappa_AC >= 0.60, f"Kappa A-C too low: {kappa_AC}"
    assert kappa_BC >= 0.60, f"Kappa B-C too low: {kappa_BC}"
