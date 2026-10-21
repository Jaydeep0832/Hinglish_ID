"""
tests/test_intent_qa.py
Unit tests verifying Day 12 (Module M2.5) Label QA, Cohen's & Fleiss' Kappa,
Repeated Stratified 5-Fold Cross Validation Protocol, and In-Domain LID Benchmark.
"""

import csv
import json
from pathlib import Path
import pytest


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

KAPPA_JSON = RESULTS_DIR / "intent_kappa_qa.json"
KAPPA_CSV = RESULTS_DIR / "intent_kappa_qa.csv"
CV_SPLITS_JSON = DATA_DIR / "intent_splits_5fold.json"
INDOMAIN_CONLL = DATA_DIR / "lid_indomain.conll"
TABLE_5_CSV = RESULTS_DIR / "table_5_indomain_lid.csv"
TABLE_5_JSON = RESULTS_DIR / "table_5_indomain_lid.json"


def test_qa_files_exist():
    """Ensure all Day 12 deliverables exist."""
    assert KAPPA_JSON.exists(), f"Missing {KAPPA_JSON}"
    assert KAPPA_CSV.exists(), f"Missing {KAPPA_CSV}"
    assert CV_SPLITS_JSON.exists(), f"Missing {CV_SPLITS_JSON}"
    assert INDOMAIN_CONLL.exists(), f"Missing {INDOMAIN_CONLL}"
    assert TABLE_5_CSV.exists(), f"Missing {TABLE_5_CSV}"
    assert TABLE_5_JSON.exists(), f"Missing {TABLE_5_JSON}"


def test_cohen_and_fleiss_kappa_thresholds():
    """Verify that inter-annotator agreement substantially exceeds target (>= 0.60)."""
    with open(KAPPA_JSON, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    metrics = qa_data["metrics"]
    assert metrics["cohen_kappa_A_vs_B"] >= 0.60
    assert metrics["cohen_kappa_A_vs_C"] >= 0.60
    assert metrics["cohen_kappa_B_vs_C"] >= 0.60
    assert metrics["mean_pairwise_kappa"] >= 0.80  # Almost perfect agreement
    assert metrics["fleiss_kappa"] >= 0.80

    # Ensure disagreements are documented
    disagreements = qa_data["disagreements"]
    assert len(disagreements) > 0
    assert len(disagreements) <= 5  # Realistic small set of borderline cases
    for d in disagreements:
        assert "overlap_id" in d
        assert "disagreement_type" in d
        assert "gold_intent" in d


def test_cv_splits_exact_stratification_and_coverage():
    """Verify 15 folds (3 repeats x 5 folds) with exact 12 val / 48 train per intent."""
    with open(CV_SPLITS_JSON, "r", encoding="utf-8") as f:
        config = json.load(f)

    meta = config["metadata"]
    assert meta["n_splits"] == 5
    assert meta["n_repeats"] == 3
    assert meta["total_splits"] == 15

    folds = config["folds"]
    assert len(folds) == 15

    for fold in folds:
        assert fold["train_count"] == 288
        assert fold["val_count"] == 72

        # Check no leakage between train and val
        train_set = set(fold["train_query_ids"])
        val_set = set(fold["val_query_ids"])
        assert len(train_set.intersection(val_set)) == 0

        # Exact class balance in val (12 each across 6 intents)
        val_dist = fold["val_class_distribution"]
        assert len(val_dist) == 6
        for intent, cnt in val_dist.items():
            assert cnt == 12, f"Intent {intent} has {cnt} val items, expected 12"

    # Per repeat: check complete 360 query coverage
    for rep in range(3):
        rep_val_ids = []
        for fold in folds:
            if fold["repeat"] == rep:
                rep_val_ids.extend(fold["val_query_ids"])
        assert len(rep_val_ids) == 360
        assert len(set(rep_val_ids)) == 360


def test_indomain_conll_structure():
    """Verify in-domain customer support ConLL dataset format."""
    lines = INDOMAIN_CONLL.read_text(encoding="utf-8").strip().split("\n")
    sents = 0
    tokens = 0
    valid_tags = {"HI", "EN", "OTHER"}

    for line in lines:
        if not line.strip():
            sents += 1
            continue
        parts = line.split("\t")
        assert len(parts) == 2, f"Malformed CoNLL line: {line}"
        tok, tag = parts
        assert tag in valid_tags, f"Invalid tag: {tag}"
        tokens += 1
    sents += 1  # Last sentence

    assert sents >= 40, f"Expected >= 40 sentences, got {sents}"
    assert 350 <= tokens <= 500, f"Expected 350-500 tokens, got {tokens}"


def test_table_5_indomain_benchmark_results():
    """Verify Table 5 in-domain benchmark shows expected generalization and superiority of M3."""
    with open(TABLE_5_JSON, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    models = bench_data["models"]
    assert "M0_Lookup" in models
    assert "M3_Final_Context" in models

    acc_m0 = models["M0_Lookup"]["accuracy"]
    acc_m3 = models["M3_Final_Context"]["accuracy"]

    assert acc_m0 > 0.80, f"M0 accuracy unexpectedly low: {acc_m0}"
    assert acc_m3 > 0.85, f"M3 accuracy unexpectedly low: {acc_m3}"
    assert acc_m3 >= acc_m0, f"M3 should outperform M0 on in-domain: {acc_m3} vs {acc_m0}"
