"""
tests/test_intent_experiments.py
Unit tests for Module M3.1 & M3.2: Downstream Intent Experiments.
Verifies Table 7 output files, metric schemas, model serializations, and statistical consistency.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import pytest


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CSV_PATH = RESULTS_DIR / "table_7_intent_experiments.csv"
JSON_PATH = RESULTS_DIR / "table_7_intent_experiments.json"
MODEL_PATH = MODELS_DIR / "intent_classifier_setup_c.joblib"


def test_table_7_csv_structure_and_folds():
    """Verify Table 7 CSV exists, has 15 evaluated folds, and valid probability ranges."""
    assert CSV_PATH.exists(), f"Missing {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 15, f"Expected 15 folds, found {len(df)}"

    expected_cols = [
        "split_id",
        "repeat",
        "fold",
        "n_train",
        "n_val",
        "Setup_A_Raw_Word_acc",
        "Setup_A_Raw_Word_f1",
        "Setup_B_Generic_Char_acc",
        "Setup_B_Generic_Char_f1",
        "Setup_C_LIDAware_Char_acc",
        "Setup_C_LIDAware_Char_f1",
        "Setup_C_Prime_LIDAware_Word_acc",
        "Setup_C_Prime_LIDAware_Word_f1",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column {col}"

    metric_cols = [
        "Setup_A_Raw_Word_acc",
        "Setup_A_Raw_Word_f1",
        "Setup_B_Generic_Char_acc",
        "Setup_B_Generic_Char_f1",
        "Setup_C_LIDAware_Char_acc",
        "Setup_C_LIDAware_Char_f1",
        "Setup_C_Prime_LIDAware_Word_acc",
        "Setup_C_Prime_LIDAware_Word_f1",
    ]
    for col in metric_cols:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Values in {col} outside [0, 1]"


def test_table_7_json_structure_and_metrics():
    """Verify Table 7 JSON schema, setups, and statistical comparisons."""
    assert JSON_PATH.exists(), f"Missing {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert data["metadata"]["folds_evaluated"] == 15
    assert data["metadata"]["dataset_size"] == 360

    assert "setups" in data
    setups = data["setups"]
    assert "Setup_A_Raw_Word" in setups
    assert "Setup_B_Generic_Char" in setups
    assert "Setup_C_LIDAware_Char" in setups
    assert "Setup_C_Prime_LIDAware_Word" in setups

    # Accuracy / F1 ranges
    for s_name, s_data in setups.items():
        assert 0.85 <= s_data["mean_acc"] <= 0.98, f"Unexpected mean_acc for {s_name}: {s_data['mean_acc']}"
        assert 0.85 <= s_data["mean_macro_f1"] <= 0.98, f"Unexpected mean_f1 for {s_name}: {s_data['mean_macro_f1']}"

    assert "statistical_comparisons" in data
    stats_comp = data["statistical_comparisons"]
    assert "Setup_Cp_minus_Setup_A" in stats_comp
    assert "Setup_C_minus_Setup_B" in stats_comp

    # Verify Setup C' vs A bootstrap 95% CI crosses zero
    cp_vs_a = stats_comp["Setup_Cp_minus_Setup_A"]
    ci_cp_a = cp_vs_a["bootstrap_95_ci"]
    assert ci_cp_a[0] < 0.0 < ci_cp_a[1], f"Expected CI to cross zero: {ci_cp_a}"

    # Verify Setup C vs B CI is strictly negative
    c_vs_b = stats_comp["Setup_C_minus_Setup_B"]
    ci_c_b = c_vs_b["bootstrap_95_ci"]
    assert ci_c_b[1] < 0.0, f"Expected CI to be entirely negative: {ci_c_b}"


def test_trained_classifier_setup_c_inference():
    """Verify production intent classifier model loads and produces valid predictions."""
    assert MODEL_PATH.exists(), f"Missing {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    valid_intents = {
        "refund",
        "cancel_order",
        "order_not_delivered",
        "delivery_delay",
        "payment_failed",
        "other",
    }

    test_queries = [
        "mera refund kab tak aayega please paise wapas karo",
        "order cancel kardo mujhe nahi chahiye",
        "paise cut gaye par order place nahi hua account se kat gaya",
        "parcel abhi tak deliver nahi hua delivery boy missing hai",
    ]

    preds = model.predict(test_queries)
    assert len(preds) == len(test_queries)
    for p in preds:
        assert p in valid_intents, f"Unexpected predicted intent {p}"

    probs = model.predict_proba(test_queries)
    assert probs.shape == (len(test_queries), 6)
    for row in probs:
        assert abs(sum(row) - 1.0) < 1e-4
