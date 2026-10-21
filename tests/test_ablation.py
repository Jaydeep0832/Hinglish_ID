"""
tests/test_ablation.py
Unit tests for Module M1.6 Week-1 evaluation, ablation studies, and plotting functions.
Run with: pytest tests/test_ablation.py -v
"""

from pathlib import Path
import pytest

from src.evaluate_lid_ablation import (
    plot_single_confusion_matrix,
    plot_comparison_confusion_matrices,
    plot_oov_vs_invocab_accuracy,
    plot_ablation_curves,
)
from src.lid_tagger import CharNgramLIDTagger


@pytest.fixture
def toy_dataset():
    train = [
        (["mera", "order", "nahi", "aaya"], ["HI", "EN", "HI", "HI"]),
        (["please", "refund", "karo", "mera"], ["EN", "EN", "HI", "HI"]),
        (["kya", "update", "hai"], ["HI", "EN", "HI"]),
        (["good", "service", "bhai"], ["EN", "EN", "HI"]),
    ]
    dev = [
        (["mera", "refund", "nahi", "hua"], ["HI", "EN", "HI", "HI"]),
        (["urgent", "help", "chahiye"], ["EN", "EN", "HI"]),
    ]
    return train, dev


def test_fit_prior_ablation(toy_dataset):
    """Test that fit_prior=True and fit_prior=False both train and produce valid predictions."""
    train, dev = toy_dataset
    
    # Empirical prior
    tagger_emp = CharNgramLIDTagger(model_type="multinomial_nb", fit_prior=True)
    tagger_emp.fit(train)
    ev_emp = tagger_emp.evaluate(dev)
    assert ev_emp["accuracy"] >= 0.0
    assert "confusion_matrix" in ev_emp
    assert len(ev_emp["confusion_matrix"]) == 2

    # Uniform prior
    tagger_unif = CharNgramLIDTagger(model_type="multinomial_nb", fit_prior=False)
    tagger_unif.fit(train)
    ev_unif = tagger_unif.evaluate(dev)
    assert ev_unif["accuracy"] >= 0.0
    assert "confusion_matrix" in ev_unif


def test_plot_single_confusion_matrix(tmp_path: Path):
    """Verify single confusion matrix plotting produces a valid image file."""
    cm = [[100, 10], [5, 50]]
    labels = ["HI", "EN"]
    out_file = tmp_path / "test_cm.png"
    plot_single_confusion_matrix(cm, labels, "Test Matrix", out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_comparison_confusion_matrices(tmp_path: Path):
    """Verify side-by-side comparison plotting works."""
    cm1 = [[100, 10], [5, 50]]
    cm2 = [[95, 15], [8, 47]]
    models = [("Model A", cm1, ["HI", "EN"]), ("Model B", cm2, ["HI", "EN"])]
    out_file = tmp_path / "test_comp_cm.png"
    plot_comparison_confusion_matrices(models, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_oov_vs_invocab_accuracy(tmp_path: Path):
    """Verify In-Vocab vs OOV accuracy bar chart generates correctly."""
    models = [
        {"short_name": "M0", "in_vocab_accuracy": 0.98, "oov_accuracy": 0.76},
        {"short_name": "M1", "in_vocab_accuracy": 0.97, "oov_accuracy": 0.78},
    ]
    out_file = tmp_path / "test_oov_bar.png"
    plot_oov_vs_invocab_accuracy(models, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_ablation_curves(tmp_path: Path):
    """Verify sensitivity curves plot generation."""
    ngram_results = [
        {"hyperparams": "char_wb=(1, 2), alpha=0.1", "macro_f1": 0.85, "accuracy": 0.88},
        {"hyperparams": "char_wb=(1, 4), alpha=0.1", "macro_f1": 0.89, "accuracy": 0.91},
    ]
    alpha_results = [
        {"alpha": 0.1, "macro_f1": 0.89, "accuracy": 0.91},
        {"alpha": 1.0, "macro_f1": 0.87, "accuracy": 0.90},
    ]
    out_file = tmp_path / "test_ablation_curves.png"
    plot_ablation_curves(ngram_results, alpha_results, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
