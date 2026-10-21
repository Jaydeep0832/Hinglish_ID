"""
tests/test_baseline_tagger.py
Unit tests for WordLookupTagger (Module M1.4).
Run with: pytest tests/test_baseline_tagger.py -v
"""

from pathlib import Path
import pytest

from src.baseline_tagger import WordLookupTagger


@pytest.fixture
def sample_corpus():
    """Toy corpus with known majority labels, ambiguous words, and class imbalance."""
    train_data = [
        (["mera", "order", "kahan", "hai"], ["HI", "EN", "HI", "HI"]),
        (["please", "refund", "karo"], ["EN", "EN", "HI"]),
        (["order", "cancel", "nahi", "hua"], ["EN", "EN", "HI", "HI"]),
        (["to", "kya", "hua"], ["HI", "HI", "HI"]),
        (["go", "to", "cart"], ["EN", "EN", "EN"]),  # 'to' has 1 HI and 1 EN, let's add one more
        (["to", "bolo"], ["HI", "HI"]),  # 'to' now has 2 HI, 1 EN -> majority HI
    ]
    dev_data = [
        (["mera", "order", "cancel", "karo"], ["HI", "EN", "EN", "HI"]),
        (["unseen_word_123", "please", "to"], ["EN", "EN", "EN"]),
    ]
    return train_data, dev_data


class TestWordLookupTagger:
    """Test suite for WordLookupTagger."""

    def test_fit_and_vocab_size(self, sample_corpus):
        train_data, _ = sample_corpus
        tagger = WordLookupTagger()
        tagger.fit(train_data)

        assert tagger.is_fitted is True
        assert tagger.vocab_size > 0
        assert "mera" in tagger.lookup_dict
        assert "order" in tagger.lookup_dict
        assert tagger.default_label == "HI"  # HI has highest count in train_data

    def test_majority_voting(self, sample_corpus):
        train_data, _ = sample_corpus
        tagger = WordLookupTagger()
        tagger.fit(train_data)

        assert tagger.predict_token("order") == "EN"
        assert tagger.predict_token("mera") == "HI"
        assert tagger.predict_token("to") == "HI"

    def test_case_insensitivity(self, sample_corpus):
        train_data, _ = sample_corpus
        tagger = WordLookupTagger()
        tagger.fit(train_data)

        assert tagger.predict_token("ORDER") == "EN"
        assert tagger.predict_token("Mera") == "HI"

    def test_oov_handling(self, sample_corpus):
        train_data, _ = sample_corpus
        tagger = WordLookupTagger()
        tagger.fit(train_data)

        assert tagger.is_oov("unseen_word_123") is True
        assert tagger.predict_token("unseen_word_123") == "HI"
        assert tagger.is_oov("order") is False

    def test_predict_sentence(self, sample_corpus):
        train_data, _ = sample_corpus
        tagger = WordLookupTagger()
        tagger.fit(train_data)

        preds = tagger.predict_sentence(["mera", "order", "unseen_token"])
        assert preds == ["HI", "EN", "HI"]

    def test_evaluation_metrics(self, sample_corpus):
        train_data, dev_data = sample_corpus
        tagger = WordLookupTagger()
        tagger.fit(train_data)

        metrics = tagger.evaluate(dev_data)
        assert metrics["total_tokens"] == 7
        assert metrics["oov_tokens"] == 1  # 'unseen_word_123'
        assert pytest.approx(metrics["oov_rate"], 0.01) == 0.1429
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "per_class" in metrics
        assert "HI" in metrics["per_class"]
        assert "EN" in metrics["per_class"]
        assert metrics["in_vocab_errors"] >= 0
        assert metrics["oov_errors"] >= 0

    def test_unfitted_model_raises_error(self, sample_corpus):
        _, dev_data = sample_corpus
        tagger = WordLookupTagger()
        with pytest.raises(RuntimeError, match="fitted before evaluation"):
            tagger.evaluate(dev_data)

    def test_save_and_load(self, sample_corpus, tmp_path: Path):
        train_data, _ = sample_corpus
        tagger = WordLookupTagger()
        tagger.fit(train_data)

        model_path = tmp_path / "model.joblib"
        tagger.save(model_path)
        assert model_path.exists()

        loaded_tagger = WordLookupTagger.load(model_path)
        assert loaded_tagger.vocab_size == tagger.vocab_size
        assert loaded_tagger.predict_token("order") == "EN"
        assert loaded_tagger.predict_token("mera") == "HI"
