"""
tests/test_lid_tagger.py
Unit tests for CharNgramLIDTagger (Module M1.5).
Run with: pytest tests/test_lid_tagger.py -v
"""

from pathlib import Path
import pytest

from src.lid_tagger import CharNgramLIDTagger


@pytest.fixture
def toy_sentences():
    train_data = [
        (["mera", "order", "kahan", "hai"], ["HI", "EN", "HI", "HI"]),
        (["please", "refund", "karo"], ["EN", "EN", "HI"]),
        (["order", "cancel", "nahi", "hua"], ["EN", "EN", "HI", "HI"]),
        (["aap", "kaise", "ho"], ["HI", "HI", "HI"]),
        (["track", "my", "package"], ["EN", "EN", "EN"]),
    ]
    dev_data = [
        (["mera", "package", "deliver", "karo"], ["HI", "EN", "EN", "HI"]),
        (["plzzzzz", "nahiiii"], ["EN", "HI"]),  # OOV spelling variants
    ]
    return train_data, dev_data


class TestCharNgramLIDTagger:
    """Test suite for CharNgramLIDTagger."""

    def test_fit_and_attributes(self, toy_sentences):
        train_data, _ = toy_sentences
        tagger = CharNgramLIDTagger(model_type="multinomial_nb", alpha=0.5)
        tagger.fit(train_data)

        assert tagger.is_fitted is True
        assert len(tagger.classes_) == 2
        assert "HI" in tagger.classes_
        assert "EN" in tagger.classes_
        assert len(tagger.train_vocab_) > 0

    def test_predict_token_known(self, toy_sentences):
        train_data, _ = toy_sentences
        tagger = CharNgramLIDTagger()
        tagger.fit(train_data)

        assert tagger.predict_token("order") == "EN"
        assert tagger.predict_token("kahan") == "HI"

    def test_oov_generalization_spelling_variants(self, toy_sentences):
        """Char n-grams should generalize across OOV elongations and morphological variants."""
        train_data, _ = toy_sentences
        tagger = CharNgramLIDTagger(model_type="multinomial_nb", alpha=0.1)
        tagger.fit(train_data)

        # 'plzzzzz' shares char n-grams with 'please'
        # 'nahiiii' shares char n-grams with 'nahi'
        pred_plz = tagger.predict_token("plzzzzz")
        pred_nahi = tagger.predict_token("nahiiii")
        assert pred_plz == "EN"
        assert pred_nahi == "HI"

    def test_predict_proba(self, toy_sentences):
        train_data, _ = toy_sentences
        tagger = CharNgramLIDTagger()
        tagger.fit(train_data)

        probs = tagger.predict_proba_token("order")
        assert "HI" in probs
        assert "EN" in probs
        assert pytest.approx(probs["HI"] + probs["EN"], 0.01) == 1.0

    def test_predict_sentence(self, toy_sentences):
        train_data, _ = toy_sentences
        tagger = CharNgramLIDTagger()
        tagger.fit(train_data)

        preds = tagger.predict_sentence(["mera", "order"])
        assert len(preds) == 2
        assert preds[0] == "HI"
        assert preds[1] == "EN"

    def test_evaluate_structure(self, toy_sentences):
        train_data, dev_data = toy_sentences
        tagger = CharNgramLIDTagger(alpha=0.5)
        tagger.fit(train_data)

        metrics = tagger.evaluate(dev_data)
        assert metrics["total_tokens"] == 6
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "in_vocab_errors" in metrics
        assert "oov_errors" in metrics

    def test_logistic_regression_variant(self, toy_sentences):
        train_data, dev_data = toy_sentences
        tagger = CharNgramLIDTagger(model_type="logistic_regression", c_param=1.0)
        tagger.fit(train_data)

        assert tagger.is_fitted is True
        preds = tagger.predict_sentence(["mera", "order"])
        assert len(preds) == 2
        metrics = tagger.evaluate(dev_data)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics

    def test_save_and_load(self, toy_sentences, tmp_path: Path):
        train_data, _ = toy_sentences
        tagger = CharNgramLIDTagger(model_type="multinomial_nb", alpha=0.5)
        tagger.fit(train_data)

        model_path = tmp_path / "lid_model.joblib"
        tagger.save(model_path)
        assert model_path.exists()

        loaded_tagger = CharNgramLIDTagger.load(model_path)
        assert loaded_tagger.model_type == "multinomial_nb"
        assert loaded_tagger.alpha == 0.5
        assert loaded_tagger.predict_token("order") == tagger.predict_token("order")
