"""
tests/test_context_tagger.py
Unit tests for ContextLIDTagger and context window feature extraction (Module M2.1).
Run with: pytest tests/test_context_tagger.py -v
"""

from pathlib import Path
import pytest

from src.lid_context import (
    ContextLIDTagger,
    extract_context_features_for_sentence,
)


@pytest.fixture
def sample_sentences():
    train = [
        (["mera", "order", "deliver", "karo"], ["HI", "EN", "EN", "HI"]),
        (["please", "give", "me", "refund"], ["EN", "EN", "EN", "EN"]),  # 'me' as English
        (["ye", "mere", "ko", "do"], ["HI", "HI", "HI", "HI"]),          # 'do' as Hindi
        (["please", "do", "it", "fast"], ["EN", "EN", "EN", "EN"]),      # 'do' as English
        (["kya", "karu", "main", "batao"], ["HI", "HI", "HI", "HI"]),
        (["mujhe", "bhi", "do"], ["HI", "HI", "HI"]),
        (["urgent", "help", "needed", "sir"], ["EN", "EN", "EN", "EN"]),
    ]
    dev = [
        (["please", "do", "refund"], ["EN", "EN", "EN"]),
        (["mujhe", "do", "kitabein"], ["HI", "HI", "HI"]),
        (["give", "me", "my", "money"], ["EN", "EN", "EN", "EN"]),
    ]
    return train, dev


def test_extract_context_features_boundary():
    tokens = ["karo", "to", "sahi"]
    feats = extract_context_features_for_sentence(tokens)

    assert len(feats) == 3

    # First token boundary
    assert feats[0]["p=<S>"] == 1
    assert feats[0]["w=karo"] == 1
    assert feats[0]["n=to"] == 1

    # Middle token
    assert feats[1]["p=karo"] == 1
    assert feats[1]["w=to"] == 1
    assert feats[1]["n=sahi"] == 1
    assert feats[1]["p_w=karo_to"] == 1
    assert feats[1]["w_n=to_sahi"] == 1

    # Last token boundary
    assert feats[2]["p=to"] == 1
    assert feats[2]["w=sahi"] == 1
    assert feats[2]["n=</S>"] == 1


def test_context_tagger_char_only(sample_sentences):
    train, dev = sample_sentences
    tagger = ContextLIDTagger(feature_mode="char_only", min_df_char=1, min_df_ctx=1)
    tagger.fit(train)

    assert tagger.is_fitted is True
    preds = tagger.predict_sentence(["please", "refund"])
    assert len(preds) == 2
    assert preds[0] == "EN"

    ev = tagger.evaluate(dev, ambiguous_words={"do", "me"})
    assert "accuracy" in ev
    assert "ambiguous_accuracy" in ev


def test_context_tagger_context_words(sample_sentences):
    train, dev = sample_sentences
    tagger = ContextLIDTagger(feature_mode="context_words", min_df_char=1, min_df_ctx=1)
    tagger.fit(train)

    assert tagger.is_fitted is True
    preds = tagger.predict_sentence(["mujhe", "do"])
    assert len(preds) == 2
    assert preds[0] == "HI"

    ev = tagger.evaluate(dev, ambiguous_words={"do", "me"})
    assert "ambiguous_word_stats" in ev
    assert "do" in ev["ambiguous_word_stats"]


def test_context_tagger_two_pass(sample_sentences):
    train, dev = sample_sentences
    tagger = ContextLIDTagger(feature_mode="context_plus_labels", min_df_char=1, min_df_ctx=1)
    tagger.fit(train)

    assert tagger.is_fitted is True
    assert tagger.first_pass_tagger is not None
    preds = tagger.predict_sentence(["please", "do", "it"])
    assert len(preds) == 3
    assert preds[0] == "EN"


def test_save_and_load(sample_sentences, tmp_path: Path):
    train, dev = sample_sentences
    tagger = ContextLIDTagger(feature_mode="context_words", min_df_char=1, min_df_ctx=1)
    tagger.fit(train)

    model_file = tmp_path / "context_tagger.joblib"
    tagger.save(model_file)
    assert model_file.exists()

    loaded = ContextLIDTagger.load(model_file)
    assert loaded.is_fitted is True
    assert loaded.predict_sentence(["refund"]) == tagger.predict_sentence(["refund"])
