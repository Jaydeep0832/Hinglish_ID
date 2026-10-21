"""
tests/test_demo_tagger.py
Unit tests for HinglishDemoTagger and query tagging interface (Module M1.7).
Run with: pytest tests/test_demo_tagger.py -v
"""

from pathlib import Path
import pytest

from src.demo_tagger import HinglishDemoTagger


@pytest.fixture
def demo_tagger():
    # Use trained logistic regression model if available, otherwise test fallback
    model_path = Path("models/lid_lr.joblib")
    if model_path.is_file():
        return HinglishDemoTagger(model_path=str(model_path))
    return HinglishDemoTagger(model_path=None)


def test_tag_query_structure(demo_tagger):
    query = "Mera order deliver nahi hua please help"
    res = demo_tagger.tag_query(query)

    assert "tokens" in res
    assert "tags" in res
    assert "token_details" in res
    assert "cmi" in res
    assert "is_code_mixed" in res
    assert len(res["tokens"]) == len(res["tags"])
    assert res["language_distribution"]["HI"] > 0
    assert res["language_distribution"]["EN"] > 0
    assert res["is_code_mixed"] is True


def test_tag_query_placeholders(demo_tagger):
    query = "Order #12345 track karo on https://track.com 😡"
    res = demo_tagger.tag_query(query)

    tags = res["tags"]
    tokens = res["tokens"]

    # Placeholders <ORDER_ID>, <URL>, <EMOJI> must be tagged as OTHER
    assert "<ORDER_ID>" in tokens
    assert "<URL>" in tokens
    assert "<EMOJI>" in tokens

    oid_idx = tokens.index("<ORDER_ID>")
    url_idx = tokens.index("<URL>")
    emoji_idx = tokens.index("<EMOJI>")

    assert tags[oid_idx] == "OTHER"
    assert tags[url_idx] == "OTHER"
    assert tags[emoji_idx] == "OTHER"


def test_tag_query_monolingual_cmi(demo_tagger):
    query = "Please cancel my order immediately"
    res = demo_tagger.tag_query(query)

    assert res["language_distribution"]["HI"] == 0
    assert res["cmi"] == 0.0
    assert res["is_code_mixed"] is False


def test_tag_query_empty_string(demo_tagger):
    res = demo_tagger.tag_query("")
    assert res["tokens"] == []
    assert res["tags"] == []
    assert res["cmi"] == 0.0


def test_format_colored_tokens(demo_tagger):
    res = demo_tagger.tag_query("refund karo")
    colored = demo_tagger.format_colored_tokens(res)
    assert "[EN]" in colored or "[HI]" in colored


def test_format_html_pills(demo_tagger):
    res = demo_tagger.tag_query("refund karo")
    html = demo_tagger.format_html_pills(res)
    assert "<span" in html
    assert "refund" in html
