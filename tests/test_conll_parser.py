"""
tests/test_conll_parser.py
Comprehensive test suite for src/conll_parser.py (M1.2)
Run with: pytest tests/test_conll_parser.py
"""

from pathlib import Path
import pytest

from src.conll_parser import (
    calculate_cmi,
    compute_corpus_stats,
    is_code_mixed,
    read_conll,
    split_sentences,
    write_conll,
)


class TestConllIO:
    """Tests for reading and writing CoNLL formatted files."""

    def test_read_conll_basic(self, tmp_path: Path):
        test_file = tmp_path / "sample.conll"
        content = "mera\tHI\norder\tEN\nkahan\tHI\nhai\tHI\n\nplease\tEN\nhelp\tEN\n"
        test_file.write_text(content, encoding="utf-8")

        sents = read_conll(test_file)
        assert len(sents) == 2
        assert sents[0] == (
            ["mera", "order", "kahan", "hai"],
            ["HI", "EN", "HI", "HI"],
        )
        assert sents[1] == (["please", "help"], ["EN", "EN"])

    def test_read_conll_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_conll("non_existent_file_path_12345.conll")

    def test_read_conll_malformed_line(self, tmp_path: Path):
        test_file = tmp_path / "bad.conll"
        test_file.write_text("singletokenwithoutlabel\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed line"):
            read_conll(test_file)

    def test_write_conll_and_roundtrip(self, tmp_path: Path):
        test_file = tmp_path / "output.conll"
        original_sents = [
            (["kya", "refund", "milega"], ["HI", "EN", "HI"]),
            (["yes", "sure"], ["EN", "EN"]),
        ]
        write_conll(original_sents, test_file)

        assert test_file.exists()
        reloaded = read_conll(test_file)
        assert reloaded == original_sents

    def test_write_conll_length_mismatch(self, tmp_path: Path):
        test_file = tmp_path / "mismatch.conll"
        with pytest.raises(ValueError, match="Mismatched tokens"):
            write_conll([(["word1", "word2"], ["HI"])], test_file)


class TestCMI:
    """Tests for Code-Mixing Index (CMI) calculation."""

    def test_cmi_monolingual_hindi(self):
        labels = ["HI", "HI", "HI", "HI"]
        assert calculate_cmi(labels) == 0.0

    def test_cmi_monolingual_english(self):
        labels = ["EN", "EN", "EN"]
        assert calculate_cmi(labels) == 0.0

    def test_cmi_perfectly_balanced(self):
        labels = ["HI", "EN"]
        # n=2, u=0, max(1, 1)=1 -> 100 * (1 - 1/2) = 50.0
        assert calculate_cmi(labels) == 50.0

    def test_cmi_with_other_label(self):
        labels = ["HI", "HI", "EN", "OTHER"]
        # n=4, u=1, valid=3, max(2, 1)=2 -> 100 * (1 - 2/3) = 33.333...
        cmi = calculate_cmi(labels)
        assert pytest.approx(cmi, 0.01) == 33.33

    def test_cmi_empty_and_only_other(self):
        assert calculate_cmi([]) == 0.0
        assert calculate_cmi(["OTHER", "OTHER"]) == 0.0


class TestCodeMixedFlag:
    """Tests for code-mixed sentence detection."""

    def test_is_code_mixed_true(self):
        assert is_code_mixed(["HI", "EN"]) is True
        assert is_code_mixed(["HI", "OTHER", "EN"]) is True

    def test_is_code_mixed_false(self):
        assert is_code_mixed(["HI", "HI", "OTHER"]) is False
        assert is_code_mixed(["EN", "EN"]) is False
        assert is_code_mixed(["OTHER"]) is False
        assert is_code_mixed([]) is False


class TestCorpusStats:
    """Tests for corpus statistics computation."""

    def test_empty_corpus(self):
        stats = compute_corpus_stats([])
        assert stats["num_sentences"] == 0
        assert stats["num_tokens"] == 0
        assert stats["avg_sentence_length"] == 0.0

    def test_simple_corpus_stats(self):
        corpus = [
            (["mera", "order"], ["HI", "EN"]),
            (["kahan", "hai"], ["HI", "HI"]),
        ]
        stats = compute_corpus_stats(corpus)
        assert stats["num_sentences"] == 2
        assert stats["num_tokens"] == 4
        assert stats["avg_sentence_length"] == 2.0
        assert stats["min_sentence_length"] == 2
        assert stats["max_sentence_length"] == 2
        assert stats["label_counts"]["HI"] == 3
        assert stats["label_counts"]["EN"] == 1
        assert stats["num_code_mixed_sentences"] == 1
        assert stats["pct_code_mixed_sentences"] == 50.0


class TestSplitSentences:
    """Tests for deterministic sentence splitting."""

    def test_split_proportions(self):
        data = list(range(100))
        train, dev, test = split_sentences(
            data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=42
        )
        assert len(train) == 80
        assert len(dev) == 10
        assert len(test) == 10

    def test_no_data_leakage(self):
        data = list(range(100))
        train, dev, test = split_sentences(data, seed=42)
        set_train = set(train)
        set_dev = set(dev)
        set_test = set(test)
        assert len(set_train & set_dev) == 0
        assert len(set_train & set_test) == 0
        assert len(set_dev & set_test) == 0
        assert len(set_train | set_dev | set_test) == 100

    def test_deterministic_seed(self):
        data = list(range(50))
        t1, d1, s1 = split_sentences(data, seed=42)
        t2, d2, s2 = split_sentences(data, seed=42)
        assert t1 == t2
        assert d1 == d2
        assert s1 == s2

    def test_invalid_ratios(self):
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_sentences(list(range(10)), 0.7, 0.1, 0.1)
