"""
src/conll_parser.py
Modular CoNLL-2003 / CoNLL-style data parser, serializer, and corpus statistics engine
for Hinglish LID (Language Identification).

Conforms to HinglishID-Router specification:
- Labels: HI, EN, OTHER
- CoNLL format: token<TAB>label, blank line = sentence boundary
- CMI formula: Gambäck & Das (2014) / Das & Gambäck (2014)
"""

from collections import Counter
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Standard label definitions
VALID_LABELS = {"HI", "EN", "OTHER"}


def read_conll(
    filepath: Union[str, Path],
    encoding: str = "utf-8",
    delimiter: str = "\t",
) -> List[Tuple[List[str], List[str]]]:
    """
    Parse a CoNLL-format file where each line is 'token<delimiter>label'
    and blank lines delimit sentences.

    Args:
        filepath: Path to the CoNLL file.
        encoding: File encoding (default 'utf-8').
        delimiter: Column separator (default '\\t').

    Returns:
        List of tuples: (tokens, labels) for each sentence.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If a non-blank line cannot be split by delimiter.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"CoNLL file not found at: {path}")

    sentences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\r\n")

            # Blank line marks sentence boundary
            if not line.strip():
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue

            parts = line.split(delimiter)
            if len(parts) < 2:
                # Fallback to whitespace split if tab split fails
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"Malformed line {line_num} in {path.name}: expected "
                        f"'token{delimiter}label', got '{line}'"
                    )

            token, label = parts[0], parts[1]
            tokens.append(token)
            labels.append(label)

    # Append any remaining sentence without trailing newline
    if tokens:
        sentences.append((tokens, labels))

    return sentences


def write_conll(
    sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    filepath: Union[str, Path],
    encoding: str = "utf-8",
    delimiter: str = "\t",
) -> None:
    """
    Write sentences to a CoNLL-format file.

    Args:
        sentences: Sequence of (tokens, labels) tuples.
        filepath: Output file path.
        encoding: File encoding (default 'utf-8').
        delimiter: Separator between token and label (default '\\t').

    Raises:
        ValueError: If tokens and labels lengths mismatch in any sentence.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding=encoding, newline="\n") as f:
        for tokens, labels in sentences:
            if len(tokens) != len(labels):
                raise ValueError(
                    f"Mismatched tokens ({len(tokens)}) and labels ({len(labels)})"
                )
            for token, label in zip(tokens, labels):
                f.write(f"{token}{delimiter}{label}\n")
            f.write("\n")


def calculate_cmi(labels: Sequence[str]) -> float:
    """
    Compute Code-Mixing Index (CMI) per Gambäck & Das (2014) / Das & Gambäck (2014):

        CMI = 100 * [ 1 - (max(N_lang1, N_lang2, ...) / (N - N_other)) ]

    For bilingual Hindi-English text:
        u = count of OTHER (punctuation, symbols, etc.)
        n = total tokens
        If n == u or n == 0: CMI = 0.0
        CMI = 100.0 * (1.0 - max(HI, EN) / (n - u))

    CMI = 0 represents a monolingual sentence.
    CMI = 50 represents a maximally balanced bilingual sentence.

    Args:
        labels: List of label strings for a single sentence.

    Returns:
        float: CMI value between 0.0 and 100.0 (typically 0.0 to 50.0 for 2 languages).
    """
    n = len(labels)
    if n == 0:
        return 0.0

    u = sum(1 for lab in labels if lab == "OTHER")
    valid_tokens = n - u
    if valid_tokens <= 0:
        return 0.0

    hi = sum(1 for lab in labels if lab == "HI")
    en = sum(1 for lab in labels if lab == "EN")

    max_lang = max(hi, en)
    return 100.0 * (1.0 - (max_lang / valid_tokens))


def is_code_mixed(labels: Sequence[str]) -> bool:
    """
    Check if a sentence is code-mixed (contains at least one 'HI' and one 'EN' token).

    Args:
        labels: Sequence of token labels in the sentence.

    Returns:
        bool: True if both 'HI' and 'EN' are present, False otherwise.
    """
    has_hi = any(lab == "HI" for lab in labels)
    has_en = any(lab == "EN" for lab in labels)
    return has_hi and has_en


def compute_corpus_stats(
    sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for a parsed corpus.

    Args:
        sentences: Sequence of (tokens, labels) tuples.
        top_k: Number of frequent tokens to return per label.

    Returns:
        Dict containing:
            - num_sentences: int
            - num_tokens: int
            - avg_sentence_length: float
            - min_sentence_length: int
            - max_sentence_length: int
            - label_counts: Dict[str, int]
            - label_shares: Dict[str, float] (fractions 0.0 to 1.0)
            - num_code_mixed_sentences: int
            - pct_code_mixed_sentences: float (percentage 0 to 100)
            - avg_cmi: float
            - cmi_values: List[float]
            - top_tokens_by_label: Dict[str, List[Tuple[str, int]]]
            - ambiguous_tokens: List[Tuple[str, int, int]] (token, hi_count, en_count)
    """
    num_sents = len(sentences)
    if num_sents == 0:
        return {
            "num_sentences": 0,
            "num_tokens": 0,
            "avg_sentence_length": 0.0,
            "min_sentence_length": 0,
            "max_sentence_length": 0,
            "label_counts": {},
            "label_shares": {},
            "num_code_mixed_sentences": 0,
            "pct_code_mixed_sentences": 0.0,
            "avg_cmi": 0.0,
            "cmi_values": [],
            "top_tokens_by_label": {},
            "ambiguous_tokens": [],
        }

    lengths: List[int] = []
    cmi_values: List[float] = []
    num_code_mixed = 0
    label_counter: Counter = Counter()
    tokens_by_label: Dict[str, Counter] = {}

    for tokens, labels in sentences:
        s_len = len(tokens)
        lengths.append(s_len)

        for tok, lab in zip(tokens, labels):
            label_counter[lab] += 1
            if lab not in tokens_by_label:
                tokens_by_label[lab] = Counter()
            tokens_by_label[lab][tok.lower()] += 1

        sent_cmi = calculate_cmi(labels)
        cmi_values.append(sent_cmi)

        if is_code_mixed(labels):
            num_code_mixed += 1

    total_tokens = sum(lengths)
    avg_len = total_tokens / num_sents if num_sents else 0.0
    label_shares = {
        lab: (count / total_tokens) if total_tokens else 0.0
        for lab, count in label_counter.items()
    }
    pct_code_mixed = (num_code_mixed / num_sents * 100.0) if num_sents else 0.0
    avg_cmi = sum(cmi_values) / num_sents if num_sents else 0.0

    top_tokens = {
        lab: counter.most_common(top_k) for lab, counter in tokens_by_label.items()
    }

    # Find tokens that appear with both HI and EN tags (ambiguous token candidates)
    hi_tokens = tokens_by_label.get("HI", Counter())
    en_tokens = tokens_by_label.get("EN", Counter())
    overlap_words = set(hi_tokens.keys()) & set(en_tokens.keys())

    # Sort overlap words by total occurrences
    ambiguous_tokens = sorted(
        [
            (w, hi_tokens[w], en_tokens[w])
            for w in overlap_words
            if hi_tokens[w] >= 5 and en_tokens[w] >= 5
        ],
        key=lambda item: item[1] + item[2],
        reverse=True,
    )

    return {
        "num_sentences": num_sents,
        "num_tokens": total_tokens,
        "avg_sentence_length": round(avg_len, 2),
        "min_sentence_length": min(lengths) if lengths else 0,
        "max_sentence_length": max(lengths) if lengths else 0,
        "label_counts": dict(label_counter),
        "label_shares": {k: round(v, 4) for k, v in label_shares.items()},
        "num_code_mixed_sentences": num_code_mixed,
        "pct_code_mixed_sentences": round(pct_code_mixed, 2),
        "avg_cmi": round(avg_cmi, 2),
        "cmi_values": cmi_values,
        "top_tokens_by_label": top_tokens,
        "ambiguous_tokens": ambiguous_tokens[:top_k],
    }


def split_sentences(
    sentences: Sequence[Any],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split sentences into train, dev, and test subsets at the sentence level
    with a deterministic random seed.

    Args:
        sentences: Sequence of items (sentences) to split.
        train_ratio: Fraction of sentences for training (default 0.8).
        dev_ratio: Fraction of sentences for validation/dev (default 0.1).
        test_ratio: Fraction of sentences for test (default 0.1).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple of (train_sentences, dev_sentences, test_sentences).

    Raises:
        ValueError: If ratios do not sum to approximately 1.0 or are negative.
    """
    if abs((train_ratio + dev_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + dev_ratio + test_ratio}"
        )
    if train_ratio < 0 or dev_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative")

    sents_copy = list(sentences)
    rng = random.Random(seed)
    rng.shuffle(sents_copy)

    n = len(sents_copy)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train_data = sents_copy[:train_end]
    dev_data = sents_copy[train_end:dev_end]
    test_data = sents_copy[dev_end:]

    return train_data, dev_data, test_data
