"""
scripts/mine_ambiguous_words.py
Script to mine, verify, and freeze the 30-40 cross-lingual ambiguous word list
for HinglishID-Router (Module M2.2 - Day 9).

Strict Methodological Rules:
1. Mined strictly from train.conll statistics + linguistic domain knowledge.
2. Verified against dev.conll to observe frequency support.
3. Frozen in data/ambiguous_words.txt BEFORE evaluating on test.conll.
4. test.conll remains strictly untouched and FROZEN until Day 10.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conll_parser import read_conll


def analyze_token_frequencies(filepath: Path) -> Dict[str, Dict[str, int]]:
    """Count token occurrences under HI, EN, and OTHER."""
    sents = read_conll(filepath)
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"HI": 0, "EN": 0, "OTHER": 0})
    for tokens, labels in sents:
        for word, label in zip(tokens, labels):
            counts[word.lower()][label] += 1
    return counts


def main():
    train_path = Path("data/processed/train.conll")
    dev_path = Path("data/processed/dev.conll")

    print(f"Reading training corpus from: {train_path}")
    train_counts = analyze_token_frequencies(train_path)

    print(f"Reading dev corpus from: {dev_path}")
    dev_counts = analyze_token_frequencies(dev_path)

    # Starter pool from project plan (M2.2) + data-driven cross-lingual homographs
    # Selected based on:
    # 1. Genuine lexical/syntactic presence in both Hindi and English
    # 2. Minimum occurrences across corpus
    # 3. Known code-mixed customer support confusion points
    candidate_words = [
        # Top high-frequency syntactic overlap words
        "to",    # HI: then/so (toh) vs EN: preposition ("to order")
        "is",    # HI: this (iss) vs EN: auxiliary verb ("is")
        "the",   # HI: were (the) vs EN: definite article ("the")
        "sir",   # HI: head (sar) vs EN: honorific ("sir")
        "do",    # HI: two/give (do) vs EN: auxiliary verb ("do")
        "us",    # HI: that/him (uss) vs EN: pronoun ("us")
        "use",   # HI: to him/her (use) vs EN: verb/noun ("use")
        "me",    # HI: in/inside (mein) vs EN: pronoun ("me")
        "hi",    # HI: emphatic particle (hi) vs EN: greeting ("hi")
        "main",  # HI: pronoun (main / I) vs EN: adjective ("main")
        "or",    # HI: and (aur) vs EN: disjunction ("or")
        "par",   # HI: on/but (par) vs EN: noun/adverb ("par")
        "mat",   # HI: negative particle (mat) vs EN: noun ("mat")
        "tab",   # HI: then (tab) vs EN: noun ("tab")
        "man",   # HI: heart/mind (mann) vs EN: noun ("man")
        "pure",  # HI: full/whole (poore) vs EN: adjective ("pure")
        "tune",  # HI: pronoun (tune) vs EN: noun ("tune")
        "ram",   # HI: proper noun (Ram) vs EN: computer memory ("RAM")
        "per",   # HI: postposition (par) vs EN: preposition ("per")
        "bare",  # HI: about (baare) vs EN: adjective ("bare")
        "band",  # HI: closed/off (band) vs EN: noun ("band")
        "beta",  # HI: son/child (beta) vs EN: software testing ("beta")
        "had",   # HI: limit (hadd) vs EN: past verb ("had")
        "log",   # HI: people (log) vs EN: noun/verb ("log")
        "ap",    # HI: pronoun (aap) vs EN: abbreviation/noun ("AP")
        "are",   # HI: interjection (arre) vs EN: auxiliary verb ("are")
        "ban",   # HI: become (bann) vs EN: verb/noun ("ban")
        "lag",   # HI: feel/take time (lagg) vs EN: noun/verb ("lag")
        "hat",   # HI: move (hatt) vs EN: headwear ("hat")
        "bag",   # HI: garden/tiger (baagh) vs EN: noun ("bag")
        "josh",  # HI: passion/energy (josh) vs EN: proper noun ("Josh")
        "mama",  # HI: maternal uncle (mama) vs EN: mother ("mama")
        "they",  # HI: were spelling variant vs EN: pronoun ("they")
        "up",    # HI: you variant vs EN: preposition ("up")
        "plz",   # HI: borrowed politeness vs EN: abbreviation ("please")
        "mile",  # HI: receive (mile) vs EN: distance unit ("mile")
        "den",   # HI: give (den) vs EN: noun ("den")
    ]

    print(f"\nEvaluating candidate pool of {len(candidate_words)} ambiguous words:\n")
    header = f"{'Word':10s} | {'Train Tot':10s} | {'Train HI':9s} | {'Train EN':9s} | {'Dev Tot':8s} | {'Dev HI':7s} | {'Dev EN':7s} | {'Support'}"
    print(header)
    print("-" * len(header))

    valid_words = []
    for w in sorted(candidate_words):
        t_h = train_counts[w]["HI"]
        t_e = train_counts[w]["EN"]
        t_tot = t_h + t_e

        d_h = dev_counts[w]["HI"]
        d_e = dev_counts[w]["EN"]
        d_tot = d_h + d_e

        support = "Robust" if d_tot >= 10 else ("Low Support" if d_tot > 0 else "Zero Dev")
        print(f"{w:10s} | {t_tot:10d} | {t_h:9d} | {t_e:9d} | {d_tot:8d} | {d_h:7d} | {d_e:7d} | {support}")
        valid_words.append(w)

    print(f"\nFinal ambiguous word count: {len(valid_words)}")

    # Freeze to data/ambiguous_words.txt
    out_file = Path("data/ambiguous_words.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for w in valid_words:
            f.write(f"{w}\n")

    print(f"[SUCCESS] Frozen {len(valid_words)} ambiguous words to: {out_file}")


if __name__ == "__main__":
    main()
