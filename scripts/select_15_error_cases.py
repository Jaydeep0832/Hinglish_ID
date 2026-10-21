"""
scripts/select_15_error_cases.py
Selects and formats 15 concrete, diverse, and representative error cases across
linguistic error categories for docs/error_analysis.md.
"""

from collections import defaultdict
from pathlib import Path
import pandas as pd


def select_cases():
    df = pd.read_csv("results/dev_errors_raw.csv")

    # Categories to populate:
    # 1. Ambiguous Homographs (e.g. 'to', 'is', 'do', 'pure', 'tune')
    # 2. Romanized Hindi Spelling Variations (e.g. 'jahar', 'riksha', 'jmeen')
    # 3. Named Entities / Brand / Proper Nouns (e.g. 'Saudia', 'PMOP', 'Divyang')
    # 4. English Out-Of-Vocabulary Technical / Rare Words (e.g. 'gallantry', 'specialist')
    # 5. Social Media Slang / Abbreviations / Interjections (e.g. 'cch', 'yrr', 'plzz')
    # 6. Morphological Affixation / Hybrid Code-Mixing (e.g. Hindi verb root + English suffix)
    # 7. Annotation Inconsistency / Corpus Noise

    print(f"Total dev errors available: {len(df)}")

    # Specific selections to ensure diverse representation:
    target_tokens = [
        # Ambiguous Homographs
        ("to", "Ambiguous Cross-Lingual Homograph"),
        ("is", "Ambiguous Cross-Lingual Homograph"),
        ("do", "Ambiguous Cross-Lingual Homograph"),
        ("pure", "Ambiguous Cross-Lingual Homograph"),
        ("tune", "Ambiguous Cross-Lingual Homograph"),
        # Spelling Variants in Romanized Hindi
        ("jahar", "Romanized Hindi Spelling Variation"),
        ("riksha", "Romanized Hindi Spelling Variation"),
        ("jmeen", "Romanized Hindi Spelling Variation"),
        # Named Entities & Proper Nouns
        ("saudia", "Brand Name / Named Entity"),
        ("divyang", "Domain-Specific Named Entity"),
        ("pmop", "Acronym / Political Named Entity"),
        # English OOV Vocabulary
        ("gallantry", "English Out-of-Vocabulary (OOV)"),
        # Slang & Informal Internet Orthography
        ("cch", "Informal Slang / Phonetic Exclamation"),
        ("yrr", "Colloquial Code-Mixed Slang"),
        # Corpus Annotation Inconsistency
        ("beta", "Ambiguous Kinship Noun / Annotation Inconsistency"),
    ]

    selected_rows = []
    seen_tokens = set()

    for tok_target, cat in target_tokens:
        sub = df[df["token_lower"] == tok_target]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            row["category"] = cat
            selected_rows.append(row)
            seen_tokens.add(tok_target)

    # If fewer than 15, fill from remaining
    if len(selected_rows) < 15:
        remaining = df[~df["token_lower"].isin(seen_tokens)]
        for _, r in remaining.head(15 - len(selected_rows)).iterrows():
            row = r.to_dict()
            row["category"] = "General Lexical Misclassification"
            selected_rows.append(row)

    df_selected = pd.DataFrame(selected_rows)
    print("\nSelected 15 Cases:")
    for i, r in df_selected.iterrows():
        print(f"#{i+1:02d} [{r['category']}] Word: '{r['token']}' (Gold: {r['gold_label']} -> Pred: {r['pred_label']})")
        print(f"    Snippet: \"... {r['context_snippet']} ...\"")
        print(f"    Confidence: {r['confidence']:.3f} | Gold Prob: {r['gold_prob']:.3f}")

    df_selected.to_json("results/selected_15_errors.json", orient="records", indent=2)
    print("\nSaved selected cases to results/selected_15_errors.json")


if __name__ == "__main__":
    select_cases()
