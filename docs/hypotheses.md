# Hypotheses — HinglishID-Router
# Written BEFORE running any experiments (Day 8 deadline)
# DO NOT edit after Day 8.

## H1 — Char N-gram vs Word Lookup

**Hypothesis:** Char n-gram NB (M1) achieves higher macro-F1 on the dev set
than the word-lookup majority-label baseline (M0).

**Rationale:** Romanized Hindi has high spelling variation (nahi/nhi/nahin/nahii).
Character n-grams generalize across spelling variants, while lookup requires
exact match. OOV words (unseen in training) default to majority label in M0,
but n-gram features can still signal language from subword patterns.

**Expected direction:** M1 macro-F1 > M0 macro-F1

---

## H2 — Context Features on Ambiguous Words

**Hypothesis:** Adding neighbouring word features (M3) improves accuracy
on the ambiguous-word subset (30-40 words) compared to char-n-gram only (M1).

**Rationale:** Words like `is`, `to`, `me`, `hi`, `do`, `main`, `par` are valid in
both Hindi and English. Their correct tag depends on surrounding context.
A model with access to prev/next words can resolve these ambiguities.

**Expected direction:** M3 ambiguous-word accuracy > M1 ambiguous-word accuracy

---

## H3 — LID-aware Normalization Improves Intent Classification

**Hypothesis:** Setup C (LID-aware normalization → char_wb TF-IDF) achieves
higher macro-F1 on intent classification than Setup B (raw lowercase → char_wb TF-IDF).

**Rationale:** LID-aware processing:
- Removes language-specific stopwords (ka, ki, ke for HI; the, a, is for EN)
- Normalizes Hindi spelling variants to canonical form
- Applies English stemming only to EN tokens

This should produce a cleaner, more consistent feature space for the intent classifier.

**Expected direction:** mean macro-F1(C) ≥ mean macro-F1(B) ≥ mean macro-F1(A)

**Caveat:** char n-grams in B already absorb spelling variation through n-gram overlap.
Gain may be small or zero. If gain ≈ 0, run oracle-LID experiment to diagnose
whether the issue is LID errors propagating vs the idea being weak.

---

*If results do not match hypotheses, report honestly. Negative result + diagnosis
is acceptable. Do not re-run experiments until a hypothesis is confirmed.*
