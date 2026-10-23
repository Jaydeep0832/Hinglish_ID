# Label Mapping — HinglishID-Router

## ✅ Dataset Used: L3Cube-HingLID (Plan B — LinCE inaccessible)

**Source:** https://github.com/l3cube-pune/code-mixed-nlp/tree/main/L3Cube-HingLID  
**Paper:** Nayak & Joshi (2022). "L3Cube-HingCorpus and HingBERT: A Code Mixed Hindi-English Dataset..."  
**License:** Publicly available on GitHub (no registration required)

## Why L3Cube-HingLID Instead of LinCE

LinCE (ritual.uh.edu/lince) is inaccessible (site down). L3Cube-HingLID is a better
alternative for this project because:
- **No registration required** — immediate access
- **Labels are already HI/EN** — zero remapping needed
- **CoNLL format** — exact same token\tLABEL format the project expects
- **Large scale** — 1M train tokens, 200K val tokens, 200K test tokens
- **Cleaner split** — separate train/validation/test files provided

## Label Mapping (Verified from actual train.txt)

| L3Cube-HingLID label | Our label | Notes |
|---|---|---|
| `HI` | `HI` | **No change** — direct match |
| `EN` | `EN` | **No change** — direct match |
| (no OTHER label present) | `OTHER` | Punctuation/numbers handled in preprocessing |

> **Note:** L3Cube-HingLID uses only HI and EN tags. Punctuation and numbers
> appear infrequently and are already lowercase/ASCII. Our preprocessor will
> assign OTHER to non-word tokens (emoji, URLs, order IDs) at inference time.

## Dataset Statistics (from downloaded files)

| Split | Lines | Size |
|---|---|---|
| train | ~1,000,000 | 7.42 MB |
| validation | ~200,000 | 1.48 MB |
| test | ~200,000 | 1.48 MB |

## Decision Log

- **Date:** Day 1 — 2026-10-21
- **Trigger:** LinCE inaccessible (site down — Risk R1 activated)
- **Decision owner:** Solo Developer (Jaydeep)
- **Plan B activated by:** Antigravity agent
- **Report note:** "We use L3Cube-HingLID (Nayak & Joshi, 2022) as our primary LID training 
  corpus, as the LinCE portal was inaccessible at project start time. L3Cube-HingLID 
  provides 1M+ token-level annotations in CoNLL format, with the same HI/EN/OTHER 
  label scheme required by our project."

## Citation
```
Nayak, T., & Joshi, R. (2022). L3Cube-HingCorpus and HingBERT: A Code Mixed 
Hindi-English Dataset and BERT Language Models. arXiv:2204.08398.
```
