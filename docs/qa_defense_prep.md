# HinglishID-Router: Technical Defense & Q&A Preparation

**10 Tough Technical Questions and Authoritative Answers for Project Defense**

---

### Question 1: Why use lightweight character n-grams and linear models instead of fine-tuning multilingual transformers like mBERT, XLM-RoBERTa, or IndicBERT?

**Answer:**
1. **Ultra-Low Latency & High Throughput:** In high-volume customer support operations (e.g., tier-1 e-commerce routing processing millions of messages daily), latency budgets are sub-10 milliseconds. Our character $n$-gram Logistic Regression pipeline runs end-to-end in **$< 2.5\text{ ms}$ per query** on standard commodity CPU.
2. **Zero GPU / Cloud Hardware Barrier:** Deep transformer models require dedicated GPU instances (e.g., NVIDIA T4/A10G) incurring substantial infrastructure costs. Our production model weighs **$< 300\text{ KB}$** and executes deterministically in memory without GPU acceleration.
3. **Subword Robustness to Novel Slang:** Large pre-trained multilingual language models frequently suffer from subword token fragmentation on novel Romanized transliterations (e.g., splitting *bahuuuut* into 5+ pieces), leading to embedding dilution. In contrast, character boundary $n$-grams (`char_wb` 1–4) naturally capture morphemic affixes (*-kar*, *-raha*, *-unga*) consistently.

---

### Question 2: What is your single weakest empirical result, and how do you defend it?

**Answer:**
**The Result:** The apples-to-apples word-level comparison between LID-aware normalization (Setup C') and the raw word baseline (Setup A) yielded a mean $\Delta \text{ Macro-F1} = -0.0076$ with a 95% Bootstrap Confidence Interval of **$[-0.0163, +0.0007]$** ($p = 0.117$). **The confidence interval crosses zero**, meaning the difference is statistically indistinguishable.

**Our Defense:**
1. **Unflinching Scientific Honesty:** Rather than cherry-picking a single favorable train/test split or tuning until reaching a spurious $p < 0.05$, we pre-registered our hypothesis ($H3$), conducted 15 locked CV folds, and computed 5,000 bootstrap resamples. We openly report that manual phonetic word normalization alone provides no statistically significant gain over raw words on moderate-sized corpora.
2. **Why It Happens:** In customer support domains with bounded intent vocabularies, high-impact keywords appear with sufficient frequency that linear classifiers learn direct associations without needing variant mapping.
3. **The High-Stakes Asymmetry:** When examining granular grievance intents, Setup C beats all baselines on the two highest-cost categories: `refund` (**0.9606** C vs **0.9555** B vs **0.8992** A) and `cancel_order` (**0.9723** C vs **0.9697** B vs **0.9560** A).

---

### Question 3: Why did Setup B (generic subwords without LID) outperform Setup C (LID-aware normalization + subwords) overall (93.08% vs 91.85%)?

**Answer:**
Character boundary $n$-grams (`char_wb` 1–4) inherently decompose words into overlapping character substrings that absorb spelling noise automatically:
- *kro* decomposes into: `[' k', 'kr', 'kro', 'ro ']`
- *karo* decomposes into: `[' k', 'ka', 'kar', 'karo', 'ar', 'aro', 'ro ']`

Both tokens share the edge n-grams `[' k', 'ro ']`. The linear classifier assigns positive weights directly to these subword fragments. Consequently, the model naturally groups spelling variants without requiring an explicit upstream phonetic map or LID tagger. Furthermore, Setup B avoids any cascading errors where an upstream LID misclassification might lead to an improper normalization rule being applied.

---

### Question 4: If Setup B is statistically superior overall, why build an upstream LID tagger at all?

**Answer:**
An enterprise customer support system requires capabilities beyond raw text classification:
1. **Asymmetric Grievance Protection:** Setup C outperforms Setup B on the two most critical operational intents—`refund` (+0.0051) and `cancel_order` (+0.0026)—where misclassification leads directly to financial loss or transit waste.
2. **Deterministic Bilingual Routing:** Downstream support dispatch requires knowing *when* to route a query to a bilingual agent. Calculating the Code-Mixing Index (CMI) is mathematically impossible without token-level language identification. Setup B is an uninterpretable subword bag that cannot calculate CMI.
3. **Regulatory Auditability & Explainability:** In financial dispute resolution, support managers must be able to audit why a decision was reached. Setup C provides a complete token-by-token trace (`[token, tag, normalization_action]`).

---

### Question 5: How did you ensure that there is no data leakage across splits and folds?

**Answer:**
1. **Sentence-Level Splitting:** The 1.35-million-token LinCE dataset was partitioned strictly at the **sentence level** using `seed=42`. Token-level random splitting was strictly avoided to ensure that tokens from the same utterance never appear in both train and test splits.
2. **Nested Vectorizer Fitting:** In the downstream 15-fold cross-validation experiment, all TF-IDF vectorizers were fit strictly inside the fold loops on the training split only (`Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])`).
3. **Zero Intent Exposure:** The upstream LID tagger was trained exclusively on the LinCE social media corpus and was **never exposed** to the customer support intent dataset.
4. **Pre-Frozen Ambiguous Words:** The 36 ambiguous cross-lingual homographs were locked in `data/ambiguous_words.txt` on Day 9 before the final test evaluation on Day 10.

---

### Question 6: LinCE is Twitter data, but your target domain is customer support queries. How do you address the domain gap?

**Answer:**
1. **In-Domain Test Verification:** On Day 12, we evaluated the frozen LID tagger on 45 hand-tagged customer support queries (500+ tokens) in `data/processed/lid_indomain.conll`. The tagger achieved **96.2% accuracy** in-domain, proving that character boundary patterns transfer effectively across domains.
2. **Regex Masking Normalization:** Domain-specific noise such as URLs, Order IDs (`OD123456789`), and phone numbers are normalized to deterministic pseudo-tokens (`<URL>`, `<ORDER_ID>`, `<PHONE>`) before reaching the tagger, eliminating out-of-domain entity distractions.
3. **Lexical Domain Protection:** Domain-specific commercial brand names (*Zomato, Zepto, Paytm, Swiggy*) were identified as boundary edge cases in our error analysis and documented transparently in `docs/error_analysis.md`.

---

### Question 7: How was the CMI threshold of 30.0% chosen? Is it arbitrary?

**Answer:**
The threshold is **strictly data-derived** from the empirical distribution of our 360 customer support queries:
- 25th percentile: $0.00\%$ (Monolingual queries)
- 50th percentile (Median): $25.00\%$
- **75th percentile:** **$30.00\%$**

Queries below $30.0\%$ typically contain only an isolated English loanword (e.g., *"Mera parcel nahi aaya"* has $\text{CMI} = 20.0\%$), which standard agents can easily handle. Queries with $\text{CMI} \ge 30.0\%$ represent true syntactic alternation across clauses and verbs. This threshold routes **37.8% of queries (136/360)** to bilingual queues, perfectly matching standard call-center bilingual staffing capacities (35–40%).

---

### Question 8: How reliable are your intent labels, and how was inter-annotator agreement measured?

**Answer:**
The dataset consists of 360 queries across 6 balanced classes (60 queries per class). Two annotators independently labeled the corpus following written guidelines (`docs/intent_guidelines.md`).
- **Observed Agreement ($P_o$):** $0.9500$ (95.0%)
- **Expected Chance Agreement ($P_e$):** $0.1667$ (1/6)
- **Cohen's Kappa ($\kappa$):** **$0.9399$**

A Kappa score of $0.9399$ denotes **near-perfect agreement** under Landis & Koch (1977) standards. Disagreements were systematically logged in `results/intent_kappa_qa.json` and adjudicated prior to finalization.

---

### Question 9: How did you validate that surrounding context actually helps disambiguate homographs?

**Answer:**
Across 36 pre-frozen ambiguous words (8,428 test occurrences):
- Majority-word lookup baseline: **77.83%**
- Character n-gram without context (M1): **77.83%**
- Character n-gram with context $[w_{i-1}, w_{i+1}]$ (M3): **88.62%**
- **Net Gain:** **+10.79%**
- **5,000 Bootstrap Resamples 95% CI:** **$[+6.84\%, +14.86\%]$** ($p < 0.001$)

Because the lower bound ($+6.84\%$) is strictly positive, the benefit of surrounding lexical context is statistically proven beyond random fluctuation.

---

### Question 10: What are the primary future improvements planned for this architecture?

**Answer:**
1. **Sequence Modeling with Conditional Random Fields (CRF):** While local context windows ($w_{i-1}, w_{i+1}$) provide $+10.79\%$ gain, a linear-chain CRF or structured HMM would explicitly enforce valid transition paths between language tags.
2. **Semi-Supervised Expansion of Normalization Lexicon:** Expanding the 224-rule dictionary using unsupervised character edit-distance clustering over large unlabelled e-commerce query logs.
3. **Hybrid Architecture:** Employing our ultra-fast CPU pipeline as a **tier-1 triage filter** that handles 90%+ of queries in $<2.5\text{ ms}$, escalating only high-ambiguity or low-confidence queries ($< 0.50$) to an LLM.
