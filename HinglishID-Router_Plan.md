# HinglishID-Router: 3-Week Project Plan + Demo Notes

**Word-Level Language Identification with Character N-Grams and Naive Bayes to Improve Intent Classification of Romanized Hindi-English Customer Support Queries**

3 weeks · 21 modules · Small team (3 recommended) · Modest compute (CPU only) · Honest evaluation, negative results OK

*Ye Markdown version `HinglishID-Router_Plan_and_Demo.html` ka text copy hai. Interactive demo aur progress tracking sirf HTML mein hain; yahan checkboxes (`- [ ]`) apne editor mein tick kar sakte ho.*

---

## 1. Project Summary and Goals

Short mein: kya banana hai, kyun, aur success ka matlab kya hai.

-   **Problem:** Indian users support queries aise likhte hain: *"Mera order abhi tak nahi aaya, please refund karo"*. Standard English pipeline (stopwords, stemmer, intent model) Hindi words ko noise samajhta hai.
-   **Core (Part 1):** Word-level tagger jo har word ko `HI` `EN` `OTHER` label kare, using char n-grams (1-4) + Naive Bayes.
-   **Proof of value (Part 2):** Intent classifier ko 3 setups mein compare karna: A (raw, English pipeline), B (raw, char n-gram TF-IDF, no LID), C (LID-aware normalization, then TF-IDF).
-   **Honest rule:** Gain chhota ya zero ho sakta hai. Negative result + analysis acceptable hai. Fabricated gain kabhi nahi.
-   **Timeline:** Week 1 tagger, Week 2 context + data + normalization, Week 3 comparison + demo + report.

### Pipeline ek nazar mein

-   **1. Tokenize & Clean:** Emoji, URL, order ID mask
-   **2. Word-level LID:** HI / EN / OTHER (char n-gram NB)
-   **3. Language-aware processing:** EN: stopwords + stem. HI: spelling normalize + stopwords
-   **4. Intent classification:** TF-IDF + LR / NB
-   **5. Routing:** Intent + code-mix flag

### Measurable outcomes

| # | Outcome | Kaise measure hoga |
| --- | --- | --- |
| 1 | Tagger quality | Per-class P/R/F1 for HI and EN. Expect roughly 88-95%, lekin run karne se pehle number promise mat karo. |
| 2 | Feature ablation | Char n-gram vs whole-word lookup; neighbouring-word context ka effect on ambiguous words. |
| 3 | Ambiguity result | 30-40 hand-picked ambiguous words par accuracy, alag se report. |
| 4 | Downstream value | Intent accuracy / macro-F1 for A vs B vs C. |
| 5 | Corpus statistics | HI vs EN token share, % code-mixed messages, CMI distribution. |

### Syllabus mapping

**Unit 1**  
Tokenization, stemming, morphological analysis (spelling variation in romanized Hindi).

**Unit 2**  
Corpus analysis (language ratio, CMI), categorical attributes (HI/EN/OTHER), dataset prep, CoNLL token-per-line format.

**Unit 3**  
Pre-processing, ambiguity handling (`is, to, me, hai, main...`).

**Unit 4**  
Char n-grams (1-4), Bag of Words, TF-IDF, Naive Bayes, Logistic Regression, rule-based vs ML.

### 21 din ka timeline

| Day | Module | Task | Owner |
|---|---|---|---|
| Day 1 | M1.1 | Setup + Dataset Access Check | All (A leads) |
| Day 2 | M1.2 | Corpus Exploration + CoNLL Parser + Splits | A (verify: B) |
| Day 3 | M1.3 | Preprocessing: Tokenizer + Masking | C (verify: A) |
| Day 4 | M1.4 | Baseline: Word-Lookup Tagger | A (verify: B) |
| Day 5 | M1.5 | Char N-Gram Naive Bayes Tagger | A (verify: C) |
| Day 6 | M1.6 | Week-1 Evaluation + Ablation Table 1 | A + B |
| Day 7 | M1.7 | Buffer + Week-1 Review + Guidelines Kickoff | All |
| Day 8 | M2.1 | Context Features (neighbouring words) | A (verify: B) |
| Day 9 | M2.2 | Ambiguous Word List (30-40) + Evaluation | A + B |
| Day 10 | M2.3 | Error Analysis (15 examples) + Freeze Tagger | All |
| Day 11 | M2.4 | Intent Dataset Writing (300-400 queries) | B leads, All write |
| Day 12 | M2.5 | Label QA, Kappa, CV Folds + In-domain LID Set | B (verify: C) |
| Day 13 | M2.6 | Normalization Lexicon + Stopwords | C (verify: B) |
| Day 14 | M2.7 | Pipeline Integration + Week-2 Review | C + All |
| Day 15 | M3.1 | Implement Setups A, B, C (fair comparison) | B (verify: C) |
| Day 16 | M3.2 | Results, Statistics, Per-intent Analysis | B + A |
| Day 17 | M3.3 | Routing + Code-Mixing Flag | C (verify: A) |
| Day 18 | M3.4 | Streamlit Demo | C (verify: B) |
| Day 19 | M3.5 | Report Writing | All (A: LID, B: intent, C: intro/method/demo) |
| Day 20 | M3.6 | Presentation + Repo Cleanup | C + All |
| Day 21 | M3.7 | Final Review + Rehearsal (buffer) | All |

> **Mentor note:** Sirf tagger banane se project "standard exercise" lagega. Asli value Part 2 se aati hai: tagger intent classification ko sach mein improve karta hai ya nahi. Isliye Week 3 ko skip ya compress mat karna.

## 2. Team Roles and Task Division

Suggested split 3 logon ke liye. Har module ka ek **owner** hai aur ek alag **verifier**, taaki koi bhi kaam bina review ke "done" na ho.

Member A

### Tagger Lead (LID core)

-   Data loading, CoNLL parser, splits
-   Lookup baseline, char n-gram NB/LR tagger
-   Context features, ablations
-   Ambiguous-word list + evaluation

Member B

### Data & Intent Lead

-   Intent guidelines, dataset coordination
-   Label QA, Cohen's kappa, CV folds
-   A/B/C experiments and statistics
-   Per-intent analysis

Member C

### Pipeline, Demo & Report Lead

-   `preprocess.py`, normalization lexicon
-   `pipeline.py` integration, routing
-   Streamlit demo
-   Repo hygiene, slides, report assembly

### Working rules

-   **Build-and-verify buddy:** A banata hai → B verify karta hai; B → C verify; C → A verify.
-   **Definition of Done:** code fresh clone se chalta hai, tests pass, deliverable commit hua, checkpoint question ka jawab `docs/checkpoints.md` mein likha.
-   **Shared kaam:** intent queries likhna (har member ~100-130), error analysis (Day 10), report writing (Day 19).
-   **Sync:** har 2 din mein 10-minute standup. Slip hone par risk register update karo.
-   **Git:** feature branches + small PRs; `main` hamesha runnable.

### Agar team size alag ho

**2 members**  
A+C merge: tagger + pipeline. B: data + experiments. Streamlit demo simple rakho (text box + colored tokens + intent).

**Solo / 4 members**  
Solo: kappa ke badle 3 din baad khud dobara label karo (intra-annotator agreement), in-domain LID mini-set chhota rakho (~25 queries). 4 members: 4th person demo + report, C sirf pipeline + routing.

## 3. Week 1: Foundation and Tagger

Goal: dataset access confirm, corpus samajhna, aur ek kaam karne wala char n-gram tagger dev set par.

### M1.1 · Day 1 · Setup + Dataset Access Check

- [x] Done
- **Owner:** All (A leads)
- **Syllabus:** Unit 2
- **Effort:** ~3 hrs

**Objective:** Repo, environment aur dataset access Day 1 par confirm karna, taaki baad mein surprise na ho.

**Steps:**

1. GitHub repo banao: `data/raw`, `data/processed`, `src`, `notebooks`, `app`, `docs`, `results`, `tests`.
2. Virtual env + `requirements.txt` (pandas, scikit-learn, nltk, streamlit, matplotlib, joblib, regex).
3. LinCE (ritual.uh.edu/lince) par register karo aur Hindi-English LID download page check karo. Dhyan do: **test labels hidden hain**, sirf train/dev labels use honge.
4. Parallel mein SemEval-2020 Task 9 (SentiMix) availability verify karo. Link dead ho sakta hai, isliye ek hi source par depend mat karo.
5. Label scheme finalize karo: LinCE ke labels (lang1, lang2, other, ne, mixed, ...) ko HI / EN / OTHER mein map karo. Dataset README se verify karo ki lang1/lang2 kaunsi language hai, assume mat karo.
6. Fallback trigger decide karo: Day 2 ke end tak data nahi mila to Risk R1 ka plan B chalu.

**Tools / code hints:**

```text
# docs/label_mapping.md  (example: dataset README se verify karo)
lang1 -> EN        # verify
lang2 -> HI        # verify
ne, other, unk, mixed, fw, ambiguous -> OTHER   # decision likh ke rakho
```

**Deliverable:** Repo link, data access status (Yes/No + screenshot), `docs/label_mapping.md`.

> **Checkpoint question:** Agar LinCE access Day 2 tak nahi mila, to hamara plan B kya hai aur use kaun trigger karega?

### M1.2 · Day 2 · Corpus Exploration + CoNLL Parser + Splits

- [x] Done
- **Owner:** A (verify: B)
- **Syllabus:** Unit 2
- **Effort:** ~4 hrs

**Objective:** Corpus samajhna: kitne sentences/tokens, language ratio, code-mixing index. Aur train/dev/test split fix karna.

**Steps:**

1. CoNLL-style file parse karo: har line `token<TAB>label`, blank line = sentence boundary. Actual file ka format check karke parser adjust karo.
2. Table banao: #sentences, #tokens, avg length, label distribution (HI/EN/OTHER).
3. Per-sentence CMI nikalo aur % code-mixed sentences (HI aur EN dono present) report karo.
4. Har class ke top-20 frequent tokens dekho. Same token dono classes mein ho to ambiguity ka early sign hai.
5. **Split by sentence** 80/10/10, `seed=42`, files save karo: `data/processed/{train,dev,test}.conll`. Test ko Day 10 tak mat chhuo.
6. Plots: label distribution bar chart, CMI histogram.

**Tools / code hints:**

```python
def read_conll(path):
    sents, toks, labs = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if toks: sents.append((toks, labs)); toks, labs = [], []
            continue
        t, l = line.split("\t")[:2]
        toks.append(t); labs.append(l)
    if toks: sents.append((toks, labs))
    return sents

def cmi(labels):               # Das & Gambäck; 0 = monolingual
    u = labels.count("OTHER"); n = len(labels)
    if n == u: return 0.0
    hi, en = labels.count("HI"), labels.count("EN")
    return 100 * (1 - max(hi, en) / (n - u))
```

**Deliverable:** `01_corpus_stats.ipynb`, stats table, 2 plots, split files.

> **Checkpoint question:** Corpus mein HI vs EN token share kya hai, aur kitne % sentences actually code-mixed hain?

### M1.3 · Day 3 · Preprocessing: Tokenizer + Masking

- [x] Done
- **Owner:** C (verify: A)
- **Syllabus:** Unit 1, 3
- **Effort:** ~3 hrs

**Objective:** Ek reusable `preprocess.py` banana jo URL, emoji, order IDs mask kare aur tokens nikale. Same code LID aur intent dono ke liye.

**Steps:**

1. Regex rules: URL → `<URL>`, emoji → `<EMOJI>`, order IDs (`OD123456789`, `#45678`, 6+ digits) → `<ORDER_ID>`.
2. Elongation normalization: `plzzzzz` → `plzz` (3+ repeated letters ko 2 tak).
3. Tokenizer: placeholders ek token, words, numbers, punctuation alag.
4. Lowercase sirf features ke liye; original casing alag rakho (Capitalized words brand/NE ka hint ho sakte hain).
5. 20 asserts/unit tests likho (weird inputs: emoji-only, empty string, mixed script).

**Tools / code hints:**

```python
import regex as re   # pip install regex  (Unicode property support)
URL   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMOJI = re.compile(r"\p{Extended_Pictographic}")
OID   = re.compile(r"#\d{4,}|\b[A-Za-z]{1,4}[-_]?\d{5,}\b|\b\d{6,}\b")
ELONG = re.compile(r"([A-Za-z])\1{2,}")

def clean(t):
    t = URL.sub(" <URL> ", t); t = EMOJI.sub(" <EMOJI> ", t)
    t = OID.sub(" <ORDER_ID> ", t); return ELONG.sub(r"\1\1", t)
```

**Deliverable:** `src/preprocess.py` + `tests/test_preprocess.py` (20 asserts passing).

> **Checkpoint question:** Order ID aur normal number (jaise "2 din") mein difference kaise karoge? Kaunsa example tumhara regex galat pakad sakta hai?

### M1.4 · Day 4 · Baseline: Word-Lookup Tagger

- [x] Done
- **Owner:** A (verify: B)
- **Syllabus:** Unit 4
- **Effort:** ~2 hrs

**Objective:** Simple baseline banana taaki char n-gram ka fayda measurable ho.

**Steps:**

1. Train split se word → majority label dictionary banao (lowercased).
2. OOV (unseen) words ka default label decide karo aur **OOV rate** report karo.
3. Dev set par per-class P/R/F1 nikalo.
4. Optional: top-100 Hindi function words ki tiny hand lexicon se compare karo.
5. OOV errors ke examples dekho: romanized spelling variation kitna bada problem hai?

**Tools / code hints:**

```python
from collections import Counter, defaultdict
counts = defaultdict(Counter)
for toks, labs in train:
    for t, l in zip(toks, labs): counts[t.lower()][l] += 1
lookup  = {w: c.most_common(1)[0][0] for w, c in counts.items()}
default = Counter(l for _, labs in train for l in labs).most_common(1)[0][0]
pred = [lookup.get(t.lower(), default) for t in dev_tokens]
```

**Deliverable:** Baseline results table (dev) + OOV rate.

> **Checkpoint question:** OOV rate kitna hai, aur kya baseline ka zyada error OOV words se aa raha hai?

### M1.5 · Day 5 · Char N-Gram Naive Bayes Tagger

- [x] Done
- **Owner:** A (verify: C)
- **Syllabus:** Unit 4
- **Effort:** ~4 hrs

**Objective:** Main tagger: `char_wb` 1-4 grams + MultinomialNB, aur comparison ke liye LogisticRegression.

**Steps:**

1. Har token (lowercased) ek "document" hai; `CountVectorizer(analyzer="char_wb", ngram_range=(1,4))`.
2. `make_pipeline(vectorizer, MultinomialNB(alpha=...))`; alpha dev par tune karo (0.1, 0.5, 1.0).
3. Same features par `LogisticRegression(max_iter=1000)` train karo.
4. Class imbalance check; `class_weight` / prior try karo aur dono report karo.
5. Model `joblib.dump` se save karo; `fit / predict / save / load` wali class banao.

**Tools / code hints:**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

tagger = make_pipeline(
    CountVectorizer(analyzer="char_wb", ngram_range=(1, 4)),
    MultinomialNB(alpha=0.5))
tagger.fit(train_words, train_labels)
```

**Deliverable:** `src/lid_tagger.py` + dev metrics.

> **Checkpoint question:** Char n-grams word lookup se kyun better hone chahiye? Dev par actual result kya bata raha hai? Agar nahi hai, to kyun?

### M1.6 · Day 6 · Week-1 Evaluation + Ablation Table 1

- [x] Done
- **Owner:** A + B
- **Syllabus:** Unit 4
- **Effort:** ~3 hrs

**Objective:** Lookup vs NB vs LR ka clean comparison, test set ko touch kiye bina.

**Steps:**

1. `classification_report` se per-class P/R/F1; macro-F1 over HI+EN.
2. Confusion matrix plot; OOV vs in-vocab accuracy alag nikalo.
3. Best config dev par choose karo; test set **untouched**.
4. Results ko `results/lid_dev.csv` mein save karo (report ke liye script se tables banenge).

**Tools / code hints:**

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(dev_labels, tagger.predict(dev_words), digits=3))
print(confusion_matrix(dev_labels, tagger.predict(dev_words), labels=["HI","EN","OTHER"]))
```

**Deliverable:** Table 1 (dev) + confusion matrix image + `results/lid_dev.csv`.

> **Checkpoint question:** Kaunsi class sabse zyada confuse ho rahi hai, aur kya wo corpus ki ambiguity hai ya model ki kamzori?

### M1.7 · Day 7 · Buffer + Week-1 Review + Guidelines Kickoff

- [x] Done
- **Owner:** All
- **Syllabus:** Unit 2
- **Effort:** ~3 hrs

**Objective:** Gaps fix karna, buddy review, aur intent guidelines ka pehla draft.

**Steps:**

1. Har module buddy se verify karwao (fresh clone par run).
2. Notebook mein 10 sample queries par tagger demo karo (colored output).
3. `docs/intent_guidelines.md` draft: 6 intents ki definitions + borderline rules.
4. 15-min retro: kya slow tha, kaunsa module slip hua.

**Deliverable:** Git tag `v0.1`, guidelines draft.

> **Checkpoint question:** Kya hum Week 2 ke liye on track hain? Kaunsa module slip hua aur uska plan kya hai?

## 4. Week 2: Context, Error Analysis, Dataset, Normalization

Goal: ambiguity handle karna, intent dataset banana, aur language-aware normalization ready karna.

### M2.1 · Day 8 · Context Features (neighbouring words)

- [x] Done
- **Owner:** A (verify: B)
- **Syllabus:** Unit 3, 4
- **Effort:** ~4 hrs

**Objective:** Neighbouring words ka context add karke ambiguous words (`is, to, me, main`) par improvement check karna.

**Steps:**

1. Hypotheses H1-H3 pehle `docs/hypotheses.md` mein likh do (results dekhne se pehle).
2. Features: char n-grams + prev word + next word (boundary par `<S>`, `</S>`).
3. Variant M4: neighbours ke **predicted** labels (2-pass). Train ke liye gold labels mat lo, `cross_val_predict` ke predicted labels lo, warna train/test mismatch hoga.
4. Teen variants compare karo: char-only / +context words / +neighbour labels. Splits same rakho.
5. Sparse matrices ko `scipy.sparse.hstack` se jodo (char block + context block).

**Tools / code hints:**

```python
def ctx_feats(tokens, i):
    p = tokens[i-1].lower() if i > 0 else "<S>"
    n = tokens[i+1].lower() if i + 1 < len(tokens) else "</S>"
    return {"p=" + p: 1, "n=" + n: 1}
# X = hstack([char_vec.transform(words), DictVectorizer().fit_transform(ctx_dicts)])
```

**Deliverable:** `src/lid_context.py` + Table 2 (char-only vs +context).

> **Checkpoint question:** Context se overall F1 badha ya sirf ambiguous words par? Dono alag kyun report karne chahiye?

### M2.2 · Day 9 · Ambiguous Word List (30-40) + Evaluation

- [x] Done
- **Owner:** A + B
- **Syllabus:** Unit 3
- **Effort:** ~4 hrs

**Objective:** Hard cases ke liye alag test: words jo dono languages mein valid hain.

**Steps:**

1. Data-driven candidates: train corpus mein jo words HI aur EN dono label ke saath dikhte hain (minority ≥ 10%, freq ≥ 20; thresholds tune karo).
2. Domain knowledge se add karo. Starter pool (corpus se verify karo): `is, to, me, hi, or, do, main, par, the, tab, mat, us, mile, den`.
3. Final list 30-40 words, **test results dekhne se pehle freeze**: `data/ambiguous_words.txt` commit karo.
4. Test tokens mein se list wale tokens ka subset nikalo; per-word count report karo, n < 5 ko "low support" flag karo.
5. Models M0, M1, M3 + **majority-label-per-word baseline** evaluate karo; pooled micro-accuracy + bootstrap CI.

**Tools / code hints:**

```python
# candidate mining (train only)
amb = [w for w, c in counts.items()
       if sum(c.values()) >= 20 and c["HI"] and c["EN"]
       and min(c["HI"], c["EN"]) / sum(c.values()) >= 0.10]
```

**Deliverable:** `ambiguous_words.txt` + Table 3 (per-word + pooled).

> **Checkpoint question:** Ye list tumne test results dekhne se pehle freeze ki thi? Ye kyun zaroori hai?

### M2.3 · Day 10 · Error Analysis (15 examples) + Freeze Tagger

- [x] Done
- **Owner:** All
- **Syllabus:** Unit 1, 3
- **Effort:** ~4 hrs

**Objective:** Errors ko categorize karna, tagger freeze karna, aur test set par **ek baar** final evaluation.

**Steps:**

1. Dev errors collect karo aur categorize karo: spelling variants, ambiguous, brand/NE, slang/abbreviation, numerals, elongation, annotation noise.
2. 15 concrete examples sentence context ke saath chuno.
3. Sasta fix (feature/alpha) sirf dev par try karo, phir **freeze**: `lid_final.joblib`.
4. Test set par final evaluation **ek baar** chalao; number kabhi dobara tune mat karo.

**Deliverable:** `error_analysis.md` (15 examples), frozen model, final test table.

> **Checkpoint question:** Kaunsi error category sabse badi hai, aur kya normalization (Day 13) usse address kar sakta hai?

### M2.4 · Day 11 · Intent Dataset Writing (300-400 queries)

- [x] Done
- **Owner:** B leads, All write
- **Syllabus:** Unit 2
- **Effort:** ~4 hrs each

**Objective:** 6 intents × ~60 queries, diverse spellings aur language mix ke saath.

**Steps:**

1. Guidelines freeze: har intent ki definition. **Primary-intent rule:** query mein do intents ho ("order nahi aaya, refund karo") to label = customer jo action maang raha hai (refund).
2. Har member sab 6 intents likhe (~100-130 queries each) taaki writer style bias kam ho; `writer` column rakho.
3. Variety quota: ~⅓ Hindi-dominant, ⅓ balanced, ⅓ English-dominant. Spellings vary karo (nahi/nhi/nai), shorthand (plz/pls), typos, kabhi emoji/order ID, short + long queries.
4. Ek template 60 baar copy-paste mat karo; near-duplicates baad mein dedupe honge.
5. Agar AI-assisted drafting use karo to report mein disclose karo aur manually edit karo; bulk generated queries mein realistic spellings kam aate hain.

**Tools / code hints:**

```csv
id,text,intent,writer,mix_style
1,"Mera order abhi tak nahi aaya, please refund karo",refund,A,balanced
2,"payment fail ho gaya but paise kat gaye",payment_failed,B,balanced
3,"order cancel karna hai plz jaldi",cancel_order,C,hindi
4,"My order is delayed, kab tak aayega?",delivery_delay,A,english
```

**Deliverable:** `data/intent_raw.csv` (≥ 300 rows) + guidelines v1.

> **Checkpoint question:** Kya har intent ke ~60 examples hain? Kya queries sach mein alag spellings aur lengths mein hain?

### M2.5 · Day 12 · Label QA, Kappa, CV Folds + In-domain LID Set

- [x] Done
- **Owner:** B (verify: C)
- **Syllabus:** Unit 2
- **Effort:** ~4 hrs

**Objective:** Dataset ko trustworthy banana aur evaluation protocol lock karna.

**Steps:**

1. Random 10-15% (~40-50) queries kisi doosre se bina original label dekhe relabel karwao; `cohen_kappa_score` nikalo; confusions discuss karo; guidelines update karo.
2. Near-duplicates dedupe; class balance check.
3. Protocol fix: **stratified 5-fold × 3 repeats**, fixed seeds, same folds A/B/C ke liye (chhote dataset par single split reliable nahi).
4. Recommended: 40-50 support queries (~400 tokens) khud token-level tag karo. Ye **sirf evaluation** ke liye hai (LinCE Twitter vs support domain gap), training mein nahi.

**Tools / code hints:**

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(labels_annotator1, labels_annotator2)
# folds
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

**Deliverable:** `intent_clean.csv`, kappa value, fold config, `lid_indomain.conll`.

> **Checkpoint question:** Kappa kitna aaya? Agar < 0.6 hai, to guidelines mein kya badlaav karoge?

### M2.6 · Day 13 · Normalization Lexicon + Stopwords

- [x] Done
- **Owner:** C (verify: B)
- **Syllabus:** Unit 1, 3
- **Effort:** ~4 hrs

**Objective:** Hindi spelling variants ko ek canonical form dena aur language-specific stopwords handle karna.

**Steps:**

1. Intent data + LinCE Hindi tokens se frequent Hindi tokens nikalo; spelling variants cluster karo (manual + edit distance ≤ 2 suggestions).
2. `norm_map.json` (100-200 variants → canonical): `nahin/nai/nhi/nahii → nahi`, `hain/hay → hai`, `kro → karo`, `bohot/bahot → bahut`. Ye **spelling normalization** hai, lemmatization nahi.
3. Hindi stopwords (~40-60): `ka, ki, ke, ko, se, mein, par, hai...`. **Negation/question words (nahi, mat, kab, kya) mat hatao**, ye intent signal hain.
4. English: NLTK stopwords lo lekin negations (`not, no, nor, never`) rakho; stemming Porter/Snowball.
5. **LID-aware collision handling:** `do` English stopword hai, lekin Hindi "do" (dena) content word hai. Tag ke hisaab se decide karo.
6. 15 before/after pairs ke unit tests.

**Tools / code hints:**

```python
def process(tokens, tags):
    out = []
    for t, tag in zip(tokens, tags):
        w = t.lower()
        if tag == "HI":
            w = NORM.get(w, w)
            if w in HI_STOP: continue
        elif tag == "EN":
            if w in EN_STOP: continue
            w = stemmer.stem(w)
        out.append(w)
    return out
```

**Deliverable:** `norm_map.json`, `stopwords_hi.txt`, `src/lang_aware.py`, tests.

> **Checkpoint question:** Kaunse Hindi words ko stopword list se **nahi** hatana chahiye aur kyun?

### M2.7 · Day 14 · Pipeline Integration + Week-2 Review

- [x] Done
- **Owner:** C + All
- **Syllabus:** Unit 1-4
- **Effort:** ~4 hrs

**Objective:** Saare modules ko ek single `analyze(query)` function mein jodna.

**Steps:**

1. `pipeline.py`: preprocess → LID → language-aware processing → return tokens, tags, normalized string, CMI.
2. 20 sample queries par end-to-end run; output table save karo; latency check.
3. Buddy code review (A reviews C). Fresh clone par run.
4. Risk register update; decide karo ki intent dataset size theek hai ya aur queries chahiye.

**Tools / code hints:**

```python
def analyze(query):
    toks = tokenize(clean(query))
    tags = tagger.predict(toks)            # frozen LID model
    out  = process(toks, tags)             # language-aware
    return {"tokens": toks, "tags": tags, "text_C": " ".join(out), "cmi": cmi(tags)}
```

**Deliverable:** `src/pipeline.py`, 20-query sample table, tag `v0.2`.

> **Checkpoint question:** Pipeline ek function call mein chal raha hai? Kya LID tagger intent data par kabhi train hua? (Nahi hona chahiye.)

## 5. Week 3: Downstream Comparison, Demo, Report

Goal: A vs B vs C fair comparison, routing + Streamlit demo, aur honest report.

### M3.1 · Day 15 · Implement Setups A, B, C (fair comparison)

- [x] Done
- **Owner:** B (verify: C)
- **Syllabus:** Unit 4
- **Effort:** ~4 hrs

**Objective:** A/B/C ko ek-ek variable change karke build karna.

**Steps:**

1. **A:** lowercase → English stopwords + stemmer saare tokens par (no LID) → word TF-IDF (1-2 grams).
2. **B:** raw lowercase text → `TfidfVectorizer(analyzer="char_wb", ngram_range=(1,4))`.
3. **C:** pipeline output (normalized text) → **same** char\_wb TF-IDF jaisa B mein. B vs C ka difference sirf LID-aware normalization se aayega. Optional C′: C ke saath word TF-IDF (A se fair comparison).
4. Classifier same: LogisticRegression (second run: MultinomialNB). Same small hyperparameter grid.
5. Vectorizer ko `Pipeline` ke andar rakho taaki fold ke andar fit ho (leakage nahi).

**Tools / code hints:**

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def make_model(analyzer, ngram):
    return Pipeline([("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True)),
                     ("clf", LogisticRegression(max_iter=2000))])
A = make_model("word", (1, 2)); B = make_model("char_wb", (1, 4)); C = make_model("char_wb", (1, 4))
# inputs: A <- text_A, B <- raw lowercase, C <- text_C
```

**Deliverable:** `src/run_intent_experiments.py` + fold-level score CSV.

> **Checkpoint question:** B aur C mein **sirf** ek cheez different hai. Kaunsi? Agar do cheezein badal jaayein to conclusion kyun kamzor hoga?

### M3.2 · Day 16 · Results, Statistics, Per-intent Analysis

- [x] Done
- **Owner:** B + A
- **Syllabus:** Unit 4
- **Effort:** ~4 hrs

**Objective:** Numbers ko honest tareeke se analyze karna.

**Steps:**

1. 15 fold scores se mean ± std (accuracy, macro-F1) nikalo.
2. Paired difference (C−B, B−A) + bootstrap 95% CI. Folds overlap karte hain, isliye CI ko p-value se zyada importance do.
3. Per-intent F1 table + best model ka confusion matrix.
4. Queries jahan C ne B ko fix kiya ya tod diya, unke examples collect karo.
5. Agar gain ≈ 0: **oracle-LID** check (in-domain hand-tagged queries par gold tags se C) karo, taaki "LID error" aur "idea kamzor" alag ho sakein.

**Tools / code hints:**

```python
import numpy as np
diff = scores_C - scores_B                      # shape (15,)
boot = [np.mean(np.random.choice(diff, len(diff))) for _ in range(5000)]
print(diff.mean(), np.percentile(boot, [2.5, 97.5]))
```

**Deliverable:** Table 4 + figures + `analysis_notes.md`.

> **Checkpoint question:** Kya CI zero ko cross kar raha hai? Agar haan, to claim kya karoge?

### M3.3 · Day 17 · Routing + Code-Mixing Flag

- [x] Done
- **Owner:** C (verify: A)
- **Syllabus:** Unit 4
- **Effort:** ~3 hrs

**Objective:** Rule-based routing layer: intent → queue, aur heavy code-mixing flag.

**Steps:**

1. Routing table: intent → queue (rule-based system, Unit 4).
2. CMI threshold **data se** choose karo (train/dev intent queries ka percentile), gut feeling se nahi.
3. Low-confidence predictions ko "human agent" par bhejo (threshold dev par choose karo).
4. Stretch: heavily-mixed queries par misclassification zyada hai? CMI vs error plot.
5. `route()` ke unit tests.

**Tools / code hints:**

```python
QUEUE = {"order_not_delivered": "Logistics", "refund": "Refunds", "payment_failed": "Payments",
         "cancel_order": "Order Ops", "delivery_delay": "Logistics", "other": "General FAQ"}
def route(intent, conf, cmi_val, cmi_thr=30, conf_thr=0.5):
    queue = QUEUE[intent] if conf >= conf_thr else "Human agent (low confidence)"
    return {"queue": queue, "flag_heavy_mix": cmi_val >= cmi_thr}
```

**Deliverable:** `src/routing.py` + tests, CMI-vs-error plot (agar time ho).

> **Checkpoint question:** CMI threshold ka basis kya hai, gut feeling ya data?

### M3.4 · Day 18 · Streamlit Demo

- [x] Done
- **Owner:** C (verify: B)
- **Syllabus:** Unit 1-4
- **Effort:** ~4 hrs

**Objective:** Ek simple, reliable demo jo pipeline ke har stage ko dikhaye (is page ke demo jaisa).

**Steps:**

1. Layout: text area + 5-6 sample query buttons; colored tokens (HTML spans); normalized text; intent + top-3 probabilities; routing + CMI.
2. `@st.cache_resource` se models ek baar load karo.
3. Edge cases: empty input, very long input, sirf emoji.
4. Ek **"Limitations"** expander jisme known failure cases hon.
5. Local run: `streamlit run app/demo.py`. Optional: Streamlit Community Cloud deploy.

**Tools / code hints:**

```python
import streamlit as st
from src.pipeline import analyze

q = st.text_area("Customer query", "Mera order abhi tak nahi aaya, please refund karo")
if st.button("Analyze"):
    res = analyze(q)
    st.markdown(render_tokens(res), unsafe_allow_html=True)
    st.write(res["intent"], res["queue"], f"CMI = {res['cmi']:.1f}")
```

**Deliverable:** Working demo, 3 screenshots, 60-second screen recording.

> **Checkpoint question:** Demo mein tumhara system galat kab hota hai? Kya demo wo failure bhi dikhata hai?

### M3.5 · Day 19 · Report Writing

- [ ] Done
- **Owner:** All (A: LID, B: intent, C: intro/method/demo)
- **Syllabus:** -
- **Effort:** ~5 hrs

**Objective:** Section 9 ke outline ke hisaab se 6-8 page report ka draft v1.

**Steps:**

1. Tables/figures script se generate karo (`results/*.csv` → `make_figures.py`), hand se copy nahi.
2. Har claim ke saath number + CI. Negative result ko apna section do.
3. Limitations honestly likho: chhota data, domain gap, annotation noise.
4. Datasets cite karo: LinCE (Aguilar et al., 2020), SemEval-2020 Task 9 (Patwa et al., 2020). Reference details apni copy se verify karo.
5. AI-assisted drafting use hui ho to disclose karo.

**Deliverable:** Report draft v1.

> **Checkpoint question:** Kya abstract ke teen numbers results CSV se match karte hain?

### M3.6 · Day 20 · Presentation + Repo Cleanup

- [ ] Done
- **Owner:** C + All
- **Syllabus:** -
- **Effort:** ~4 hrs

**Objective:** Reproducible repo aur 8-10 slides.

**Steps:**

1. README: setup, data download, run commands, results reproduce karne ke steps.
2. `requirements.txt` pinned versions, seeds fixed, `run_all.sh` / Makefile.
3. Fresh clone test kisi doosre laptop par.
4. LinCE raw data repo se hatao (license); download instructions do.
5. Slides: problem → pipeline → 3 results → demo → limitations → future work.

**Deliverable:** Clean repo, slides.

> **Checkpoint question:** Kya koi bhi bina tumhari madad ke repo se results reproduce kar sakta hai?

### M3.7 · Day 21 · Final Review + Rehearsal (buffer)

- [ ] Done
- **Owner:** All
- **Syllabus:** -
- **Effort:** ~3 hrs

**Objective:** Last checks aur presentation practice.

**Steps:**

1. Report ke har number ko results CSV se cross-check karo.
2. 2 mock presentations; 10 expected Q&A (why Naive Bayes? why char n-grams? agar gain zero ho to kya? domain gap?).
3. Final tag `v1.0`; submission checklist (Section 8) poora karo.

**Deliverable:** Submission-ready package.

> **Checkpoint question:** Agar evaluator pooche "tumhara sabse kamzor result kaunsa hai?", to tumhara jawab kya hoga?

## 6. Evaluation Plan

Exact metrics, splits, ambiguous-word reporting, aur negative results ka protocol.

### 6.1 LID tagger: splits aur metrics

| Item | Decision |
| --- | --- |
| Data | LinCE Hindi-English LID train + dev (test labels hidden). Fallback: SemEval-2020 Task 9 ya self-tagged tokens. |
| Split | **Sentence level** par 80/10/10 (train/dev/test), `seed=42`. Token level split mat karo, warna same sentence ke tokens leak honge. |
| Use | Train = fit. Dev = hyperparameters (alpha, n-gram range, features). **Test = sirf ek baar**, Day 10 par jab model freeze ho. |
| Primary metric | Per-class precision / recall / F1 for HI and EN; macro-F1 over HI+EN. |
| Secondary | OTHER class F1, overall accuracy (imbalance ki wajah se primary nahi), confusion matrix, OOV vs in-vocabulary accuracy. |
| In-domain check | 40-50 hand-tagged support queries par tagger evaluate (train mein kabhi nahi). Twitter vs support-query domain gap dikhata hai. |

### 6.2 Feature ablation table (Outcome 2)

| Model | Features | HI F1 | EN F1 | Macro-F1 | Ambig. acc |
| --- | --- | --- | --- | --- | --- |
| M0 | Whole-word lookup (majority label) | … | … | … | … |
| M1 | char\_wb 1-4 + MultinomialNB | … | … | … | … |
| M2 | char\_wb 1-4 + LogisticRegression | … | … | … | … |
| M3 | M1/M2 + prev/next word | … | … | … | … |
| M4 | M1/M2 + neighbour predicted labels | … | … | … | … |

### 6.3 Ambiguous words (30-40): kaise report karna hai (Outcome 3)

1.  **List kaise banti hai:** train corpus se data-driven candidates (word dono labels ke saath dikhe, minority label ≥ 10%, freq ≥ 20) + domain knowledge. Starter pool (corpus se verify karo): `is, to, me, hi, or, do, main, par, the, tab, mat, us, mile, den`.
2.  **Freeze:** list test results dekhne se **pehle** commit karo (`data/ambiguous_words.txt`).
3.  **Table columns:** word | n\_test | gold HI count | gold EN count | acc (majority-label baseline) | acc (M1) | acc (M3).
4.  **Pooled number:** micro-accuracy on all ambiguous tokens + bootstrap 95% CI (1000 resamples).
5.  **Small-n flag:** jin words ke test occurrences < 5, unhe "low support" mark karo aur pooled number ke saath alag se bhi dikhao.
6.  **Must-have baseline:** "har word ko uska majority label do". Agar model isse beat nahi karta, to context ka fayda claim mat karo.

### 6.4 Downstream intent evaluation (Outcome 4)

| Item | Decision |
| --- | --- |
| Protocol | Stratified 5-fold CV × 3 repeats (15 scores per setup). Same folds A/B/C ke liye. Vectorizer fold ke andar fit. |
| Metrics | Accuracy, macro-F1 (primary, kyunki "other" class ho sakti hai), per-intent F1. |
| Comparison | Per-fold paired difference (C−B, B−A) with bootstrap 95% CI. Folds overlap karte hain, isliye p-values optimistic hote hain; CI ko prefer karo. |
| Fairness | B aur C mein sirf LID-aware normalization alag ho. Same classifier, same vectorizer, same hyperparameter grid. |
| Leakage guard | LID tagger intent data par kabhi train nahi hota. Near-duplicate queries same fold mein rakho (ya dedupe). |

### 6.5 Corpus statistics (Outcome 5)

LinCE aur intent dataset dono ke liye: #sentences, #tokens, HI/EN/OTHER share, % code-mixed sentences (HI aur EN dono present), CMI histogram. Dono datasets ka comparison table banao; agar intent data zyada Hindi-heavy ya English-heavy hai to report mein likho.

### 6.6 Negative results ka protocol

> **Pehle hypotheses likh do (Day 8 tak, `docs/hypotheses.md`):**
> 
> -   **H1:** char n-gram NB lookup se zyada macro-F1 deta hai (dev).
> -   **H2:** context features ambiguous-word accuracy badhate hain.
> -   **H3:** intent macro-F1: C ≥ B ≥ A.

**Agar result expected nahi aaya:**

1.  **Jaisa hai waisa report karo.** Example: "C ne B se 1.2 points behtar kiya, lekin 95% CI \[−2.1, +4.3\] hai, to improvement reliable nahi hai."
2.  **Test set par tune mat karo,** aur "jab tak achha result na aaye tab tak re-run" mat karo. Kitne experiments chale, wo report mein likho.
3.  **Diagnose karo (checklist):**
    -   Kya B ke char n-grams already spelling variation absorb kar lete hain? (Plausible: nahi/nhi/nahin ke n-grams overlap karte hain.)
    -   Kya LID errors aage propagate ho rahe hain? In-domain hand-tagged queries par **oracle LID** (gold tags) se upper bound nikalo.
    -   Kya intents keyword-driven hain (e.g. "refund" word hi kaafi hai)? To LID se extra fayda kam hoga.
    -   Kya dataset chhota hai (300-400)? To CI chaudi hogi; conclusion "inconclusive" hai, "no effect" nahi.
4.  **Negative result ko report mein ek proper section do:** kya try kiya, kya nahi chala, aur likely kyun.

## 7. Risk Register

Har risk ka early signal aur mitigation. Har standup mein is table ko 1 minute dekho.

| ID | Risk | Likelihood / Impact | Early signal | Mitigation | Owner |
| --- | --- | --- | --- | --- | --- |
| R1 | **Dataset access failure** (LinCE registration delay, SemEval link dead) | Medium / High | Day 2 tak download nahi hua | Day 1 par dono sources check. Fallback: intent-style queries se 1,000-2,000 tokens khud tag karo (do log, overlap par agreement check). Dono sources fail hon to LID evaluation chhota hoga, report mein likho. | A |
| R2 | **Label noise** (LinCE ki annotation + apni intent labels) | Medium / Medium | Kappa < 0.6; same word ke inconsistent tags | Written guidelines, 10-15% double-annotation, disagreements discuss. LID noise ke liye error analysis mein "annotation noise" category rakho. | B |
| R3 | **Small intent data** (300-400 queries) | High / High | CV std bahut zyada; fold-to-fold swings | Repeated stratified CV, CI report karo, single-number claim mat karo. Mix styles ensure karo. Time bache to 100 aur queries likho (leave-one-writer-out ke saath). | B |
| R4 | **Weak ya zero gains** (C ≈ B) | High / Medium | Pilot run (Day 15) mein C−B ≈ 0 | Section 6.6 protocol. Oracle-LID experiment, per-intent analysis, spelling-heavy subset par alag evaluation. Report mein negative result ko seriously analyze karo. | B + A |
| R5 | **Domain mismatch** (LinCE = Twitter, test = support queries) | High / Medium | In-domain F1 << LinCE F1 | 40-50 in-domain queries hand-tag karke evaluate. Brand/product words ke liye small domain lexicon; limitation likho. | A |
| R6 | **Data leakage** (neighbour labels gold vs predicted, vectorizer fit on all data, duplicates) | Medium / High | Suspiciously high scores (>98%) | Sklearn `Pipeline` use karo, `cross_val_predict` neighbour labels ke liye, dedupe, buddy code review. | All |
| R7 | **Timeline slip / member unavailable** | Medium / Medium | Koi module 1 din se zyada late | Day 7, 14, 21 buffer days. Optional items (in-domain tagging, extensions) pehle cut hote hain. Modules independent hain, to reassign possible. | C |
| R8 | **Licensing / redistribution** (LinCE data repo mein daalna) | Low / Medium | Repo public karna hai | Raw dataset commit mat karo; download instructions do. Sirf apna intent dataset share karo. | C |

## 8. Final Deliverables Checklist

Submission se pehle sab tick hona chahiye.

### Code repo

- [ ] README (setup + reproduce steps)
- [ ] src/ (preprocess, lid_tagger, lid_context, lang_aware, pipeline, routing)
- [ ] tests/ (preprocess, lang_aware, routing) passing
- [ ] requirements.txt pinned + seeds fixed
- [ ] Git tags v0.1 / v0.2 / v1.0

### Labeled data

- [ ] intent_clean.csv (300-400 rows, 6 intents)
- [ ] Intent guidelines + kappa value
- [ ] ambiguous_words.txt (frozen before test)
- [ ] norm_map.json + stopwords_hi.txt
- [ ] lid_indomain.conll (hand-tagged queries)

### Report sections

- [ ] Abstract + Introduction
- [ ] Data & corpus statistics (Outcome 5)
- [ ] LID tagger + ablation (Outcomes 1, 2)
- [ ] Ambiguity analysis (Outcome 3)
- [ ] Downstream A/B/C with CI (Outcome 4)
- [ ] Error analysis (15 examples)
- [ ] Limitations & negative results
- [ ] References + appendix

### Streamlit demo

- [ ] Runs from fresh clone
- [ ] Colored tokens, normalized text, intent, routing, CMI
- [ ] Limitations expander
- [ ] Screenshots + 60s recording

### Presentation

- [ ] 8-10 slides
- [ ] Q&A prep (10 questions)
- [ ] 2 rehearsals done

## 9. Report Outline

6-8 pages. Structure teen outcomes (tagger, ambiguity, downstream) ke around hai.

| Section | Content | Pages | Outcome |
| --- | --- | --- | --- |
| **1. Abstract** | Problem, method, 3 key numbers (with CI), ek line honest limitation. | 0.25 | All |
| **2. Introduction** | Code-mixed support queries, English pipeline ki failure example, contributions (2 parts), research questions H1-H3. | 0.75 | \- |
| **3. Data & Corpus Statistics** | LinCE (+ SemEval), label mapping, intent dataset construction, guidelines, kappa, HI/EN share, % code-mixed, CMI histogram. | 1 | 5 |
| **4. Method** | Preprocessing/masking, tagger (char n-gram NB/LR), context features, language-aware processing, setups A/B/C, routing, CMI. | 1.25 | \- |
| **5. Exp 1: LID Tagger** | Table 1 (lookup vs NB vs LR), per-class F1, confusion matrix, in-domain result. | 0.75 | 1, 2 |
| **6. Exp 2: Ambiguity** | Table 3 (per-word + pooled with CI), majority-label baseline, effect of context. | 0.75 | 3 |
| **7. Exp 3: Downstream Intent** | A vs B vs C: mean ± std, paired-difference CI, per-intent F1, examples where C fixed or broke predictions. | 1.25 | 4 |
| **8. Error Analysis** | 15 concrete examples by category (spelling, ambiguity, brands, slang, noise). | 0.5 | 1-4 |
| **9. Demo & Routing** | Screenshots, routing rules, CMI flag, failure case. | 0.5 | \- |
| **10. Limitations & Negative Results** | Small data, domain gap, what did not work, oracle-LID diagnosis. | 0.5 | 4 |
| **11. Conclusion & Future Work** | Honest takeaways; HMM over tags, CMI vs error, bigger intent data. | 0.25 | \- |
| **References / Appendix** | LinCE (Aguilar et al., LREC 2020), SemEval-2020 Task 9 (Patwa et al.), scikit-learn, NLTK. Appendix: guidelines, norm lexicon, ambiguous list. | excl. | \- |

> Har table aur figure script se generate karo (`results/*.csv` → `make_figures.py`). Hand se number copy karoge to report aur results mein mismatch aa jayega.

## Bonus: Live Demo: Pipeline Preview

> **Honest disclosure:** Interactive demo sirf HTML file mein hai. Wo ek **illustrative toy** hai: ~420 seed words ka lookup + unseen words ke liye char n-gram (1-4) Naive Bayes, rule-based intent scorer, aur simplified stemmer. Neeche ke outputs project ke **results nahi** hain aur ye galtiyan kar sakta hai (jaise unseen slang). Asli project mein tagger LinCE par train hoga aur intent classifier TF-IDF + LR/NB hoga.

Demo har query par 5 stages dikhata hai (Tokenize & Clean → LID → Language-aware processing → Intent → Routing) aur Setup A / B / C ka input side-by-side dikhata hai. Toy ke sample outputs (context rule on, CMI threshold 30):

| Query | Tagged (token/tag) | Setup C text | Toy intent | CMI | Queue |
|---|---|---|---|---|---|
| Mera order abhi tak nahi aaya, please refund karo | `Mera/HI order/EN abhi/HI tak/HI nahi/HI aaya/HI ,/OTHER please/EN refund/EN karo/HI` | `order abhi tak nahi aaya please refund karo` | refund | 33 | Refunds queue |
| payment fail ho gaya but paise kat gaye 😡 OD123456789 | `payment/EN fail/EN ho/HI gaya/HI but/EN paise/HI kat/HI gaye/HI <EMOJI>/OTHER <ORDER_ID>/OTHER` | `payment fail gaya paise kat gaye <emoji> <order_id>` | payment_failed | 38 | Payments queue |
| order cancel karna hai plz jaldi | `order/EN cancel/EN karna/HI hai/HI plz/EN jaldi/HI` | `order cancel karna please jaldi` | cancel_order | 50 | Order Ops queue |
| My order is delayed, kab tak aayega? https://track.example.com/x | `My/EN order/EN is/EN delayed/EN ,/OTHER kab/HI tak/HI aayega/HI ?/OTHER <URL>/OTHER` | `order delay kab tak aayega <url>` | delivery_delay | 43 | Logistics queue |
| mujhe is order ka refund chahiye, wrong item aaya hai | `mujhe/HI is/HI order/EN ka/HI refund/EN chahiye/HI ,/OTHER wrong/EN item/EN aaya/HI hai/HI` | `order refund chahiye wrong item aaya` | refund | 40 | Refunds queue |
| Thanks, app bahut accha hai | `Thanks/EN ,/OTHER app/EN bahut/HI accha/HI hai/HI` | `thank app bahut accha` | other | 40 | General FAQ / bot |

Interactive version ke liye `HinglishID-Router_Plan_and_Demo.html` browser mein kholo.
