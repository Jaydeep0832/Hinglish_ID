"""
app/demo.py
Module M3.4 (Day 18): Interactive Streamlit Customer Support Pipeline & Router Demo.

Features:
- Live Word-Level Language Identification (LID) with Production M3 Context Model.
- Phonetic Hindi Normalization with Protected Intent Signals (Setup C).
- End-to-End Support Intent Classification (6 classes) with Probability Distribution.
- Enterprise Support Queue Routing & Code-Mixing Index (CMI >= 30%) Escalation.
- Side-by-Side Comparison of Benchmark Setups (Setup A vs Setup B vs Setup C).
- Transparent Scientific Findings & Failure Taxonomy Expander.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.conll_parser import calculate_cmi
from src.lid_context import ContextLIDTagger
from src.normalize import (
    ENGLISH_GRAMMATICAL_STOPWORDS,
    HINDI_GRAMMATICAL_STOPWORDS,
    PROTECTED_NEGATIONS,
    PROTECTED_QUESTIONS,
    HinglishNormalizer,
)
from src.pipeline import HinglishPipeline
from src.preprocess import HinglishPreprocessor, tokenize
from src.routing import (
    DEFAULT_CMI_THRESHOLD,
    DEFAULT_CONF_THRESHOLD,
    HUMAN_AGENT_QUEUE,
    QUEUE_MAPPING,
    route_intent,
)


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HinglishID-Router Demo",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Header badges */
.badge-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 16px 0;
}
.badge-pill {
    font-size: 0.8rem;
    padding: 3px 10px;
    border-radius: 99px;
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    font-weight: 600;
}
.badge-pill.ok {
    background: #ecfdf5;
    color: #047857;
    border-color: #6ee7b7;
}
.badge-pill.accent {
    background: #eef2ff;
    color: #4338ca;
    border-color: #c7d2fe;
}
/* Token badges */
.tok-box {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 12px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin: 8px 0;
}
.tok {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    padding: 4px 10px 3px 10px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.25;
    border-bottom: 3px solid transparent;
}
.tok small {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    opacity: 0.85;
}
.tok-HI {
    background: #fef0d9;
    color: #b45309;
    border-color: #d97706;
}
.tok-EN {
    background: #dde8fd;
    color: #1d4ed8;
    border-color: #2563eb;
}
.tok-OTHER {
    background: #eceef2;
    color: #4b5563;
    border-color: #6b7280;
}
.tok-amb {
    outline: 2px dashed #4338ca;
    outline-offset: 1px;
}
/* Routing cards */
.queue-card {
    padding: 16px;
    border-radius: 12px;
    border-left: 6px solid #4338ca;
    background: #f8fafc;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin: 8px 0;
}
.queue-card.escalate {
    border-left-color: #d97706;
    background: #fffbeb;
}
.queue-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1e293b;
}
.queue-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline() -> HinglishPipeline:
    """Load unified Hinglish preprocessing and LID pipeline."""
    return HinglishPipeline(
        model_path=ROOT_DIR / "models" / "lid_final.joblib",
        norm_map_path=ROOT_DIR / "data" / "norm_map.json",
    )


@st.cache_resource
def load_intent_classifiers() -> Dict[str, Any]:
    """Load trained intent classification pipelines for Setups A, B, and C."""
    models = {}
    models_dir = ROOT_DIR / "models"

    model_c_path = models_dir / "intent_classifier_setup_c.joblib"
    if model_c_path.exists():
        models["Setup_C"] = joblib.load(model_c_path)

    model_b_path = models_dir / "intent_classifier_setup_b.joblib"
    if model_b_path.exists():
        models["Setup_B"] = joblib.load(model_b_path)

    model_a_path = models_dir / "intent_classifier_setup_a.joblib"
    if model_a_path.exists():
        models["Setup_A"] = joblib.load(model_a_path)

    return models


# Initialize singletons
pipeline = load_pipeline()
classifiers = load_intent_classifiers()
stemmer = PorterStemmer()


# -----------------------------------------------------------------------------
# Pre-defined Benchmark Queries
# -----------------------------------------------------------------------------
SAMPLE_QUERIES = [
    {
        "label": "Sample 1: Refund Dispute (Mixed)",
        "text": "Mera order abhi tak nahi aaya, please refund karo",
        "expected": "refund (High Grievance)",
    },
    {
        "label": "Sample 2: Payment Deducted (Emoji + Order ID)",
        "text": "payment fail ho gaya but paise kat gaye 😡 OD123456789",
        "expected": "payment_failed",
    },
    {
        "label": "Sample 3: Fast Cancellation (Hindi Heavy)",
        "text": "order cancel karna hai plz jaldi, galat item order ho gaya",
        "expected": "cancel_order",
    },
    {
        "label": "Sample 4: Delivery Delay (English Heavy + URL)",
        "text": "My order is delayed by 4 days, kab tak aayega? https://track.delivery.in/OD998",
        "expected": "delivery_delay",
    },
    {
        "label": "Sample 5: Address Modification (Code-Mixed)",
        "text": "mujhe shipping address change karwana hai naya flat number add karo",
        "expected": "cancel_order / Order Ops",
    },
    {
        "label": "Sample 6: Undelivered Order (Courier Issue)",
        "text": "4 din ho gaye parcel nahi mila, delivery boy phone pick nahi kar raha",
        "expected": "order_not_delivered",
    },
    {
        "label": "Sample 7: Positive Feedback (General FAQ)",
        "text": "Thanks a lot, customer care agent was very helpful app accha hai",
        "expected": "other",
    },
]


# -----------------------------------------------------------------------------
# Sidebar: Configuration, Controls & Benchmark Insights
# -----------------------------------------------------------------------------
st.sidebar.title("🇮🇳 HinglishID-Router")
st.sidebar.markdown(
    "**End-to-End Word-Level Language ID, Phonetic Normalization & Enterprise Support Routing**"
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Routing Controls")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold (Triage)",
    min_value=0.20,
    max_value=0.90,
    value=DEFAULT_CONF_THRESHOLD,
    step=0.05,
    help="Predictions below this confidence level are routed to 'Human agent (low confidence)'.",
)

cmi_threshold = st.sidebar.slider(
    "Bilingual Queue CMI Threshold (%)",
    min_value=10.0,
    max_value=60.0,
    value=DEFAULT_CMI_THRESHOLD,
    step=5.0,
    help="Queries with Code-Mixing Index >= threshold are flagged for bilingual support routing (Empirical 75th percentile = 30.0%).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Empirical Highlights (15 Folds)")
st.sidebar.markdown(
    """
    - **LID Test Accuracy:** `98.42%` (Macro-F1 `0.9805`)
    - **Homograph Disambiguation Gain:** `+10.79%`
    - **Annotator Agreement:** `κ = 0.9399`
    - **Setup B Macro-F1:** `0.9308 ± 0.0272`
    - **Setup C Macro-F1:** `0.9185 ± 0.0258`
    - **Setup A Macro-F1:** `0.8975 ± 0.0280`
    - **Grievance Wins (C vs B):**
      - `refund`: `0.9606` vs `0.9555`
      - `cancel_order`: `0.9723` vs `0.9697`
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "[GitHub: Jaydeep0832/Hinglish_ID](https://github.com/Jaydeep0832/Hinglish_ID)"
)


# -----------------------------------------------------------------------------
# Main Application Header & Live Badges
# -----------------------------------------------------------------------------
st.title("HinglishID-Router: Enterprise Customer Support Pipeline")
st.caption(
    "Word-Level Language Identification, Phonetic Normalization, and Intent Routing for Romanized Hindi-English Queries"
)

badge_html = """
<div class="badge-bar">
    <span class="badge-pill ok">● Status: Day 18 Operational</span>
    <span class="badge-pill ok">✓ 121 / 121 Tests Passing</span>
    <span class="badge-pill accent">Frozen M3 Context Tagger (98.42%)</span>
    <span class="badge-pill">224-Rule Phonetic Normalizer</span>
    <span class="badge-pill">6 Enterprise Support Queues</span>
    <span class="badge-pill">15 Locked CV Folds</span>
</div>
"""
st.markdown(badge_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Query Input & Sample Quick Fill
# -----------------------------------------------------------------------------
st.markdown("### 📝 Input Customer Support Query")

# Quick button selectors
cols = st.columns(len(SAMPLE_QUERIES))
selected_sample = None
for i, s in enumerate(SAMPLE_QUERIES):
    if cols[i].button(f"Sample {i+1}", key=f"sample_btn_{i}", help=f"{s['text']} ({s['expected']})"):
        selected_sample = s["text"]

# Set default or selected query
default_text = selected_sample if selected_sample else "Mera order abhi tak nahi aaya, please refund karo"

user_query = st.text_area(
    "Type or edit a customer support query in romanized Hindi-English (Hinglish):",
    value=default_text,
    height=80,
    help="Supports conversational Hinglish, order numbers, emojis, phonetic spellings, and abbreviations.",
)


# -----------------------------------------------------------------------------
# Analysis Execution
# -----------------------------------------------------------------------------
if not user_query.strip():
    st.info("👆 Please enter a customer query above or click any sample button.")
    st.stop()

# Step 1: Execute pipeline analysis
t0 = time.perf_counter()
analysis = pipeline.analyze(user_query)
latency_ms = (time.perf_counter() - t0) * 1000.0

tokens = analysis["tokens"]
tags = analysis["tags"]
query_cmi = analysis["cmi"]
text_A = analysis["text_A"]
text_B = analysis["text_B"]
text_C = analysis["text_C"]


# -----------------------------------------------------------------------------
# Display Pipeline Stages
# -----------------------------------------------------------------------------
tab_stages, tab_comparison, tab_evidence = st.tabs([
    "🔍 End-to-End Pipeline Stages",
    "⚖️ 3-Way Setup Comparison (A vs B vs C)",
    "📖 Scientific Findings & Limitations",
])

with tab_stages:
    # -------------------------------------------------------------------------
    # Stage 1: Cleaning & Masking
    # -------------------------------------------------------------------------
    st.markdown("#### 1. Tokenization & Masking")
    clean_text = analysis["clean_text"]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Cleaned Text:** `{clean_text}`")
    with col2:
        st.metric("Total Tokens", len(tokens))

    # -------------------------------------------------------------------------
    # Stage 2: Word-Level LID Tagging
    # -------------------------------------------------------------------------
    st.markdown("#### 2. Word-Level Language Identification (Frozen M3 Model)")
    
    # Render colored token badges
    ambiguous_words_set = set(["is", "to", "me", "hi", "or", "do", "main", "par", "the", "tab", "mat", "us", "mile", "den"])
    
    tokens_html = []
    for tok, tag in zip(tokens, tags):
        is_ambig = tok.lower() in ambiguous_words_set
        amb_class = " tok-amb" if is_ambig else ""
        amb_icon = " ⚑" if is_ambig else ""
        tokens_html.append(
            f'<span class="tok tok-{tag}{amb_class}" title="{tok} -> {tag}">'
            f'{tok}<small>{tag}{amb_icon}</small></span>'
        )
    
    st.markdown(f'<div class="tok-box">{"".join(tokens_html)}</div>', unsafe_allow_html=True)
    
    hi_count = sum(1 for t in tags if t == "HI")
    en_count = sum(1 for t in tags if t == "EN")
    other_count = sum(1 for t in tags if t == "OTHER")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Hindi (HI) Tokens", hi_count)
    m_col2.metric("English (EN) Tokens", en_count)
    m_col3.metric("Other Tokens", other_count)
    m_col4.metric(
        "Code-Mixing Index (CMI)",
        f"{query_cmi:.1f}%",
        help="0% = Monolingual, 50% = Perfectly Balanced Code-Mixing (Das & Gambäck)."
    )

    # -------------------------------------------------------------------------
    # Stage 3: Language-Aware Normalization
    # -------------------------------------------------------------------------
    st.markdown("#### 3. Language-Aware Processing & Intent Signal Preservation")
    
    norm_rows = []
    for tok, tag in zip(tokens, tags):
        norm_tok = pipeline.normalizer.normalize_token(tok)
        is_stop = pipeline.normalizer.is_stopword(tok, lid_tag=tag)
        
        if is_stop:
            action = f"Dropped ({tag} Stopword)"
        elif norm_tok != tok.lower():
            action = f"Phonetically Normalized: '{tok}' → '{norm_tok}'"
        else:
            action = "Preserved"
            
        norm_rows.append({
            "Token": tok,
            "LID Tag": tag,
            "Normalized Form": norm_tok if not is_stop else "-",
            "Action Applied": action,
        })
    
    with st.expander("View Token-by-Token Normalization Audit Trail", expanded=False):
        st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)
    
    st.markdown(f"**Setup C Normalized Representation:** `{text_C if text_C else '(empty)'}`")

    # -------------------------------------------------------------------------
    # Stage 4: Support Intent Classification
    # -------------------------------------------------------------------------
    st.markdown("#### 4. Downstream Support Intent Classification")
    
    model_c = classifiers.get("Setup_C")
    if model_c is not None and text_C:
        probs = model_c.predict_proba([text_C])[0]
        classes = model_c.classes_
        
        top_idx = int(np.argmax(probs))
        pred_intent = str(classes[top_idx])
        pred_conf = float(probs[top_idx])
        
        # Display top prediction
        i_col1, i_col2 = st.columns([2, 1])
        with i_col1:
            st.markdown(f"### Predicted Intent: `{pred_intent}`")
        with i_col2:
            st.metric("Model Confidence", f"{pred_conf*100:.1f}%")
            
        # Probability chart
        df_probs = pd.DataFrame({
            "Intent": classes,
            "Probability": [round(float(p), 4) for p in probs]
        }).sort_values(by="Probability", ascending=False)
        
        st.bar_chart(df_probs.set_index("Intent"), color="#4338ca")
    else:
        pred_intent = "other"
        pred_conf = 0.50
        st.warning("Setup C model not available or query was empty after normalization.")

    # -------------------------------------------------------------------------
    # Stage 5: Enterprise Queue Routing
    # -------------------------------------------------------------------------
    st.markdown("#### 5. Enterprise Support Queue Routing & Bilingual Escalation")
    
    routing = route_intent(
        intent=pred_intent,
        confidence=pred_conf,
        cmi=query_cmi,
        conf_threshold=conf_threshold,
        cmi_threshold=cmi_threshold,
    )
    
    is_escalated = routing["flag_heavy_mix"]
    is_low_conf = routing["escalated_to_human"]
    assigned_queue = routing["queue"]
    
    card_class = "queue-card escalate" if (is_escalated or is_low_conf) else "queue-card"
    
    queue_html = f"""
    <div class="{card_class}">
        <div class="queue-title">Assigned Queue: {assigned_queue}</div>
        <div class="queue-sub">
            <b>Routing Reason:</b> {routing["routing_reason"]}<br>
            <b>Code-Mixing Assessment:</b> {"⚠️ Heavy Mix Flagged (CMI " + str(query_cmi) + "% >= " + str(cmi_threshold) + "%) → Dispatched to Bilingual Tier" if is_escalated else "Standard Monolingual Workflow (CMI " + str(query_cmi) + "% < " + str(cmi_threshold) + "%)"}
        </div>
    </div>
    """
    st.markdown(queue_html, unsafe_allow_html=True)


with tab_comparison:
    st.markdown("### ⚖️ Side-by-Side Comparison of Benchmark Setups")
    st.markdown(
        "Evaluate how each representation pipeline transforms the same input query "
        "and how the underlying classifier performs on it."
    )

    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    # -------------------------------------------------------------------------
    # Setup A
    # -------------------------------------------------------------------------
    with comp_col1:
        st.markdown("#### Setup A: Monolingual English Baseline")
        st.caption("Lowercased + English Stopwords + Porter Stemmer + Word TF-IDF (1, 2)")
        st.code(text_A if text_A else "(empty)", language="text")
        
        model_a = classifiers.get("Setup_A")
        if model_a is not None and text_A:
            probs_a = model_a.predict_proba([text_A])[0]
            top_a_idx = int(np.argmax(probs_a))
            pred_a = str(model_a.classes_[top_a_idx])
            conf_a = float(probs_a[top_a_idx])
            st.metric("Setup A Prediction", f"{pred_a} ({conf_a*100:.1f}%)")
        else:
            st.write("Model A prediction: N/A")
            
        st.info("Benchmark Macro-F1: **0.8975 ± 0.0280** (Acc: 89.82%)")
        st.markdown(
            "- Blinds itself to Hindi function words.\n"
            "- Treats Hindi transliterations as OOV noise."
        )

    # -------------------------------------------------------------------------
    # Setup B
    # -------------------------------------------------------------------------
    with comp_col2:
        st.markdown("#### Setup B: Generic Subword TF-IDF")
        st.caption("Clean Masked Text + char_wb TF-IDF (1, 4), NO LID Tagger")
        st.code(text_B if text_B else "(empty)", language="text")
        
        model_b = classifiers.get("Setup_B")
        if model_b is not None and text_B:
            probs_b = model_b.predict_proba([text_B])[0]
            top_b_idx = int(np.argmax(probs_b))
            pred_b = str(model_b.classes_[top_b_idx])
            conf_b = float(probs_b[top_b_idx])
            st.metric("Setup B Prediction", f"{pred_b} ({conf_b*100:.1f}%)")
        else:
            st.write("Model B prediction: N/A")
            
        st.success("Benchmark Macro-F1: **0.9308 ± 0.0272** (Acc: 93.06%)")
        st.markdown(
            "- Naturally absorbs spelling variations (*kro* / *karo* share `kr`, `ro`).\n"
            "- Strongest generalist baseline across 15 folds."
        )

    # -------------------------------------------------------------------------
    # Setup C
    # -------------------------------------------------------------------------
    with comp_col3:
        st.markdown("#### Setup C: LID-Aware Normalization (Ours)")
        st.caption("224 Phonetic Rules + Intent Signals + char_wb (1, 4)")
        st.code(text_C if text_C else "(empty)", language="text")
        
        if model_c is not None and text_C:
            probs_c = model_c.predict_proba([text_C])[0]
            top_c_idx = int(np.argmax(probs_c))
            pred_c = str(model_c.classes_[top_c_idx])
            conf_c = float(probs_c[top_c_idx])
            st.metric("Setup C Prediction", f"{pred_c} ({conf_c*100:.1f}%)")
        else:
            st.write("Model C prediction: N/A")
            
        st.warning("Benchmark Macro-F1: **0.9185 ± 0.0258** (Acc: 91.85%)")
        st.markdown(
            "- **Top Performer on Grievance Categories:**\n"
            "  - `refund`: **0.9606** (C) > 0.9555 (B) > 0.8992 (A)\n"
            "  - `cancel_order`: **0.9723** (C) > 0.9697 (B) > 0.9560 (A)\n"
            "- Disambiguates homographs (*do* give vs *do* auxiliary)."
        )


with tab_evidence:
    st.markdown("### 🔬 Empirical Findings & Scientific Disclosures")
    
    st.markdown("#### 1. Why Generic Subwords (Setup B) are so Competitive on Hinglish")
    st.write(
        """
        Our 15-fold cross validation revealed that generic subword character n-grams (`char_wb` 1-4)
        reach **93.08% Macro-F1**, outperforming the monolingual English pipeline (+3.33% gain, $p < 0.001$).
        Because subword n-grams decompose tokens into sub-character sequences, minor transliteration noise 
        (e.g., *kro* vs *karo*, *nhi* vs *nahi*, *plz* vs *please*) naturally overlap in subword space 
        without needing external dictionary lookups.
        """
    )

    st.markdown("#### 2. The Statistical Reality of Word Normalization (Setup C' vs Setup A)")
    st.write(
        """
        Comparing word-level normalization against raw word features yielded a paired difference of 
        **-0.0076** with a 95% Bootstrap Confidence Interval of **[-0.0163, +0.0007]** ($p = 0.117$).
        Because this confidence interval **crosses zero**, we report honestly that word-level phonetic 
        normalization does not statistically outperform unnormalized text on small customer support corpora.
        """
    )

    st.markdown("#### 3. Where LID-Aware Processing Provides Actionable Value")
    st.write(
        """
        1. **High-Stakes Grievance Resolution:** On critical financial complaints (`refund` and `cancel_order`), 
           Setup C consistently beats Setup B by avoiding the loss of key Hindi verbs.
        2. **Homograph Collision Prevention:** Resolves lexical collisions where English stopwords like 'do' 
           or 'me' would otherwise be pruned, preserving crucial Hindi imperatives like *'paise wapas do'*.
        3. **Data-Derived Routing:** The Code-Mixing Index (CMI) accurately flags 37.8% of queries with 
           substantial multi-token language switching for bilingual human agents.
        """
    )

    st.markdown("#### 4. Model Failure Modes & Taxonomy")
    st.write(
        """
        - **Sparse Context Failure:** Single-word ambiguous homographs (e.g. typing just *'is'* or *'to'*) 
          lack neighbouring tokens for context disambiguation.
        - **Novel Slang & Transliterations:** Unseen conversational slang outside our 224 rules relies 
          entirely on character n-grams.
        - **Named Entity Confusions:** Obscure brand and product names may occasionally be tagged as 
          Hindi or English rather than OTHER.
        """
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"HinglishID-Router • Latency: {latency_ms:.2f} ms • "
    "CPU-Only Inference • 100% Reproducible Research"
)
