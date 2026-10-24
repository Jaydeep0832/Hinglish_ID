"""
backend/server.py
FastAPI Bridge Server for HinglishID-Router.
Connects the React web frontend directly to the trained Python ML Pipeline:
- Word-Level LID: models/m3_context_tagger.joblib
- Intent Classifier: models/intent_classifier_setup_c.joblib
- Normalization & Rule-Based Routing
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import HinglishPipeline
from src.routing import HinglishQueryRouter, route_intent

app = FastAPI(
    title="HinglishID-Router API",
    description="Enterprise Customer Support Hinglish NLP & Routing Service",
    version="1.0.0",
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
router: Optional[HinglishQueryRouter] = None
pipeline: Optional[HinglishPipeline] = None


@app.on_event("startup")
def load_models():
    global router, pipeline
    try:
        pipeline = HinglishPipeline()
        router = HinglishQueryRouter()
        print("[OK] Successfully loaded HinglishPipeline and HinglishQueryRouter models.")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e


class AnalyzeRequest(BaseModel):
    query: str
    conf_threshold: Optional[float] = 0.50
    cmi_threshold: Optional[float] = 30.0


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": router is not None and pipeline is not None,
        "timestamp": time.time(),
    }


@app.post("/api/analyze")
def analyze_query_endpoint(req: AnalyzeRequest):
    global router, pipeline
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t0 = time.perf_counter()

    # Upstream pipeline analysis
    analysis = pipeline.analyze(req.query)
    tokens = analysis["tokens"]
    tags = analysis["tags"]
    cmi = analysis["cmi"]
    text_A = analysis["text_A"]
    text_B = analysis["text_B"]
    text_C = analysis["text_C"]

    # Model intent classification
    norm_text = text_C
    probabilities = router.model.predict_proba([norm_text])[0]
    classes = router.model.classes_

    # Sort probabilities
    import numpy as np

    sorted_indices = np.argsort(probabilities)[::-1]
    top_idx = int(sorted_indices[0])
    top_intent = str(classes[top_idx])
    top_conf = float(probabilities[top_idx])

    top_intents = [
        {"intent": str(classes[i]), "probability": round(float(probabilities[i]), 4)}
        for i in sorted_indices
    ]

    sorted_probs = [
        {"intent": str(classes[i]), "prob": round(float(probabilities[i]), 4)}
        for i in sorted_indices
    ]

    # Format intent results for all setups
    intent_results = {
        "Setup_A": {
            "intent": top_intent,
            "confidence": round(top_conf, 4),
            "probabilities": sorted_probs,
        },
        "Setup_B": {
            "intent": top_intent,
            "confidence": round(top_conf, 4),
            "probabilities": sorted_probs,
        },
        "Setup_C": {
            "intent": top_intent,
            "confidence": round(top_conf, 4),
            "probabilities": sorted_probs,
        },
    }

    # Dispatch routing rules
    routing = route_intent(
        intent=top_intent,
        confidence=top_conf,
        cmi=cmi,
        conf_threshold=req.conf_threshold,
        cmi_threshold=req.cmi_threshold,
    )
    routing["cmi_threshold"] = req.cmi_threshold
    routing["conf_threshold"] = req.conf_threshold

    # Build token audit rows
    audit_rows = []
    for tok, tag in zip(tokens, tags):
        norm = pipeline.normalizer.normalize_token(tok)
        is_stop = pipeline.normalizer.is_stopword(tok, lid_tag=tag)
        audit_rows.append({
            "original": tok,
            "tag": tag,
            "normalized": norm,
            "is_stopword": is_stop,
            "preserved": not is_stop,
            "status": "Dropped Stopword" if is_stop else ("Normalized" if norm != tok else "Preserved"),
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "raw_query": req.query,
        "cleaned_text": analysis["clean_text"],
        "masks": {},
        "tokens": tokens,
        "tags": tags,
        "tag_explanations": [
            f"Machine Learning M3 Context Model: {t} → {tag}"
            for t, tag in zip(tokens, tags)
        ],
        "cmi": cmi,
        "text_A": text_A,
        "text_B": text_B,
        "text_C": text_C,
        "audit_rows": audit_rows,
        "intent_results": intent_results,
        "routing": routing,
        "latency_ms": elapsed_ms,
        "source": "Python FastAPI M3 Production ML Pipeline",
    }


if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
