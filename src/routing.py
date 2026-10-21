"""
src/routing.py
Module M3.3 (Day 17): Rule-Based Intent Routing & Code-Mixing Flagging.

Integrates:
1. Downstream Intent Classification (using trained Setup C model).
2. Queue Routing Rules:
   - 'refund' -> 'Refunds'
   - 'cancel_order' -> 'Order Ops'
   - 'order_not_delivered' -> 'Logistics'
   - 'delivery_delay' -> 'Logistics'
   - 'payment_failed' -> 'Payments'
   - 'other' -> 'General FAQ'
3. Confidence thresholding: Predictions with confidence < 0.50 are routed to 'Human agent (low confidence)'.
4. Data-driven Code-Mixing Flag: CMI >= 30.0 flags queries with substantial Hindi-English mixing
   (derived from empirical 60th percentile of dataset queries / 35th percentile of mixed queries).
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import HinglishPipeline


# Target departmental queues for customer support
QUEUE_MAPPING: Dict[str, str] = {
    "refund": "Refunds",
    "cancel_order": "Order Ops",
    "order_not_delivered": "Logistics",
    "delivery_delay": "Logistics",
    "payment_failed": "Payments",
    "other": "General FAQ",
}

# Empirical thresholds
DEFAULT_CONF_THRESHOLD: float = 0.50
DEFAULT_CMI_THRESHOLD: float = 30.0
HUMAN_AGENT_QUEUE: str = "Human agent (low confidence)"


def route_intent(
    intent: str,
    confidence: float,
    cmi: float,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    cmi_threshold: float = DEFAULT_CMI_THRESHOLD,
) -> Dict[str, Any]:
    """
    Route an intent prediction and CMI score to the appropriate support queue.

    Args:
        intent: Predicted intent category.
        confidence: Prediction probability [0.0, 1.0].
        cmi: Code-Mixing Index [0.0, 100.0].
        conf_threshold: Minimum confidence required for automated routing.
        cmi_threshold: Threshold above which heavy code-mixing flag is set.

    Returns:
        Dictionary containing destination queue, heavy mix flag, and routing metadata.
    """
    if intent not in QUEUE_MAPPING:
        base_queue = "General FAQ"
    else:
        base_queue = QUEUE_MAPPING[intent]

    is_low_conf = confidence < conf_threshold
    queue = HUMAN_AGENT_QUEUE if is_low_conf else base_queue
    flag_heavy_mix = cmi >= cmi_threshold

    if is_low_conf:
        reason = f"Low confidence ({confidence:.2f} < {conf_threshold:.2f}); escalated to human agent"
    else:
        reason = f"Routed to {queue} based on high-confidence '{intent}' ({confidence:.2f})"

    return {
        "intent": intent,
        "confidence": round(float(confidence), 4),
        "queue": queue,
        "flag_heavy_mix": bool(flag_heavy_mix),
        "cmi": round(float(cmi), 2),
        "escalated_to_human": bool(is_low_conf),
        "routing_reason": reason,
    }


class HinglishQueryRouter:
    """End-to-end Router combining LID Pipeline, Intent Classifier, and Rule-Based Dispatch."""

    def __init__(
        self,
        classifier_path: Optional[Path] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        cmi_threshold: float = DEFAULT_CMI_THRESHOLD,
    ):
        self.conf_threshold = conf_threshold
        self.cmi_threshold = cmi_threshold
        self.pipeline = HinglishPipeline()

        if classifier_path is None:
            classifier_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "intent_classifier_setup_c.joblib"
            )

        self.classifier_path = Path(classifier_path)
        if not self.classifier_path.exists():
            raise FileNotFoundError(f"Intent classifier not found at {self.classifier_path}")

        self.model = joblib.load(self.classifier_path)

    def route(self, query: str) -> Dict[str, Any]:
        """
        Process a raw customer query end-to-end through LID, intent classification, and queue routing.

        Args:
            query: Raw user query string.

        Returns:
            Dictionary with tokens, LID tags, CMI, normalized text, intent probabilities, and assigned queue.
        """
        # Step 1: Upstream LID and linguistic analysis
        analysis = self.pipeline.analyze(query)
        norm_text = analysis["text_C"]
        cmi = analysis["cmi"]

        # Step 2: Intent classification with probability estimation
        probabilities = self.model.predict_proba([norm_text])[0]
        classes = self.model.classes_

        top_idx = int(np.argmax(probabilities))
        pred_intent = str(classes[top_idx])
        pred_conf = float(probabilities[top_idx])

        # Step 3: Top-3 intent ranking
        sorted_indices = np.argsort(probabilities)[::-1]
        top_intents = [
            {"intent": str(classes[i]), "probability": round(float(probabilities[i]), 4)}
            for i in sorted_indices[:3]
        ]

        # Step 4: Dispatch routing rules
        routing = route_intent(
            intent=pred_intent,
            confidence=pred_conf,
            cmi=cmi,
            conf_threshold=self.conf_threshold,
            cmi_threshold=self.cmi_threshold,
        )

        return {
            "query": query,
            "tokens": analysis["tokens"],
            "lid_tags": analysis["tags"],
            "cmi": cmi,
            "normalized_text": norm_text,
            "predicted_intent": pred_intent,
            "confidence": round(pred_conf, 4),
            "top_intents": top_intents,
            "queue": routing["queue"],
            "flag_heavy_mix": routing["flag_heavy_mix"],
            "escalated_to_human": routing["escalated_to_human"],
            "routing_reason": routing["routing_reason"],
        }


def route(
    query: str,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    cmi_threshold: float = DEFAULT_CMI_THRESHOLD,
) -> Dict[str, Any]:
    """Convenience function to route a customer query in a single call."""
    router = HinglishQueryRouter(
        conf_threshold=conf_threshold,
        cmi_threshold=cmi_threshold,
    )
    return router.route(query)


if __name__ == "__main__":
    sample_queries = [
        "Mera refund abhi tak nahi aaya please paise wapas karo",
        "Order cancel kardo mujhe nahi chahiye",
        "Money cut from account but order nahi mila",
        "Parcel kab aayega delay ho gaya hai",
        "Hello good morning",
    ]

    router = HinglishQueryRouter()
    print("=" * 80)
    print("HINGLISH INTENT ROUTING DEMO (Module M3.3)")
    print("=" * 80)
    for q in sample_queries:
        out = router.route(q)
        print(f"\nQuery: '{q}'")
        print(f"  Intent:     {out['predicted_intent']} (Conf: {out['confidence']:.2%})")
        print(f"  CMI:        {out['cmi']:.1f}% | Heavy Code-Mix Flag: {out['flag_heavy_mix']}")
        print(f"  Queue:      {out['queue']}")
        print(f"  Escalated:  {out['escalated_to_human']}")
