"""
tests/test_routing.py
Unit tests for Module M3.3: Rule-Based Routing & Code-Mixing Flagging.
"""

import pytest

from src.routing import (
    DEFAULT_CMI_THRESHOLD,
    DEFAULT_CONF_THRESHOLD,
    HUMAN_AGENT_QUEUE,
    QUEUE_MAPPING,
    HinglishQueryRouter,
    route,
    route_intent,
)


def test_queue_mapping_all_intents():
    """Verify each supported intent routes to its designated operational queue."""
    expected_queues = {
        "refund": "Refunds",
        "cancel_order": "Order Ops",
        "order_not_delivered": "Logistics",
        "delivery_delay": "Logistics",
        "payment_failed": "Payments",
        "other": "General FAQ",
    }
    for intent, expected_queue in expected_queues.items():
        res = route_intent(intent=intent, confidence=0.85, cmi=15.0)
        assert res["queue"] == expected_queue, f"Failed for intent {intent}"
        assert not res["escalated_to_human"]
        assert not res["flag_heavy_mix"]


def test_low_confidence_escalation():
    """Verify predictions below confidence threshold route to human agent."""
    res_low = route_intent(
        intent="refund",
        confidence=0.42,
        cmi=10.0,
        conf_threshold=0.50,
    )
    assert res_low["queue"] == HUMAN_AGENT_QUEUE
    assert res_low["escalated_to_human"] is True

    res_high = route_intent(
        intent="refund",
        confidence=0.51,
        cmi=10.0,
        conf_threshold=0.50,
    )
    assert res_high["queue"] == "Refunds"
    assert res_high["escalated_to_human"] is False


def test_heavy_code_mixing_flag_threshold():
    """Verify CMI threshold behaves correctly at boundary conditions."""
    # Under threshold
    res_under = route_intent(intent="cancel_order", confidence=0.90, cmi=29.9, cmi_threshold=30.0)
    assert res_under["flag_heavy_mix"] is False

    # Exact threshold
    res_exact = route_intent(intent="cancel_order", confidence=0.90, cmi=30.0, cmi_threshold=30.0)
    assert res_exact["flag_heavy_mix"] is True

    # Well above threshold
    res_above = route_intent(intent="cancel_order", confidence=0.90, cmi=45.5, cmi_threshold=30.0)
    assert res_above["flag_heavy_mix"] is True


def test_unknown_intent_handling():
    """Verify handling of unexpected/unknown intent string."""
    res = route_intent(intent="unknown_intent_xyz", confidence=0.80, cmi=12.0)
    assert res["queue"] == "General FAQ"


def test_end_to_end_query_router():
    """Verify HinglishQueryRouter end-to-end execution on live queries."""
    router = HinglishQueryRouter()

    # Query 1: Clear refund intent
    res1 = router.route("Mera refund abhi tak nahi aaya please paise wapas karo")
    assert res1["predicted_intent"] == "refund"
    assert res1["queue"] == "Refunds"
    assert res1["confidence"] > 0.50
    assert len(res1["top_intents"]) == 3
    assert res1["top_intents"][0]["intent"] == "refund"

    # Query 2: Clear cancel intent
    res2 = router.route("Order cancel kardo mujhe nahi chahiye")
    assert res2["predicted_intent"] == "cancel_order"
    assert res2["queue"] == "Order Ops"

    # Query 3: Payment failure intent
    res3 = router.route("Paise account se kat gaye par order confirm nahi hua payment failed")
    assert res3["predicted_intent"] == "payment_failed"
    assert res3["queue"] == "Payments"

    # Query 4: Code-mixed query with high CMI
    res4 = router.route("Please delay order update batao kab tak deliver hoga parcel")
    assert res4["cmi"] > 0.0
    assert "tokens" in res4
    assert "lid_tags" in res4
    assert len(res4["tokens"]) == len(res4["lid_tags"])


def test_convenience_route_function():
    """Verify single-call route() function operates cleanly."""
    res = route("Paise refund kardo")
    assert "queue" in res
    assert "predicted_intent" in res
    assert "confidence" in res
    assert "flag_heavy_mix" in res
