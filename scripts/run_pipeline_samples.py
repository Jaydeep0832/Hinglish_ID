"""
scripts/run_pipeline_samples.py
Module M2.7 (Day 14): Runs 20 representative customer support queries through
the unified pipeline to benchmark latency, CMI distribution, and compare
Setups A, B, and C.
"""

import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import HinglishPipeline


SAMPLE_QUERIES = [
    # Refund queries
    "Mera order abhi tak nahi aaya, please refund karo",
    "Khana bohot late ho gaya, order cancel karke paise wapas do",
    "Product was completely damaged, I need full refund immediately",
    "Refund initiate nahi hua abhi tak, transaction #948271",
    
    # Cancel order queries
    "Order cancel karna hai plz jaldi",
    "Please cancel my order #994821 immediately",
    "Mujhe yeh order nahi chahiye, cancel kar do",
    "I want to cancel this order as delivery is too late",
    
    # Order not delivered queries
    "Status delivered dikha raha hai but mujhe item nahi mila!",
    "4 din ho gaye parcel gayab hai, delivery boy switch off hai",
    "I never received this delivery, tracking is wrong",
    "Delivery marked without OTP, mujhe saman nahi mila",
    
    # Delivery delay queries
    "My order is delayed, kab tak aayega?",
    "Expected delivery kal thi, ab kab milega?",
    "Bhai bohot der ho gayi, delivery time update karo",
    "Out for delivery hokar bhi 6 ghante lag gaye, kab aayega?",
    
    # Payment failed queries
    "Payment fail ho gaya but bank se paise kat gaye",
    "Money deducted from account but order nahi aaya",
    "Card charged twice for order #482910, please check",
    
    # Other inquiries
    "Thank you so much team, customer support service bohot acchi thi!",
]


def main():
    print("Executing Pipeline Demonstration on 20 Sample Customer Support Queries...")
    pipeline = HinglishPipeline()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    latencies = []
    rows = []
    full_output = []

    for idx, q in enumerate(SAMPLE_QUERIES, 1):
        res = pipeline.analyze(q)
        latencies.append(res["latency_ms"])
        
        row = {
            "ID": idx,
            "Raw_Query": q,
            "CMI": res["cmi"],
            "Tags": " ".join(res["tags"]),
            "Setup_A_Raw": res["text_A"],
            "Setup_B_Generic": res["text_B"],
            "Setup_C_LIDAware": res["text_C"],
            "Latency_ms": res["latency_ms"],
        }
        rows.append(row)
        full_output.append(res)

    df = pd.DataFrame(rows)
    csv_path = results_dir / "table_6_pipeline_samples.csv"
    json_path = results_dir / "table_6_pipeline_samples.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {csv_path}")
    print(f"[SAVED] {json_path}")
    print(f"\n=== Latency Benchmark ({len(SAMPLE_QUERIES)} queries) ===")
    print(f"Mean Latency   : {sum(latencies)/len(latencies):.2f} ms")
    print(f"Min Latency    : {min(latencies):.2f} ms")
    print(f"Max Latency    : {max(latencies):.2f} ms")
    print(f"Sub-10ms Gate  : {'PASSED' if (sum(latencies)/len(latencies)) < 10.0 else 'CHECK'}")
    
    print("\n=== Sample Comparison (First 5 Queries) ===")
    print(df[["ID", "Raw_Query", "Setup_B_Generic", "Setup_C_LIDAware", "Latency_ms"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
