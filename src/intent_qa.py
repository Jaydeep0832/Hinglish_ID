"""
src/intent_qa.py
Module M2.5 (Day 12): Label QA, Inter-Annotator Agreement (Cohen's & Fleiss' Kappa),
Repeated Stratified 5-Fold Cross-Validation Protocol, and In-Domain Customer Support LID Benchmark.

Key Objectives:
1. Inter-annotator agreement evaluation on 40-query overlap benchmark:
   - Pairwise Cohen's Kappa (A vs B, A vs C, B vs C)
   - Multi-rater Fleiss' Kappa
   - Linguistic profiling of borderline disagreements (Primary Intent Rule, Delays vs Missing, Payment Drops)
2. Stratified 5-Fold x 3 Repeats Cross-Validation Split Protocol:
   - Deterministic splits fixed across Setups A, B, and C
   - Perfectly balanced: 12 val / 48 train per intent (72 val / 288 train total per fold)
3. In-Domain Customer Support LID Benchmark:
   - 45 customer support sentences (~400 tokens) tagged with HI, EN, OTHER
   - Quantitative evaluation of production tagger vs word lookup on customer support queries
"""

import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_tagger import WordLookupTagger
from src.conll_parser import read_conll
from src.lid_context import ContextLIDTagger
from src.lid_tagger import CharNgramLIDTagger
from src.preprocess import HinglishPreprocessor

# Register classes for joblib unpickling
import __main__
__main__.WordLookupTagger = WordLookupTagger
__main__.CharNgramLIDTagger = CharNgramLIDTagger
__main__.ContextLIDTagger = ContextLIDTagger


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa for inter-rater agreement across multiple annotators.
    ratings_matrix: shape (N_subjects, k_categories) with counts of raters.
    """
    N, k = ratings_matrix.shape
    n = np.sum(ratings_matrix[0])  # number of raters per subject
    
    # Proportion of all assignments to j-th category
    p_j = np.sum(ratings_matrix, axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)
    
    # Extent of agreement for i-th subject
    P_i = (np.sum(ratings_matrix * (ratings_matrix - 1), axis=1)) / (n * (n - 1))
    P_bar = np.mean(P_i)
    
    if np.isclose(1.0 - P_e, 0.0):
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


def evaluate_inter_annotator_qa(
    overlap_csv_path: Path = Path("data/intent_overlap_40.csv"),
    output_json_path: Path = Path("results/intent_kappa_qa.json"),
    output_csv_path: Path = Path("results/intent_kappa_qa.csv"),
) -> Dict[str, Any]:
    """
    Calculate Cohen's Kappa, Fleiss' Kappa, and disagreement profiling on overlap set.
    """
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(overlap_csv_path)
    total_samples = len(df)
    
    ann_A = df["annotator_A"].tolist()
    ann_B = df["annotator_B"].tolist()
    ann_C = df["annotator_C"].tolist()
    gold = df["gold_intent"].tolist()
    
    # Pairwise Cohen's Kappa
    kappa_AB = float(cohen_kappa_score(ann_A, ann_B))
    kappa_AC = float(cohen_kappa_score(ann_A, ann_C))
    kappa_BC = float(cohen_kappa_score(ann_B, ann_C))
    mean_pairwise_kappa = float((kappa_AB + kappa_AC + kappa_BC) / 3.0)
    
    # Accuracies vs Gold
    acc_A = float(accuracy_score(gold, ann_A))
    acc_B = float(accuracy_score(gold, ann_B))
    acc_C = float(accuracy_score(gold, ann_C))
    
    # Fleiss Kappa Matrix
    unique_intents = sorted(list(set(gold)))
    intent_to_idx = {name: i for i, name in enumerate(unique_intents)}
    
    matrix = np.zeros((total_samples, len(unique_intents)))
    for i, row in df.iterrows():
        for ann_col in ["annotator_A", "annotator_B", "annotator_C"]:
            label = row[ann_col]
            matrix[i, intent_to_idx[label]] += 1
            
    fleiss_kappa = compute_fleiss_kappa(matrix)
    
    # Profiling Disagreements
    disagreements = []
    for idx, row in df.iterrows():
        votes = [row["annotator_A"], row["annotator_B"], row["annotator_C"]]
        if len(set(votes)) > 1:
            disagreements.append({
                "overlap_id": int(row["overlap_id"]),
                "text": row["text"],
                "gold_intent": row["gold_intent"],
                "annotator_A": row["annotator_A"],
                "annotator_B": row["annotator_B"],
                "annotator_C": row["annotator_C"],
                "disagreement_type": "Primary Intent Rule (Refund vs Action)" if "refund" in votes and "cancel_order" in votes
                else ("Delay vs Permanent Loss" if "delivery_delay" in votes and "order_not_delivered" in votes
                else "Payment Drop vs Missing Order"),
            })
            
    qa_results = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overlap_file": str(overlap_csv_path),
            "sample_count": total_samples,
            "rater_count": 3,
            "categories": unique_intents,
        },
        "metrics": {
            "cohen_kappa_A_vs_B": round(kappa_AB, 4),
            "cohen_kappa_A_vs_C": round(kappa_AC, 4),
            "cohen_kappa_B_vs_C": round(kappa_BC, 4),
            "mean_pairwise_kappa": round(mean_pairwise_kappa, 4),
            "fleiss_kappa": round(fleiss_kappa, 4),
            "annotator_accuracy_A_vs_gold": round(acc_A, 4),
            "annotator_accuracy_B_vs_gold": round(acc_B, 4),
            "annotator_accuracy_C_vs_gold": round(acc_C, 4),
            "total_disagreements": len(disagreements),
            "agreement_rate": round((total_samples - len(disagreements)) / total_samples, 4),
        },
        "disagreements": disagreements,
    }
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=2, ensure_ascii=False)
        
    summary_df = pd.DataFrame([{
        "Rater_Pair": "Annotator A vs Annotator B",
        "Metric": "Cohen's Kappa",
        "Score": round(kappa_AB, 4),
        "Interpretation": "Almost Perfect Agreement (> 0.81)" if kappa_AB >= 0.81 else "Substantial",
    }, {
        "Rater_Pair": "Annotator A vs Annotator C",
        "Metric": "Cohen's Kappa",
        "Score": round(kappa_AC, 4),
        "Interpretation": "Almost Perfect Agreement (> 0.81)" if kappa_AC >= 0.81 else "Substantial",
    }, {
        "Rater_Pair": "Annotator B vs Annotator C",
        "Metric": "Cohen's Kappa",
        "Score": round(kappa_BC, 4),
        "Interpretation": "Almost Perfect Agreement (> 0.81)" if kappa_BC >= 0.81 else "Substantial",
    }, {
        "Rater_Pair": "All 3 Annotators (A, B, C)",
        "Metric": "Fleiss' Kappa",
        "Score": round(fleiss_kappa, 4),
        "Interpretation": "Almost Perfect Agreement (> 0.81)" if fleiss_kappa >= 0.81 else "Substantial",
    }])
    summary_df.to_csv(output_csv_path, index=False)
    
    print("\n=== Inter-Annotator Agreement QA Results ===")
    print(f"Cohen's Kappa (A vs B): {kappa_AB:.4f}")
    print(f"Cohen's Kappa (A vs C): {kappa_AC:.4f}")
    print(f"Cohen's Kappa (B vs C): {kappa_BC:.4f}")
    print(f"Mean Pairwise Kappa   : {mean_pairwise_kappa:.4f}")
    print(f"Fleiss' Kappa (A,B,C) : {fleiss_kappa:.4f}")
    print(f"Consensus Agreement   : {total_samples - len(disagreements)} / {total_samples} ({((total_samples - len(disagreements))/total_samples)*100:.1f}%)")
    print(f"[SAVED] {output_json_path}")
    print(f"[SAVED] {output_csv_path}")
    
    return qa_results


def generate_repeated_stratified_cv_splits(
    dataset_csv_path: Path = Path("data/intent_dataset.csv"),
    output_splits_path: Path = Path("data/intent_splits_5fold.json"),
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Generate deterministic 5-fold cross-validation splits with 3 repeats (15 total folds).
    Enforces exact stratified balance: 12 val / 48 train per intent in every fold.
    """
    df = pd.read_csv(dataset_csv_path)
    X = df["text"].values
    y = df["intent"].values
    ids = df["id"].astype(str).values
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    folds = []
    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        repeat_idx = split_idx // n_splits
        fold_idx = split_idx % n_splits
        
        train_ids = ids[train_idx].tolist()
        val_ids = ids[val_idx].tolist()
        
        val_intents = pd.Series(y[val_idx]).value_counts().to_dict()
        train_intents = pd.Series(y[train_idx]).value_counts().to_dict()
        
        folds.append({
            "split_id": split_idx,
            "repeat": repeat_idx,
            "fold": fold_idx,
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "train_query_ids": train_ids,
            "val_query_ids": val_ids,
            "val_class_distribution": val_intents,
            "train_class_distribution": train_intents,
        })
        
    config = {
        "metadata": {
            "generator": "RepeatedStratifiedKFold",
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "total_splits": len(folds),
            "random_state": random_state,
            "total_queries": len(df),
            "intents": sorted(list(set(y))),
            "val_queries_per_fold": len(df) // n_splits,
            "train_queries_per_fold": len(df) - (len(df) // n_splits),
        },
        "folds": folds,
    }
    
    with open(output_splits_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    print(f"\n[CV PROTOCOL] Generated {len(folds)} stratified folds ({n_repeats} repeats x {n_splits} folds).")
    print(f"[SAVED] {output_splits_path}")
    return config


def generate_indomain_lid_dataset(
    output_conll_path: Path = Path("data/lid_indomain.conll"),
) -> Path:
    """
    Generate 45 customer support queries tagged token-by-token (HI, EN, OTHER)
    to benchmark domain transfer resilience from social media Twitter text.
    """
    output_conll_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 45 Customer Support Sentences with expert ground-truth token annotations:
    support_corpus: List[List[Tuple[str, str]]] = [
        # Refund queries
        [("Mera", "HI"), ("order", "EN"), ("abhi", "HI"), ("tak", "HI"), ("nahi", "HI"), ("aaya", "HI"), (",", "OTHER"), ("please", "EN"), ("refund", "EN"), ("karo", "HI")],
        [("Please", "EN"), ("refund", "EN"), ("my", "EN"), ("money", "EN"), (",", "OTHER"), ("wrong", "EN"), ("item", "EN"), ("deliver", "EN"), ("hua", "HI"), ("hai", "HI")],
        [("Maine", "HI"), ("item", "EN"), ("return", "EN"), ("kar", "HI"), ("diya", "HI"), ("par", "HI"), ("paise", "HI"), ("wapas", "HI"), ("nahi", "HI"), ("mile", "HI")],
        [("Refund", "EN"), ("kab", "HI"), ("tak", "HI"), ("aayega", "HI"), ("mere", "HI"), ("bank", "EN"), ("account", "EN"), ("mein", "HI"), ("?", "OTHER")],
        [("Product", "EN"), ("was", "EN"), ("damaged", "EN"), (",", "OTHER"), ("I", "EN"), ("need", "EN"), ("full", "EN"), ("refund", "EN"), ("now", "EN")],
        [("Paise", "HI"), ("wapas", "HI"), ("chahiye", "HI"), ("mujhe", "HI"), ("abhi", "HI"), ("ke", "HI"), ("abhi", "HI")],
        [("Refund", "EN"), ("initiate", "EN"), ("nahi", "HI"), ("hua", "HI"), ("abhi", "HI"), ("tak", "HI"), (",", "OTHER"), ("txn", "EN"), ("<ORDER_ID>", "OTHER")],
        [("Item", "EN"), ("returned", "EN"), ("to", "EN"), ("delivery", "EN"), ("boy", "EN"), (",", "OTHER"), ("refund", "EN"), ("please", "EN")],

        # Cancel order queries
        [("Order", "EN"), ("cancel", "EN"), ("karna", "HI"), ("hai", "HI"), ("plz", "EN"), ("jaldi", "HI")],
        [("Please", "EN"), ("cancel", "EN"), ("my", "EN"), ("order", "EN"), ("<ORDER_ID>", "OTHER"), ("immediately", "EN")],
        [("Mujhe", "HI"), ("yeh", "HI"), ("order", "EN"), ("nahi", "HI"), ("chahiye", "HI"), (",", "OTHER"), ("cancel", "EN"), ("kar", "HI"), ("do", "HI")],
        [("Mera", "HI"), ("order", "EN"), ("cancel", "EN"), ("kardo", "HI"), ("maine", "HI"), ("galti", "HI"), ("se", "HI"), ("order", "EN"), ("kar", "HI"), ("diya", "HI")],
        [("I", "EN"), ("want", "EN"), ("to", "EN"), ("cancel", "EN"), ("this", "EN"), ("order", "EN"), ("as", "EN"), ("delivery", "EN"), ("is", "EN"), ("late", "EN")],
        [("Order", "EN"), ("abhi", "HI"), ("cancel", "EN"), ("karo", "HI"), ("please", "EN")],
        [("Galti", "HI"), ("se", "HI"), ("wrong", "EN"), ("address", "EN"), ("daal", "HI"), ("diya", "HI"), (",", "OTHER"), ("order", "EN"), ("cancel", "EN"), ("karna", "HI"), ("hai", "HI")],
        [("Can", "EN"), ("you", "EN"), ("please", "EN"), ("cancel", "EN"), ("order", "EN"), ("<ORDER_ID>", "OTHER"), ("?", "OTHER")],

        # Order not delivered queries
        [("Status", "EN"), ("delivered", "EN"), ("dikha", "HI"), ("raha", "HI"), ("hai", "HI"), ("but", "EN"), ("mujhe", "HI"), ("item", "EN"), ("nahi", "HI"), ("mila", "HI"), ("!", "OTHER")],
        [("Mera", "HI"), ("package", "EN"), ("missing", "EN"), ("hai", "HI"), (",", "OTHER"), ("kahan", "HI"), ("gaya", "HI"), ("?", "OTHER")],
        [("Kisi", "HI"), ("ne", "HI"), ("deliver", "EN"), ("nahi", "HI"), ("kiya", "HI"), ("aur", "HI"), ("delivered", "EN"), ("mark", "EN"), ("kar", "HI"), ("diya", "HI")],
        [("4", "OTHER"), ("din", "HI"), ("ho", "HI"), ("gaye", "HI"), ("parcel", "EN"), ("gayab", "HI"), ("hai", "HI")],
        [("I", "EN"), ("never", "EN"), ("received", "EN"), ("this", "EN"), ("delivery", "EN"), (",", "OTHER"), ("tracking", "EN"), ("is", "EN"), ("wrong", "EN")],
        [("Delivery", "EN"), ("marked", "EN"), ("without", "EN"), ("OTP", "EN"), (",", "OTHER"), ("mujhe", "HI"), ("saman", "HI"), ("nahi", "HI"), ("mila", "HI")],
        [("Order", "EN"), ("missing", "EN"), ("from", "EN"), ("doorstep", "EN"), (",", "OTHER"), ("please", "EN"), ("investigate", "EN")],
        [("Courier", "EN"), ("wala", "HI"), ("bina", "HI"), ("call", "EN"), ("kiye", "HI"), ("return", "EN"), ("kar", "HI"), ("diya", "HI")],

        # Delivery delay queries
        [("My", "EN"), ("order", "EN"), ("is", "EN"), ("delayed", "EN"), (",", "OTHER"), ("kab", "HI"), ("tak", "HI"), ("aayega", "HI"), ("?", "OTHER")],
        [("Expected", "EN"), ("delivery", "EN"), ("kal", "HI"), ("thi", "HI"), (",", "OTHER"), ("ab", "HI"), ("kab", "HI"), ("milega", "HI"), ("?", "OTHER")],
        [("Kitne", "HI"), ("din", "HI"), ("aur", "HI"), ("lagenge", "HI"), ("shipment", "EN"), ("pahunchne", "HI"), ("mein", "HI"), ("?", "OTHER")],
        [("Bhai", "HI"), ("bohot", "HI"), ("der", "HI"), ("ho", "HI"), ("gayi", "HI"), (",", "OTHER"), ("delivery", "EN"), ("time", "EN"), ("update", "EN"), ("karo", "HI")],
        [("Out", "EN"), ("for", "EN"), ("delivery", "EN"), ("hokar", "HI"), ("bhi", "HI"), ("6", "OTHER"), ("ghante", "HI"), ("lag", "HI"), ("gaye", "HI")],
        [("Please", "EN"), ("expedite", "EN"), ("my", "EN"), ("parcel", "EN"), ("delivery", "EN"), (",", "OTHER"), ("running", "EN"), ("very", "EN"), ("late", "EN")],
        [("Delivery", "EN"), ("date", "EN"), ("roz", "HI"), ("aage", "HI"), ("kyun", "HI"), ("badha", "HI"), ("rahe", "HI"), ("ho", "HI"), ("?", "OTHER")],

        # Payment failed queries
        [("Payment", "EN"), ("fail", "EN"), ("ho", "HI"), ("gaya", "HI"), ("but", "EN"), ("bank", "EN"), ("se", "HI"), ("paise", "HI"), ("kat", "HI"), ("gaye", "HI")],
        [("Money", "EN"), ("deducted", "EN"), ("from", "EN"), ("account", "EN"), ("but", "EN"), ("order", "EN"), ("nahi", "HI"), ("aaya", "HI")],
        [("Transaction", "EN"), ("failed", "EN"), ("phir", "HI"), ("bhi", "HI"), ("amount", "EN"), ("debited", "EN"), ("dikh", "HI"), ("raha", "HI"), ("hai", "HI")],
        [("Double", "EN"), ("payment", "EN"), ("cut", "EN"), ("ho", "HI"), ("gaya", "HI"), ("single", "EN"), ("item", "EN"), ("ke", "HI"), ("liye", "HI")],
        [("UPI", "EN"), ("payment", "EN"), ("timeout", "EN"), ("hua", "HI"), ("par", "HI"), ("account", "EN"), ("se", "HI"), ("paise", "HI"), ("gaye", "HI")],
        [("Card", "EN"), ("charged", "EN"), ("twice", "EN"), ("for", "EN"), ("order", "EN"), ("<ORDER_ID>", "OTHER"), (",", "OTHER"), ("please", "EN"), ("check", "EN")],
        [("Amount", "EN"), ("deducted", "EN"), ("from", "EN"), ("wallet", "EN"), ("but", "EN"), ("cart", "EN"), ("is", "EN"), ("still", "EN"), ("full", "EN")],

        # Other inquiries
        [("Thank", "EN"), ("you", "EN"), ("so", "EN"), ("much", "EN"), ("team", "EN"), (",", "OTHER"), ("support", "EN"), ("bohot", "HI"), ("acchi", "HI"), ("thi", "HI"), ("!", "OTHER")],
        [("Your", "EN"), ("working", "EN"), ("hours", "EN"), ("kya", "HI"), ("hain", "HI"), ("?", "OTHER")],
        [("App", "EN"), ("bahut", "HI"), ("accha", "HI"), ("hai", "HI"), (",", "OTHER"), ("great", "EN"), ("experience", "EN")],
        [("How", "EN"), ("do", "EN"), ("I", "EN"), ("change", "EN"), ("my", "EN"), ("registered", "EN"), ("phone", "EN"), ("number", "EN"), ("?", "OTHER")],
        [("Can", "EN"), ("I", "EN"), ("get", "EN"), ("a", "EN"), ("discount", "EN"), ("code", "EN"), ("for", "EN"), ("next", "EN"), ("order", "EN"), ("?", "OTHER")],
        [("Mujhe", "HI"), ("invoice", "EN"), ("download", "EN"), ("karna", "HI"), ("hai", "HI"), (",", "OTHER"), ("option", "EN"), ("kahan", "HI"), ("hai", "HI"), ("?", "OTHER")],
        [("Executive", "EN"), ("se", "HI"), ("baat", "HI"), ("karao", "HI"), ("please", "EN")],
    ]

    total_tokens = sum(len(sent) for sent in support_corpus)
    
    with open(output_conll_path, "w", encoding="utf-8") as f:
        for sent in support_corpus:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")
            
    print(f"[CONLL] Saved in-domain customer support LID dataset: {output_conll_path}")
    print(f"[CONLL] Total Sentences: {len(support_corpus)} | Total Tokens: {total_tokens}")
    return output_conll_path


def evaluate_indomain_lid_benchmark(
    conll_path: Path = Path("data/lid_indomain.conll"),
    models_dir: Path = Path("models"),
    results_csv_path: Path = Path("results/table_5_indomain_lid.csv"),
    results_json_path: Path = Path("results/table_5_indomain_lid.json"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate M0 Word Lookup and M3 Production Context Tagger on the in-domain customer support dataset.
    Quantifies domain shift resilience from social media Twitter text to customer support.
    """
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse in-domain ConLL
    sents = read_conll(conll_path)
    flat_tokens = [tok for s_toks, _ in sents for tok in s_toks]
    flat_gold = [tag for _, s_tags in sents for tag in s_tags]
    y_true = np.array(flat_gold)
    
    # 2. Load models
    m0_lookup = WordLookupTagger.load(models_dir / "baseline_lookup.joblib")
    m3_final = ContextLIDTagger.load(models_dir / "lid_final.joblib")
    
    # 3. Predict with M0 Lookup
    preds_m0 = [m0_lookup.predict_token(t) for t in flat_tokens]
    
    # 4. Predict with M3 Context
    preds_m3 = []
    for s_toks, _ in sents:
        preds_m3.extend(m3_final.predict_sentence(s_toks))
        
    labels = ["HI", "EN"]
    if "OTHER" in set(flat_gold):
        labels.append("OTHER")
        
    # 5. Compute Metrics
    def compute_metrics(y_pred):
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "hi_f1": round(float(report["HI"]["f1-score"]), 4),
            "en_f1": round(float(report["EN"]["f1-score"]), 4),
            "other_f1": round(float(report["OTHER"]["f1-score"]), 4) if "OTHER" in report else 0.0,
            "errors": int(np.sum(y_true != y_pred)),
        }
        
    m0_metrics = compute_metrics(np.array(preds_m0))
    m3_metrics = compute_metrics(np.array(preds_m3))
    
    table_rows = [
        {
            "Model_ID": "M0_Lookup",
            "Model_Name": "M0: Word Lookup Baseline",
            "Domain": "Customer Support (In-Domain)",
            "Tokens": len(flat_tokens),
            "Accuracy": f"{m0_metrics['accuracy']*100:.2f}%",
            "Macro_F1": m0_metrics["macro_f1"],
            "HI_F1": m0_metrics["hi_f1"],
            "EN_F1": m0_metrics["en_f1"],
            "OTHER_F1": m0_metrics["other_f1"],
            "Total_Errors": m0_metrics["errors"],
        },
        {
            "Model_ID": "M3_Final_Context",
            "Model_Name": "M3: Production Context Tagger (Ours)",
            "Domain": "Customer Support (In-Domain)",
            "Tokens": len(flat_tokens),
            "Accuracy": f"{m3_metrics['accuracy']*100:.2f}%",
            "Macro_F1": m3_metrics["macro_f1"],
            "HI_F1": m3_metrics["hi_f1"],
            "EN_F1": m3_metrics["en_f1"],
            "OTHER_F1": m3_metrics["other_f1"],
            "Total_Errors": m3_metrics["errors"],
        },
    ]
    
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(results_csv_path, index=False)
    
    report_dict = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conll_file": str(conll_path),
            "total_sentences": len(sents),
            "total_tokens": len(flat_tokens),
            "label_distribution": pd.Series(flat_gold).value_counts().to_dict(),
        },
        "models": {
            "M0_Lookup": m0_metrics,
            "M3_Final_Context": m3_metrics,
        },
    }
    
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
    print("\n=== Table 5: In-Domain Customer Support LID Benchmark ===")
    print(df_table[["Model_ID", "Accuracy", "Macro_F1", "HI_F1", "EN_F1", "Total_Errors"]].to_string(index=False))
    print(f"[SAVED] {results_csv_path}")
    print(f"[SAVED] {results_json_path}")
    
    return df_table, report_dict


def main():
    print("Executing Module M2.5 (Day 12): QA, Cohen's Kappa, CV Folds & In-Domain LID...")
    
    # 1. Inter-Annotator Agreement QA
    evaluate_inter_annotator_qa()
    
    # 2. Repeated Stratified 5-Fold Splits
    generate_repeated_stratified_cv_splits()
    
    # 3. Generate In-domain LID Dataset
    generate_indomain_lid_dataset()
    
    # 4. Evaluate In-Domain LID Benchmark
    evaluate_indomain_lid_benchmark()


if __name__ == "__main__":
    main()
