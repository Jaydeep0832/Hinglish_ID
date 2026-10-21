"""
src/demo_tagger.py
Interactive Demonstration & Query Tagger for HinglishID-Router (Module M1.7).

Integrates:
- HinglishPreprocessor (src/preprocess.py) for entity masking (<URL>, <EMOJI>, <ORDER_ID>)
- CharNgramLIDTagger (src/lid_tagger.py) for subword word-level language identification
- Gambäck & Das CMI calculation (src/conll_parser.py)
- ANSI terminal colored visualization & structured JSON export
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.conll_parser import calculate_cmi, is_code_mixed
from src.lid_tagger import CharNgramLIDTagger
from src.preprocess import HinglishPreprocessor, is_special_token


# ANSI Color Codes for Terminal Rendering
COLOR_HI = "\033[94m"     # Blue for Hindi
COLOR_EN = "\033[92m"     # Green for English
COLOR_OTHER = "\033[93m"  # Yellow for Other / Entities / Punctuation
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"


# 10 Representative Customer Support Sample Queries covering the 6 support intents
SAMPLE_QUERIES = [
    {
        "id": 1,
        "intent": "refund",
        "text": "Mera order abhi tak nahi aaya, please refund karo mera paisa #987654!",
    },
    {
        "id": 2,
        "intent": "refund",
        "text": "Plzzzzz refund my money, wrong product delivered hua hai 😡",
    },
    {
        "id": 3,
        "intent": "cancel_order",
        "text": "Order cancel karna hai urgently, galat address dal gaya tha.",
    },
    {
        "id": 4,
        "intent": "cancel_order",
        "text": "Please cancel my order ORD-55412 immediately, I changed my mind.",
    },
    {
        "id": 5,
        "intent": "delivery_delay",
        "text": "Package bohot late ho gaya hai, kab tak deliver hoga check https://track.in?",
    },
    {
        "id": 6,
        "intent": "delivery_delay",
        "text": "Expected delivery date 2 din pehle thi, why so much delay sir?",
    },
    {
        "id": 7,
        "intent": "order_not_delivered",
        "text": "Status delivered dikha raha hai but mujhe item nahi mila!",
    },
    {
        "id": 8,
        "intent": "payment_failed",
        "text": "Payment fail ho gaya but bank account se 500 rs cut gaye.",
    },
    {
        "id": 9,
        "intent": "payment_failed",
        "text": "Transaction failed during checkout, order placed nahi hua.",
    },
    {
        "id": 10,
        "intent": "other",
        "text": "Thank you so much team, customer support service bohot acchi thi! ❤️",
    },
]


class HinglishDemoTagger:
    """
    High-level query-level Language Identification and Code-Mixing Analyzer.
    """

    def __init__(self, model_path: Optional[str] = "models/lid_lr.joblib"):
        self.preprocessor = HinglishPreprocessor(lower=False)
        self.model_path = Path(model_path) if model_path else None
        
        if self.model_path and self.model_path.is_file():
            self.tagger = CharNgramLIDTagger.load(self.model_path)
        else:
            self.tagger = None

    def tag_query(self, query: str) -> Dict[str, Any]:
        """
        Clean, tokenize, and tag a customer support query.
        
        Returns:
            Dictionary containing tokens, tags, token-level probabilities, CMI, and metadata.
        """
        if not query or not query.strip():
            return {
                "raw_query": query,
                "cleaned_text": "",
                "tokens": [],
                "tags": [],
                "token_details": [],
                "cmi": 0.0,
                "is_code_mixed": False,
                "language_distribution": {"HI": 0, "EN": 0, "OTHER": 0},
            }

        cleaned = self.preprocessor.clean(query)
        tokens = self.preprocessor.tokenize(cleaned)

        tags: List[str] = []
        token_details: List[Dict[str, Any]] = []

        for tok in tokens:
            # Masked placeholders and punctuation are marked as OTHER
            if is_special_token(tok) or not any(c.isalnum() for c in tok):
                tag = "OTHER"
                probs = {"HI": 0.0, "EN": 0.0, "OTHER": 1.0}
            elif self.tagger:
                tag = self.tagger.predict_token(tok)
                probs = self.tagger.predict_proba_token(tok)
                probs["OTHER"] = 0.0
            else:
                tag = "OTHER"
                probs = {"HI": 0.0, "EN": 0.0, "OTHER": 1.0}

            tags.append(tag)
            token_details.append({
                "token": tok,
                "tag": tag,
                "probabilities": probs,
            })

        cmi_score = calculate_cmi(tags)
        code_mixed = is_code_mixed(tags)

        counts = {
            "HI": tags.count("HI"),
            "EN": tags.count("EN"),
            "OTHER": tags.count("OTHER"),
        }

        return {
            "raw_query": query,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "tags": tags,
            "token_details": token_details,
            "cmi": round(cmi_score, 2),
            "is_code_mixed": code_mixed,
            "language_distribution": counts,
        }

    def format_colored_tokens(self, result: Dict[str, Any]) -> str:
        """Format tokens with ANSI color highlighting for console display."""
        colored_parts = []
        for d in result["token_details"]:
            tok = d["token"]
            tag = d["tag"]
            if tag == "HI":
                colored_parts.append(f"{COLOR_HI}{tok}[HI]{COLOR_RESET}")
            elif tag == "EN":
                colored_parts.append(f"{COLOR_EN}{tok}[EN]{COLOR_RESET}")
            else:
                colored_parts.append(f"{COLOR_OTHER}{tok}[OTHER]{COLOR_RESET}")
        return " ".join(colored_parts)

    def format_html_pills(self, result: Dict[str, Any]) -> str:
        """Format tokens as modern HTML badges for notebook and UI rendering."""
        pills = []
        for d in result["token_details"]:
            tok = d["token"]
            tag = d["tag"]
            if tag == "HI":
                bg, fg = "#2b5c8f", "#ffffff"
            elif tag == "EN":
                bg, fg = "#28a745", "#ffffff"
            else:
                bg, fg = "#6c757d", "#ffffff"
            pills.append(
                f'<span style="background-color: {bg}; color: {fg}; '
                f'padding: 3px 8px; border-radius: 4px; font-weight: 500; '
                f'margin: 2px 3px; display: inline-block; font-size: 13px;">'
                f'{tok} <sub style="font-size: 9px; opacity: 0.85;">{tag}</sub></span>'
            )
        return " ".join(pills)


def run_demo(model_path: str = "models/lid_lr.joblib") -> None:
    """Run demonstration over the 10 sample customer support queries."""
    tagger_demo = HinglishDemoTagger(model_path=model_path)
    
    print("=" * 80)
    print("HINGLISH-ID ROUTER: MODULE M1.7 INTERACTIVE TAGGER DEMONSTRATION")
    print(f"Model: {model_path} | Tagger: LogisticRegression (char_wb 1-4)")
    print("Legend: [HI] Blue (Hindi) | [EN] Green (English) | [OTHER] Yellow (Entities / Punct)")
    print("=" * 80)

    for item in SAMPLE_QUERIES:
        res = tagger_demo.tag_query(item["text"])
        colored_str = tagger_demo.format_colored_tokens(res)
        
        print(f"\n[Query #{item['id']}] Intent Category: {item['intent'].upper()}")
        print(f"Raw Input:   {item['text']}")
        print(f"Tagged Text: {colored_str}")
        print(f"Metrics:     CMI = {res['cmi']:.1f} | Code-Mixed = {res['is_code_mixed']} | HI: {res['language_distribution']['HI']}, EN: {res['language_distribution']['EN']}, OTHER: {res['language_distribution']['OTHER']}")
        print("-" * 80)


if __name__ == "__main__":
    run_demo()
