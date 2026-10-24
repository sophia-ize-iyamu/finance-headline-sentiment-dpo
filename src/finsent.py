"""Shared code for the financial-headline sentiment project.

The model must answer with JSON such as {"label": "bullish"}. A deterministic checker scores each answer,
and a word-list challenger flags the headlines a person should review, the same design as Reconcile.
"""
import json
import random
import re

LABELS = ["bearish", "bullish", "neutral"]  # dataset labels 0, 1 and 2
# Run 2 prompt. Run 1 showed the labels as {"label": "bearish" | "bullish" | "neutral"}, and the base model
# copied that pattern with a quote missing, so this version names the labels and gives one plain example.
PROMPT = ('Classify the sentiment of this financial news headline as bearish, bullish or neutral. '
          'Reply with JSON only, for example {"label": "neutral"}.\n\n')
PROMPT_RUN1 = ('Classify the sentiment of this financial news. '
               'Reply only with JSON: {"label": "bearish" | "bullish" | "neutral"}.\n\n')


def to_pair(example, index, rng=random):
    """Builds one DPO preference pair in conversational format.

    The rejected answer alternates between a wrong label in valid JSON (odd rows) and the right label
    wrapped in prose, which isn't valid JSON (even rows).
    """
    gold = LABELS[example["label"]]
    wrong = rng.choice([lab for lab in LABELS if lab != gold])
    rejected = json.dumps({"label": wrong}) if index % 2 else f"Sure! The sentiment is {gold}."
    return {
        "prompt": [{"role": "user", "content": PROMPT + example["text"]}],
        "chosen": [{"role": "assistant", "content": json.dumps({"label": gold})}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def check(raw, gold):
    """Strict checker: the whole answer must parse as JSON with an allowed label."""
    try:
        obj = json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError):
        return {"valid_json": False, "allowed_label": False, "label": None, "correct": False}
    label = str(obj.get("label", "")).strip().lower() if isinstance(obj, dict) else ""
    allowed = label in LABELS
    return {"valid_json": True, "allowed_label": allowed, "label": label if allowed else None,
            "correct": allowed and label == gold}


BULLISH = ["beat", "beats", "raise", "raises", "raised", "upgrade", "upgrades", "upgraded", "surge", "surges",
           "soar", "soars", "jump", "jumps", "gain", "gains", "rally", "rallies", "record", "higher", "rise",
           "rises", "buy", "outperform", "bullish", "boost", "boosts", "strong", "tops", "climbs"]
BEARISH = ["miss", "misses", "missed", "cut", "cuts", "downgrade", "downgrades", "downgraded", "fall", "falls",
           "drop", "drops", "plunge", "plunges", "slump", "slumps", "tumble", "tumbles", "lower", "loss",
           "losses", "sell", "underperform", "bearish", "decline", "declines", "slide", "slides", "warns",
           "weak", "layoffs", "lawsuit"]
_WORD = re.compile(r"[a-z]+")


def rules_label(text):
    """Word-list challenger: more bullish words than bearish ones means bullish, and so on."""
    words = _WORD.findall(text.lower())
    score = sum(w in BULLISH for w in words) - sum(w in BEARISH for w in words)
    return "bullish" if score > 0 else "bearish" if score < 0 else "neutral"


def summarize(outputs, golds, texts):
    """Scores a model's raw answers. Disagreements with the challenger form the review queue."""
    rows = [check(o, g) for o, g in zip(outputs, golds)]
    n = len(rows)
    review = [r["label"] != rules_label(t) for r, t in zip(rows, texts)]
    wrong = [not r["correct"] for r in rows]
    caught = sum(w and q for w, q in zip(wrong, review))
    return {
        "n": n,
        "valid_json_rate": sum(r["valid_json"] for r in rows) / n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "review_rate": sum(review) / n,
        "errors": sum(wrong),
        "errors_in_review": caught,
    }
