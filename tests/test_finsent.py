import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finsent import LABELS, PROMPT, check, rules_label, summarize, to_pair  # noqa: E402


def test_check_accepts_exact_json():
    assert check('{"label": "bullish"}', "bullish") == {
        "valid_json": True, "allowed_label": True, "label": "bullish", "correct": True}


def test_check_is_case_and_space_tolerant_inside_json():
    assert check('  {"label": " Bearish "}\n', "bearish")["correct"]


def test_check_rejects_prose_and_code_fences():
    assert not check("Sure! The sentiment is bullish.", "bullish")["valid_json"]
    assert not check('```json\n{"label": "bullish"}\n```', "bullish")["valid_json"]


def test_check_flags_unknown_labels_and_non_objects():
    r = check('{"label": "positive"}', "bullish")
    assert r["valid_json"] and not r["allowed_label"] and not r["correct"]
    assert not check('["bullish"]', "bullish")["allowed_label"]
    assert not check('{"sentiment": "bullish"}', "bullish")["allowed_label"]


def test_check_wrong_label():
    r = check('{"label": "neutral"}', "bearish")
    assert r["allowed_label"] and not r["correct"]


def test_to_pair_alternates_rejected_answers():
    rng = random.Random(0)
    ex = {"text": "Acme beats estimates", "label": 1}
    even, odd = to_pair(ex, 0, rng), to_pair(ex, 1, rng)
    assert even["prompt"][0]["content"] == PROMPT + "Acme beats estimates"
    assert json.loads(even["chosen"][0]["content"]) == {"label": "bullish"}
    assert not check(even["rejected"][0]["content"], "bullish")["valid_json"]
    wrong = json.loads(odd["rejected"][0]["content"])["label"]
    assert wrong in LABELS and wrong != "bullish"


def test_rules_label():
    assert rules_label("Acme beats estimates and raises guidance") == "bullish"
    assert rules_label("Shares plunge after Acme misses on revenue") == "bearish"
    assert rules_label("Acme to hold annual meeting on Tuesday") == "neutral"


def test_summarize_counts_errors_caught_by_review():
    texts = ["Acme beats estimates", "Acme to hold meeting", "Shares plunge at Acme"]
    golds = ["bullish", "neutral", "bearish"]
    outputs = ['{"label": "bullish"}', '{"label": "bearish"}', "bearish"]
    s = summarize(outputs, golds, texts)
    assert s["n"] == 3
    assert s["valid_json_rate"] == 2 / 3
    assert s["accuracy"] == 1 / 3
    # Rows 2 and 3 disagree with the challenger (neutral vs bearish, invalid vs bearish), and both are errors.
    assert s["review_rate"] == 2 / 3
    assert (s["errors"], s["errors_in_review"]) == (2, 2)
