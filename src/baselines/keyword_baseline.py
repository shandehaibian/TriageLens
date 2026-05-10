"""
Keyword Baseline: rule-based triage using curated term lists.

Known limitation: this baseline cannot produce REUSE_PRIOR_CHECK.
It has no access to a knowledge base and therefore cannot detect
whether a claim is a variant of an existing review. All claims not
matched by HIGH_RISK_TERMS fall through to DO_NOT_CHECK.
"""

from __future__ import annotations

# ── Term lists ─────────────────────────────────────────────────────────────────

HIGH_RISK_TERMS = [
    "cure", "cures", "curative",
    "detox", "detoxify", "detoxifies",
    "eliminate", "eliminates",
    "prevent cancer", "treat cancer",
    "shrink tumor", "kill cancer",
    "reverse diabetes", "reverse disease",
    "boost immune", "supercharge immune",
    "miracle", "secret remedy",
    "doctors don't want you to know",
    "suppress", "suppress virus",
    "natural cure", "herbal cure",
]

ESTABLISHED_FACT_TERMS = [
    "regular exercise reduces",
    "vaccines prevent",
    "smoking causes",
    "hand washing reduces",
    "balanced diet",
    "sleep is important",
    "alcohol increases risk",
    "obesity is associated with",
    "screening reduces mortality",
]

UNFALSIFIABLE_TERMS = [
    "should think for themselves",
    "listen to your body",
    "natural is better",
    "everyone is different",
    "do your own research",
    "what works for me",
    "feel better",
    "holistic approach",
]


# ── Predictor ──────────────────────────────────────────────────────────────────

def predict(claim_text: str) -> str:
    lower = claim_text.lower()

    # Rule 1 — established fact
    if any(term in lower for term in ESTABLISHED_FACT_TERMS):
        return "DO_NOT_CHECK"

    # Rule 2 — unfalsifiable opinion
    if any(term in lower for term in UNFALSIFIABLE_TERMS):
        return "DO_NOT_CHECK"

    # Rule 3 — high-risk misinformation signal
    if any(term in lower for term in HIGH_RISK_TERMS):
        return "RECOMMEND_NEW_CHECK"

    # Rule 4 — default
    return "DO_NOT_CHECK"


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        (
            "Drinking lemon water detoxifies the liver",
            "RECOMMEND_NEW_CHECK",
            None,
        ),
        (
            "Regular cardiovascular exercise reduces heart disease risk",
            "DO_NOT_CHECK",
            None,
        ),
        (
            "Some researchers suggest high-dose vitamin C might help certain cancer patients",
            "DO_NOT_CHECK",
            "Known limitation: softened misinformation bypasses keyword matching",
        ),
    ]
    for text, expected, note in cases:
        result = predict(text)
        status = "PASS" if result == expected else "FAIL"
        line = f"[{status}] expected={expected:25s} got={result:25s} | {text[:70]}"
        if note:
            line += f"\n       NOTE: {note}"
        print(line)
