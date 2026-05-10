"""
Triage Card Assembler: apply a deterministic rule cascade to produce a
routing decision (triage card) for a health claim.
"""

from __future__ import annotations

from datetime import datetime, timezone

from claim_scorer import ClaimScore

# ── Rule-layer thresholds ──────────────────────────────────────────────────────
# Below this checkability score the claim is structurally uncheckable
# (too vague, opinion-like, or unresolvable with evidence).
CHECKABILITY_LOW = 2

# Below this harm score the claim poses negligible risk even if wrong.
HARM_LOW = 2

# virality_score is an external signal (platform data) not available at triage
# time in the current version; it is intentionally excluded from routing logic.

# Known blind spots
# 1. Softened-language misinformation may score low on both checkability and harm
#    and slip through to DO_NOT_CHECK.
# 2. True facts absent from the knowledge base can be mis-routed to
#    RECOMMEND_NEW_CHECK if their checkability score is high.
# 3. related_but_different always routes to RECOMMEND_NEW_CHECK — a conservative
#    fallback; editors must decide whether a prior review is reusable.

# ── Rating keywords that signal an established-fact review ────────────────────
_ESTABLISHED_FACT_TOKENS = frozenset(
    ["true", "correct", "accurate", "already established"]
)


def _prior_confirms_fact(prior_top3: list[dict]) -> bool:
    for hit in prior_top3:
        rating = (hit.get("rating") or "").lower()
        if any(token in rating for token in _ESTABLISHED_FACT_TOKENS):
            return True
    return False


# ── Main assembler ─────────────────────────────────────────────────────────────

def assemble_triage_card(
    claim_text: str,
    prior_match_label: str,
    prior_top3: list[dict],
    claim_score: ClaimScore,
) -> dict:
    action: str
    reasoning: str
    rule: str

    # Rule 1 — RAG confirms established fact
    if _prior_confirms_fact(prior_top3):
        action    = "DO_NOT_CHECK"
        reasoning = "Prior review confirms established fact"
        rule      = "Rule 1"

    # Rule 2 — Clear variant of an existing review
    elif prior_match_label == "variant_of_existing":
        action    = "REUSE_PRIOR_CHECK"
        reasoning = "High similarity to existing review"
        rule      = "Rule 2"

    # Rule 3 — Structurally uncheckable and low harm
    elif (
        claim_score.checkability <= CHECKABILITY_LOW
        and claim_score.harm <= HARM_LOW
    ):
        action    = "DO_NOT_CHECK"
        reasoning = (
            "Low checkability and low harm: "
            "likely opinion or vague generality"
        )
        rule = "Rule 3"

    # Rule 4 — Related prior exists; conservative routing for editor review
    elif prior_match_label == "related_but_different":
        action    = "RECOMMEND_NEW_CHECK"
        reasoning = (
            "Related prior reviews exist but "
            "claim may warrant independent check"
        )
        rule = "Rule 4"

    # Rule 5 — No prior, but claim is checkable and potentially harmful
    elif (
        prior_match_label == "no_prior"
        and claim_score.checkability >= 3
        and claim_score.harm >= 3
    ):
        action    = "RECOMMEND_NEW_CHECK"
        reasoning = "No prior review, checkable and potentially harmful"
        rule      = "Rule 5"

    # Rule 6 — Default fallback
    else:
        action    = "DO_NOT_CHECK"
        reasoning = "Insufficient signal to recommend review"
        rule      = "Rule 6"

    return {
        "claim_text":         claim_text,
        "recommended_action": action,
        "action_reasoning":   reasoning,
        "rule_triggered":     rule,
        "prior_match_label":  prior_match_label,
        "prior_reviews":      prior_top3,
        "scores": {
            "checkability":            claim_score.checkability,
            "harm":                    claim_score.harm,
            "clarity":                 claim_score.clarity,
            "checkability_rationale":  claim_score.checkability_rationale,
            "harm_rationale":          claim_score.harm_rationale,
            "clarity_rationale":       claim_score.clarity_rationale,
        },
        "assembled_at": datetime.now(timezone.utc).isoformat(),
    }


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scenarios = [
        {
            "label":       "Scenario 1",
            "claim_text":  "Drinking lemon water detoxifies the liver",
            "prior_label": "variant_of_existing",
            "scores":      dict(checkability=4, harm=4, clarity=4,
                                checkability_rationale="Specific claim",
                                harm_rationale="Could delay treatment",
                                clarity_rationale="Unambiguous"),
            "expected":    "Rule 2 / REUSE_PRIOR_CHECK",
        },
        {
            "label":       "Scenario 2",
            "claim_text":  "Eating more vegetables is generally good for you",
            "prior_label": "no_prior",
            "scores":      dict(checkability=1, harm=1, clarity=2,
                                checkability_rationale="Too vague to test",
                                harm_rationale="Negligible risk",
                                clarity_rationale="Somewhat ambiguous"),
            "expected":    "Rule 3 / DO_NOT_CHECK",
        },
        {
            "label":       "Scenario 3",
            "claim_text":  "Ozempic causes irreversible muscle loss after stopping",
            "prior_label": "no_prior",
            "scores":      dict(checkability=4, harm=4, clarity=4,
                                checkability_rationale="Falsifiable with clinical data",
                                harm_rationale="Could deter beneficial treatment",
                                clarity_rationale="Clear and specific"),
            "expected":    "Rule 5 / RECOMMEND_NEW_CHECK",
        },
    ]

    all_passed = True
    for s in scenarios:
        score = ClaimScore(**s["scores"])
        card  = assemble_triage_card(
            claim_text        = s["claim_text"],
            prior_match_label = s["prior_label"],
            prior_top3        = [],
            claim_score       = score,
        )
        actual   = f"{card['rule_triggered']} / {card['recommended_action']}"
        passed   = actual == s["expected"]
        status   = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"[{status}] {s['label']}")
        print(f"       expected : {s['expected']}")
        print(f"       actual   : {actual}")
        print(f"       reason   : {card['action_reasoning']}")
        print()

    if not all_passed:
        raise SystemExit("One or more scenarios failed.")
