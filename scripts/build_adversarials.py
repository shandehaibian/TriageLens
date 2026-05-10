"""
Generate test_set/adversarials.jsonl and append to test_set/test_set.jsonl.

15 adversarial claims across 5 types (3 each).

Usage:
    python scripts/build_adversarials.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT            = Path(__file__).parent.parent
OUT_DIR         = ROOT / "test_set"
ADV_PATH        = OUT_DIR / "adversarials.jsonl"
TEST_SET_PATH   = OUT_DIR / "test_set.jsonl"

ALL_KEYS = [
    "id", "claim_text", "gold_label", "adversarial_type", "challenge",
    "is_variant", "category", "type", "source", "review_url",
]

# ── Adversarial claims ─────────────────────────────────────────────────────────

ADVERSARIALS = [

    # ── Type 1: Softened misinformation ────────────────────────────────────────
    # Hedging language wraps a checkable (and likely false) health claim.
    # Risk: low checkability score causes system to route to DO_NOT_CHECK.
    {
        "adversarial_type": "softened_misinformation",
        "gold_label": "RECOMMEND_NEW_CHECK",
        "claim_text": (
            "Some researchers have suggested that high-dose vitamin C "
            "might be worth exploring as a complementary approach for "
            "certain cancer patients."
        ),
        "challenge": (
            "Hedging phrases ('some researchers', 'might be worth exploring') "
            "may suppress checkability score below threshold, causing "
            "DO_NOT_CHECK mis-route despite a specific, testable cancer claim."
        ),
    },
    {
        "adversarial_type": "softened_misinformation",
        "gold_label": "RECOMMEND_NEW_CHECK",
        "claim_text": (
            "There is growing anecdotal evidence that intermittent fasting "
            "could potentially reverse type 2 diabetes in some individuals, "
            "though more research may be needed."
        ),
        "challenge": (
            "'Anecdotal evidence' and 'could potentially' soften a specific, "
            "checkable claim about diabetes reversal that warrants scrutiny."
        ),
    },
    {
        "adversarial_type": "softened_misinformation",
        "gold_label": "RECOMMEND_NEW_CHECK",
        "claim_text": (
            "It has been reported by some holistic practitioners that "
            "frankincense essential oil may have properties that support "
            "the immune system in fighting certain types of cancer."
        ),
        "challenge": (
            "Attribution to 'some practitioners' and modal 'may have' disguise "
            "a specific anti-cancer claim; system may score harm or checkability "
            "too low and route to DO_NOT_CHECK."
        ),
    },

    # ── Type 2: Opinion mixed with established fact ─────────────────────────────
    # The factual component is already established; the value-judgment component
    # makes the whole statement non-checkable as a unit.
    # Risk: system flags the fact clause and routes to RECOMMEND_NEW_CHECK.
    {
        "adversarial_type": "fact_opinion_blend",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "While studies show omega-3 fatty acids reduce inflammation markers, "
            "people should ultimately decide for themselves whether supplements "
            "are worth the cost."
        ),
        "challenge": (
            "The factual first clause may trigger RECOMMEND_NEW_CHECK even though "
            "the overall statement is non-actionable due to the appended "
            "personal-choice opinion."
        ),
    },
    {
        "adversarial_type": "fact_opinion_blend",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "Research has demonstrated that vaccines are highly effective at "
            "preventing serious illness, but parents have every right to weigh "
            "the potential risks for their own children."
        ),
        "challenge": (
            "Established scientific consensus paired with a rights-based opinion "
            "creates a mixed statement; the factual opener may falsely elevate "
            "checkability and harm scores."
        ),
    },
    {
        "adversarial_type": "fact_opinion_blend",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "Even though clinical trials confirm statins lower LDL cholesterol, "
            "individuals should feel empowered to explore natural alternatives "
            "before starting any medication."
        ),
        "challenge": (
            "Confirmed clinical fact combined with treatment-choice advice; "
            "the implication that evidence is insufficient may not be caught "
            "and the system may route on the factual premise alone."
        ),
    },

    # ── Type 3: Grammatical ambiguity ──────────────────────────────────────────
    # Sentence structure permits two distinct interpretations.
    # Risk: system assigns high clarity score to an inherently ambiguous claim.
    {
        "adversarial_type": "grammatical_ambiguity",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "Doctors who don't recommend vitamins to their patients are missing "
            "something important about modern nutrition."
        ),
        "challenge": (
            "Could mean doctors are uninformed (value judgment) or that vitamins "
            "are medically necessary (testable claim); low clarity should trigger "
            "DO_NOT_CHECK but may not if checkability is scored on surface keywords."
        ),
    },
    {
        "adversarial_type": "grammatical_ambiguity",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "The sugar in fruit is just as dangerous as refined sugar when it "
            "comes to cancer."
        ),
        "challenge": (
            "'Just as dangerous' is ambiguous between glucose metabolism "
            "broadly and cancer risk specifically — two distinct checkable "
            "claims with different evidence bases."
        ),
    },
    {
        "adversarial_type": "grammatical_ambiguity",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "Hospitals that prioritize profit over patient care are responsible "
            "for more deaths than any single disease."
        ),
        "challenge": (
            "'Responsible for deaths' conflates causal mortality data (checkable) "
            "with moral responsibility (opinion); the disease-comparison claim "
            "is additionally too vague to fact-check."
        ),
    },

    # ── Type 4: Negation of known misinformation ───────────────────────────────
    # Correct scientific rebuttals of claims already reviewed in the knowledge
    # base. Risk: RAG retrieves the original misinformation review at high
    # similarity and mis-classifies the correction as a variant.
    {
        "adversarial_type": "negation_of_known_misinfo",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "There is no scientific evidence that lemon water has any "
            "detoxifying effect on the liver or kidneys."
        ),
        "challenge": (
            "High RAG similarity to 'lemon water detoxifies the liver' reviews "
            "may trigger variant_of_existing and route to REUSE_PRIOR_CHECK "
            "instead of recognising this as an established-fact rebuttal."
        ),
    },
    {
        "adversarial_type": "negation_of_known_misinfo",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "Contrary to viral claims, mRNA COVID-19 vaccines do not alter "
            "human DNA and cannot be integrated into the genome."
        ),
        "challenge": (
            "Semantic overlap with mRNA vaccine misinformation in the index "
            "may cause the RAG to return high-similarity hits and incorrectly "
            "classify this correction as a variant of the misinformation."
        ),
    },
    {
        "adversarial_type": "negation_of_known_misinfo",
        "gold_label": "DO_NOT_CHECK",
        "claim_text": (
            "Scientific consensus, confirmed by dozens of large-scale studies, "
            "establishes that the MMR vaccine does not cause autism."
        ),
        "challenge": (
            "'MMR vaccine' and 'autism' co-occur in existing reviews of the "
            "misinformation; RAG similarity may incorrectly flag this rebuttal "
            "as a variant rather than an established-fact statement."
        ),
    },

    # ── Type 5: Time-sensitive claims ──────────────────────────────────────────
    # Recent or emerging research — no prior review likely exists but hedging
    # language may suppress checkability score.
    # Risk: system routes to DO_NOT_CHECK due to low checkability rather than
    # RECOMMEND_NEW_CHECK.
    {
        "adversarial_type": "time_sensitive_claim",
        "gold_label": "RECOMMEND_NEW_CHECK",
        "claim_text": (
            "Recent studies suggest that the gut microbiome may play a role "
            "in regulating mood disorders, including depression and anxiety."
        ),
        "challenge": (
            "'Recent studies' and 'may play a role' are vague enough to "
            "suppress checkability below threshold despite describing a "
            "specific, verifiable scientific hypothesis."
        ),
    },
    {
        "adversarial_type": "time_sensitive_claim",
        "gold_label": "RECOMMEND_NEW_CHECK",
        "claim_text": (
            "Emerging research indicates that long-term use of proton pump "
            "inhibitors may be associated with an increased risk of dementia."
        ),
        "challenge": (
            "'Emerging research' signals uncertainty; system may penalise "
            "checkability despite this being a specific drug-outcome claim "
            "suitable for independent fact-checking."
        ),
    },
    {
        "adversarial_type": "time_sensitive_claim",
        "gold_label": "RECOMMEND_NEW_CHECK",
        "claim_text": (
            "A growing body of evidence suggests that ultra-processed foods "
            "significantly accelerate biological aging at the cellular level."
        ),
        "challenge": (
            "'Growing body of evidence' obscures specificity; the cellular-aging "
            "claim is testable and novel but hedging may push checkability "
            "below the routing threshold."
        ),
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Build adversarials with sequential IDs
    adv_records = [
        {
            "id":               f"adv_{i+1:03d}",
            "claim_text":       item["claim_text"],
            "gold_label":       item["gold_label"],
            "adversarial_type": item["adversarial_type"],
            "challenge":        item["challenge"],
            "source":           "adversarial",
        }
        for i, item in enumerate(ADVERSARIALS)
    ]

    write_jsonl(ADV_PATH, adv_records)
    print(f"Saved → test_set/adversarials.jsonl  ({len(adv_records)} records)")

    # Type breakdown
    type_counts: dict[str, int] = {}
    for r in adv_records:
        t = r["adversarial_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, n in type_counts.items():
        print(f"  {t:35s} {n}")

    # Append to test_set.jsonl
    existing = read_jsonl(TEST_SET_PATH)
    existing_ids = {r["id"] for r in existing}

    to_append = [
        {k: r.get(k) for k in ALL_KEYS}
        for r in adv_records
        if r["id"] not in existing_ids
    ]
    updated = existing + to_append
    write_jsonl(TEST_SET_PATH, updated)

    print(f"\ntest_set/test_set.jsonl updated.")
    print(f"  appended : {len(to_append)}")
    print(f"  total    : {len(updated)}")


if __name__ == "__main__":
    main()
