"""
Decision Logger: append triage decisions to a JSONL audit log and
reload them for downstream analysis.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT     = Path(__file__).parent.parent
LOG_PATH = ROOT / "logs" / "decisions.jsonl"

VALID_EDITOR_ACTIONS  = {"ACCEPT", "OVERRIDE"}
VALID_OVERRIDE_ACTIONS = {
    "RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK",
    "DO_NOT_CHECK", "INVALID_CLAIM",
}


def log_decision(
    claim_id: str,
    triage_card: dict,
    editor_action: str,
    override_action: str | None = None,
    override_reason: str | None = None,
    rejection_cause: str | None = None,
) -> None:
    if editor_action not in VALID_EDITOR_ACTIONS:
        raise ValueError(
            f"editor_action must be one of {VALID_EDITOR_ACTIONS}, got {editor_action!r}"
        )
    if editor_action == "OVERRIDE":
        if not override_action:
            raise ValueError("override_action is required when editor_action is 'OVERRIDE'")
        if override_action not in VALID_OVERRIDE_ACTIONS:
            raise ValueError(
                f"override_action must be one of {VALID_OVERRIDE_ACTIONS}, got {override_action!r}"
            )
        if not override_reason:
            raise ValueError("override_reason is required when editor_action is 'OVERRIDE'")
        if override_action == "INVALID_CLAIM" and not rejection_cause:
            raise ValueError("rejection_cause is required when override_action is 'INVALID_CLAIM'")

    record = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "claim_id":           claim_id,
        "claim_text":         triage_card["claim_text"],
        "recommended_action": triage_card["recommended_action"],
        "rule_triggered":     triage_card["rule_triggered"],
        "editor_action":      editor_action,
        "override_action":    override_action,
        "override_reason":    override_reason,
        "rejection_cause":    rejection_cause if override_action == "INVALID_CLAIM" else None,
        "scores":             triage_card["scores"],
        "prior_match_label":  triage_card["prior_match_label"],
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_decisions() -> list[dict]:
    if not LOG_PATH.exists():
        return []

    records: list[dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[logger] WARNING: skipping malformed line {lineno}: {exc}")

    return sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: minimal triage_card
    card = {
        "claim_text":         "Drinking lemon water detoxifies the liver",
        "recommended_action": "REUSE_PRIOR_CHECK",
        "rule_triggered":     "Rule 2",
        "prior_match_label":  "variant_of_existing",
        "prior_reviews":      [],
        "scores": {
            "checkability": 4, "harm": 4, "clarity": 4,
            "checkability_rationale": "Specific and falsifiable",
            "harm_rationale":         "Could deter proven treatment",
            "clarity_rationale":      "Unambiguous claim",
        },
        "assembled_at": "2026-05-09T00:00:00+00:00",
    }

    # Step 2: write records
    log_decision("smoke_001", card, "ACCEPT")
    print("Logged: ACCEPT")

    log_decision(
        "smoke_001", card, "OVERRIDE",
        override_action="RECOMMEND_NEW_CHECK",
        override_reason="Claim is more specific than system recognized",
    )
    print("Logged: OVERRIDE → RECOMMEND_NEW_CHECK")

    log_decision(
        "smoke_001", card, "OVERRIDE",
        override_action="INVALID_CLAIM",
        override_reason="Claim is not a factual statement",
        rejection_cause="Text is a personal testimonial, not a verifiable health claim",
    )
    print("Logged: OVERRIDE → INVALID_CLAIM")

    # Step 3: reload and print summary
    decisions = load_decisions()
    print(f"\nload_decisions() returned {len(decisions)} record(s).")
    first = decisions[0]
    print(f"Most recent — timestamp: {first['timestamp']}  editor_action: {first['editor_action']}")
    print(f"rejection_cause: {first.get('rejection_cause')}")

    # Step 4: error cases
    print()
    try:
        log_decision("smoke_001", card, "OVERRIDE")
    except ValueError as e:
        print(f"Caught expected error (OVERRIDE no action+reason): {e}")

    try:
        log_decision("smoke_001", card, "OVERRIDE",
                     override_action="RECOMMEND_NEW_CHECK")
    except ValueError as e:
        print(f"Caught expected error (OVERRIDE no reason): {e}")

    try:
        log_decision("smoke_001", card, "OVERRIDE",
                     override_action="INVALID_CLAIM",
                     override_reason="Not a claim")
    except ValueError as e:
        print(f"Caught expected error (INVALID_CLAIM no rejection_cause): {e}")

    try:
        log_decision("smoke_001", card, "REJECT")
    except ValueError as e:
        print(f"Caught expected error (invalid editor_action): {e}")
