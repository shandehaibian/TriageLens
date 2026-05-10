"""
Grid-search HIGH_THRESH / LOW_THRESH for the Prior-Check Matcher.

Optimisation priority:
    1. variant_of_existing recall  (missing a true variant is the costlier error)
    2. overall 3-class accuracy

Usage:
    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_matcher import retrieve_top3  # noqa: E402

FIXTURES_PATH = ROOT / "fixtures" / "threshold_pairs_v2.json"
RESULTS_DIR   = ROOT / "results"
OUTPUT_CSV    = RESULTS_DIR / "threshold_calibration.csv"

LABELS = ["variant_of_existing", "related_but_different", "no_prior"]

# Grid ranges (inclusive endpoints)
HIGH_VALS = [round(0.70 + i * 0.01, 2) for i in range(16)]  # 0.70 … 0.85
LOW_VALS  = [round(0.55 + i * 0.01, 2) for i in range(18)]  # 0.55 … 0.72


# ── helpers ────────────────────────────────────────────────────────────────────

def _classify(top3: list[dict], high: float, low: float) -> str:
    if not top3:
        return "no_prior"
    score = top3[0]["similarity_score"]
    if score >= high:
        return "variant_of_existing"
    if score >= low:
        return "related_but_different"
    return "no_prior"


def _confusion(cached: list[dict], high: float, low: float) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        exp: {pred: 0 for pred in LABELS} for exp in LABELS
    }
    for c in cached:
        pred = _classify(c["top3"], high, low)
        matrix[c["expected"]][pred] += 1
    return matrix


def _print_confusion(matrix: dict[str, dict[str, int]]) -> None:
    col_w = 26
    short = {"variant_of_existing": "variant", "related_but_different": "related", "no_prior": "no_prior"}
    print(f"{'':26}" + "".join(f"{short[p]:^26}" for p in LABELS))
    print("─" * (26 + 26 * len(LABELS)))
    for exp in LABELS:
        row = f"{short[exp]:26}" + "".join(f"{matrix[exp][p]:^26}" for p in LABELS)
        print(row)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    fixtures: list[dict] = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(fixtures)} fixture claims.\n")

    # ── Step 1: retrieve once per claim, cache scores ─────────────────────────
    print("Retrieving top-3 embeddings (one API call per claim)…")
    cached: list[dict] = []
    for i, item in enumerate(fixtures, 1):
        top3 = retrieve_top3(item["claim_text"])
        top1 = top3[0]["similarity_score"] if top3 else 0.0
        print(
            f"  [{i:02d}/{len(fixtures)}] {item['id']}  "
            f"expected={item['expected_label']:25s}  top1={top1:.4f}"
        )
        cached.append({
            "id":       item["id"],
            "expected": item["expected_label"],
            "top3":     top3,
        })

    variants = [c for c in cached if c["expected"] == "variant_of_existing"]
    print(f"\nVariant subset: {len(variants)} claims\n")

    # ── Step 2: grid search ───────────────────────────────────────────────────
    valid_combos = [
        (h, l)
        for h in HIGH_VALS
        for l in LOW_VALS
        if h > l + 0.08          # enforce minimum gap
    ]
    print(f"Grid-searching {len(valid_combos)} valid (HIGH, LOW) combinations…")

    rows: list[dict] = []
    for high, low in valid_combos:
        # overall 3-class accuracy
        n_correct = sum(
            _classify(c["top3"], high, low) == c["expected"] for c in cached
        )
        accuracy = n_correct / len(cached)

        # variant_of_existing recall
        v_correct = sum(
            _classify(c["top3"], high, low) == "variant_of_existing"
            for c in variants
        )
        variant_recall = v_correct / len(variants)

        # recall@3: ≥1 of top-3 hits scores >= high for variant claims
        r3_hits = sum(
            any(h_["similarity_score"] >= high for h_ in c["top3"])
            for c in variants
        )
        recall_at_3 = r3_hits / len(variants)

        rows.append({
            "high_thresh":    high,
            "low_thresh":     low,
            "accuracy":       round(accuracy,       4),
            "variant_recall": round(variant_recall, 4),
            "recall_at_3":    round(recall_at_3,    4),
        })

    # ── Step 3: select best ───────────────────────────────────────────────────
    best = max(rows, key=lambda r: (r["variant_recall"], r["accuracy"]))

    print(f"\n{'='*60}")
    print("BEST THRESHOLD COMBINATION")
    print(f"{'='*60}")
    print(f"  HIGH_THRESH     = {best['high_thresh']}")
    print(f"  LOW_THRESH      = {best['low_thresh']}")
    print(f"  variant recall  = {best['variant_recall']:.4f}")
    print(f"  overall acc     = {best['accuracy']:.4f}")
    print(f"  recall@3        = {best['recall_at_3']:.4f}")

    # ── Step 4: confusion matrix ──────────────────────────────────────────────
    print(f"\nCONFUSION MATRIX  (HIGH={best['high_thresh']}  LOW={best['low_thresh']})")
    print("rows = expected label   cols = predicted label\n")
    matrix = _confusion(cached, best["high_thresh"], best["low_thresh"])
    _print_confusion(matrix)

    # per-class recall summary
    print()
    for label in LABELS:
        total = sum(matrix[label].values())
        hit   = matrix[label][label]
        print(f"  {label:30s}  recall = {hit}/{total} = {hit/total if total else 0:.2f}")

    # ── Step 5: save full grid to CSV ─────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["high_thresh", "low_thresh", "accuracy", "variant_recall", "recall_at_3"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (-r["variant_recall"], -r["accuracy"])))

    print(f"\nFull grid ({len(rows)} combinations) saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
