"""
Test score_claim consistency by running each claim three times
(force_refresh=True) and measuring per-dimension standard deviation.

Usage:
    python scripts/test_scorer_consistency.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claim_scorer import score_claim  # noqa: E402

FIXTURES_PATH = ROOT / "fixtures" / "candidate_claims.json"
RESULTS_DIR   = ROOT / "results"
OUTPUT_CSV    = RESULTS_DIR / "scorer_consistency.csv"

N_RUNS      = 3
UNSTABLE_TH = 0.8   # std threshold for flagging a claim as unstable
N_CLAIMS    = 10


def main() -> None:
    claims = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))[:N_CLAIMS]
    print(f"Loaded {len(claims)} claims — {N_RUNS} runs each, force_refresh=True\n")

    rows: list[dict] = []
    unstable: list[dict] = []

    for item in claims:
        cid   = item["id"]
        text  = item["claim_text"]
        print(f"[{cid}] {text[:80]!r}")

        scores = [
            score_claim(text, prior_top3=[], force_refresh=True)
            for _ in range(N_RUNS)
        ]

        check_vals   = [s.checkability for s in scores]
        harm_vals    = [s.harm         for s in scores]
        clarity_vals = [s.clarity      for s in scores]

        check_std   = round(statistics.stdev(check_vals),   4)
        harm_std    = round(statistics.stdev(harm_vals),    4)
        clarity_std = round(statistics.stdev(clarity_vals), 4)

        is_unstable = any(v >= UNSTABLE_TH for v in (check_std, harm_std, clarity_std))

        tag = "  *** UNSTABLE ***" if is_unstable else ""
        print(
            f"  checkability {check_vals}  std={check_std:.2f}  |"
            f"  harm {harm_vals}  std={harm_std:.2f}  |"
            f"  clarity {clarity_vals}  std={clarity_std:.2f}{tag}"
        )

        row = {
            "id":             cid,
            "claim_preview":  text[:80],
            "check_r1":       check_vals[0],
            "check_r2":       check_vals[1],
            "check_r3":       check_vals[2],
            "harm_r1":        harm_vals[0],
            "harm_r2":        harm_vals[1],
            "harm_r3":        harm_vals[2],
            "clarity_r1":     clarity_vals[0],
            "clarity_r2":     clarity_vals[1],
            "clarity_r3":     clarity_vals[2],
            "checkability_std": check_std,
            "harm_std":         harm_std,
            "clarity_std":      clarity_std,
            "unstable":         is_unstable,
        }
        rows.append(row)
        if is_unstable:
            unstable.append(row)

    # ── summary ───────────────────────────────────────────────────────────────
    mean_check   = round(statistics.mean(r["checkability_std"] for r in rows), 4)
    mean_harm    = round(statistics.mean(r["harm_std"]         for r in rows), 4)
    mean_clarity = round(statistics.mean(r["clarity_std"]      for r in rows), 4)

    print(f"\n{'='*60}")
    print("SUMMARY — mean std across all claims")
    print(f"  checkability_std : {mean_check:.4f}")
    print(f"  harm_std         : {mean_harm:.4f}")
    print(f"  clarity_std      : {mean_clarity:.4f}")

    if unstable:
        print(f"\nUNSTABLE claims (any dimension std >= {UNSTABLE_TH}):")
        for r in unstable:
            print(
                f"  [{r['id']}] check_std={r['checkability_std']}  "
                f"harm_std={r['harm_std']}  clarity_std={r['clarity_std']}"
            )
            print(f"         {r['claim_preview']!r}")
    else:
        print(f"\nNo claims exceeded the instability threshold ({UNSTABLE_TH}).")

    # ── save CSV ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "claim_preview",
        "check_r1", "check_r2", "check_r3",
        "harm_r1",  "harm_r2",  "harm_r3",
        "clarity_r1", "clarity_r2", "clarity_r3",
        "checkability_std", "harm_std", "clarity_std",
        "unstable",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # summary row
        writer.writerow({
            "id": "SUMMARY",
            "claim_preview": f"mean std across {N_CLAIMS} claims",
            "checkability_std": mean_check,
            "harm_std":         mean_harm,
            "clarity_std":      mean_clarity,
            "unstable":         len(unstable) > 0,
        })

    print(f"\nResults saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
