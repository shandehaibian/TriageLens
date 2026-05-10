"""
Interactive variant annotation for test_set/positives.jsonl.

For each positive claim, retrieve top-3 similar reviews and ask the
annotator whether the claim is a variant of an existing review.

  y    → is_variant=True, gold_label="REUSE_PRIOR_CHECK"
  n    → keep current values unchanged
  skip → skip this record, keep current values unchanged

Results are written back to positives.jsonl and test_set.jsonl after
every decision so progress is never lost on interrupt.

Usage:
    python scripts/annotate_variants.py [--start N]   # resume from record N
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_matcher import retrieve_top3  # noqa: E402

POSITIVES_PATH  = ROOT / "test_set" / "positives.jsonl"
TEST_SET_PATH   = ROOT / "test_set" / "test_set.jsonl"
NEG_FACTS_PATH  = ROOT / "test_set" / "negatives_facts.jsonl"
NEG_UNFAL_PATH  = ROOT / "test_set" / "negatives_unfalsifiable.jsonl"

ALL_KEYS = [
    "id", "claim_text", "gold_label",
    "is_variant", "category", "type", "source", "review_url",
]


# ── I/O helpers ────────────────────────────────────────────────────────────────

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
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rebuild_test_set(positives: list[dict]) -> None:
    neg_facts = read_jsonl(NEG_FACTS_PATH)
    neg_unfal = read_jsonl(NEG_UNFAL_PATH)
    merged = [
        {k: r.get(k) for k in ALL_KEYS}
        for r in positives + neg_facts + neg_unfal
    ]
    write_jsonl(TEST_SET_PATH, merged)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", type=int, default=1,
        help="1-based index of the record to start from (for resuming)"
    )
    args = parser.parse_args()

    positives = read_jsonl(POSITIVES_PATH)
    if not positives:
        print(f"[error] {POSITIVES_PATH} is empty or missing.")
        sys.exit(1)

    total     = len(positives)
    start_idx = max(0, args.start - 1)

    print(f"Loaded {total} positives. Starting from record {start_idx + 1}.\n")
    print("Commands:  y = variant   n = not a variant   skip = skip\n")

    changed = 0

    for i in range(start_idx, total):
        record = positives[i]
        print(f"{'─' * 70}")
        print(f"[{i+1}/{total}]  id={record['id']}  "
              f"current is_variant={record.get('is_variant')}  "
              f"gold_label={record.get('gold_label')}")
        print(f"Claim: {record['claim_text']}\n")

        top3 = retrieve_top3(record["claim_text"])
        if top3:
            print("Top-3 similar reviews:")
            for j, hit in enumerate(top3, 1):
                print(f"  [{j}] score={hit['similarity_score']:.4f}  "
                      f"{hit['claim_text'][:100]}")
        else:
            print("  (no results returned)")

        print()
        while True:
            raw = input("Mark as variant? (y/n/skip): ").strip().lower()
            if raw in ("y", "n", "skip"):
                break
            print("  Please enter y, n, or skip.")

        if raw == "y":
            positives[i]["is_variant"]  = True
            positives[i]["gold_label"]  = "REUSE_PRIOR_CHECK"
            changed += 1
            write_jsonl(POSITIVES_PATH, positives)
            rebuild_test_set(positives)
            print("  → Marked as variant. Files updated.\n")
        elif raw == "n":
            print("  → Kept as-is.\n")
        else:
            print("  → Skipped.\n")

    print(f"{'='*70}")
    print(f"Annotation complete. {changed} record(s) updated out of {total - start_idx} reviewed.")


if __name__ == "__main__":
    main()
