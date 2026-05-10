"""
Convert claim_reviews.json → data/raw/claimreview_all.jsonl

Source fields:
  claim_text   : claim_text[0]            (list → first element)
  review_url   : review_url
  rating       : label                    (normalised label)
  date         : reviews[0].date_published
  language     : fact_checker.language
  publisher    : fact_checker.name
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC_FILE = ROOT / "claim_reviews.json"
OUT_FILE = ROOT / "data" / "raw" / "claimreview_all.jsonl"


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _extract(record: dict) -> dict:
    reviews = record.get("reviews") or []
    first_review = reviews[0] if reviews else {}
    fact_checker = record.get("fact_checker") or {}

    claim_text_raw = record.get("claim_text", "")
    if isinstance(claim_text_raw, list):
        claim_text = claim_text_raw[0] if claim_text_raw else ""
    else:
        claim_text = claim_text_raw

    return {
        "claim_text": claim_text,
        "review_url": record.get("review_url", ""),
        "rating": record.get("label", ""),
        "date": first_review.get("date_published", ""),
        "language": fact_checker.get("language", ""),
        "publisher": fact_checker.get("name", ""),
    }


def main() -> None:
    if not SRC_FILE.exists():
        print(f"[error] source file not found: {SRC_FILE}", file=sys.stderr)
        sys.exit(1)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip if output already exists and is non-empty
    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 0:
        existing = _count_lines(OUT_FILE)
        print(f"[skip] {OUT_FILE.name} already exists with {existing:,} records — nothing to do.")
        sys.exit(0)

    print(f"[load] reading {SRC_FILE.name} …")
    with SRC_FILE.open("r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    total = len(data)
    print(f"[info] {total:,} records found — writing to {OUT_FILE} …")

    report_every = max(1, total // 20)  # ~5 % increments

    written = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for i, record in enumerate(data):
            row = _extract(record)
            out.write(json.dumps(row, ensure_ascii=False))
            out.write("\n")
            written += 1

            if (i + 1) % report_every == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(f"  {pct:5.1f}%  {i + 1:,} / {total:,}", flush=True)

    print(f"[done] wrote {written:,} records → {OUT_FILE}")


if __name__ == "__main__":
    main()
