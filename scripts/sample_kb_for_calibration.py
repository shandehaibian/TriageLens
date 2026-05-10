"""
Sample 10 health-related ClaimReview records from the local ChromaDB index.

Strategy:
    1. Use ChromaDB's where_document $contains filter with each keyword
       to pull a pool of matching documents (limit=2000 per keyword).
    2. Merge results, deduplicate by doc-id, then randomly sample 10.

Usage:
    python scripts/sample_kb_for_calibration.py [--seed N]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chromadb
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

CHROMA_PATH     = ROOT / "data" / "chroma"
COLLECTION_NAME = "hf_reviews"
FIXTURES_DIR    = ROOT / "fixtures"
OUTPUT_PATH     = FIXTURES_DIR / "kb_samples.json"

KEYWORDS = ["health", "vitamin", "vaccine", "cancer",
            "diet", "cure", "treatment", "supplement"]

POOL_PER_KEYWORD = 2000   # max records fetched per keyword before dedup
TARGET_SAMPLES   = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default 42)")
    args = parser.parse_args()
    random.seed(args.seed)

    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has {collection.count():,} docs total.")

    # ── Step 1: pull a pool for each keyword, deduplicate ─────────────────────
    pool: dict[str, dict] = {}   # doc_id → {text, meta}

    for kw in KEYWORDS:
        results = collection.get(
            where_document={"$contains": kw},
            include=["documents", "metadatas"],
            limit=POOL_PER_KEYWORD,
        )
        added = 0
        for doc_id, text, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            if doc_id not in pool:
                pool[doc_id] = {"text": text, "meta": meta}
                added += 1
        print(f"  keyword={kw!r:12s}  fetched={len(results['ids']):5,}  new_unique={added:5,}")

    print(f"\nDeduped pool size: {len(pool):,}")

    if len(pool) < TARGET_SAMPLES:
        raise RuntimeError(
            f"Pool has only {len(pool)} records — need at least {TARGET_SAMPLES}."
        )

    # ── Step 2: random sample ─────────────────────────────────────────────────
    sampled_ids = random.sample(list(pool.keys()), TARGET_SAMPLES)

    # ── Step 3: print + build output ──────────────────────────────────────────
    print(f"\nSampled {TARGET_SAMPLES} records (seed={args.seed}):\n")
    output: list[dict] = []

    for i, doc_id in enumerate(sampled_ids, 1):
        item = pool[doc_id]
        claim = item["text"]
        meta  = item["meta"]
        label = f"kb{i:03d}"

        print(f"[{label}]")
        print(f"  claim_text  : {claim[:120]!r}")
        print(f"  review_url  : {meta.get('review_url', '')}")
        print(f"  date        : {meta.get('date', '')}")
        print()

        output.append({
            "id":             label,
            "original_claim": claim,
            "review_url":     meta.get("review_url", ""),
            "date":           meta.get("date", ""),
        })

    # ── Step 4: save ──────────────────────────────────────────────────────────
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
