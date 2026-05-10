"""
Embed ClaimReview records and store them in a local ChromaDB collection.

Usage:
    python scripts/build_index.py [--limit N]

Filters applied:
    - language == "English"
    - len(claim_text) >= 20
    - date < "2024-01-01"   (isolates 2015-2023 as training corpus)

Supports interrupted restarts: records whose doc-id already exists in the
collection are skipped automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
JSONL_PATH = ROOT / "data" / "raw" / "claimreview_all.jsonl"
CHROMA_PATH = ROOT / "data" / "chroma"

# ── constants ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "hf_reviews"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
DATE_CUTOFF = "2024-01-01"
MIN_CLAIM_LEN = 20


# ── helpers ────────────────────────────────────────────────────────────────────

def doc_id(review_url: str) -> str:
    return hashlib.md5(review_url.encode()).hexdigest()


def _safe_meta(value: str | None) -> str:
    return value if isinstance(value, str) else ""


def load_and_filter(path: Path, limit: int | None) -> list[dict]:
    raw_count = 0
    filtered: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_count += 1

            record = json.loads(line)

            if record.get("language") != "English":
                continue

            claim_text = record.get("claim_text") or ""
            if len(claim_text) < MIN_CLAIM_LEN:
                continue

            date = record.get("date") or ""
            if not date or date >= DATE_CUTOFF:
                continue

            filtered.append(record)

            if limit and len(filtered) >= limit:
                break

    print(f"[filter] raw={raw_count:,}  after filter={len(filtered):,}")
    return filtered


def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    count = collection.count()
    if count == 0:
        return set()
    print(f"[resume] collection already has {count:,} docs — loading existing IDs …")
    result = collection.get(include=[])
    ids = set(result["ids"])
    print(f"[resume] {len(ids):,} IDs loaded — these will be skipped")
    return ids


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N records after filtering (for quick smoke tests)"
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not JSONL_PATH.exists():
        print(f"[error] {JSONL_PATH} not found — run download_data.py first", file=sys.stderr)
        sys.exit(1)

    # ── load + filter ──────────────────────────────────────────────────────────
    records = load_and_filter(JSONL_PATH, args.limit)
    if not records:
        print("[warn] no records passed the filter — nothing to do.")
        return

    # ── chroma setup ──────────────────────────────────────────────────────────
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── resume: skip already indexed ──────────────────────────────────────────
    existing_ids = get_existing_ids(collection)

    seen_ids: set[str] = set(existing_ids)
    pending: list[dict] = []
    for r in records:
        rid = doc_id(r["review_url"])
        if rid not in seen_ids:
            seen_ids.add(rid)
            pending.append(r)
    skipped = len(records) - len(pending)
    if skipped:
        print(f"[resume] skipping {skipped:,} already-indexed or duplicate records")
    print(f"[index]  {len(pending):,} records to embed and insert")

    # ── openai client ─────────────────────────────────────────────────────────
    oai = OpenAI()

    if not pending:
        print("[done]   nothing new to index.")
    else:
        # ── batch embed + insert ──────────────────────────────────────────────
        total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(total_batches):
            batch = pending[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

            texts = [r["claim_text"] for r in batch]
            ids = [doc_id(r["review_url"]) for r in batch]
            metadatas = [
                {
                    "review_url": _safe_meta(r.get("review_url")),
                    "date":       _safe_meta(r.get("date")),
                    "publisher":  _safe_meta(r.get("publisher")),
                    "language":   _safe_meta(r.get("language")),
                }
                for r in batch
            ]

            embeddings = embed_batch(oai, texts)

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            print(
                f"  batch {batch_idx + 1:>{len(str(total_batches))}}/{total_batches}"
                f"  (+{len(batch):3d} docs)"
                f"  total in collection: {collection.count():,}"
            )

        print(f"[done]  indexed {len(pending):,} new records — "
              f"collection '{COLLECTION_NAME}' now has {collection.count():,} docs")

    _smoke_test(oai, collection)


def _smoke_test(oai: OpenAI, collection: chromadb.Collection) -> None:
    query = "drinking lemon water every morning boosts immunity and detoxifies the liver"
    print(f"\n[smoke] query: {query!r}")

    embedding = embed_batch(oai, [query])[0]
    results = collection.query(query_embeddings=[embedding], n_results=3, include=["metadatas", "documents"])

    hits = results["metadatas"][0]
    assert len(hits) == 3, f"expected 3 results, got {len(hits)}"
    for i, meta in enumerate(hits):
        assert meta.get("review_url"), f"result {i} missing review_url"
        print(f"  [{i+1}] {meta['review_url']}")

    print("[smoke] PASSED — 3 results returned, all have review_url")


if __name__ == "__main__":
    main()
