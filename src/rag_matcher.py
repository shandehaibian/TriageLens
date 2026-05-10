"""
Prior-Check Matcher: embed a claim, retrieve the top-3 most similar
ClaimReview records from the local ChromaDB index, and classify the
claim against existing prior coverage.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION_NAME = "hf_reviews"
EMBED_MODEL = "text-embedding-3-small"

# ── thresholds ─────────────────────────────────────────────────────────────────
# Calibrated: 2026-05-09
# Calibration set: threshold_pairs_v2.json (20 variant_of_existing + 10 no_prior)
# variant recall = 0.55,  recall@3 = 0.55
# Misclassification pattern: 9/20 variants were labelled related_but_different
#   (not no_prior), so the boundary error is conservative — missed variants are
#   still routed to RECOMMEND_NEW_CHECK with top-3 reviews surfaced, rather than
#   being silently dropped.
# related_but_different routing in assembler: RECOMMEND_NEW_CHECK + display top-3
HIGH_THRESH = 0.70
LOW_THRESH = 0.57


# ── module-level singletons (lazy-initialised on first call) ───────────────────
_oai: OpenAI | None = None
_collection: chromadb.Collection | None = None


def _get_oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


# ── public API ─────────────────────────────────────────────────────────────────

def retrieve_top3(claim_text: str) -> list[dict]:
    """Return the 3 most similar ClaimReview records for *claim_text*.

    Each result dict contains:
        claim_text, review_url, date, publisher, similarity_score
    similarity_score is a cosine similarity in [0, 1] converted from the
    cosine *distance* stored by ChromaDB (score = 1 - distance).
    """
    oai = _get_oai()
    collection = _get_collection()

    embedding = oai.embeddings.create(model=EMBED_MODEL, input=[claim_text]).data[0].embedding

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "claim_text": doc,
                "review_url": meta.get("review_url", ""),
                "date": meta.get("date", ""),
                "publisher": meta.get("publisher", ""),
                "similarity_score": round(1.0 - dist, 6),
            }
        )

    return hits


def classify_prior_match(top3: list[dict]) -> str:
    """Classify a claim based on its top-3 retrieval results.

    Returns:
        "variant_of_existing"  – similarity_score >= HIGH_THRESH
        "related_but_different"– similarity_score >= LOW_THRESH
        "no_prior"             – below LOW_THRESH or top3 is empty
    """
    if not top3:
        return "no_prior"

    score = top3[0]["similarity_score"]
    if score >= HIGH_THRESH:
        return "variant_of_existing"
    if score >= LOW_THRESH:
        return "related_but_different"
    return "no_prior"


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_claims = [
        "Drinking lemon water detoxifies the liver",
        "A new study shows intermittent fasting reverses type 2 diabetes in 90 days",
    ]

    for claim in test_claims:
        print(f"\n{'─' * 70}")
        print(f"Claim : {claim}")
        top3 = retrieve_top3(claim)
        for i, hit in enumerate(top3, 1):
            print(
                f"  [{i}] score={hit['similarity_score']:.4f}  "
                f"url={hit['review_url']}\n"
                f"      text={hit['claim_text'][:90]!r}"
            )
        label = classify_prior_match(top3)
        print(f"Label : {label}")
