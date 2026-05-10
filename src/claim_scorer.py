"""
Claim Scorer: score a health claim on checkability, harm, and clarity
using GPT-4o-mini with a diskcache layer to avoid redundant API calls.

Consistency test: 2026-05-09
  Claims tested : 10  (candidate_claims.json c001–c010, 3 runs each)
  checkability  mean std = 0.0000
  harm          mean std = 0.1155
  clarity       mean std = 0.0000
  Verdict       : consistent — all dimensions std < 0.8
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import diskcache
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

CACHE_DIR  = ROOT / ".cache" / "scorer"
SCORE_MODEL = "gpt-4o-mini"

# ── Pydantic schema ────────────────────────────────────────────────────────────

class ClaimScore(BaseModel):
    checkability: int
    harm: int
    clarity: int
    checkability_rationale: str
    harm_rationale: str
    clarity_rationale: str

    @field_validator("checkability", "harm", "clarity")
    @classmethod
    def must_be_1_to_5(cls, v: int, info) -> int:
        if not (1 <= v <= 5):
            raise ValueError(
                f"{info.field_name} must be between 1 and 5, got {v}"
            )
        return v


# ── Scoring prompt ─────────────────────────────────────────────────────────────

SCORER_PROMPT = """\
You are a triage assistant for a health fact-checking organization. \
Your task is to score a health claim on three dimensions.

Score each dimension from 1 to 5:

CHECKABILITY: Is this claim specific and falsifiable?
  1 = pure opinion or vague generality
  5 = precise, testable claim with clear parameters

HARM: If false, how harmful could this claim be?
  1 = negligible (general wellness advice)
  5 = severe (could cause people to avoid proven treatment or take dangerous action)

CLARITY: Is the claim clearly stated?
  1 = ambiguous or self-contradictory
  5 = unambiguous, one clear interpretation

Context: the following prior reviews may be related.
Use them only to calibrate your harm score.
{prior_context}

Claim to score:
{claim_text}

Respond ONLY with a JSON object matching this schema:
{schema}"""


# ── helpers ────────────────────────────────────────────────────────────────────

def _cache_key(claim_text: str) -> str:
    return hashlib.md5(claim_text.encode()).hexdigest()


def _format_prior_context(prior_top3: list[dict]) -> str:
    if not prior_top3:
        return "(none)"
    lines = [
        f"- {hit['claim_text'][:120]} (score: {hit['similarity_score']:.2f})"
        for hit in prior_top3
    ]
    return "\n".join(lines)


# ── public API ─────────────────────────────────────────────────────────────────

def score_claim(
    claim_text: str,
    prior_top3: list[dict],
    force_refresh: bool = False,
) -> ClaimScore:
    key = _cache_key(claim_text)

    with diskcache.Cache(str(CACHE_DIR)) as cache:
        if not force_refresh and key in cache:
            return ClaimScore(**cache[key])

        prior_context = _format_prior_context(prior_top3)
        schema_str = json.dumps({
            "checkability":            "<integer 1-5>",
            "harm":                    "<integer 1-5>",
            "clarity":                 "<integer 1-5>",
            "checkability_rationale":  "<one-sentence explanation>",
            "harm_rationale":          "<one-sentence explanation>",
            "clarity_rationale":       "<one-sentence explanation>",
        }, indent=2)

        prompt = SCORER_PROMPT.format(
            prior_context=prior_context,
            claim_text=claim_text,
            schema=schema_str,
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=SCORE_MODEL,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[{"role": "system", "content": prompt}],
        )

        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            score = ClaimScore(**data)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse ClaimScore from API response: {exc}\nRaw output: {raw}"
            ) from exc

        cache[key] = score.model_dump()

    return score


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    claims = [
        "Drinking lemon water detoxifies the liver",
        "Regular cardiovascular exercise reduces heart disease risk",
    ]

    for claim in claims:
        print(f"\n{'─' * 70}")
        print(f"Claim : {claim}")
        result = score_claim(claim, prior_top3=[])
        print(f"  checkability : {result.checkability}  — {result.checkability_rationale}")
        print(f"  harm         : {result.harm}  — {result.harm_rationale}")
        print(f"  clarity      : {result.clarity}  — {result.clarity_rationale}")
