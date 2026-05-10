"""
Zero-shot Baseline: single GPT-4o-mini call with no retrieval context.

Uses the same diskcache pattern as claim_scorer.py.
Cache directory: .cache/zeroshot/
Cache key: MD5 of claim_text (results are deterministic at temperature=0).

Known limitations
─────────────────
1. REUSE_PRIOR_CHECK is not grounded in any actual knowledge base.
   The label is produced entirely from the LLM's implicit memory of
   previously fact-checked topics, making it susceptible to hallucinated
   matches — the model may confidently assert a prior review exists when
   none does.

2. Softened misinformation (e.g. "some researchers suggest X might…")
   may be classified as REUSE_PRIOR_CHECK rather than the correct
   RECOMMEND_NEW_CHECK if the LLM recognises the underlying topic as
   familiar, conflating "topic has been discussed before" with "this
   specific claim has been reviewed before".

3. Core difference from the main TriageLens system: the main system's
   REUSE_PRIOR_CHECK decisions are backed by Chroma vector-search
   results with explicit similarity scores and review URLs. This
   baseline's REUSE_PRIOR_CHECK carries no such evidence and should
   not be treated as equivalent in downstream workflows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import diskcache
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / ".cache" / "zeroshot"
MODEL     = "gpt-4o-mini"

VALID_ACTIONS = {"RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK", "DO_NOT_CHECK"}

ZEROSHOT_PROMPT = """\
You are a triage assistant for a health fact-checking organization.

Given a health claim, classify it into exactly one of three categories:

RECOMMEND_NEW_CHECK
  The claim is specific, falsifiable, and potentially harmful if false.
  No prior fact-check appears to exist.

REUSE_PRIOR_CHECK
  The claim is a variant of a health claim that has likely been
  fact-checked before by organizations like Health Feedback, Snopes,
  or AFP Fact Check.

DO_NOT_CHECK
  The claim is either already well-established scientific consensus,
  too vague to fact-check, or a matter of personal opinion.

Respond ONLY with a JSON object:
{{"action": "RECOMMEND_NEW_CHECK" | "REUSE_PRIOR_CHECK" | "DO_NOT_CHECK",
  "reasoning": "one sentence explanation"}}

Claim: {claim_text}"""


# ── Predictor ──────────────────────────────────────────────────────────────────

def predict(claim_text: str, force_refresh: bool = False) -> str:
    key = hashlib.md5(claim_text.encode()).hexdigest()

    with diskcache.Cache(str(CACHE_DIR)) as cache:
        if not force_refresh and key in cache:
            return cache[key]

        prompt = ZEROSHOT_PROMPT.format(claim_text=claim_text)
        client = OpenAI()
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.choices[0].message.content
        try:
            data   = json.loads(raw)
            action = data["action"]
        except Exception as exc:
            raise ValueError(
                f"Failed to parse zeroshot response: {exc}\nRaw: {raw}"
            ) from exc

        if action not in VALID_ACTIONS:
            raise ValueError(
                f"action {action!r} is not one of {VALID_ACTIONS}\nRaw: {raw}"
            )

        cache[key] = action

    return action


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        (
            "Drinking lemon water detoxifies the liver",
            "RECOMMEND_NEW_CHECK",
            None,
        ),
        (
            "Regular cardiovascular exercise reduces heart disease risk",
            "DO_NOT_CHECK",
            None,
        ),
        (
            "Some researchers suggest high-dose vitamin C might help certain cancer patients",
            "RECOMMEND_NEW_CHECK",
            "LLM should recognise the checkable claim behind softened phrasing",
        ),
    ]
    for text, expected, note in cases:
        result = predict(text)
        status = "PASS" if result == expected else "FAIL"
        line = f"[{status}] expected={expected:25s} got={result:25s} | {text[:70]}"
        if note:
            line += f"\n       NOTE: {note}"
        print(line)
