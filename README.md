# TriageLens

A health claim triage system for fact-checking organizations. Given a free-text health claim, TriageLens routes it to one of three actions:

| Action | Meaning |
|---|---|
| `RECOMMEND_NEW_CHECK` | Specific, falsifiable, potentially harmful — needs a new review |
| `REUSE_PRIOR_CHECK` | Close variant of an already fact-checked claim |
| `DO_NOT_CHECK` | Established consensus, too vague, or a matter of opinion |

---

## Data Source

The knowledge base is built from the [claimreview-data](https://github.com/MartinoMensio/claimreview-data) dataset by Martino Mensio, which aggregates ClaimReview markup from fact-checking publishers worldwide.

Neither the raw source file nor the processed JSONL is included in this repository. Follow these steps to prepare the data locally:

**Step 1 — Download the source file**

From the [claimreview-data releases page](https://github.com/MartinoMensio/claimreview-data), download `claim_reviews.json` and place it in the project root:

```
TriageLens/
└── claim_reviews.json   ← place here (~336 MB)
```

**Step 2 — Convert to JSONL**

```bash
python scripts/download_data.py
```

This reads `claim_reviews.json` from the project root and writes the normalised output to `data/raw/claimreview_all.jsonl` (~98 MB). The script skips the conversion if the output file already exists.

**Step 3 — Build the vector index**

```bash
python scripts/build_index.py
```

This embeds health-related English ClaimReview records into a local ChromaDB collection at `data/chroma/`. Requires a valid `OPENAI_API_KEY` in `.env`.

---

## Architecture

```
claim_text
    │
    ├─► RAG Matcher          retrieve_top3() → classify_prior_match()
    │     ChromaDB (cosine)   HIGH_THRESH=0.70 / LOW_THRESH=0.57
    │
    ├─► Claim Scorer          score_claim()  →  ClaimScore(checkability, harm, clarity)
    │     GPT-4o-mini          diskcache keyed by MD5(claim_text)
    │
    └─► Assembler             assemble_triage_card()
          6-rule cascade      recommended_action + rule_triggered + scores
```

**Stack:** ChromaDB · OpenAI `text-embedding-3-small` · GPT-4o-mini · Pydantic v2 · Streamlit · diskcache

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/shandehaibian/TriageLens.git
cd TriageLens
pip install -e ".[dev]"
```

### 2. Set API key

This project calls the OpenAI API for embedding (index build) and claim scoring (runtime). An API key is required.

Copy the example file and add your key:

```bash
cp .env.example .env
```

Then open `.env` and set:

```
OPENAI_API_KEY=sk-...your-key-here...
```

The key is read at startup via `python-dotenv`; no code changes are needed. The `.env` file is listed in `.gitignore` and will never be committed.

### 3. Prepare data and build the vector index

See the [Data Source](#data-source) section above for the full three-step process:
download `claim_reviews.json` → run `scripts/download_data.py` → run `scripts/build_index.py`.

### 4. Launch the editor UI

```bash
streamlit run app.py
```

---

## Project Structure

```
TriageLens/
├── app.py                          # Streamlit editor interface
├── src/
│   ├── rag_matcher.py              # ChromaDB retrieval + prior-match classification
│   ├── claim_scorer.py             # GPT-4o-mini claim scoring (checkability/harm/clarity)
│   ├── assembler.py                # 6-rule triage card assembler
│   ├── logger.py                   # Decision logger → logs/decisions.jsonl
│   └── baselines/
│       ├── keyword_baseline.py     # Rule-based keyword triage
│       └── zeroshot_baseline.py    # Single GPT-4o-mini call, no retrieval
├── scripts/
│   ├── download_data.py            # Convert claim_reviews.json → claimreview_all.jsonl
│   ├── build_index.py              # Embed ClaimReview records into ChromaDB
│   ├── build_test_set.py           # Build 3-category evaluation test set
│   ├── build_adversarials.py       # Generate 15 adversarial test claims
│   ├── annotate_variants.py        # Interactive REUSE_PRIOR variant annotation
│   ├── calibrate_threshold.py      # Grid-search HIGH/LOW similarity thresholds
│   ├── eval_harness.py             # Full evaluation + statistical tests
│   └── test_scorer_consistency.py  # Scorer temperature-0 consistency check
├── test_set/
│   ├── positives.jsonl             # 50 real Health Feedback 2024 reviews
│   ├── negatives_facts.jsonl       # 35 synthesised established-fact claims
│   ├── negatives_unfalsifiable.jsonl # 20 synthesised opinion claims
│   ├── adversarials.jsonl          # 15 adversarial claims (5 failure types)
│   └── test_set.jsonl              # Merged test set (120 claims)
├── fixtures/
│   ├── candidate_claims.json       # 30 labelled demo claims for the UI
│   ├── kb_samples.json             # Knowledge-base samples for calibration
│   └── threshold_pairs_v2.json     # 30 annotated pairs for threshold search
├── results/
│   ├── eval_report.txt             # Full evaluation report
│   ├── final_summary.md            # Summary with failure analysis
│   ├── predictions.jsonl           # Per-claim predictions from all 3 systems
│   ├── scorer_consistency.csv      # Consistency test results
│   └── threshold_calibration.csv  # Calibration grid-search results
├── data/
│   ├── raw/                        # claimreview_all.jsonl (not in repo, ~98 MB)
│   └── chroma/                     # ChromaDB index (not in repo, rebuild locally)
└── logs/
    └── decisions.jsonl             # Editor decisions (not in repo)
```

---

## Evaluation

### What was tested

120 claims across three categories, run through all three systems and scored against gold labels:

- **50 positives** — real health misinformation claims from Health Feedback / AFP Fact Check 2024 reviews. Gold label: `RECOMMEND_NEW_CHECK`.
- **55 negatives** — 35 established medical consensus statements (CDC/WHO style) + 20 unfalsifiable personal-opinion statements. Gold label: `DO_NOT_CHECK`.
- **15 adversarial cases** — hand-crafted claims targeting five known failure modes: softened misinformation, fact-opinion blends, grammatical ambiguity, negations of known misinfo, and time-sensitive emerging claims.

### What counted as good output

A correct prediction matches the gold label. The primary metric is **macro-F1** (unweighted average across all three classes), which penalises ignoring any class. For the adversarial subset, per-type accuracy was used instead, since each type has only three examples.

### Two baselines

| System | What it does |
|---|---|
| **Keyword baseline** | Rule-based: matches curated term lists; cannot produce `REUSE_PRIOR_CHECK` |
| **Zeroshot baseline** | Single GPT-4o-mini call with no retrieval; `REUSE_PRIOR_CHECK` is produced from model memory only |
| **Full system** | RAG retrieval + GPT-4o-mini scorer + 6-rule assembler |

### What the comparison showed

| System | Macro-F1 | RECOMMEND_NEW F1 | REUSE_PRIOR F1 | DO_NOT_CHECK F1 |
|---|---|---|---|---|
| **Full system** | **0.5928** | 0.5766 | 0.5625 | 0.6392 |
| Zeroshot baseline | 0.5314 | 0.5405 | 0.1538 | 0.9000 |
| Keyword baseline | 0.3171 | 0.2273 | 0.0000 | 0.7241 |

The full system is the only one that meaningfully detects `REUSE_PRIOR_CHECK` (F1 0.56 vs 0.15 / 0.00). Its largest advantage over both baselines is on **softened misinformation** and **time-sensitive claims** — cases where hedging language hides a checkable assertion. The scorer prompt is explicitly designed to look past that hedging; neither baseline has this capability.

Bootstrap 95% CI on macro-F1 difference (1 000 resamples): full vs keyword is [+0.15, +0.39] — the gap is real. Full vs zeroshot is [−0.06, +0.18] — the point estimate favours the full system but the interval straddles zero; the advantage is not conclusive at this sample size.

### Where it broke down

**Established-fact misclassification (32 / 64 DO_NOT_CHECK wrong).** Well-established medical consensus statements — "The MMR vaccine does not cause autism", "Handwashing prevents infection" — score high on `checkability` and `harm` because the scorer reads them as important health claims. Rule 1 (which would short-circuit to `DO_NOT_CHECK`) only fires when the knowledge base contains a review with a positive `rating` field; the current index holds only misinformation reviews, so Rule 1 never triggers.

**REUSE_PRIOR recall is low (9 / 22, recall 0.41).** The similarity threshold was calibrated to be conservative — precision is 0.90, but 9 true variant claims fall below the threshold and get rerouted to `RECOMMEND_NEW_CHECK`. This was a deliberate design choice to avoid falsely reusing a review that does not actually cover the new claim.

**Negations of known misinformation fail completely (accuracy 0.00).** Claims like "mRNA vaccines do not alter human DNA" retrieve high-similarity hits from existing misinformation reviews (same topic, opposite stance). The system has no mechanism to distinguish a scientific rebuttal from a variant of the misinformation itself.

**Adversarial subset — per-type accuracy**

| Type | Full | Keyword | Zeroshot |
|---|---|---|---|
| softened_misinformation | **1.00** | 0.00 | 0.33 |
| time_sensitive_claim | **1.00** | 0.00 | 0.00 |
| negation_of_known_misinfo | 0.00 | 0.67 | **1.00** |
| grammatical_ambiguity | 0.33 | **1.00** | 0.67 |
| fact_opinion_blend | 0.33 | **1.00** | **1.00** |

---

## Reproducing the Evaluation

```bash
# Full run (makes API calls, ~120 × GPT-4o-mini scorer calls)
python scripts/eval_harness.py

# Quick smoke test on first 10 claims only
python scripts/eval_harness.py --sample 10

# Reuse existing predictions, regenerate report only
python scripts/eval_harness.py --reuse
```

Report is saved to `results/eval_report.txt`.
