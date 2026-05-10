# TriageLens

A health claim triage system for fact-checking organizations. Given a free-text health claim, TriageLens routes it to one of three actions:

| Action | Meaning |
|---|---|
| `RECOMMEND_NEW_CHECK` | Specific, falsifiable, potentially harmful — needs a new review |
| `REUSE_PRIOR_CHECK` | Close variant of an already fact-checked claim |
| `DO_NOT_CHECK` | Established consensus, too vague, or a matter of opinion |

---

## Data Source

The knowledge base is built from the [claimreview-data](https://github.com/MartinoMensio/claimreview-data) dataset by Martino Mensio, which aggregates ClaimReview markup from fact-checking publishers worldwide. The file `data/raw/claimreview_all.jsonl` is not included in this repository (98 MB); download it from the source and place it at that path before running `scripts/build_index.py`.

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

```bash
cp .env.example .env
# edit .env and fill in your OpenAI API key
```

### 3. Build the vector index

Download the ClaimReview source data and place it at `data/raw/claimreview_all.jsonl`, then:

```bash
python scripts/build_index.py
```

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

## Evaluation Results (120 claims)

### Macro-F1

| System | Macro-F1 |
|---|---|
| **Full system** | **0.5928** |
| Zeroshot baseline | 0.5314 |
| Keyword baseline | 0.3171 |

### Per-class F1

| | RECOMMEND_NEW | REUSE_PRIOR | DO_NOT_CHECK |
|---|---|---|---|
| Full system | 0.5766 | 0.5625 | 0.6392 |
| Zeroshot BL | 0.5405 | 0.1538 | 0.9000 |
| Keyword BL | 0.2273 | 0.0000 | 0.7241 |

### Adversarial subset (per-type accuracy)

| Type | Full | Keyword | Zeroshot |
|---|---|---|---|
| softened_misinformation | **1.00** | 0.00 | 0.33 |
| time_sensitive_claim | **1.00** | 0.00 | 0.00 |
| negation_of_known_misinfo | 0.00 | 0.67 | **1.00** |
| grammatical_ambiguity | 0.33 | **1.00** | 0.67 |
| fact_opinion_blend | 0.33 | **1.00** | **1.00** |

### Statistical tests

- **Full vs Keyword** — Bootstrap 95% CI [0.1532, 0.3911], does not contain zero: full system is significantly better.
- **Full vs Zeroshot** — Bootstrap 95% CI [−0.0569, 0.1801], straddles zero: advantage is in the right direction but not conclusive at this sample size.

---

## Known Limitations

1. **Established-fact blind spot** — 32/64 DO_NOT_CHECK claims misrouted to RECOMMEND_NEW_CHECK because Rule 1 requires a positive `rating` field in the knowledge base, which only contains misinformation reviews.
2. **Conservative REUSE_PRIOR routing** — HIGH_THRESH=0.70 maximises precision (0.90) at the cost of recall (0.41); 9 true variants are downgraded.
3. **negation_of_known_misinfo failure** — Scientific rebuttals of known misinformation retrieve high-similarity hits from the wrong reviews and are misclassified.
4. **No virality signal** — Queue prioritisation has no empirical basis without a real social-signal data source.
5. **Zeroshot REUSE hallucination** — Zero-shot baseline can assert a prior review exists with no supporting evidence.

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
