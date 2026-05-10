# TriageLens — Final Evaluation Summary

Evaluation date: 2026-05-10  
Test set: 120 claims (50 positives · 35 neg_facts · 20 neg_unfalsifiable · 15 adversarials)

---

## 1. Macro-F1 Comparison

| System | RECOMMEND_NEW F1 | REUSE_PRIOR F1 | DO_NOT_CHECK F1 | **Macro-F1** |
|---|---|---|---|---|
| Full system | 0.5766 | 0.5625 | 0.6392 | **0.5928** |
| Zeroshot baseline | 0.5405 | 0.1538 | 0.9000 | 0.5314 |
| Keyword baseline | 0.2273 | 0.0000 | 0.7241 | 0.3171 |

---

## 2. Statistical Test Conclusions

**Full vs Keyword** — Bootstrap 95 % CI on macro-F1 difference: [0.1532, 0.3911].
The interval does not contain zero; the advantage is real and the full system is significantly better than the keyword baseline.

**Full vs Zeroshot** — Bootstrap 95 % CI: [−0.0569, 0.1801].
The interval straddles zero; at this sample size we cannot rule out random variation, though the point estimate (+0.0614) favours the full system.

---

## 3. Three Key Failure Modes

### 3.1 Established-fact misclassification (neg_fact)
32 of 64 DO_NOT_CHECK claims were routed to RECOMMEND_NEW_CHECK (recall 0.48).
Root cause: Rule 1 in the assembler fires only when the top retrieved review carries a
positive `rating` field (e.g. "TRUE"). The knowledge base contains almost exclusively
misinformation reviews, so established-fact claims are never matched by Rule 1 and fall
through to the scoring cascade, where high `checkability` and `harm` scores push them
toward RECOMMEND_NEW_CHECK.

### 3.2 Low REUSE_PRIOR recall (9/22, recall 0.41)
Root cause: a deliberate conservative design choice. The calibrated HIGH_THRESH (0.70)
was set to minimise false REUSE assignments (precision 0.90 achieved), accepting lower
recall. Nine true variants were downgraded to `related_but_different` by the RAG matcher
and conservatively re-routed to RECOMMEND_NEW_CHECK.

### 3.3 negation_of_known_misinfo — complete failure (accuracy 0.00)
Root cause: Rule 1 is never triggered (same knowledge-base gap as 3.1). These claims
score high similarity to existing misinformation reviews, which pushes
`variant_of_existing` classification; simultaneously their high `checkability` score
causes the cascade to route them to RECOMMEND_NEW_CHECK instead of DO_NOT_CHECK.
Without a dedicated "scientific rebuttal" signal in either the retrieval result or the
scoring prompt, the system has no path to the correct label.

---

## 4. Two Clear Advantages

### 4.1 softened_misinformation — accuracy 1.00 (vs 0.00 keyword / 0.33 zeroshot)
The scorer prompt explicitly instructs the model to look past hedging language
("some researchers suggest", "may have properties…") and evaluate the underlying
medical claim. The keyword baseline has no such logic; the zeroshot baseline partially
succeeds but fails on two of three cases.

### 4.2 time_sensitive_claim — accuracy 1.00 (vs 0.00 keyword / 0.00 zeroshot)
Claims framed as "emerging research indicates…" or "growing body of evidence…" are
correctly routed to RECOMMEND_NEW_CHECK. The scorer treats specificity of the
drug/outcome pair as the primary signal, not the epistemic hedge, while both baselines
systematically miss this pattern.

---

## 5. Top-Priority Improvements

**Priority 1 — Fix the established-fact blind spot**
Expand the `rating` vocabulary in the assembler (currently matches only a narrow set
of positive-verdict strings) or train a lightweight binary "established medical consensus"
classifier to run before the RAG step. Either approach would capture the 32 misclassified
neg_fact claims without requiring knowledge-base expansion.

**Priority 2 — Integrate a real virality data source**
`virality_score` was removed from the pipeline; queue prioritisation currently has no
empirical basis. Connecting to a social-signal API (e.g. CrowdTangle export, Meltwater,
or a simple share-count endpoint) would make the triage output actionable for editors
managing a real workload.
