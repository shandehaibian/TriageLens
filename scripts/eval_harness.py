"""
Evaluation harness: run full TriageLens pipeline and two baselines on
test_set/test_set.jsonl, compute metrics, statistical tests, and save
a comparison report.

Usage:
    python scripts/eval_harness.py            # full run
    python scripts/eval_harness.py --reuse    # skip inference, reuse predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from assembler import assemble_triage_card                        # noqa: E402
from baselines.keyword_baseline import predict as kw_predict      # noqa: E402
from baselines.zeroshot_baseline import predict as zs_predict     # noqa: E402
from claim_scorer import score_claim                               # noqa: E402
from rag_matcher import classify_prior_match, retrieve_top3       # noqa: E402

TEST_SET_PATH    = ROOT / "test_set" / "test_set.jsonl"
PREDICTIONS_PATH = ROOT / "results"  / "predictions.jsonl"
REPORT_PATH      = ROOT / "results"  / "eval_report.txt"
RESULTS_DIR      = ROOT / "results"

LABELS = ["RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK", "DO_NOT_CHECK"]
SHORT  = {
    "RECOMMEND_NEW_CHECK": "RECOMMEND_NEW",
    "REUSE_PRIOR_CHECK":   "REUSE_PRIOR  ",
    "DO_NOT_CHECK":        "DO_NOT_CHECK ",
}

# GPT-4o-mini pricing
INPUT_RATE         = 0.15 / 1_000_000   # USD per token
OUTPUT_RATE        = 0.60 / 1_000_000
BASE_PROMPT_TOKENS = 600
OUTPUT_TOKENS_EST  = 150


# ── I/O ────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Cost estimation ────────────────────────────────────────────────────────────

def estimate_cost(claim_text: str, prior_top3: list[dict]) -> float:
    claim_tokens = len(claim_text.split()) * 1.3
    prior_tokens = sum(len(h["claim_text"].split()) * 1.3 + 10 for h in prior_top3)
    input_tokens = BASE_PROMPT_TOKENS + claim_tokens + prior_tokens
    return round(input_tokens * INPUT_RATE + OUTPUT_TOKENS_EST * OUTPUT_RATE, 8)


# ── Part 1: inference ──────────────────────────────────────────────────────────

def run_inference(claims: list[dict]) -> list[dict]:
    predictions: list[dict] = []
    total = len(claims)

    for i, item in enumerate(claims, 1):
        cid  = item.get("id", f"item_{i:03d}")
        text = item["claim_text"]

        # ── Full TriageLens pipeline ──────────────────────────────────────────
        t0          = time.time()
        top3        = retrieve_top3(text)
        prior_label = classify_prior_match(top3)
        cscore      = score_claim(text, top3)
        card        = assemble_triage_card(
            claim_text        = text,
            prior_match_label = prior_label,
            prior_top3        = top3,
            claim_score       = cscore,
        )
        latency = round(time.time() - t0, 3)

        full_pred   = card["recommended_action"]
        full_rule   = card["rule_triggered"]
        full_scores = card["scores"]
        cost        = estimate_cost(text, top3)

        # ── Baselines ─────────────────────────────────────────────────────────
        kw_pred = kw_predict(text)
        zs_pred = zs_predict(text)

        predictions.append({
            "claim_id":           cid,
            "claim_text":         text,
            "gold_label":         item.get("gold_label", ""),
            "adversarial_type":   item.get("adversarial_type"),
            "full_system":        full_pred,
            "keyword_baseline":   kw_pred,
            "zeroshot_baseline":  zs_pred,
            "full_system_rule":   full_rule,
            "full_system_scores": full_scores,
            "latency_seconds":    latency,
            "estimated_cost_usd": cost,
        })

        print(
            f"[{i:3d}/{total}] {cid:12s} | "
            f"full={full_pred:22s} | "
            f"kw={kw_pred:22s} | "
            f"zs={zs_pred}"
        )

    return predictions


# ── Part 2: metrics ────────────────────────────────────────────────────────────

def per_class_metrics(
    gold: list[str], pred: list[str]
) -> tuple[dict[str, dict], float]:
    metrics: dict[str, dict] = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[label] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   tp + fn,
        }
    macro_f1 = round(sum(m["f1"] for m in metrics.values()) / len(LABELS), 4)
    return metrics, macro_f1


def confusion_matrix(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        if g in matrix:
            matrix[g][p] = matrix[g].get(p, 0) + 1
    return matrix


# ── Part 3: statistical tests ──────────────────────────────────────────────────

def mcnemar_test(
    gold: list[str], pred1: list[str], pred2: list[str]
) -> tuple[float, float]:
    import statsmodels.stats.contingency_tables as ct  # lazy: not needed in --sample mode

    c1 = [g == p for g, p in zip(gold, pred1)]
    c2 = [g == p for g, p in zip(gold, pred2)]
    n00 = sum(not a and not b for a, b in zip(c1, c2))
    n01 = sum(not a and     b for a, b in zip(c1, c2))
    n10 = sum(    a and not b for a, b in zip(c1, c2))
    n11 = sum(    a and     b for a, b in zip(c1, c2))

    if (n01 + n10) == 0:
        return 0.0, 1.0  # no discordant pairs — no difference detectable

    table  = [[n00, n01], [n10, n11]]
    result = ct.mcnemar(table, exact=False, correction=True)
    return round(float(result.statistic), 4), round(float(result.pvalue), 4)


def bootstrap_f1_diff(
    gold: list[str], pred1: list[str], pred2: list[str],
    n: int = 1000, seed: int = 42,
) -> tuple[float, float]:
    rng     = np.random.default_rng(seed)
    n_items = len(gold)
    diffs   = []
    for _ in range(n):
        idx = rng.integers(0, n_items, n_items)
        g  = [gold[i]  for i in idx]
        p1 = [pred1[i] for i in idx]
        p2 = [pred2[i] for i in idx]
        _, f1_1 = per_class_metrics(g, p1)
        _, f1_2 = per_class_metrics(g, p2)
        diffs.append(f1_1 - f1_2)
    arr = np.array(diffs)
    return round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)


# ── Part 4: performance stats ──────────────────────────────────────────────────

def perf_stats(preds: list[dict]) -> dict:
    lat  = np.array([p["latency_seconds"]    for p in preds])
    cost = np.array([p["estimated_cost_usd"] for p in preds])
    return {
        "p50":          round(float(np.percentile(lat, 50)), 3),
        "p95":          round(float(np.percentile(lat, 95)), 3),
        "total_cost":   round(float(cost.sum()), 6),
        "cost_per_100": round(float(cost.mean()) * 100, 4),
    }


# ── Part 5: report ─────────────────────────────────────────────────────────────

COL_W = 16

def _row(label: str, vals: list) -> str:
    return f"  {label:12s}" + "".join(f"{str(v):>{COL_W}}" for v in vals)


def format_per_class(metrics: dict, name: str) -> str:
    header = f"  {name}\n"
    header += f"  {'':12}" + "".join(f"{SHORT[l]:>{COL_W}}" for l in LABELS)
    rows = [header]
    for metric in ("precision", "recall", "f1", "support"):
        rows.append(_row(metric, [metrics[l][metric] for l in LABELS]))
    return "\n".join(rows)


def format_confusion(matrix: dict, name: str) -> str:
    header = f"  {name} (rows=gold, cols=predicted)\n"
    header += f"  {'':20}" + "".join(f"{SHORT[l]:>{COL_W}}" for l in LABELS)
    rows = [header]
    for g in LABELS:
        rows.append(f"  {SHORT[g]:20}" + "".join(
            f"{matrix[g].get(p, 0):>{COL_W}}" for p in LABELS
        ))
    return "\n".join(rows)


def build_report(
    preds: list[dict],
    systems: dict[str, list[str]],
    gold: list[str],
    skip_stats: bool = False,
) -> str:
    lines: list[str] = []

    def h(t: str = "") -> None:
        lines.append(t)

    h("=" * 44)
    h("TRIAGELENS EVALUATION REPORT")
    h("=" * 44)
    h()

    # Dataset
    gc = Counter(gold)
    adv_n = sum(1 for p in preds if p.get("adversarial_type"))
    h("DATASET")
    h(f"  Total claims    : {len(gold)}")
    h(f"  RECOMMEND_NEW   : {gc.get('RECOMMEND_NEW_CHECK', 0)}")
    h(f"  REUSE_PRIOR     : {gc.get('REUSE_PRIOR_CHECK', 0)}")
    h(f"  DO_NOT_CHECK    : {gc.get('DO_NOT_CHECK', 0)}")
    h(f"  Adversarial     : {adv_n}")
    h()

    # Macro-F1 summary
    h("-" * 44)
    h("MACRO-F1 SUMMARY")
    h("-" * 44)
    mf1 = {k: per_class_metrics(gold, v)[1] for k, v in systems.items()}
    h(f"  Full system     : {mf1['full']:.4f}")
    h(f"  Keyword BL      : {mf1['keyword']:.4f}")
    h(f"  Zeroshot BL     : {mf1['zeroshot']:.4f}")
    h()

    # Per-class metrics
    h("-" * 44)
    h("PER-CLASS METRICS")
    h("-" * 44)
    for sys_key, sys_name in [("full", "Full system"), ("keyword", "Keyword BL"), ("zeroshot", "Zeroshot BL")]:
        m, _ = per_class_metrics(gold, systems[sys_key])
        h(format_per_class(m, sys_name))
        h()

    # Confusion matrices
    h("-" * 44)
    h("CONFUSION MATRICES")
    h("-" * 44)
    for sys_key, sys_name in [("full", "Full system"), ("keyword", "Keyword BL"), ("zeroshot", "Zeroshot BL")]:
        cm = confusion_matrix(gold, systems[sys_key])
        h(format_confusion(cm, sys_name))
        h()

    # Adversarial subset
    h("-" * 44)
    h("ADVERSARIAL SUBSET")
    h("-" * 44)
    adv = [p for p in preds if p.get("adversarial_type")]
    if adv:
        ag = [p["gold_label"]        for p in adv]
        af = [p["full_system"]       for p in adv]
        ak = [p["keyword_baseline"]  for p in adv]
        az = [p["zeroshot_baseline"] for p in adv]
        h(f"  Full system macro-F1  : {per_class_metrics(ag, af)[1]:.4f}")
        h(f"  Keyword BL macro-F1   : {per_class_metrics(ag, ak)[1]:.4f}")
        h(f"  Zeroshot BL macro-F1  : {per_class_metrics(ag, az)[1]:.4f}")
        h()
        h("  Per-type accuracy:")

        type_groups: dict[str, list] = defaultdict(list)
        for p in adv:
            type_groups[p["adversarial_type"]].append(p)

        cw = 10
        h(f"  {'type':37s}{'full':>{cw}}{'kw':>{cw}}{'zs':>{cw}}")
        h("  " + "─" * (37 + cw * 3))
        for adv_type, group in sorted(type_groups.items()):
            g_ = [p["gold_label"]        for p in group]
            f_ = [p["full_system"]       for p in group]
            k_ = [p["keyword_baseline"]  for p in group]
            z_ = [p["zeroshot_baseline"] for p in group]
            acc = lambda gs, ps: f"{sum(a==b for a,b in zip(gs,ps))/len(gs):.2f}"
            h(f"  {adv_type:37s}{acc(g_,f_):>{cw}}{acc(g_,k_):>{cw}}{acc(g_,z_):>{cw}}")
    else:
        h("  No adversarial records found.")
    h()

    # Statistical tests
    h("-" * 44)
    h("STATISTICAL TESTS")
    h("-" * 44)
    if skip_stats:
        h("  (skipped — sample mode)")
    else:
        chi_fk, p_fk = mcnemar_test(gold, systems["full"], systems["keyword"])
        chi_fz, p_fz = mcnemar_test(gold, systems["full"], systems["zeroshot"])
        h(f"  Full vs Keyword  : chi2={chi_fk:.4f} p={p_fk:.4f}")
        h(f"  Full vs Zeroshot : chi2={chi_fz:.4f} p={p_fz:.4f}")
        h()
        h("  Bootstrap 95% CI (macro-F1 difference, 1000 resamples):")
        lo_fk, hi_fk = bootstrap_f1_diff(gold, systems["full"], systems["keyword"])
        lo_fz, hi_fz = bootstrap_f1_diff(gold, systems["full"], systems["zeroshot"])
        h(f"  Full - Keyword   : [{lo_fk:.4f}, {hi_fk:.4f}]")
        h(f"  Full - Zeroshot  : [{lo_fz:.4f}, {hi_fz:.4f}]")
    h()

    # Performance
    h("-" * 44)
    h("PERFORMANCE")
    h("-" * 44)
    perf = perf_stats(preds)
    h(f"  Latency p50     : {perf['p50']:.3f}s")
    h(f"  Latency p95     : {perf['p95']:.3f}s")
    h(f"  Cost per 100    : ${perf['cost_per_100']:.4f}")
    h()

    # Known limitations
    h("-" * 44)
    h("KNOWN LIMITATIONS")
    h("-" * 44)
    for line in [
        "  1. RAG recall@3 = 0.55：知识库覆盖不完整，",
        "     9 条真变体被降级为 related_but_different",
        "     并保守路由至 RECOMMEND_NEW_CHECK",
        "",
        "  2. virality_score 未纳入：外部传播信号",
        "     缺失，队列优先级排序无实际依据",
        "",
        "  3. 已知事实盲区：checkability 高的已确立",
        "     事实若不在知识库中，可能被误判为",
        "     RECOMMEND_NEW_CHECK",
        "",
        "  4. Zeroshot REUSE_PRIOR_CHECK 幻觉性风险：",
        "     无知识库支撑的变体判断不可验证",
        "",
        "  5. 软化措辞漏检：keyword baseline 对",
        "     defensive phrasing 系统性失效，",
        "     主系统部分缓解但未完全解决",
    ]:
        h(line)
    h()
    h("=" * 44)

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse", action="store_true",
        help="Skip inference and reload existing predictions.jsonl",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Only process the first N claims (skips statistical tests)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reuse and PREDICTIONS_PATH.exists():
        print(f"Reusing {PREDICTIONS_PATH}")
        preds = read_jsonl(PREDICTIONS_PATH)
        if args.sample is not None:
            preds = preds[: args.sample]
            print(f"--sample {args.sample}: using first {len(preds)} records")
    else:
        claims = read_jsonl(TEST_SET_PATH)
        if args.sample is not None:
            claims = claims[: args.sample]
            print(f"--sample {args.sample}: running inference on {len(claims)} claims\n")
        else:
            print(f"Running inference on {len(claims)} claims…\n")
        preds = run_inference(claims)
        write_jsonl(PREDICTIONS_PATH, preds)
        print(f"\nSaved → {PREDICTIONS_PATH}")

    gold    = [p["gold_label"]        for p in preds]
    systems = {
        "full":    [p["full_system"]       for p in preds],
        "keyword": [p["keyword_baseline"]  for p in preds],
        "zeroshot":[p["zeroshot_baseline"] for p in preds],
    }

    report = build_report(preds, systems, gold, skip_stats=args.sample is not None)

    print("\n" + report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
