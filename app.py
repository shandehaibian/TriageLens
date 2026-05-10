"""TriageLens – Streamlit editor interface."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from assembler import assemble_triage_card          # noqa: E402
from claim_scorer import score_claim                # noqa: E402
from logger import load_decisions, log_decision     # noqa: E402
from rag_matcher import classify_prior_match, retrieve_top3  # noqa: E402
from ui_styles import (                             # noqa: E402
    ACTION_CONFIG, SCORE_CONFIG,
    sim_badge_html, decision_badge_html,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TriageLens", layout="wide")
st.title("TriageLens")

# ── Constants ──────────────────────────────────────────────────────────────────
OVERRIDE_ACTIONS   = ["RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK", "DO_NOT_CHECK", "INVALID_CLAIM"]
ACTION_PLACEHOLDER = "-- Select action --"


def _queue_marker(info: dict) -> str:
    ea = info["editor_action"]
    oa = info.get("override_action")
    if ea == "ACCEPT":
        return " ✓"
    if ea == "OVERRIDE" and oa == "INVALID_CLAIM":
        return " ✗"
    if ea == "OVERRIDE":
        return " ↩"
    return ""

# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_claims() -> list[dict]:
    path = ROOT / "fixtures" / "candidate_claims.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_decided_map() -> dict[str, dict]:
    """Return {claim_id: {editor_action, override_action}} (newest-first)."""
    decided: dict[str, dict] = {}
    for d in load_decisions():
        if d["claim_id"] not in decided:
            decided[d["claim_id"]] = {
                "editor_action":  d["editor_action"],
                "override_action": d.get("override_action"),
            }
    return decided


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(claim_item: dict) -> dict:
    key = f"card_{claim_item['id']}"
    if key not in st.session_state:
        with st.spinner("Running pipeline…"):
            top3        = retrieve_top3(claim_item["claim_text"])
            prior_label = classify_prior_match(top3)
            cscore      = score_claim(claim_item["claim_text"], top3)
            card        = assemble_triage_card(
                claim_text        = claim_item["claim_text"],
                prior_match_label = prior_label,
                prior_top3        = top3,
                claim_score       = cscore,
            )
        st.session_state[key] = card
    return st.session_state[key]


def run_pipeline_ad_hoc(claim_text: str) -> dict:
    with st.spinner("Running pipeline…"):
        top3        = retrieve_top3(claim_text)
        prior_label = classify_prior_match(top3)
        cscore      = score_claim(claim_text, top3)
        return assemble_triage_card(
            claim_text        = claim_text,
            prior_match_label = prior_label,
            prior_top3        = top3,
            claim_score       = cscore,
        )


# ── UI helpers ─────────────────────────────────────────────────────────────────

def render_card_body(card: dict, claim_item: dict) -> None:
    st.markdown(f"**{card['claim_text']}**")
    st.caption(f"Source: {claim_item.get('source', 'N/A')}")

    cfg = ACTION_CONFIG[card["recommended_action"]]
    getattr(st, cfg["streamlit_type"])(
        f"{cfg['icon']} **{cfg['label']}**"
        f" · `{card['rule_triggered']}`  \n"
        f"_{card['action_reasoning']}_"
    )

    st.markdown("---")
    st.markdown("**── Dimension Scores ──**")
    col1, col2, col3 = st.columns(3)
    for col, (label, key, _color) in zip([col1, col2, col3], SCORE_CONFIG):
        with col:
            score = card["scores"][key]
            st.metric(label, f"{score} / 5")
            st.progress(score / 5)
            st.caption(card["scores"][f"{key}_rationale"])

    st.markdown("---")
    st.markdown("**── Prior Reviews ──**")
    if card["prior_reviews"]:
        for review in card["prior_reviews"]:
            sim   = review["similarity_score"]
            badge = sim_badge_html(sim)
            url   = review["review_url"]
            text  = html.escape(review["claim_text"])
            st.markdown(
                f'{badge} {text}  \n'
                f'<a href="{url}" target="_blank"'
                f' style="font-size:11px">{url}</a>',
                unsafe_allow_html=True,
            )
            st.divider()
    else:
        st.caption("No prior reviews found.")


# ── Session state init ─────────────────────────────────────────────────────────
for _k, _v in [
    ("selected_claim_id", None),
    ("pending_action",    None),
    ("action_message",    None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Triage Queue", "Audit Log"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Triage Queue
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    claims      = load_claims()
    decided_map = get_decided_map()

    col_queue, col_card = st.columns([1, 2])

    # ── Left: candidate queue ──────────────────────────────────────────────────
    with col_queue:
        st.subheader("Candidate Queue")
        for claim in claims:
            cid         = claim["id"]
            is_decided  = cid in decided_map
            is_selected = (cid == st.session_state["selected_claim_id"])
            escaped     = html.escape(claim["claim_text"][:60])

            if is_decided:
                ea    = decided_map[cid]["editor_action"]
                oa    = decided_map[cid].get("override_action")
                badge = decision_badge_html(ea, oa)
                st.markdown(
                    f'<div style="opacity:0.4;padding:4px 8px 4px 12px;'
                    f'border-left:2px solid transparent;font-size:12px;'
                    f'line-height:1.4">'
                    f'{escaped}… {badge}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if is_selected:
                    st.markdown(
                        '<div style="border-left:2px solid #378ADD;'
                        'padding-left:4px;margin-bottom:-8px"></div>',
                        unsafe_allow_html=True,
                    )
                if st.button(
                    f"{claim['claim_text'][:60]}…",
                    key=f"btn_{cid}",
                    use_container_width=True,
                ):
                    st.session_state["selected_claim_id"] = cid
                    st.session_state["pending_action"]    = None
                    st.session_state["action_message"]    = None

    # ── Right: triage card ─────────────────────────────────────────────────────
    with col_card:
        selected_id = st.session_state["selected_claim_id"]

        if not selected_id:
            st.info("← Select a claim to begin")
        else:
            claim_item = next((c for c in claims if c["id"] == selected_id), None)
            if claim_item is None:
                st.error("Claim not found.")
            else:
                card = run_pipeline(claim_item)

                # One-shot action message
                if st.session_state["action_message"]:
                    st.success(st.session_state["action_message"])
                    st.session_state["action_message"] = None

                render_card_body(card, claim_item)

                # ── Editor action buttons ──────────────────────────────────
                st.markdown("---")
                already_decided = decided_map.get(selected_id)
                buttons_disabled = already_decided is not None

                if buttons_disabled:
                    st.info(f"Already decided: **{already_decided['editor_action']}**")

                b1, b2 = st.columns(2)

                with b1:
                    if st.button("Accept", key="btn_accept",
                                 disabled=buttons_disabled):
                        log_decision(selected_id, card, "ACCEPT")
                        st.session_state["action_message"] = "Decision logged: ACCEPT"
                        st.session_state["pending_action"] = None
                        st.rerun()

                with b2:
                    if st.button("Override", key="btn_override",
                                 disabled=buttons_disabled):
                        st.session_state["pending_action"] = "OVERRIDE"

                # ── Override form ──────────────────────────────────────────
                if st.session_state["pending_action"] == "OVERRIDE":
                    st.markdown("---")
                    override_action = st.selectbox(
                        "Your decision:",
                        [ACTION_PLACEHOLDER] + OVERRIDE_ACTIONS,
                        key="override_action_select",
                    )
                    override_reason = st.text_area(
                        "Reason for override:",
                        placeholder=(
                            "Explain why you disagree "
                            "with the recommendation..."
                        ),
                        key="override_reason_text",
                    )

                    rejection_cause: str | None = None
                    if override_action == "INVALID_CLAIM":
                        rejection_cause = st.text_area(
                            "Rejection cause:",
                            placeholder=(
                                "Specify why this claim is "
                                "invalid: duplicate, off-topic, "
                                "malformed, non-health claim..."
                            ),
                            key="rejection_cause_text",
                        )

                    confirm_disabled = (
                        override_action == ACTION_PLACEHOLDER
                        or not override_reason.strip()
                        or (
                            override_action == "INVALID_CLAIM"
                            and not (rejection_cause or "").strip()
                        )
                    )

                    cf1, cf2 = st.columns(2)
                    with cf1:
                        if st.button("Confirm Override", key="btn_confirm",
                                     disabled=confirm_disabled):
                            log_decision(
                                selected_id, card, "OVERRIDE",
                                override_action = override_action,
                                override_reason = override_reason.strip(),
                                rejection_cause = (
                                    rejection_cause.strip()
                                    if rejection_cause else None
                                ),
                            )
                            st.session_state["action_message"] = (
                                f"Decision logged: OVERRIDE → {override_action}"
                            )
                            st.session_state["pending_action"] = None
                            st.rerun()
                    with cf2:
                        if st.button("Cancel", key="btn_cancel"):
                            st.session_state["pending_action"] = None
                            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Audit Log
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Audit Log")
    all_decisions = load_decisions()

    if not all_decisions:
        st.info("No decisions logged yet.")
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            editor_filter = st.selectbox(
                "Editor action",
                ["ALL", "ACCEPT", "OVERRIDE"],
                key="filter_editor",
            )
        with fc2:
            action_filter = st.selectbox(
                "Recommended action",
                ["ALL", "RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK", "DO_NOT_CHECK"],
                key="filter_action",
            )
        with fc3:
            show_override_filter = editor_filter in ("ALL", "OVERRIDE")
            override_filter = st.selectbox(
                "Override action",
                ["ALL", "RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK",
                 "DO_NOT_CHECK", "INVALID_CLAIM"],
                key="filter_override",
                disabled=not show_override_filter,
            ) if show_override_filter else "ALL"

        filtered = [
            d for d in all_decisions
            if (editor_filter == "ALL" or d["editor_action"]      == editor_filter)
            and (action_filter  == "ALL" or d["recommended_action"] == action_filter)
            and (override_filter == "ALL" or d.get("override_action") == override_filter)
        ]

        # ── Table ─────────────────────────────────────────────────────────────
        df = pd.DataFrame([
            {
                "timestamp":          d["timestamp"][:16],
                "claim":              d["claim_text"][:50],
                "recommended_action": d["recommended_action"],
                "rule_triggered":     d["rule_triggered"],
                "editor_action":      d["editor_action"],
                "override_action":    d.get("override_action"),
                "override_reason":    d.get("override_reason"),
                "rejection_cause":    d.get("rejection_cause"),
            }
            for d in filtered
        ])
        st.dataframe(df, use_container_width=True)

        # ── Stats ─────────────────────────────────────────────────────────────
        total      = len(all_decisions)
        n_accept   = sum(1 for d in all_decisions if d["editor_action"] == "ACCEPT")
        n_override = sum(1 for d in all_decisions if d["editor_action"] == "OVERRIDE")
        st.caption(
            f"Total: {total} | Filtered: {len(filtered)} | "
            f"ACCEPT {n_accept/total:.0%} | "
            f"OVERRIDE {n_override/total:.0%}"
        )

        # ── Override analysis ──────────────────────────────────────────────
        overrides = [d for d in all_decisions if d["editor_action"] == "OVERRIDE"]
        if overrides:
            st.markdown("---")

            # Part 1 — substantive disagreements (non-INVALID_CLAIM)
            substantive = [
                d for d in overrides
                if d.get("override_action") != "INVALID_CLAIM"
            ]
            st.markdown("**Override Analysis Part 1 — Substantive disagreements**")
            st.caption("recommended_action × override_action (excludes INVALID_CLAIM)")
            if substantive:
                matrix_df = (
                    pd.DataFrame([
                        {
                            "recommended": d["recommended_action"],
                            "override_to": d.get("override_action", "unknown"),
                        }
                        for d in substantive
                    ])
                    .pivot_table(
                        index="recommended",
                        columns="override_to",
                        aggfunc="size",
                        fill_value=0,
                    )
                )
                st.dataframe(matrix_df, use_container_width=True)
            else:
                st.info("No substantive overrides yet.")

            # Part 2 — invalid claims
            st.markdown("---")
            st.markdown("**Override Analysis Part 2 — Invalid claims**")
            invalids = [
                d for d in overrides
                if d.get("override_action") == "INVALID_CLAIM"
            ]
            st.caption(f"Total flagged as INVALID_CLAIM: {len(invalids)}")
            if invalids:
                invalid_df = pd.DataFrame([
                    {
                        "timestamp":      d["timestamp"][:16],
                        "claim":          d["claim_text"][:60],
                        "override_reason": d.get("override_reason"),
                        "rejection_cause": d.get("rejection_cause"),
                    }
                    for d in invalids
                ])
                st.dataframe(invalid_df, use_container_width=True)
            else:
                st.info("No claims flagged as invalid yet.")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Live Pipeline Input
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Try a new claim")
    sidebar_text = st.text_area("Paste any health claim:", key="sidebar_claim_text")

    if st.button("Run Pipeline", key="btn_sidebar_run"):
        if sidebar_text.strip():
            st.session_state["sidebar_card"] = run_pipeline_ad_hoc(sidebar_text.strip())

    if "sidebar_card" in st.session_state:
        sc  = st.session_state["sidebar_card"]
        cfg = ACTION_CONFIG[sc["recommended_action"]]
        getattr(st, cfg["streamlit_type"])(
            f"{cfg['icon']} **{cfg['label']}**"
            f" · `{sc['rule_triggered']}`  \n"
            f"_{sc['action_reasoning']}_"
        )

        st.markdown("**Scores**")
        for label, key, _ in SCORE_CONFIG:
            score = sc["scores"][key]
            st.caption(label)
            st.progress(score / 5)
            st.write(f"{score} / 5 — {sc['scores'][f'{key}_rationale']}")

        st.markdown("**Prior reviews**")
        if sc["prior_reviews"]:
            for hit in sc["prior_reviews"]:
                sim   = hit["similarity_score"]
                badge = sim_badge_html(sim)
                text  = html.escape(hit["claim_text"][:80])
                url   = hit["review_url"]
                st.markdown(
                    f'{badge} {text}  \n'
                    f'<a href="{url}" target="_blank"'
                    f' style="font-size:11px">{url}</a>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No prior reviews found.")
