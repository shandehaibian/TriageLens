"""TriageLens – Streamlit editor interface."""

from __future__ import annotations

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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TriageLens", layout="wide")
st.title("TriageLens")

# ── Constants ──────────────────────────────────────────────────────────────────
ACTIONS      = ["RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK", "DO_NOT_CHECK"]
ACTION_PLACEHOLDER = "-- Select action --"

_DECISION_MARKER = {"ACCEPT": " ✓", "OVERRIDE": " ↩", "REJECT": " ✗"}
_ACTION_WIDGET   = {
    "RECOMMEND_NEW_CHECK": st.error,
    "REUSE_PRIOR_CHECK":   st.warning,
    "DO_NOT_CHECK":        st.success,
}

# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_claims() -> list[dict]:
    path = ROOT / "fixtures" / "candidate_claims.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_decided_map() -> dict[str, str]:
    """Return {claim_id: latest editor_action} (decisions.jsonl is newest-first)."""
    decided: dict[str, str] = {}
    for d in load_decisions():
        if d["claim_id"] not in decided:
            decided[d["claim_id"]] = d["editor_action"]
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

    action = card["recommended_action"]
    _ACTION_WIDGET.get(action, st.info)(f"RECOMMENDED ACTION: {action}")
    st.write(f"**Rule triggered:** {card['rule_triggered']}")
    st.write(f"**Reasoning:** {card['action_reasoning']}")

    st.markdown("---")
    st.markdown("**── Dimension Scores ──**")
    sc = card["scores"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Checkability", f"{sc['checkability']}/5")
        st.caption(sc["checkability_rationale"])
    with c2:
        st.metric("Harm", f"{sc['harm']}/5")
        st.caption(sc["harm_rationale"])
    with c3:
        st.metric("Clarity", f"{sc['clarity']}/5")
        st.caption(sc["clarity_rationale"])

    st.markdown("---")
    st.markdown("**── Prior Reviews ──**")
    if card["prior_reviews"]:
        for hit in card["prior_reviews"]:
            st.write(
                f"similarity: {hit['similarity_score']:.2f}  |  "
                f"{hit['claim_text'][:120]}"
            )
            st.markdown(f"[{hit['review_url']}]({hit['review_url']})")
    else:
        st.write("No prior reviews found.")


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
            cid   = claim["id"]
            label = f"{claim['claim_text'][:60]}…"
            if cid in decided_map:
                label += _DECISION_MARKER.get(decided_map[cid], "")
            if st.button(label, key=f"btn_{cid}"):
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
                    st.info(f"Already decided: **{already_decided}**")

                b1, b2, b3 = st.columns(3)

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

                with b3:
                    if st.button("Reject", key="btn_reject",
                                 disabled=buttons_disabled):
                        log_decision(selected_id, card, "REJECT")
                        st.session_state["action_message"] = "Decision logged: REJECT"
                        st.session_state["pending_action"] = None
                        st.rerun()

                # ── Override form ──────────────────────────────────────────
                if st.session_state["pending_action"] == "OVERRIDE":
                    st.markdown("---")
                    override_action = st.selectbox(
                        "Your decision:",
                        [ACTION_PLACEHOLDER] + ACTIONS,
                        key="override_action_select",
                    )
                    override_reason = st.text_area(
                        "Reason for override:",
                        placeholder="Explain why you disagree with the recommendation…",
                        key="override_reason_text",
                    )

                    confirm_disabled = (
                        override_action == ACTION_PLACEHOLDER
                        or not override_reason.strip()
                    )

                    cf1, cf2 = st.columns(2)
                    with cf1:
                        if st.button("Confirm Override", key="btn_confirm",
                                     disabled=confirm_disabled):
                            log_decision(
                                selected_id, card, "OVERRIDE",
                                override_action = override_action,
                                override_reason = override_reason.strip(),
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
        fc1, fc2 = st.columns(2)
        with fc1:
            editor_filter = st.selectbox(
                "Editor action",
                ["ALL", "ACCEPT", "OVERRIDE", "REJECT"],
                key="filter_editor",
            )
        with fc2:
            action_filter = st.selectbox(
                "Recommended action",
                ["ALL", "RECOMMEND_NEW_CHECK", "REUSE_PRIOR_CHECK", "DO_NOT_CHECK"],
                key="filter_action",
            )

        filtered = [
            d for d in all_decisions
            if (editor_filter == "ALL" or d["editor_action"]      == editor_filter)
            and (action_filter == "ALL" or d["recommended_action"] == action_filter)
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
            }
            for d in filtered
        ])
        st.dataframe(df, use_container_width=True)

        # ── Stats ─────────────────────────────────────────────────────────────
        total      = len(all_decisions)
        n_accept   = sum(1 for d in all_decisions if d["editor_action"] == "ACCEPT")
        n_override = sum(1 for d in all_decisions if d["editor_action"] == "OVERRIDE")
        n_reject   = sum(1 for d in all_decisions if d["editor_action"] == "REJECT")
        st.caption(
            f"Total: {total} | Filtered: {len(filtered)} | "
            f"ACCEPT {n_accept/total:.0%} | "
            f"OVERRIDE {n_override/total:.0%} | "
            f"REJECT {n_reject/total:.0%}"
        )

        # ── Override analysis matrix ───────────────────────────────────────
        overrides = [d for d in all_decisions if d["editor_action"] == "OVERRIDE"]
        if overrides:
            st.markdown("---")
            st.markdown("**Override Analysis — recommended vs. editor decision**")
            matrix_df = (
                pd.DataFrame([
                    {
                        "recommended": d["recommended_action"],
                        "override_to": d.get("override_action", "unknown"),
                    }
                    for d in overrides
                ])
                .pivot_table(
                    index="recommended",
                    columns="override_to",
                    aggfunc="size",
                    fill_value=0,
                )
            )
            st.dataframe(matrix_df, use_container_width=True)

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
        sc = st.session_state["sidebar_card"]
        _ACTION_WIDGET.get(sc["recommended_action"], st.info)(
            f"**{sc['recommended_action']}**"
        )
        st.write(f"Rule: {sc['rule_triggered']}")
        scores = sc["scores"]
        st.write(
            f"Checkability: **{scores['checkability']}/5** | "
            f"Harm: **{scores['harm']}/5** | "
            f"Clarity: **{scores['clarity']}/5**"
        )
        st.markdown("**Top-3 prior reviews:**")
        if sc["prior_reviews"]:
            for hit in sc["prior_reviews"]:
                st.write(f"• {hit['similarity_score']:.2f} — {hit['claim_text'][:80]}")
                st.markdown(f"[link]({hit['review_url']})")
        else:
            st.write("No prior reviews found.")
