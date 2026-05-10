ACTION_CONFIG = {
    "RECOMMEND_NEW_CHECK": {
        "streamlit_type": "error",
        "label": "Recommend new check",
        "icon": "🔴",
    },
    "REUSE_PRIOR_CHECK": {
        "streamlit_type": "warning",
        "label": "Reuse prior review",
        "icon": "🟡",
    },
    "DO_NOT_CHECK": {
        "streamlit_type": "success",
        "label": "Do not check",
        "icon": "🟢",
    },
}

SCORE_CONFIG = [
    ("Checkability", "checkability", "#378ADD"),
    ("Harm",         "harm",         "#E24B4A"),
    ("Clarity",      "clarity",      "#1D9E75"),
]


def sim_badge_html(score: float) -> str:
    if score >= 0.70:
        bg, color = "#E1F5EE", "#085041"
    else:
        bg, color = "#FAEEDA", "#633806"
    return (
        f'<span style="background:{bg};color:{color};'
        f'padding:1px 8px;border-radius:4px;'
        f'font-size:12px;font-weight:500">'
        f'{score:.2f}</span>'
    )


def decision_badge_html(
    editor_action: str,
    override_action: str | None = None,
) -> str:
    badges = {
        "ACCEPT":           ("#E1F5EE", "#085041", "✓ accepted"),
        "OVERRIDE_INVALID": ("#FCEBEB", "#791F1F", "✗ invalid"),
        "OVERRIDE":         ("#FAEEDA", "#633806", "↩ override"),
    }
    if editor_action == "ACCEPT":
        key = "ACCEPT"
    elif editor_action == "OVERRIDE" and override_action == "INVALID_CLAIM":
        key = "OVERRIDE_INVALID"
    elif editor_action == "OVERRIDE":
        key = "OVERRIDE"
    else:
        return ""
    bg, color, text = badges[key]
    return (
        f'<span style="background:{bg};color:{color};'
        f'padding:1px 7px;border-radius:4px;'
        f'font-size:11px;font-weight:500">{text}</span>'
    )
