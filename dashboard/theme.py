import streamlit as st

TOKENS = {
    "bg": "#F7F4EE",
    "surface": "#FFFFFF",
    "surface_hover": "#FBF9F4",
    "border": "#E4DFD3",
    "text": "#2B2A26",
    "text_muted": "#6E6A61",
    "text_faint": "#9C978C",
    "forecast": "#2F6D5E",
    "anomaly": "#A8571F",
    "ai": "#3C4770",
    "good": "#2E9457",
    "bad": "#C1493F",
    "neutral": "#8C8779",
    "chart_muted": "#D3CBBC",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Public Sans', sans-serif;
}}

.stApp {{
    background: {TOKENS["bg"]};
}}
section[data-testid="stSidebar"] {{
    background: {TOKENS["surface"]};
    border-right: 1px solid {TOKENS["border"]};
}}
hr {{
    border: none;
    border-top: 1px solid {TOKENS["border"]};
    margin: 1.4rem 0;
}}

.rc-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--rc-eyebrow-color, {TOKENS["text_muted"]});
    margin-bottom: 0.35rem;
}}
.rc-eyebrow--page {{
    text-align: center;
}}
.rc-h1 {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.1rem;
    line-height: 1.15;
    color: {TOKENS["text"]};
    margin: 0 0 0.4rem 0;
    text-align: center;
}}
.rc-sub {{
    font-family: 'Public Sans', sans-serif;
    font-size: 0.98rem;
    color: {TOKENS["text_muted"]};
    width: 100%;
    margin: 0 0 1.1rem 0;
    text-align: center;
}}

.rc-card {{
    background: {TOKENS["surface"]};
    border: 1px solid {TOKENS["border"]};
    border-left: 3px solid {TOKENS["neutral"]};
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 2px rgba(43,42,38,0.04), 0 1px 6px rgba(43,42,38,0.03);
}}
.rc-card--forecast {{ border-left-color: {TOKENS["forecast"]}; }}
.rc-card--anomaly  {{ border-left-color: {TOKENS["anomaly"]}; }}
.rc-card--ai       {{ border-left-color: {TOKENS["ai"]}; }}
.rc-card--neutral  {{ border-left-color: {TOKENS["neutral"]}; }}

.rc-card-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.02rem;
    color: {TOKENS["text"]};
    margin-bottom: 0.2rem;
}}
.rc-card-body {{
    font-family: 'Public Sans', sans-serif;
    font-size: 0.9rem;
    color: {TOKENS["text_muted"]};
    line-height: 1.5;
}}

.rc-compare-card {{ min-height: 172px; }}
.st-key-nav_0, .st-key-nav_1, .st-key-nav_2, .st-key-nav_3 {{ min-height: 168px !important; }}

.rc-stat-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.9rem;
    color: {TOKENS["text"]};
    line-height: 1.1;
}}
.rc-stat-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TOKENS["text_faint"]};
    margin-top: 0.25rem;
}}

.rc-badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 0.14rem 0.55rem;
    border-radius: 3px;
    margin: 0.1rem 0.28rem 0.1rem 0;
    border: 1px solid transparent;
    white-space: nowrap;
}}
.rc-badge--neutral  {{ background: rgba(140,135,121,0.16); color: {TOKENS["text_muted"]}; border-color: {TOKENS["border"]}; }}
.rc-badge--forecast {{ background: rgba(47,109,94,0.12);  color: {TOKENS["forecast"]}; border-color: rgba(47,109,94,0.35); }}
.rc-badge--anomaly  {{ background: rgba(168,87,31,0.12);  color: {TOKENS["anomaly"]}; border-color: rgba(168,87,31,0.35); }}
.rc-badge--ai       {{ background: rgba(60,71,112,0.12);   color: {TOKENS["ai"]}; border-color: rgba(60,71,112,0.35); }}
.rc-badge--good     {{ background: rgba(46,148,87,0.12);   color: {TOKENS["good"]}; border-color: rgba(46,148,87,0.35); }}
.rc-badge--bad      {{ background: rgba(193,73,63,0.12);   color: {TOKENS["bad"]}; border-color: rgba(193,73,63,0.35); }}

.rc-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.4rem 0.85rem;
    border-radius: 4px;
    border: 1px solid;
}}
.rc-pill--good {{ background: rgba(46,148,87,0.10);  color: {TOKENS["good"]}; border-color: rgba(46,148,87,0.4); }}
.rc-pill--warn {{ background: rgba(168,87,31,0.10);  color: {TOKENS["anomaly"]}; border-color: rgba(168,87,31,0.4); }}
.rc-pill--bad  {{ background: rgba(193,73,63,0.10);   color: {TOKENS["bad"]}; border-color: rgba(193,73,63,0.4); }}
.rc-pill-dot {{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; }}

.rc-grounding-bar {{
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.15rem 0.6rem;
    padding: 0.5rem 0.85rem;
    border-radius: 4px;
    border: 1px solid;
    font-size: 0.86rem;
    line-height: 1.5;
}}
.rc-grounding-bar strong {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    white-space: nowrap;
}}

[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TOKENS["text_muted"]} !important;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {TOKENS["border"]};
    border-radius: 4px;
}}
div[data-testid="stMarkdownContainer"] {{
    width: 100%;
}}

{{accent_rail_rules}}

.st-key-selected_stores, .st-key-selected_families {{ min-height: 120px !important; }}

.st-key-model_compare_ai, .st-key-anomaly_chart_0, .st-key-anomaly_chart_1 {{ min-height: 165px !important; }}

.st-key-narrative_card p {{ text-align: justify; text-justify: inter-word; }}
</style>
"""


def inject(accent_rails: dict[str, str] | None = None) -> None:
    rules = []
    if accent_rails:
        for key, token in accent_rails.items():
            color = TOKENS.get(token, TOKENS["neutral"])
            rules.append(
                f'.st-key-{key} {{ border-left: 3px solid {color} !important; '
                f'border-radius: 6px !important; '
                f'box-shadow: 0 1px 2px rgba(43,42,38,0.04), 0 1px 6px rgba(43,42,38,0.03) !important; }}'
            )
    css = _CSS.replace("{accent_rail_rules}", "\n".join(rules))
    st.markdown(css, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "", accent: str = "neutral") -> None:
    color = TOKENS.get(accent, TOKENS["neutral"])
    sub_html = f'<p class="rc-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="rc-eyebrow rc-eyebrow--page" style="--rc-eyebrow-color:{color}">{eyebrow}</div>'
        f'<div class="rc-h1">{title}</div>'
        f'{sub_html}',
        unsafe_allow_html=True,
    )


def card(title: str, body: str, accent: str = "neutral") -> None:
    st.markdown(
        f'<div class="rc-card rc-card--{accent}">'
        f'<div class="rc-card-title">{title}</div>'
        f'<div class="rc-card-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="rc-badge rc-badge--{kind}">{text}</span>'


def badge_row(items: list[str], kind: str = "neutral") -> None:
    st.markdown("".join(badge(str(i), kind) for i in items), unsafe_allow_html=True)


def stat(value: str, label: str) -> str:
    return (
        f'<div><div class="rc-stat-value">{value}</div>'
        f'<div class="rc-stat-label">{label}</div></div>'
    )


def stat_row(stats: list[tuple[str, str]]) -> None:
    cols = st.columns(len(stats))
    for col, (value, label) in zip(cols, stats):
        with col:
            st.markdown(stat(value, label), unsafe_allow_html=True)


DEMAND_PATTERN_KIND = {
    "smooth": "good",
    "intermittent": "neutral",
    "erratic": "anomaly",
    "lumpy": "bad",
    "no_demand": "neutral",
}


def _grounding_status(ratio: float, verifiable: bool = True) -> tuple[str, str]:
    if not verifiable:
        return "warn", "no numeric claims to verify"
    pct = ratio * 100
    if pct >= 90:
        return "good", "grounded"
    if pct >= 70:
        return "warn", "partially grounded"
    return "bad", "low grounding"


def grounding_pill(ratio: float, verifiable: bool = True) -> str:
    kind, label = _grounding_status(ratio, verifiable)
    prefix = f"{ratio * 100:.0f}% " if verifiable else ""
    return (
        f'<span class="rc-pill rc-pill--{kind}">'
        f'<span class="rc-pill-dot"></span>{prefix}{label}</span>'
    )


def grounding_bar(ratio: float, verifiable: bool = True) -> str:
    kind, label = _grounding_status(ratio, verifiable)
    prefix = f"{ratio * 100:.0f}% " if verifiable else ""
    caveat = (
        "Regex-based numeric check, not full claim verification. It can miss paraphrased "
        "claims with no literal number, and can flag numbers that are correct but simply "
        "aren't in the source facts. Treat a low ratio as 'needs review,' not 'definitely wrong.'"
    )
    return (
        f'<div class="rc-grounding-bar rc-pill--{kind}">'
        f'<strong><span class="rc-pill-dot"></span>{prefix}{label}</strong>'
        f'<span style="color:{TOKENS["text_muted"]}">{caveat}</span>'
        f'</div>'
    )


def altair_theme(chart):
    return (
        chart.properties(background=TOKENS["surface"])
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelFont="IBM Plex Mono",
            labelFontSize=10.5,
            labelLimit=0,
            labelColor=TOKENS["text_muted"],
            titleFont="Public Sans",
            titleFontSize=11,
            titleColor=TOKENS["text_muted"],
            grid=True,
            gridColor=TOKENS["border"],
            domainColor=TOKENS["border"],
            tickColor=TOKENS["border"],
        )
        .configure_legend(
            labelFont="Public Sans",
            labelFontSize=11,
            labelColor=TOKENS["text_muted"],
            titleColor=TOKENS["text_muted"],
        )
        .configure_title(font="Fraunces", fontSize=13, color=TOKENS["text"])
    )
