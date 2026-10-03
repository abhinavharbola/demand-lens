import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import (
    DEMAND_PATTERN_KIND,
    TOKENS,
    altair_theme,
    badge_row,
    inject,
    page_header,
)
from src.utils.config import CONFIG

inject(accent_rails={"selected_stores": "neutral", "selected_families": "neutral",
                      "pattern_card": "neutral", "stationarity_card": "neutral"})

page_header(
    eyebrow="Dataset scope",
    title="Dataset Overview",
    subtitle="What was actually run: the store/family subset, its demand-pattern mix, "
             "and how stationary each series is once pre-activation rows are removed.",
)

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]


@st.cache_data
def load_overview_data():
    with open(DATA_DIR / FILES["subset_config"], encoding="utf-8") as f:
        config = json.load(f)
    stationarity = pd.read_csv(DATA_DIR / FILES["stationarity"])
    pattern = pd.read_csv(DATA_DIR / FILES["demand_pattern"])
    return config, stationarity, pattern


try:
    config, stationarity, pattern = load_overview_data()
except FileNotFoundError as e:
    st.error(
        f"Pipeline output not found: {e.filename}. Download the Kaggle outputs into "
        f"{DATA_DIR} as described in the README."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True, key="selected_stores"):
        st.markdown('<div class="rc-card-title">Selected stores</div>', unsafe_allow_html=True)
        badge_row(config["selected_stores"], kind="neutral")
with col2:
    with st.container(border=True, key="selected_families"):
        st.markdown('<div class="rc-card-title">Selected families</div>', unsafe_allow_html=True)
        badge_row(config["selected_families"], kind="neutral")

st.divider()
st.markdown('<div class="rc-eyebrow">Demand pattern classification</div>', unsafe_allow_html=True)
st.caption(
    "Syntetos-Boylan classification (ADI / CV\u00b2), computed on each series after its "
    "pre-activation rows are removed. MAPE is unreliable on intermittent/erratic series, "
    "so see the Forecast Explorer's WAPE/MASE for those."
)

with st.container(border=True, key="pattern_card"):
    counts = pattern["pattern"].value_counts().reset_index()
    counts.columns = ["pattern", "count"]
    pattern_order = [
        p for p in ["smooth", "intermittent", "erratic", "lumpy", "no_demand"]
        if p in set(counts["pattern"])
    ]
    pattern_colors = {p: TOKENS[DEMAND_PATTERN_KIND.get(p, "neutral")] for p in pattern_order}

    chart = (
        alt.Chart(counts)
        .mark_bar(size=28, cornerRadiusEnd=2)
        .encode(
            y=alt.Y("pattern:N", sort=pattern_order, title=None),
            x=alt.X("count:Q", title="series"),
            color=alt.Color(
                "pattern:N",
                scale=alt.Scale(domain=list(pattern_colors), range=list(pattern_colors.values())),
                legend=None,
            ),
            tooltip=["pattern", "count"],
        )
        .properties(height=160)
    )
    st.altair_chart(altair_theme(chart), width='stretch')

    def _pattern_style(row):
        color = pattern_colors.get(row["pattern"], TOKENS["neutral"])
        return [f"background-color: {color}22; color: {color}" if col == "pattern" else "" for col in row.index]

    st.dataframe(pattern.style.apply(_pattern_style, axis=1), width='stretch')

st.divider()
st.markdown('<div class="rc-eyebrow">Stationarity (ADF test)</div>', unsafe_allow_html=True)

with st.container(border=True, key="stationarity_card"):
    pct_stationary = stationarity["likely_stationary_adf"].mean() * 100
    if pct_stationary >= 70:
        pill_kind, pill_note = "good", "Most series are stationary - safe ground for classical statistical models."
    elif pct_stationary >= 40:
        pill_kind, pill_note = "warn", "Mixed - some series will need differencing or trend/seasonal handling."
    else:
        pill_kind, pill_note = "bad", "Mostly non-stationary - lean on ML/global models over classical ARIMA-style ones."
    st.markdown(
        f'<span class="rc-pill rc-pill--{pill_kind}"><span class="rc-pill-dot"></span>'
        f'{pct_stationary:.1f}% stationary by ADF</span>',
        unsafe_allow_html=True,
    )
    st.caption(pill_note)
    st.dataframe(stationarity, width='stretch')
