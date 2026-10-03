import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.theme import inject, page_header, stat_row
from src.utils.config import CONFIG

NAV_CARDS = [
    ("views/overview.py", "Overview", "neutral",
     "Dataset scope, demand pattern classification, stationarity tests."),
    ("views/forecast_explorer.py", "Forecast Explorer", "forecast",
     "Model comparison and holdout forecast vs. actual, by store and family."),
    ("views/anomaly_view.py", "Anomaly View", "anomaly",
     "Control limits vs. Isolation Forest, synthetic-injection evaluation."),
    ("views/ai_report.py", "AI Report", "ai",
     "Grounded GenAI narrative with numeric claim verification."),
]

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]
HORIZON_DAYS = CONFIG["forecasting"]["horizon_days"]
N_FOLDS = CONFIG["forecasting"]["n_folds"]


@st.cache_data
def load_scope():
    try:
        with open(DATA_DIR / FILES["subset_config"], encoding="utf-8") as f:
            subset_config = json.load(f)
        ml = pd.read_csv(DATA_DIR / FILES["ml_results"], usecols=["model"])
        anomaly = pd.read_csv(DATA_DIR / FILES["anomaly_eval_metrics"], index_col=0)
    except FileNotFoundError:
        return None
    return {
        "n_stores": len(subset_config["selected_stores"]),
        "n_families": len(subset_config["selected_families"]),
        "n_models": int(ml["model"].nunique()) + 2,
        "n_methods": len(anomaly),
    }


scope = load_scope()

inject(accent_rails={f"nav_{i}": accent for i, (_, _, accent, _) in enumerate(NAV_CARDS)})

n_series_text = f"{scope['n_stores'] * scope['n_families']} " if scope else ""
page_header(
    eyebrow="Retail forecasting & anomaly intelligence",
    title="DemandLens",
    subtitle=(
        "Demand forecasting and anomaly detection benchmarked across Prophet, SARIMA, "
        f"LightGBM, and XGBoost on {n_series_text}store-family series from the Favorita Store "
        "Sales dataset, paired with a GenAI results narrative that checks its own numbers "
        "against the underlying data and shows you the result."
    ),
)

if scope:
    stat_row([
        (str(scope["n_stores"] * scope["n_families"]), "series covered"),
        (str(scope["n_models"]), "models benchmarked"),
        (str(scope["n_methods"]), "anomaly detection methods"),
        (f"{HORIZON_DAYS}d", "walk-forward holdout"),
    ])
else:
    st.info(f"No pipeline outputs found in {DATA_DIR}. Download them from Kaggle as described in the README.")

st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="rc-eyebrow">Explore</div>', unsafe_allow_html=True)

cols = st.columns(4)
for i, (col, (path, title, accent, caption)) in enumerate(zip(cols, NAV_CARDS)):
    with col:
        with st.container(border=True, key=f"nav_{i}"):
            st.markdown(
                f'<div class="rc-card-title">{title}</div>'
                f'<div class="rc-card-body">{caption}</div>',
                unsafe_allow_html=True,
            )
            st.page_link(path, label="Open", icon=":material/arrow_forward:")

if scope:
    st.divider()
    st.caption(
        f"{scope['n_stores']} stores \u00d7 {scope['n_families']} product families \u2022 "
        f"{HORIZON_DAYS}-day walk-forward-validated forecast horizon \u2022 "
        f"expanding-window CV, {N_FOLDS} folds + holdout"
    )
