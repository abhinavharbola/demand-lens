from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import TOKENS, altair_theme, inject, page_header
from src.storage.supabase_client import fetch_forecast_runs, save_forecast_runs
from src.tracking.mlflow_utils import fetch_recent_runs
from src.utils.config import CONFIG
from src.utils.results import (
    best_full_scope_row,
    build_comparison,
    holdout_series_keys,
    load_forecast_tables,
    select_best_ml_model,
)

HORIZON_DAYS = CONFIG["forecasting"]["horizon_days"]
N_FOLDS = CONFIG["forecasting"]["n_folds"]

inject(accent_rails={"model_compare": "forecast", "series_chart": "forecast",
                     "experiment_history": "forecast"})

page_header(
    eyebrow="Model benchmark - holdout window",
    title="Forecast Explorer",
    subtitle=f"Prophet, SARIMA, LightGBM, and XGBoost compared on the same {HORIZON_DAYS}-day holdout, "
             "scored with per-series MASE averaged across series (scale-free, comparable across "
             "series of very different volume).",
    accent="forecast",
)

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]


@st.cache_data
def load_results():
    prophet, sarima, ml = load_forecast_tables(DATA_DIR, FILES)
    holdout = pd.read_parquet(DATA_DIR / FILES["final_holdout_predictions"])
    return prophet, sarima, ml, holdout


try:
    prophet, sarima, ml, holdout = load_results()
except FileNotFoundError as e:
    st.error(
        f"Pipeline output not found: {e.filename}. Download the Kaggle outputs into "
        f"{DATA_DIR} as described in the README."
    )
    st.stop()

holdout_model = holdout["model"].iloc[0] if "model" in holdout.columns and not holdout.empty else None
selected_ml_model = select_best_ml_model(ml)

comparison = build_comparison(prophet, sarima, ml)
sarima_keys = holdout_series_keys(sarima)
like_for_like = build_comparison(prophet, sarima, ml, restrict_to=sarima_keys)

best_row = best_full_scope_row(comparison)
best_mase = float(best_row["mase"])
full_scope_n = int(best_row["n_series"])

if holdout_model is not None and holdout_model != selected_ml_model:
    st.warning(
        f"final_holdout_predictions.parquet was written for {holdout_model}, but ml_results.csv "
        f"selects {selected_ml_model} on CV folds. The files likely come from different pipeline runs."
    )

st.markdown(
    f'<div class="rc-card rc-card--forecast">'
    f'<div class="rc-card-title">Best on holdout ({full_scope_n}-series scope): {str(best_row["model"]).upper()}</div>'
    f'<div class="rc-card-body">'
    f'<span class="rc-stat-value" style="font-size:1.5rem">{best_mase:.3f}</span> MASE'
    + (
        f' &nbsp;\u2022&nbsp; <span style="color:{TOKENS["good"]}">below 1.0: error under the in-sample '
        f'seasonal-naive (lag-7) scale</span>' if best_mase < 1
        else ' &nbsp;\u2022&nbsp; at or above the in-sample seasonal-naive scale (MASE \u2265 1.0)'
    )
    + f' &nbsp;\u2022&nbsp; {float(best_row["mape"]):.2f}% MAPE &nbsp;\u2022&nbsp; '
    f'{float(best_row["wape"]):.2f}% WAPE'
    f'</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Model comparison (holdout)</div>'
            .format(TOKENS["forecast"]), unsafe_allow_html=True)

with st.container(border=True, key="model_compare"):
    chart_df = comparison.copy()
    chart_df["label"] = chart_df["model"] + " (" + chart_df["n_series"].astype(str) + " series)"
    chart_df["is_best"] = chart_df["model"] == best_row["model"]

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            y=alt.Y("label:N", sort="-x", title=None),
            x=alt.X("mase:Q", title="MASE (lower is better)"),
            color=alt.condition(
                alt.datum.is_best,
                alt.value(TOKENS["forecast"]),
                alt.value(TOKENS["chart_muted"]),
            ),
            tooltip=["source", "model", "n_series", alt.Tooltip("mase:Q", format=".3f"),
                     alt.Tooltip("mape:Q", format=".2f"), alt.Tooltip("wape:Q", format=".2f")],
        )
        .properties(height=42 * len(chart_df) + 20)
    )
    baseline_rule = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(strokeDash=[4, 3], color=TOKENS["text_faint"])
        .encode(x=alt.X("x:Q", title="MASE (lower is better)"))
    )
    st.altair_chart(altair_theme(bars + baseline_rule), width='stretch')
    st.caption(
        "MASE is scaled by each series' in-sample seasonal-naive (lag-7) error, so the dashed "
        "line at 1.0 marks that in-sample scale, not a naive forecast scored on the holdout. "
        f"The highlighted bar is the best model among those covering all {full_scope_n} series."
    )

    st.dataframe(
        comparison[["source", "model", "n_series", "mape", "wape", "mase"]],
        width='stretch',
        hide_index=True,
    )
    st.caption(
        "All metrics are computed per series and then averaged, for every model. "
        f"The ML model ({selected_ml_model}) was selected on mean MASE over the {N_FOLDS} CV folds; "
        "holdout figures are not used for selection. SARIMA only ran on a few representative "
        "series (CPU cost), so its row is not comparable in scope to the others."
    )

    st.markdown(
        '<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Like-for-like on SARIMA\'s series</div>'
        .format(TOKENS["forecast"]), unsafe_allow_html=True,
    )
    st.dataframe(
        like_for_like[["source", "model", "n_series", "mape", "wape", "mase"]],
        width='stretch',
        hide_index=True,
    )
    st.caption(
        f"Every model restricted to the {len(sarima_keys)} series SARIMA was fit on, so the "
        "averages cover identical series."
    )

    if st.button("Log this comparison to Supabase"):
        to_log = pd.concat([comparison, like_for_like], ignore_index=True).drop_duplicates(
            subset=["model", "source", "fold"]
        )
        try:
            save_forecast_runs(to_log)
            st.success(f"Saved {len(to_log)} rows to Supabase (existing rows for the same model, scope and fold are updated).")
        except Exception as e:
            st.error(f"Logging failed: {e}")

    with st.expander("Past logged comparisons"):
        try:
            past_runs = fetch_forecast_runs()
        except Exception as e:
            past_runs = None
            st.caption(f"Could not load past comparisons: {e}")
        if past_runs:
            st.dataframe(pd.DataFrame(past_runs), width='stretch', hide_index=True)
        elif past_runs is not None:
            st.caption("No comparisons logged yet.")

st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Experiment history</div>'
            .format(TOKENS["forecast"]), unsafe_allow_html=True)
st.caption("Past ML training runs tracked via MLflow on DagsHub, separate from the "
           "holdout comparison above (this pulls raw run metrics/params, not just the "
           "final holdout scores).")
with st.container(border=True, key="experiment_history"):
    try:
        recent_runs = fetch_recent_runs()
    except Exception as e:
        recent_runs = None
        st.caption(f"Could not load MLflow run history: {e}")
    if recent_runs:
        st.dataframe(pd.json_normalize(recent_runs), width='stretch', hide_index=True)
    elif recent_runs is not None:
        st.caption("No MLflow run history available. Set DAGSHUB_TOKEN and DAGSHUB_REPO "
                   "in .env to enable this (optional, everything else works without it).")

st.divider()
st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Store-family forecast vs. actual</div>'
            .format(TOKENS["forecast"]), unsafe_allow_html=True)

col1, col2 = st.columns(2)
store = col1.selectbox("Store", sorted(holdout["store_nbr"].unique()))
family = col2.selectbox("Family", sorted(holdout["family"].unique()))

series = holdout[(holdout["store_nbr"] == store) & (holdout["family"] == family)].sort_values("date")
with st.container(border=True, key="series_chart"):
    if series.empty:
        st.warning("No holdout predictions for this store/family combination.")
    else:
        melted = series.melt(id_vars="date", value_vars=["sales", "forecast"],
                             var_name="series", value_name="units")
        line = (
            alt.Chart(melted)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
                y=alt.Y("units:Q", title="units"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["sales", "forecast"],
                        range=[TOKENS["text"], TOKENS["forecast"]],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["date:T", "series:N", alt.Tooltip("units:Q", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(altair_theme(line), width='stretch')
        shown = holdout_model if holdout_model is not None else selected_ml_model
        st.caption(f"Predictions shown are from {shown}, the ML model with the lowest mean MASE "
                   "over the CV folds (see the comparison above for how it stacks up against "
                   "Prophet and SARIMA).")
