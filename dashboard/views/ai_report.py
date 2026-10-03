import html
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import TOKENS, altair_theme, badge, grounding_bar, grounding_pill, inject, page_header
from src.llm.grounding_check import check_grounding
from src.llm.narrative import generate_narrative
from src.storage.supabase_client import fetch_reports, save_report
from src.utils.config import CONFIG

inject(accent_rails={
    "narrative_card": "ai", "model_compare_ai": "forecast",
    "anomaly_chart_0": "anomaly", "anomaly_chart_1": "anomaly",
})

page_header(
    eyebrow="GenAI results brief - grounding verified",
    title="AI-Generated Results Report",
    subtitle="Every number the model writes is checked against the same facts it was "
             "given, and the result is shown next to the report. Reports are displayed "
             "and saved even when the check flags claims.",
    accent="ai",
)

DATA_DIR = str(Path(CONFIG["data"]["kaggle_outputs_dir"]))
CHART_BOX_HEIGHT = 130


def build_report_markdown(text, facts, provider, grounding_ratio, created_at=None, verifiable=True):
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    grounding_text = f"{grounding_ratio * 100:.0f}%" if verifiable else "n/a (no numeric claims to verify)"
    facts_lines = "\n".join(f"- **{k}**: {v}" for k, v in facts.items())
    return f"""# DemandLens AI Report

**Generated:** {created_at}
**Provider:** {provider}
**Grounding ratio:** {grounding_text}

---

{text}

---

## Source facts

{facts_lines}
"""


def download_filename(provider, created_at):
    safe_ts = str(created_at).replace(":", "-").replace(" ", "_")
    return f"demandlens_report_{provider}_{safe_ts}.md"


def render_model_mase_chart(facts):
    sarima_label = f"sarima ({facts['n_series_sarima']} series)"
    df = pd.DataFrame({
        "model": [facts["best_ml_model"], "prophet", sarima_label],
        "mase": [facts["best_ml_mase_holdout"], facts["prophet_mase_holdout"], facts["sarima_mase_holdout"]],
    })
    df["is_best"] = df["mase"] == df["mase"].iloc[:2].min()
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            y=alt.Y("model:N", sort="-x", title=None),
            x=alt.X("mase:Q", title=None),
            color=alt.condition(alt.datum.is_best, alt.value(TOKENS["forecast"]), alt.value(TOKENS["neutral"])),
            tooltip=["model", alt.Tooltip("mase:Q", format=".3f")],
        )
        .properties(height=CHART_BOX_HEIGHT, title="MASE by model")
    )
    return altair_theme(chart)


def render_anomaly_method_chart(precision, recall, title):
    df = pd.DataFrame({"metric": ["precision", "recall"], "score": [precision, recall]})
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=2, size=20)
        .encode(
            y=alt.Y("metric:N", sort=["precision", "recall"], title=None),
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                     axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["precision", "recall"], range=[TOKENS["anomaly"], TOKENS["forecast"]]),
                legend=None,
            ),
            tooltip=["metric", alt.Tooltip("score:Q", format=".2f")],
        )
        .properties(height=CHART_BOX_HEIGHT, title=title)
    )
    return altair_theme(chart)


def _generate_report():
    with st.spinner("Calling LLM provider..."):
        try:
            result = generate_narrative(DATA_DIR)
        except Exception as e:
            st.session_state.ai_report = None
            st.session_state.ai_report_error = str(e)
            return

    grounding = check_grounding(result["text"], result["facts"])
    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        save_report(result["text"], result["facts"], result["provider"], grounding["grounded_ratio"])
        save_status = "Saved to Supabase."
    except Exception as e:
        save_status = f"Not saved to Supabase: {e}"

    st.session_state.ai_report = {
        "result": result, "grounding": grounding,
        "generated_at": generated_at, "save_status": save_status,
    }
    st.session_state.ai_report_error = None


has_report = st.session_state.get("ai_report") is not None
st.button(
    "Regenerate report" if has_report else "Generate new report",
    type="secondary" if has_report else "primary",
    on_click=_generate_report,
)

if st.session_state.get("ai_report_error"):
    st.error(f"Report generation failed: {st.session_state.ai_report_error}")

if has_report:
    result = st.session_state.ai_report["result"]
    grounding = st.session_state.ai_report["grounding"]
    facts = result["facts"]
    generated_at = st.session_state.ai_report["generated_at"]
    fallback_happened = any(a["provider"] != result["provider"] for a in result["attempts"])

    st.markdown('<div class="rc-eyebrow">Key metrics at a glance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Forecasting: per-series MASE on holdout, averaged (lower is better)")
        with st.container(border=True, key="model_compare_ai"):
            st.altair_chart(render_model_mase_chart(facts), width='stretch')
    with col2:
        st.caption("Anomaly detection: precision vs recall")
        sub1, sub2 = st.columns(2)
        with sub1:
            with st.container(border=True, key="anomaly_chart_0"):
                st.altair_chart(
                    render_anomaly_method_chart(
                        facts["control_limit_precision"], facts["control_limit_recall"], "Control limits",
                    ),
                    width='stretch',
                )
        with sub2:
            with st.container(border=True, key="anomaly_chart_1"):
                st.altair_chart(
                    render_anomaly_method_chart(
                        facts["isoforest_precision"], facts["isoforest_recall"], "Isolation Forest",
                    ),
                    width='stretch',
                )

    if "best_ml_estimated_cost_usd" in facts:
        st.markdown(
            f'<div class="rc-badge rc-badge--neutral" style="font-size:0.82rem">'
            f'Estimated cost of forecast error ({facts["best_ml_model"]}, holdout): '
            f'${facts["best_ml_estimated_cost_usd"]:,.2f}. Illustrative, based on published '
            f'margin benchmarks, not verified P&amp;L data.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.markdown(grounding_bar(grounding["grounded_ratio"], grounding["verifiable"]), unsafe_allow_html=True)

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}; margin-bottom:0">Narrative</div>'
                .format(TOKENS["ai"]), unsafe_allow_html=True)
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    with st.container(border=True, key="narrative_card"):
        st.markdown(result["text"])

    dl_col, claims_col, facts_col, fallback_col = st.columns(4)
    with dl_col:
        st.download_button(
            "Download this report (.md)",
            data=build_report_markdown(
                result["text"], facts, result["provider"], grounding["grounded_ratio"],
                generated_at, grounding["verifiable"],
            ),
            file_name=download_filename(result["provider"], generated_at),
            mime="text/markdown",
            icon=":material/download:",
            width='stretch',
        )
    with claims_col:
        with st.expander("Flagged numeric claims"):
            ungrounded = [c for c in grounding["claims"] if not c["grounded"]]
            if not grounding["verifiable"]:
                st.write("No numeric claims were extracted from the report, so nothing could be verified.")
            elif ungrounded:
                st.write(ungrounded)
            else:
                grounded_claims = [c for c in grounding["claims"] if c["grounded"]]
                st.write("None flagged. Every extracted number matched a source figure "
                         "(see which fact matched each claim below).")
                st.json([
                    {"claim": c["raw"], "matched_fact_key": c["matched_fact_key"]}
                    for c in grounded_claims
                ])
    with facts_col:
        with st.expander("Source facts used"):
            st.json(facts)
    with fallback_col:
        with st.expander("Provider fallback log", expanded=fallback_happened):
            for a in result["attempts"]:
                status_badge = badge("ok", "good") if a["success"] else badge("failed", "bad")
                st.markdown(
                    f'{status_badge} <strong>{html.escape(str(a["provider"]))}</strong> '
                    f'<span style="color:{TOKENS["text_muted"]}">{html.escape(str(a["error"] or "succeeded"))}</span>',
                    unsafe_allow_html=True,
                )

    st.caption(st.session_state.ai_report["save_status"])

st.divider()
st.markdown('<div class="rc-eyebrow">Past reports</div>', unsafe_allow_html=True)
try:
    past = fetch_reports()
    if past:
        for r in past:
            created_at = r.get("created_at", "unknown date")
            provider = r.get("provider", "unknown provider")
            ratio = r.get("grounding_ratio") or 0
            with st.expander(f"{created_at} \u2022 {provider} \u2022 {ratio * 100:.0f}% grounded"):
                st.markdown(grounding_pill(ratio), unsafe_allow_html=True)
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown(r["report_text"])
                st.download_button(
                    "Download this report (.md)",
                    data=build_report_markdown(
                        r["report_text"], r.get("facts", {}), provider,
                        ratio, created_at,
                    ),
                    file_name=download_filename(provider, created_at),
                    mime="text/markdown",
                    icon=":material/download:",
                    key=f"download_{r.get('id', created_at)}",
                )
    else:
        st.caption("No saved reports yet.")
except Exception as e:
    st.caption(f"Could not load past reports: {e}")
