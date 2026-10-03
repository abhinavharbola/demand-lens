import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import CONFIG
from src.utils.results import (
    build_comparison,
    holdout_series_keys,
    load_forecast_tables,
    select_best_ml_model,
)

MAX_ATTEMPTS_PER_PROVIDER = 2
RETRY_BACKOFF_SECONDS = 2
RETRYABLE_STATUS = {408, 425, 429}

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class ProviderError(RuntimeError):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def build_facts(results_dir, config=CONFIG):
    results_dir = Path(results_dir)
    files = config["data"]["files"]

    prophet, sarima, ml = load_forecast_tables(results_dir, files)
    anomaly_metrics = pd.read_csv(results_dir / files["anomaly_eval_metrics"], index_col=0)

    best_model = select_best_ml_model(ml)
    full = build_comparison(prophet, sarima, ml)
    like_for_like = build_comparison(prophet, sarima, ml, restrict_to=holdout_series_keys(sarima))

    def row(table, model):
        return table[table["model"] == model].iloc[0]

    best_ml = row(full, best_model)
    prophet_full = row(full, "prophet")
    sarima_full = row(full, "sarima")
    best_ml_shared = row(like_for_like, best_model)
    prophet_shared = row(like_for_like, "prophet")

    facts = {
        "best_ml_model": best_model,
        "n_series_full": int(best_ml["n_series"]),
        "n_series_sarima": int(sarima_full["n_series"]),
        "n_cv_folds": int(config["forecasting"]["n_folds"]),
        "horizon_days": int(config["forecasting"]["horizon_days"]),
        "best_ml_mase_holdout": round(float(best_ml["mase"]), 3),
        "best_ml_mape_holdout": round(float(best_ml["mape"]), 2),
        "best_ml_wape_holdout": round(float(best_ml["wape"]), 2),
        "prophet_mase_holdout": round(float(prophet_full["mase"]), 3),
        "prophet_mape_holdout": round(float(prophet_full["mape"]), 2),
        "prophet_wape_holdout": round(float(prophet_full["wape"]), 2),
        "sarima_mase_holdout": round(float(sarima_full["mase"]), 3),
        "sarima_mape_holdout": round(float(sarima_full["mape"]), 2),
        "sarima_wape_holdout": round(float(sarima_full["wape"]), 2),
        "best_ml_mase_sarima_series": round(float(best_ml_shared["mase"]), 3),
        "prophet_mase_sarima_series": round(float(prophet_shared["mase"]), 3),
        "control_limit_precision": round(float(anomaly_metrics.loc["control_limit_flag_injected", "precision"]), 3),
        "control_limit_recall": round(float(anomaly_metrics.loc["control_limit_flag_injected", "recall"]), 3),
        "isoforest_precision": round(float(anomaly_metrics.loc["isoforest_flag_injected", "precision"]), 3),
        "isoforest_recall": round(float(anomaly_metrics.loc["isoforest_flag_injected", "recall"]), 3),
    }

    cost_path = results_dir / files.get("cost_of_error", "cost_of_error.json")
    if cost_path.exists():
        with open(cost_path, encoding="utf-8") as f:
            cost_summary = json.load(f)
        best_model_cost = cost_summary.get(best_model)
        if best_model_cost and "estimated_cost_usd" in best_model_cost:
            facts["best_ml_estimated_cost_usd"] = round(float(best_model_cost["estimated_cost_usd"]), 2)

    return facts


def build_prompt(facts):
    return f"""You are a senior retail analytics consultant preparing a results brief for
leadership. Write a structured, insightful report using ONLY the numbers given below. Do
not invent, estimate, or derive new comparison numbers beyond what is directly stated (for
example "X is lower than Y" is fine; computing a new percentage difference that is not in
the facts is not).

Context you must respect:
- All error metrics are computed per series and averaged across series.
- MASE is scaled by each series' in-sample seasonal-naive (lag-7) error. A MASE below 1
  means lower error than that in-sample scale. It is not a measured win over a naive
  forecast on the holdout.
- The ML model was selected on cross-validation folds, not on the holdout. Holdout figures
  are out-of-selection.
- ML features use only information available at the forecast origin (horizon-safe).
- SARIMA covers only n_series_sarima series. Never compare SARIMA's holdout figures
  directly with the all-series figures of the other models. For a like-for-like comparison
  use best_ml_mase_sarima_series and prophet_mase_sarima_series.

Format the report in Markdown with exactly these sections:

## Executive Summary
2-3 bullet points: the single most important takeaway from forecasting, the single most
important takeaway from anomaly detection, and one clear recommendation.

## Forecasting Performance
Compare the three approaches using MASE (primary), MAPE, and WAPE, respecting the scope
rules above. Explain in plain terms what the winning model's MASE means.

## Anomaly Detection Performance
Compare control limits vs. Isolation Forest on precision and recall. Explain the practical
tradeoff between them in one or two sentences. The evaluation uses synthetic injected
anomalies on holdout data.

## Recommendation
One short paragraph: which model to deploy and why, and how the two anomaly methods could
be used together if the numbers justify it. If a cost figure is present in the facts,
you may cite it as an illustrative estimate, not a verified P&L number.

Keep total length to 300-400 words. Do not restate the raw JSON. Do not add a title (the
page already has one).

Facts:
{json.dumps(facts, indent=2)}
"""


def _post_json(url, headers, payload):
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ProviderError(type(e).__name__, retryable=True) from None
    except requests.RequestException as e:
        raise ProviderError(type(e).__name__) from None

    if not resp.ok:
        retryable = resp.status_code in RETRYABLE_STATUS or resp.status_code >= 500
        raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:400]}", retryable=retryable)

    try:
        return resp.json()
    except ValueError:
        raise ProviderError("response was not valid JSON") from None


def _extract_text(data, getter):
    try:
        text = getter(data)
    except (KeyError, IndexError, TypeError):
        raise ProviderError("unexpected response shape") from None
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("empty response")
    return text


def _call_chat_completions(url, prompt, api_key, model, max_tokens, temperature):
    data = _post_json(
        url,
        {"Authorization": f"Bearer {api_key}"},
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    return _extract_text(data, lambda d: d["choices"][0]["message"]["content"])


def _call_nim(prompt, api_key, model, max_tokens, temperature):
    return _call_chat_completions(NIM_URL, prompt, api_key, model, max_tokens, temperature)


def _call_groq(prompt, api_key, model, max_tokens, temperature):
    return _call_chat_completions(GROQ_URL, prompt, api_key, model, max_tokens, temperature)


def _call_gemini(prompt, api_key, model, max_tokens, temperature):
    data = _post_json(
        GEMINI_URL.format(model=model),
        {"x-goog-api-key": api_key},
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        },
    )
    return _extract_text(data, lambda d: d["candidates"][0]["content"]["parts"][0]["text"])


PROVIDERS = {"nim": _call_nim, "groq": _call_groq, "gemini": _call_gemini}
KEY_NAMES = {"nim": "nim_api_key", "groq": "groq_api_key", "gemini": "gemini_api_key"}


def _redact(message, api_key):
    return message.replace(api_key, "***") if api_key else message


def generate_narrative(results_dir, config=CONFIG):
    facts = build_facts(results_dir, config)
    prompt = build_prompt(facts)

    llm_cfg = config["llm"]
    env = config["env"]
    attempts = []

    for provider in llm_cfg["provider_priority"]:
        api_key = env.get(KEY_NAMES[provider])
        if not api_key:
            attempts.append({"provider": provider, "success": False,
                             "error": "no API key configured", "attempt": None})
            continue

        last_error = None
        made = 0
        for attempt_num in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
            made = attempt_num
            try:
                text = PROVIDERS[provider](
                    prompt, api_key, llm_cfg["models"][provider],
                    llm_cfg["max_tokens"], llm_cfg["temperature"],
                )
                attempts.append({"provider": provider, "success": True, "error": None,
                                 "attempt": attempt_num})
                return {"text": text, "provider": provider, "facts": facts, "attempts": attempts}
            except ProviderError as e:
                last_error = _redact(str(e), api_key)
                if not e.retryable:
                    break
            except Exception as e:
                last_error = _redact(f"{type(e).__name__}: {e}", api_key)
                break
            if attempt_num < MAX_ATTEMPTS_PER_PROVIDER:
                time.sleep(RETRY_BACKOFF_SECONDS)

        attempts.append({"provider": provider, "success": False, "error": last_error,
                         "attempt": made})

    raise RuntimeError(f"All LLM providers failed: {attempts}")
