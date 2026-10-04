import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import narrative
from src.utils.config import load_config


def _write_results(tmp_path, config):
    files = config["data"]["files"]
    series = [(1, "DAIRY"), (2, "PRODUCE")]

    def rows(model, fold, mase):
        return [
            {"model": model, "fold": fold, "store_nbr": s, "family": f, "mape": 10.0, "wape": 9.0, "mase": mase}
            for s, f in series
        ]

    ml = pd.DataFrame(
        rows("lightgbm", "fold_1", 0.6) + rows("lightgbm", "holdout", 0.9)
        + rows("xgboost", "fold_1", 0.8) + rows("xgboost", "holdout", 0.5)
    )
    ml.to_csv(tmp_path / files["ml_results"], index=False)

    prophet = pd.DataFrame(
        [{"fold": "holdout", "store_nbr": s, "family": f, "mape": 12.0, "wape": 11.0, "mase": 0.7} for s, f in series]
    )
    prophet.to_csv(tmp_path / files["prophet_results"], index=False)

    sarima = pd.DataFrame(
        [{"fold": "holdout", "store_nbr": 1, "family": "DAIRY", "mape": 9.0, "wape": 8.0, "mase": 0.4}]
    )
    sarima.to_csv(tmp_path / files["sarima_results"], index=False)

    anomaly = pd.DataFrame(
        {"precision": [0.8, 0.7], "recall": [0.5, 0.9], "f1": [0.6, 0.78]},
        index=["control_limit_flag_injected", "isoforest_flag_injected"],
    )
    anomaly.to_csv(tmp_path / files["anomaly_eval_metrics"])

    with open(tmp_path / files["cost_of_error"], "w", encoding="utf-8") as f:
        json.dump({"lightgbm": {"estimated_cost_usd": 123.456}}, f)


def test_build_facts_selects_model_on_cv_and_records_scope(tmp_path):
    config = load_config()
    _write_results(tmp_path, config)
    facts = narrative.build_facts(tmp_path, config)
    assert facts["best_ml_model"] == "lightgbm"
    assert facts["best_ml_mase_holdout"] == 0.9
    assert facts["n_series_full"] == 2
    assert facts["n_series_sarima"] == 1
    assert facts["best_ml_mase_sarima_series"] == 0.9
    assert facts["best_ml_estimated_cost_usd"] == 123.46


def test_build_facts_honours_config_argument(tmp_path):
    config = load_config()
    _write_results(tmp_path, config)
    config["forecasting"]["horizon_days"] = 21
    assert narrative.build_facts(tmp_path, config)["horizon_days"] == 21


def _config_with_key(key="secret-key-123"):
    config = load_config()
    config["env"] = {**config["env"], "groq_api_key": key, "nim_api_key": None, "gemini_api_key": None}
    config["llm"] = {**config["llm"], "provider_priority": ["groq", "nim", "gemini"]}
    return config


def _stub_facts(monkeypatch):
    monkeypatch.setattr(narrative, "build_facts", lambda results_dir, config: {"mase": 0.5})
    monkeypatch.setattr(narrative.time, "sleep", lambda s: None)


def test_non_retryable_error_is_not_retried(monkeypatch):
    _stub_facts(monkeypatch)
    calls = []

    def fail(prompt, api_key, model, max_tokens, temperature):
        calls.append(1)
        raise narrative.ProviderError("HTTP 401: bad key", retryable=False)

    monkeypatch.setitem(narrative.PROVIDERS, "groq", fail)
    with pytest.raises(RuntimeError):
        narrative.generate_narrative("unused", _config_with_key())
    assert len(calls) == 1


def test_retryable_error_is_retried_once(monkeypatch):
    _stub_facts(monkeypatch)
    calls = []

    def flaky(prompt, api_key, model, max_tokens, temperature):
        calls.append(1)
        if len(calls) == 1:
            raise narrative.ProviderError("HTTP 429: slow down", retryable=True)
        return "report text"

    monkeypatch.setitem(narrative.PROVIDERS, "groq", flaky)
    result = narrative.generate_narrative("unused", _config_with_key())
    assert result["text"] == "report text"
    assert len(calls) == 2


def test_api_key_is_redacted_from_recorded_errors(monkeypatch):
    _stub_facts(monkeypatch)

    def leaky(prompt, api_key, model, max_tokens, temperature):
        raise RuntimeError(f"failed calling https://example.test/?key={api_key}")

    monkeypatch.setitem(narrative.PROVIDERS, "groq", leaky)
    with pytest.raises(RuntimeError) as excinfo:
        narrative.generate_narrative("unused", _config_with_key("secret-key-123"))
    assert "secret-key-123" not in str(excinfo.value)
