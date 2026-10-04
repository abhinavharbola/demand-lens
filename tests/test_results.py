import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.results import (
    best_full_scope_row,
    build_comparison,
    holdout_series_keys,
    select_best_ml_model,
)


def _rows(model, fold, mase_by_series):
    return [
        {"model": model, "fold": fold, "store_nbr": s, "family": f, "mape": 10.0, "wape": 10.0, "mase": m}
        for (s, f), m in mase_by_series.items()
    ]


def _fixtures():
    series = [(1, "DAIRY"), (2, "DAIRY"), (3, "PRODUCE")]
    ml = pd.DataFrame(
        _rows("lightgbm", "fold_1", {k: 0.6 for k in series})
        + _rows("lightgbm", "holdout", {k: 0.9 for k in series})
        + _rows("xgboost", "fold_1", {k: 0.8 for k in series})
        + _rows("xgboost", "holdout", {k: 0.5 for k in series})
    )
    prophet = pd.DataFrame(
        [{"fold": "holdout", "store_nbr": s, "family": f, "mape": 12.0, "wape": 12.0, "mase": 0.7} for s, f in series]
    )
    sarima = pd.DataFrame(
        [{"fold": "holdout", "store_nbr": 1, "family": "DAIRY", "mape": 9.0, "wape": 9.0, "mase": 0.2}]
    )
    return prophet, sarima, ml


def test_model_selection_uses_cv_folds_not_holdout():
    _, _, ml = _fixtures()
    assert select_best_ml_model(ml) == "lightgbm"


def test_comparison_records_series_scope():
    prophet, sarima, ml = _fixtures()
    comparison = build_comparison(prophet, sarima, ml)
    scopes = dict(zip(comparison["model"], comparison["n_series"]))
    assert scopes == {"lightgbm": 3, "xgboost": 3, "prophet": 3, "sarima": 1}


def test_best_row_ignores_partial_scope_models():
    prophet, sarima, ml = _fixtures()
    comparison = build_comparison(prophet, sarima, ml)
    assert comparison.iloc[0]["model"] == "sarima"
    assert best_full_scope_row(comparison)["model"] == "xgboost"


def test_like_for_like_restricts_every_model_to_sarima_series():
    prophet, sarima, ml = _fixtures()
    comparison = build_comparison(prophet, sarima, ml, restrict_to=holdout_series_keys(sarima))
    assert set(comparison["n_series"]) == {1}


def test_selection_refuses_to_fall_back_to_holdout_rows():
    _, _, ml = _fixtures()
    with pytest.raises(ValueError):
        select_best_ml_model(ml[ml["fold"] == "holdout"])
