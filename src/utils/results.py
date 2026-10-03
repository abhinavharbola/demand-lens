from pathlib import Path

import pandas as pd

METRICS = ["mape", "wape", "mase"]
SERIES_KEYS = ["store_nbr", "family"]


def load_forecast_tables(data_dir, files):
    data_dir = Path(data_dir)
    prophet = pd.read_csv(data_dir / files["prophet_results"])
    sarima = pd.read_csv(data_dir / files["sarima_results"])
    ml = pd.read_csv(data_dir / files["ml_results"])
    return prophet, sarima, ml


def select_best_ml_model(ml):
    cv = ml[ml["fold"] != "holdout"]
    if cv.empty:
        raise ValueError("ml_results.csv has no cross-validation rows to select a model from")
    return str(cv.groupby("model")["mase"].mean().idxmin())


def holdout_series_keys(df):
    rows = df[df["fold"] == "holdout"]
    return list(rows[SERIES_KEYS].drop_duplicates().itertuples(index=False, name=None))


def _holdout_rows(df, restrict_to):
    rows = df[df["fold"] == "holdout"]
    if restrict_to is not None:
        index = pd.MultiIndex.from_frame(rows[SERIES_KEYS])
        rows = rows[index.isin(restrict_to)]
    return rows


def _summarize(rows, model, kind):
    n_series = int(rows[SERIES_KEYS].drop_duplicates().shape[0])
    record = {"model": model, "source": f"{kind}, {n_series} series", "n_series": n_series}
    for metric in METRICS:
        record[metric] = float(rows[metric].mean())
    return record


def build_comparison(prophet, sarima, ml, restrict_to=None):
    records = []
    for model, rows in _holdout_rows(ml, restrict_to).groupby("model"):
        records.append(_summarize(rows, str(model), "ML"))
    records.append(_summarize(_holdout_rows(prophet, restrict_to), "prophet", "Prophet"))
    records.append(_summarize(_holdout_rows(sarima, restrict_to), "sarima", "SARIMA"))
    comparison = pd.DataFrame(records)
    comparison["fold"] = "holdout"
    return comparison.sort_values("mase").reset_index(drop=True)


def best_full_scope_row(comparison):
    full = comparison[comparison["n_series"] == comparison["n_series"].max()]
    return full.iloc[0]
