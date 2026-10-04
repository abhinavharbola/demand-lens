import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from _notebook_utils import extract_functions, parse_notebook, read_notebook, top_level_constants

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.metrics import mape, mase, wape

NOTEBOOKS = ROOT / "kaggle" / "notebooks"
NB02 = NOTEBOOKS / "02_feature_engineering.ipynb"
NB03 = NOTEBOOKS / "03_statistical_models.ipynb"
NB04 = NOTEBOOKS / "04_ml_models.ipynb"
NB05 = NOTEBOOKS / "05_anomaly_detection.ipynb"


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def constants_of(path):
    return top_level_constants(parse_notebook(read_notebook(path))[1])


def load(path, names, namespace):
    ns = dict(namespace)
    funcs = extract_functions(path, names)
    for name in names:
        exec(funcs[name], ns)
    return ns


def check_injection_baseline():
    ns = load(NB05, ["inject_synthetic_anomalies"], {"np": np, "pd": pd})
    inject = ns["inject_synthetic_anomalies"]
    df = pd.DataFrame([
        {"store_nbr": 1, "family": "DAIRY", "sales": 0, "forecast": 20},
        {"store_nbr": 1, "family": "DAIRY", "sales": 22, "forecast": 20},
        {"store_nbr": 1, "family": "DAIRY", "sales": 18, "forecast": 20},
        {"store_nbr": 1, "family": "GROCERY I", "sales": 500, "forecast": 500},
        {"store_nbr": 1, "family": "GROCERY I", "sales": 510, "forecast": 500},
        {"store_nbr": 1, "family": "GROCERY I", "sales": 490, "forecast": 500},
    ])
    for seed in range(50):
        eval_df = inject(df, n_anomalies=len(df), seed=seed)
        row = eval_df.iloc[0]
        if row["is_synthetic_anomaly"] == 1 and row["sales_injected"] > row["sales"]:
            return check(
                f"injected DAIRY spike (seed={seed}) uses the series' own scale "
                f"(got {row['sales_injected']:.1f}, expected under 200)",
                row["sales_injected"] < 200,
            )
    print("[SKIP] no seed produced a spike on the target row")
    return True


def check_clean_reference_scaling():
    ns = load(
        NB05,
        ["inject_synthetic_anomalies", "build_scaled_iso_features_from_clean_reference"],
        {"np": np, "pd": pd},
    )
    inject = ns["inject_synthetic_anomalies"]
    build_scaled = ns["build_scaled_iso_features_from_clean_reference"]

    rng = np.random.default_rng(0)
    rows = []
    for store in [1, 2]:
        for family in ["GROCERY I", "DAIRY"]:
            for _ in range(15):
                rows.append({
                    "store_nbr": store, "family": family,
                    "sales": max(0, rng.normal(100, 10)),
                    "forecast": max(0, rng.normal(100, 10)),
                })
    holdout_preds = pd.DataFrame(rows)
    holdout_preds["residual"] = holdout_preds["sales"] - holdout_preds["forecast"]
    eval_df = inject(holdout_preds, n_anomalies=10, seed=1)

    feats = build_scaled(eval_df, holdout_preds, value_cols={"sales_injected": "sales"})
    mean = holdout_preds.groupby(["store_nbr", "family"])["sales"].transform("mean")
    std = holdout_preds.groupby(["store_nbr", "family"])["sales"].transform("std")
    expected = (eval_df["sales_injected"].values - mean.values) / std.values
    return check(
        "isolation forest eval features are scaled from clean group stats",
        np.allclose(feats["sales_injected"].values, expected, equal_nan=True),
    )


def check_isolation_forest_not_capped():
    ns = load(NB05, ["isolation_forest_flags"], {"np": np, "IsolationForest": IsolationForest})
    flags_fn = ns["isolation_forest_flags"]
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, size=(400, 3))
    eval_features = np.vstack([rng.normal(0, 1, size=(160, 3)), rng.normal(0, 1, size=(40, 3)) + 8])
    flags = flags_fn(clean, predict_features=eval_features, contamination=0.05)
    injected_recall = flags[160:].mean()
    return check(
        "isolation forest fit on clean data is not capped by contamination on the eval set "
        f"(flagged {int(flags.sum())} of 200, injected recall {injected_recall:.2f})",
        flags.sum() > 0.05 * len(eval_features) and injected_recall > 0.9,
    )


def check_control_limits_are_centered():
    ns = load(NB05, ["control_limit_flags"], {"np": np, "pd": pd})
    flags_fn = ns["control_limit_flags"]

    rng = np.random.default_rng(5)
    df = pd.DataFrame({"store_nbr": 1, "family": "DAIRY", "residual": 40.0 + rng.normal(0, 5, size=300)})
    biased_flag_rate = flags_fn(df)["control_limit_flag"].mean()

    spiked = df.copy()
    spiked.loc[0, "residual"] = 100.0
    spike_flagged = int(flags_fn(spiked)["control_limit_flag"].iloc[0])

    reference = df.assign(clean_mean=40.0, clean_std=5.0)
    reference["residual"] = reference["residual"] + 100.0
    reference_flags = flags_fn(
        reference, center_reference_col="clean_mean", std_reference_col="clean_std"
    )["control_limit_flag"]

    return check(
        "control limits are centered on the clean mean residual, so a constant forecast bias is not flagged "
        f"(flag rate under bias {biased_flag_rate:.1%}, spike flagged, reference stats honored)",
        biased_flag_rate < 0.05 and spike_flagged == 1 and bool((reference_flags == 1).all()),
    )


def check_run_fold_raises():
    ns = load(NB04, ["run_fold"], {"np": np, "pd": pd, "FEATURE_COLS": [], "TARGET": "sales", "CATEGORICAL_COLS": []})
    dummy = pd.DataFrame({"sales": [1, 2, 3]})
    try:
        ns["run_fold"]("not_a_real_model", dummy, dummy)
    except ValueError:
        return check("run_fold raises ValueError on an unknown model_type", True)
    except Exception as e:
        return check(f"run_fold raised {type(e).__name__}, expected ValueError", False)
    return check("run_fold raises on an unknown model_type", False)


def check_features_have_no_forecast_window_leakage():
    consts = constants_of(NB02)
    ns = load(
        NB02,
        ["add_lag_rolling_features"],
        {"np": np, "pd": pd, "HORIZON": consts["HORIZON"],
         "FEATURE_LAGS": consts["FEATURE_LAGS"], "ROLLING_WINDOWS": consts["ROLLING_WINDOWS"]},
    )
    build = ns["add_lag_rolling_features"]
    horizon = consts["HORIZON"]

    rng = np.random.default_rng(3)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    frames = []
    for store, family in [(1, "DAIRY"), (2, "PRODUCE")]:
        frames.append(pd.DataFrame({
            "store_nbr": store, "family": family, "date": dates,
            "sales": rng.poisson(50, size=len(dates)).astype(float),
        }))
    df = pd.concat(frames, ignore_index=True)

    cutoff = dates[-1] - pd.Timedelta(days=horizon)
    perturbed = df.copy()
    perturbed.loc[perturbed["date"] > cutoff, "sales"] += 1000.0

    base = build(df)
    changed = build(perturbed)
    feature_cols = [c for c in base.columns if c.startswith(("lag_", "rolling_"))]
    window = base["date"] > cutoff
    same = np.allclose(
        base.loc[window, feature_cols].to_numpy(),
        changed.loc[window, feature_cols].to_numpy(),
        equal_nan=True,
    )
    return check(
        f"features of rows inside a {horizon}-day forecast window ignore actuals inside that window",
        same,
    )


def check_per_series_metrics():
    ns = load(
        NB04,
        ["per_series_metrics"],
        {"np": np, "pd": pd, "mape": mape, "wape": wape, "mase": mase,
         "TARGET": "sales", "SERIES_KEYS": ["store_nbr", "family"]},
    )
    fn = ns["per_series_metrics"]

    rng = np.random.default_rng(1)
    train_dates = pd.date_range("2024-01-01", periods=60, freq="D")
    test_dates = pd.date_range("2024-03-01", periods=5, freq="D")
    train_rows, test_rows = [], []
    for store, family, scale in [(1, "DAIRY", 10), (2, "PRODUCE", 1000)]:
        for d in train_dates:
            train_rows.append({"store_nbr": store, "family": family, "date": d, "sales": rng.normal(scale, scale * 0.1)})
        for d in test_dates:
            test_rows.append({"store_nbr": store, "family": family, "date": d, "sales": rng.normal(scale, scale * 0.1)})
    train_df, test_df = pd.DataFrame(train_rows), pd.DataFrame(test_rows)
    pred = test_df["sales"].to_numpy() * 1.1

    out = fn(train_df, test_df, pred)
    ok = len(out) == 2
    for _, row in out.iterrows():
        t = train_df[(train_df["store_nbr"] == row["store_nbr"])]["sales"].to_numpy()
        s = test_df[(test_df["store_nbr"] == row["store_nbr"])]
        expected = mase(s["sales"].to_numpy(), s["sales"].to_numpy() * 1.1, t)
        ok = ok and abs(row["mase"] - expected) < 1e-9
    return check("per_series_metrics scores each series against its own training history", ok)


def check_fold_definitions():
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    results = []
    for path in (NB03, NB04):
        consts = constants_of(path)
        ns = load(path, ["get_walkforward_folds"], {"pd": pd, "HORIZON": consts["HORIZON"], "N_FOLDS": consts["N_FOLDS"]})
        results.append(ns["get_walkforward_folds"](dates))
    folds, holdout = results[0]
    horizon = constants_of(NB03)["HORIZON"]

    ordered = all(
        train_end < test_start and (test_end - test_start).days == horizon - 1
        for train_end, test_start, test_end in folds + [holdout]
    )
    contiguous = all(
        (folds[i + 1][1] - folds[i][2]).days == 1 for i in range(len(folds) - 1)
    ) and (holdout[1] - folds[-1][2]).days == 1
    return check(
        "walk-forward folds are ordered, contiguous and identical in notebooks 03 and 04",
        ordered and contiguous and results[0] == results[1] and holdout[2] == dates[-1],
    )


def main():
    results = [
        check_injection_baseline(),
        check_clean_reference_scaling(),
        check_isolation_forest_not_capped(),
        check_control_limits_are_centered(),
        check_run_fold_raises(),
        check_features_have_no_forecast_window_leakage(),
        check_per_series_metrics(),
        check_fold_definitions(),
    ]
    print()
    if all(results):
        print("All notebook smoke checks passed.")
    else:
        print("Some notebook smoke checks FAILED, see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
