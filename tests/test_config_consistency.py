import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _notebook_utils import function_defaults, parse_notebook, read_notebook, top_level_constants
from src.utils.config import load_config

NOTEBOOKS = ROOT / "kaggle" / "notebooks"


def _tree(name):
    return parse_notebook(read_notebook(NOTEBOOKS / name))[1]


def test_forecasting_constants_match_notebooks():
    config = load_config()
    expected = {
        "02_feature_engineering.ipynb": {"HORIZON": config["forecasting"]["horizon_days"]},
        "03_statistical_models.ipynb": {
            "HORIZON": config["forecasting"]["horizon_days"],
            "N_FOLDS": config["forecasting"]["n_folds"],
        },
        "04_ml_models.ipynb": {
            "HORIZON": config["forecasting"]["horizon_days"],
            "N_FOLDS": config["forecasting"]["n_folds"],
        },
    }
    for name, constants in expected.items():
        actual = top_level_constants(_tree(name))
        for key, value in constants.items():
            assert actual[key] == value, f"{name}: {key} is {actual[key]}, config says {value}"


def test_selection_parameters_match_notebook_01():
    config = load_config()
    tree = _tree("01_eda.ipynb")
    assert top_level_constants(tree)["LATE_OPENING_THRESHOLD_DAYS"] == config["selection"]["late_opening_threshold_days"]
    defaults = function_defaults(tree, "first_sustained_activation")
    assert defaults["window"] == config["selection"]["sustained_activation"]["window_days"]
    assert defaults["min_active_days"] == config["selection"]["sustained_activation"]["min_active_days"]


def test_anomaly_parameters_match_notebook_05():
    config = load_config()["anomaly_detection"]
    tree = _tree("05_anomaly_detection.ipynb")
    assert function_defaults(tree, "control_limit_flags")["k"] == config["control_limit_k"]
    assert function_defaults(tree, "isolation_forest_flags")["contamination"] == config["isolation_forest_contamination"]
    injection = function_defaults(tree, "inject_synthetic_anomalies")
    assert injection["n_anomalies"] == config["synthetic_injection"]["n_anomalies"]
    assert list(injection["spike_multiplier_range"]) == config["synthetic_injection"]["spike_multiplier_range"]
    assert list(injection["drop_fraction_range"]) == config["synthetic_injection"]["drop_fraction_range"]


def test_cost_table_and_families_match_notebooks():
    config = load_config()
    nb04 = top_level_constants(_tree("04_ml_models.ipynb"))
    nb01 = top_level_constants(_tree("01_eda.ipynb"))
    assert nb04["FAMILY_COST_PER_UNIT_ERROR"] == config["cost_per_unit_error"]
    assert set(nb01["selected_families"]) == set(config["cost_per_unit_error"])


def test_ml_features_respect_the_forecast_horizon():
    config = load_config()
    horizon = config["forecasting"]["horizon_days"]
    nb02 = top_level_constants(_tree("02_feature_engineering.ipynb"))
    nb04 = top_level_constants(_tree("04_ml_models.ipynb"))

    assert min(nb02["FEATURE_LAGS"]) >= horizon

    feature_cols = set(nb04["FEATURE_COLS"])
    expected = {f"lag_{lag}" for lag in nb02["FEATURE_LAGS"]}
    expected |= {f"rolling_{stat}_{w}" for stat in ("mean", "std") for w in nb02["ROLLING_WINDOWS"]}
    assert expected <= feature_cols
    assert {c for c in feature_cols if c.startswith("lag_")} == {f"lag_{lag}" for lag in nb02["FEATURE_LAGS"]}
    assert "dcoilwtico" not in feature_cols
    assert "dcoilwtico_lag" in feature_cols


def test_every_configured_output_file_is_unique():
    files = load_config()["data"]["files"]
    assert len(set(files.values())) == len(files)


def test_cost_of_error_file_key_present():
    assert "cost_of_error" in load_config()["data"]["files"]


def test_data_dir_is_absolute():
    assert Path(load_config()["data"]["kaggle_outputs_dir"]).is_absolute()


def test_streamlit_theme_matches_dashboard_tokens():
    theme_tree = ast.parse((ROOT / "dashboard" / "theme.py").read_text(encoding="utf-8"))
    tokens = next(
        ast.literal_eval(node.value)
        for node in theme_tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "TOKENS"
    )
    toml = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    theme = dict(re.findall(r'^(\w+)\s*=\s*"(#[0-9A-Fa-f]{6})"', toml, re.M))
    assert theme["backgroundColor"].lower() == tokens["bg"].lower()
    assert theme["secondaryBackgroundColor"].lower() == tokens["surface"].lower()
    assert theme["textColor"].lower() == tokens["text"].lower()
    assert theme["primaryColor"].lower() == tokens["forecast"].lower()
