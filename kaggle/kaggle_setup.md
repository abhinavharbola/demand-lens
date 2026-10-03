# Kaggle Setup

## 1. Create the notebook
1. Open the competition page: **Store Sales - Time Series Forecasting** (Kaggle).
2. Create a new notebook from that page. The dataset is attached as input at
   `/kaggle/input/store-sales-time-series-forecasting/`.
3. Settings: Accelerator = **None/CPU** (no GPU is needed anywhere in this project).
4. Settings: Internet = **ON** (needed to `pip install` prophet, lightgbm, xgboost, mlflow).

## 2. Install packages
Each notebook's first cell runs its own `!pip install -q ...` line, scoped to what that
notebook needs. Nothing has to be installed by hand. `numpy`, `pandas`, `scipy` and
`matplotlib` are preinstalled on Kaggle. `kaggle/requirements-ipynb.txt` is a reference
list of version ranges for debugging install problems.

## 3. Input discovery
Notebooks locate their inputs with `find_input(filename)`, which searches `/kaggle/input`
recursively for a file with that name. It raises an error unless exactly one match exists,
so attach only one version of each upstream notebook output. No path editing is needed.

Inputs per notebook:

| Notebook | Attach |
|---|---|
| 01 | the competition dataset |
| 02 | output of 01 |
| 03 | output of 01 and output of 02 |
| 04 | output of 02 |
| 05 | output of 04 |

## 4. DagsHub and MLflow tracking (notebook 04)
1. Create a DagsHub repo for DemandLens.
2. Get a token: DagsHub, Settings, Tokens.
3. In Kaggle: Add-ons, Secrets, add:
   - `DAGSHUB_TOKEN`: your token
   - `DAGSHUB_REPO`: `yourusername/demand-lens`
4. Notebook 04 reads these secrets and logs to the `demandlens_ml_models` experiment.

## 5. Order of execution
Run the notebooks in order. Each saves its output to `/kaggle/working/`:
1. `01_eda.ipynb` saves `demandlens_subset.parquet`, `stationarity_results.csv`, `series_activation_diagnostics.csv`, `subset_config.json`
2. `02_feature_engineering.ipynb` saves `demand_pattern_classification.csv`, `demandlens_features.parquet`
3. `03_statistical_models.ipynb` saves `prophet_results.csv`, `sarima_results.csv`
4. `04_ml_models.ipynb` saves `ml_results.csv`, `final_holdout_predictions.parquet`, `cost_of_error.json`
5. `05_anomaly_detection.ipynb` saves `anomaly_results.parquet`, `anomaly_eval_metrics.csv`

Use "Save Version" then "Save & Run All" so outputs persist, and attach the previous
notebook's output via "Add Data", "Notebook Output".

## 6. Keeping the chain consistent
Before running notebook N, check that its input is notebook N-1's latest saved version.
If you re-run an earlier notebook, re-attach its output to every downstream notebook and
re-run those too.

The dashboard reads 10 files: `subset_config.json`, `stationarity_results.csv`,
`demand_pattern_classification.csv`, `prophet_results.csv`, `sarima_results.csv`,
`ml_results.csv`, `final_holdout_predictions.parquet`, `anomaly_results.parquet`,
`anomaly_eval_metrics.csv`, `cost_of_error.json`. Download all of them together from the
latest run of each notebook into `kaggle_outputs/`, so the dashboard never mixes files from
two different pipeline runs. The large intermediates (`demandlens_subset.parquet`,
`demandlens_features.parquet`, `series_activation_diagnostics.csv`) stay on Kaggle.
