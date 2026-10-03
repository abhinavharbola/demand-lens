import os

import mlflow

from src.utils.config import CONFIG


def configure_tracking():
    env = CONFIG["env"]
    if not env.get("dagshub_token") or not env.get("dagshub_repo"):
        return False
    os.environ["MLFLOW_TRACKING_USERNAME"] = env["dagshub_token"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = env["dagshub_token"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{env['dagshub_repo']}.mlflow")
    return True


def fetch_recent_runs(experiment_name="demandlens_ml_models", max_results=20):
    if not configure_tracking():
        return []
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id], max_results=max_results, order_by=["start_time DESC"]
    )
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName"),
            "metrics": r.data.metrics,
            "params": r.data.params,
        }
        for r in runs
    ]
