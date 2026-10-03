import json
from datetime import datetime, timezone

from supabase import create_client

from src.utils.config import CONFIG

FORECAST_RUN_COLUMNS = ["model", "source", "fold", "n_series", "mape", "wape", "mase"]
ANOMALY_FLAG_COLUMNS = [
    "date", "store_nbr", "family", "sales", "forecast", "residual",
    "control_limit_flag", "isoforest_flag",
]


def get_client():
    env = CONFIG["env"]
    if not env.get("supabase_url") or not env.get("supabase_key"):
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set in .env")
    return create_client(env["supabase_url"], env["supabase_key"])


def _json_records(df):
    return json.loads(df.to_json(orient="records"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def save_forecast_runs(runs_df):
    records = _json_records(runs_df[FORECAST_RUN_COLUMNS])
    created_at = _now()
    for record in records:
        record["created_at"] = created_at
    return (
        get_client()
        .table("forecast_runs")
        .upsert(records, on_conflict="model,source,fold")
        .execute()
    )


def fetch_forecast_runs(limit=20):
    client = get_client()
    return client.table("forecast_runs").select("*").order("created_at", desc=True).limit(limit).execute().data


def save_report(report_text: str, facts: dict, provider: str, grounding_ratio: float):
    client = get_client()
    payload = {
        "report_text": report_text,
        "facts": facts,
        "provider": provider,
        "grounding_ratio": grounding_ratio,
    }
    return client.table("reports").insert(payload).execute()


def fetch_reports(limit=20):
    client = get_client()
    return client.table("reports").select("*").order("created_at", desc=True).limit(limit).execute().data


def save_anomaly_flags(flags_df):
    records = _json_records(flags_df[ANOMALY_FLAG_COLUMNS])
    created_at = _now()
    for record in records:
        record["created_at"] = created_at
    return (
        get_client()
        .table("anomaly_flags")
        .upsert(records, on_conflict="date,store_nbr,family")
        .execute()
    )


def fetch_anomaly_flags(limit=500):
    client = get_client()
    return client.table("anomaly_flags").select("*").order("date", desc=True).limit(limit).execute().data
