import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

load_dotenv(ROOT / ".env")


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["kaggle_outputs_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    config["data"]["kaggle_outputs_dir"] = str(data_dir)

    config["env"] = {
        "nim_api_key": os.getenv("NIM_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_key": os.getenv("SUPABASE_KEY"),
        "dagshub_token": os.getenv("DAGSHUB_TOKEN"),
        "dagshub_repo": os.getenv("DAGSHUB_REPO"),
    }
    return config


CONFIG = load_config()
