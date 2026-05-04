"""Configuration loader with Google Sheets CSV source and local JSON fallback.

Expected Google Sheet format after publishing as CSV:
key,value
initial_budget,100000
use_llm,false
...

Set GOOGLE_SHEET_CSV_URL to a published Google Sheet CSV URL. Airflow refreshes this
hourly and writes config/config_cache.json. Local runs fall back to config/default_config.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

from trading_pipeline.strategy_core import CONFIG as NOTEBOOK_DEFAULTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"
CACHE_CONFIG_PATH = PROJECT_ROOT / "config" / "config_cache.json"


def _coerce_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null", ""}:
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_config(config: Dict[str, Any], path: Path = CACHE_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def fetch_google_sheet_config(sheet_csv_url: str, timeout: int = 20) -> Dict[str, Any]:
    response = requests.get(sheet_csv_url, timeout=timeout)
    response.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    if not {"key", "value"}.issubset(df.columns):
        raise ValueError("Google Sheet must contain columns: key,value")
    return {str(row["key"]).strip(): _coerce_value(row["value"]) for _, row in df.iterrows()}


def refresh_config_from_google_sheet() -> Dict[str, Any]:
    config = dict(NOTEBOOK_DEFAULTS)
    config.update(load_json_config(DEFAULT_CONFIG_PATH))
    sheet_url = os.getenv("GOOGLE_SHEET_CSV_URL", "").strip()
    if sheet_url:
        sheet_config = fetch_google_sheet_config(sheet_url)
        config.update(sheet_config)
        save_json_config(config, CACHE_CONFIG_PATH)
        return config
    cached = load_json_config(CACHE_CONFIG_PATH)
    if cached:
        config.update(cached)
    save_json_config(config, CACHE_CONFIG_PATH)
    return config


def load_config() -> Dict[str, Any]:
    config = dict(NOTEBOOK_DEFAULTS)
    config.update(load_json_config(DEFAULT_CONFIG_PATH))
    cached = load_json_config(CACHE_CONFIG_PATH)
    if cached:
        config.update(cached)
    return config
