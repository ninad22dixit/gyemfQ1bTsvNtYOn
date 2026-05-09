"""Configuration helpers with local JSON fallback."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file if it exists.

    This keeps the repo runnable without Google credentials.  A production
    Google-Sheets reader can refresh this JSON cache hourly and the pipeline can
    continue to read the local fallback.
    """
    path = Path(config_path or os.getenv("CONFIG_CACHE_PATH", "config/config_cache.json"))
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data




def load_json_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON config file and return it as a dictionary.

    This backwards-compatible helper is used by tests and CI. It is a thin
    wrapper around the same validation behavior as ``load_config`` but requires
    an explicit path so callers can use temporary files safely.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {cfg_path}")
    return data


def save_json_config(config: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as pretty-printed JSON, creating parents as needed."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_config_cache(config: Dict[str, Any], path: str = "config/config_cache.json") -> None:
    """Persist the latest config cache used by the pipeline."""
    save_json_config(config, path)


def refresh_config_from_google_sheet(
    sheet_url: Optional[str] = None,
    cache_path: str | Path = "config/config_cache.json",
) -> Dict[str, Any]:
    """Refresh the runtime config cache from Google Sheets.

    The production DAG imports this function directly. To keep local and CI
    environments runnable without Google credentials, the function falls back
    to the existing cache when no sheet URL is configured.

    Supported inputs:
    - ``GOOGLE_SHEET_ID`` plus service-account credentials in local ``.env``.
    - ``GOOGLE_SHEET_CONFIG_URL`` or ``sheet_url`` pointing to JSON.
    - A CSV/export URL with either ``key,value`` rows or one record of columns.
    """
    _load_local_env()

    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if sheet_id:
        data = _load_config_from_google_sheets_api(sheet_id)
        save_config_cache(data, str(cache_path))
        return data

    url = sheet_url or os.getenv("GOOGLE_SHEET_CONFIG_URL")
    if not url:
        return load_config(str(cache_path))

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    text = response.text.strip()

    if "json" in content_type or text.startswith("{"):
        data = response.json()
    else:
        data = _parse_csv_config(text)

    if not isinstance(data, dict):
        raise ValueError("Google Sheet config must resolve to a JSON object or CSV mapping")

    save_config_cache(data, str(cache_path))
    return data


def _load_config_from_google_sheets_api(sheet_id: str) -> Dict[str, Any]:
    """Read a private Google Sheet via the Sheets API using local credentials."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "Google Sheets API dependencies are missing. Install requirements/base.txt "
            "or add google-api-python-client and google-auth to your environment."
        ) from exc

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    if credentials_json:
        service_account_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=scopes)
    elif credentials_path:
        credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)
    else:
        raise ValueError(
            "Set GOOGLE_APPLICATION_CREDENTIALS to a local service-account JSON file path "
            "or GOOGLE_SERVICE_ACCOUNT_JSON in your private .env file."
        )

    sheet_range = os.getenv("GOOGLE_SHEET_RANGE", "Config!A:B")
    service = build("sheets", "v4", credentials=credentials, cache_discovery=False)
    result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    return _parse_sheet_values(result.get("values", []))


def _parse_csv_config(text: str) -> Dict[str, Any]:
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return {}
    if {"key", "value"}.issubset(rows[0]):
        return {row["key"]: _coerce_config_value(row.get("value")) for row in rows if row.get("key")}
    return {key: _coerce_config_value(value) for key, value in rows[0].items()}


def _parse_sheet_values(values: List[List[Any]]) -> Dict[str, Any]:
    if not values:
        return {}

    headers = [str(cell).strip().lower() for cell in values[0]]
    if len(headers) >= 2 and headers[0] == "key" and headers[1] == "value":
        return {
            str(row[0]).strip(): _coerce_config_value(row[1] if len(row) > 1 else None)
            for row in values[1:]
            if row and str(row[0]).strip()
        }

    if len(values) < 2:
        return {}
    return {
        str(key).strip(): _coerce_config_value(values[1][idx] if idx < len(values[1]) else None)
        for idx, key in enumerate(values[0])
        if str(key).strip()
    }


def _load_local_env(env_path: str | Path = ".env") -> None:
    """Load local .env values without requiring python-dotenv at import time."""
    path = Path(env_path)
    if not path.exists():
        candidate = _find_project_env()
        if candidate is None:
            return
        path = candidate

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_project_env() -> Optional[Path]:
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def _coerce_config_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped.lower() in {"none", "null", ""}:
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped
