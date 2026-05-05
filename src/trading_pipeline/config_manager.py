"""Configuration helpers with local JSON fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


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
