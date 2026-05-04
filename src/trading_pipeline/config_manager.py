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


def save_config_cache(config: Dict[str, Any], path: str = "config/config_cache.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
