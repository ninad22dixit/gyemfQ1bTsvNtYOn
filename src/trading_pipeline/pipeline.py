"""Local pipeline entrypoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import mlflow
except Exception:  # MLflow is optional for local/unit-test usage.
    mlflow = None

from trading_pipeline.config_manager import load_config
from trading_pipeline.strategy_core import (
    CONFIG,
    get_btc_historical_candles,
    load_candles_from_csv,
    prepare_feature_table,
    run_hybrid_llm_quant_strategy,
    summarize_performance,
)


def make_synthetic_candles(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-05-05", periods=rows, freq="D", tz="UTC")
    returns = rng.normal(0.001, 0.025, rows)
    close = 45000 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.002, 0.03, rows))
    low = close * (1 - rng.uniform(0.002, 0.03, rows))
    open_ = close * (1 + rng.normal(0, 0.005, rows))
    volume = rng.uniform(1000, 5000, rows)
    return pd.DataFrame({"time": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def resolve_config(config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
    resolved = dict(CONFIG)
    resolved.update(load_config(config_path))
    if config:
        resolved.update(config)
    return resolved


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_raw_candles(config: Dict[str, Any], use_live_data: bool) -> pd.DataFrame:
    if use_live_data:
        try:
            return get_btc_historical_candles(
                config["product_id"],
                int(config["granularity"]),
                int(config["days_back"]),
            )
        except Exception as exc:
            allow_fallback = _as_bool(config.get("allow_synthetic_fallback"), True)
            if not allow_fallback:
                raise
            print(f"Live data fetch failed; using synthetic candles. Error: {exc}")

    return make_synthetic_candles(max(int(config.get("days_back", 300)), 180))


def fetch_data(
    config: Optional[Dict[str, Any]] = None,
    output_path: str = "data/raw/btc_candles.csv",
    use_live_data: bool = True,
) -> pd.DataFrame:
    """Fetch or synthesize raw candle data and persist it for downstream tasks."""
    resolved = resolve_config(config)
    raw = _load_raw_candles(resolved, use_live_data)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False)
    return raw


def build_features(
    raw_path: str = "data/raw/btc_candles.csv",
    output_path: str = "data/processed/features.csv",
) -> pd.DataFrame:
    """Build feature table from persisted raw candles and save it to disk."""
    raw = load_candles_from_csv(raw_path)
    features = prepare_feature_table(raw)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return features


def run_backtest(
    config: Optional[Dict[str, Any]] = None,
    features_path: str = "data/processed/features.csv",
    output_dir: str = "reports/latest_run",
) -> Dict[str, Any]:
    """Run the trading strategy from persisted features and write report files."""
    resolved = resolve_config(config)
    features = pd.read_csv(features_path)
    features["time"] = pd.to_datetime(features["time"], utc=True)
    portfolio, equity_df, buys_df, sells_df, llm_df, guardrail_df = run_hybrid_llm_quant_strategy(features, resolved)
    summary = summarize_performance(portfolio, equity_df, buys_df, sells_df)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(out_dir / "equity_curve.csv", index=False)
    buys_df.to_csv(out_dir / "buys.csv", index=False)
    sells_df.to_csv(out_dir / "sells.csv", index=False)
    llm_df.to_csv(out_dir / "llm_decisions.csv", index=False)
    guardrail_df.to_csv(out_dir / "guardrails.csv", index=False)
    accounting_check = _accounting_check_if_available(equity_df)
    accounting_check.to_csv(out_dir / "accounting_mismatches.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def run_pipeline(config_path: Optional[str] = None, output_dir: str = "outputs", use_live_data: bool = True) -> Dict[str, Any]:
    config = resolve_config(config_path=config_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    raw = _load_raw_candles(config, use_live_data)

    features = prepare_feature_table(raw)
    portfolio, equity_df, buys_df, sells_df, llm_df, guardrail_df = run_hybrid_llm_quant_strategy(features, config)
    summary = summarize_performance(portfolio, equity_df, buys_df, sells_df)

    if mlflow is not None:
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("bitcoin-trading-agent")
            with mlflow.start_run(run_name="local-docker-run"):
                mlflow.log_param("use_live_data", use_live_data)
                mlflow.log_param("product_id", config.get("product_id"))
                mlflow.log_param("granularity", config.get("granularity"))
                mlflow.log_param("days_back", config.get("days_back"))
                for key, value in summary.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        mlflow.log_metric(key, float(value))
        except Exception as exc:
            print(f"MLflow logging skipped: {exc}")

    raw.to_csv(Path(output_dir) / "candles.csv", index=False)
    features.to_csv(Path(output_dir) / "features.csv", index=False)
    equity_df.to_csv(Path(output_dir) / "equity_curve.csv", index=False)
    buys_df.to_csv(Path(output_dir) / "buys.csv", index=False)
    sells_df.to_csv(Path(output_dir) / "sells.csv", index=False)
    llm_df.to_csv(Path(output_dir) / "llm_decisions.csv", index=False)
    guardrail_df.to_csv(Path(output_dir) / "guardrail_events.csv", index=False)
    with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _accounting_check_if_available(equity_df: pd.DataFrame) -> pd.DataFrame:
    from trading_pipeline.strategy_core import accounting_check

    return accounting_check(equity_df) if not equity_df.empty else pd.DataFrame()


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
