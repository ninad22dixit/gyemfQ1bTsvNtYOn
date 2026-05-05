"""Local pipeline entrypoint."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import mlflow
except Exception:  # MLflow is optional for local/unit-test usage.
    mlflow = None

from src.trading_pipeline.config_manager import load_config
from src.trading_pipeline.strategy_core import (
    CONFIG,
    get_btc_historical_candles,
    prepare_feature_table,
    run_hybrid_llm_quant_strategy,
    summarize_performance,
)


def make_synthetic_candles(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    returns = rng.normal(0.001, 0.025, rows)
    close = 45000 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.002, 0.03, rows))
    low = close * (1 - rng.uniform(0.002, 0.03, rows))
    open_ = close * (1 + rng.normal(0, 0.005, rows))
    volume = rng.uniform(1000, 5000, rows)
    return pd.DataFrame({"time": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def run_pipeline(config_path: Optional[str] = None, output_dir: str = "outputs", use_live_data: bool = False) -> Dict[str, Any]:
    config = dict(CONFIG)
    config.update(load_config(config_path))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if use_live_data:
        raw = get_btc_historical_candles(config["product_id"], int(config["granularity"]), int(config["days_back"]))
    else:
        raw = make_synthetic_candles(max(int(config.get("days_back", 180)), 120))

    features = prepare_feature_table(raw)
    portfolio, equity_df, buys_df, sells_df, llm_df, guardrail_df = run_hybrid_llm_quant_strategy(features, config)
    summary = summarize_performance(portfolio, equity_df, buys_df, sells_df)

    if mlflow is not None:
        try:
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


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
