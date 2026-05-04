from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import mlflow
import pandas as pd

from trading_pipeline.config_manager import load_config, refresh_config_from_google_sheet
from trading_pipeline.strategy_core import (
    accounting_check,
    get_btc_historical_candles,
    prepare_feature_table,
    run_hybrid_llm_quant_strategy,
    summarize_performance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fetch_data(config: Dict[str, Any] | None = None) -> pd.DataFrame:
    config = config or load_config()
    df = get_btc_historical_candles(
        product_id=config["product_id"],
        granularity=int(config["granularity"]),
        days_back=int(config["days_back"]),
    )
    out = PROJECT_ROOT / "data" / "raw" / "btc_candles.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def build_features(raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw_df is None:
        raw_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "btc_candles.csv", parse_dates=["time"])
    df = prepare_feature_table(raw_df)
    out = PROJECT_ROOT / "data" / "processed" / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def run_backtest(config: Dict[str, Any] | None = None, features_df: pd.DataFrame | None = None) -> Dict[str, Any]:
    config = config or load_config()
    if features_df is None:
        features_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features.csv", parse_dates=["time"])

    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "file:reports/mlruns"))
    mlflow.set_experiment(config.get("mlflow_experiment_name", "bitcoin-hybrid-llm-quant"))

    with mlflow.start_run(run_name="hybrid_llm_quant_backtest"):
        for key in ["product_id", "granularity", "days_back", "initial_budget", "use_llm", "dca_budget_fraction", "swing_budget_fraction"]:
            if key in config:
                mlflow.log_param(key, config[key])

        portfolio, equity_df, buys_df, sells_df, llm_df, guardrail_df = run_hybrid_llm_quant_strategy(features_df, config)
        summary = summarize_performance(portfolio, equity_df, buys_df, sells_df)
        mismatches = accounting_check(equity_df)
        summary["accounting_mismatches"] = int(len(mismatches))

        run_dir = PROJECT_ROOT / "reports" / "latest_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        equity_df.to_csv(run_dir / "equity_curve.csv", index=False)
        buys_df.to_csv(run_dir / "buys.csv", index=False)
        sells_df.to_csv(run_dir / "sells.csv", index=False)
        llm_df.to_csv(run_dir / "llm_decisions.csv", index=False)
        guardrail_df.to_csv(run_dir / "guardrails.csv", index=False)
        mismatches.to_csv(run_dir / "accounting_mismatches.csv", index=False)
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        for k, v in summary.items():
            if isinstance(v, (int, float)) and pd.notna(v):
                mlflow.log_metric(k, float(v))
        mlflow.log_artifacts(str(run_dir), artifact_path="latest_run")
        return summary


def run_full_pipeline(refresh_config: bool = True) -> Dict[str, Any]:
    config = refresh_config_from_google_sheet() if refresh_config else load_config()
    raw = fetch_data(config)
    features = build_features(raw)
    return run_backtest(config, features)
