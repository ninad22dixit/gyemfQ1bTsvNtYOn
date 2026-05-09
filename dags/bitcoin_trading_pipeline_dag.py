from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="bitcoin_hybrid_llm_quant_pipeline",
    start_date=datetime(2026, 5, 5),
    schedule="@hourly",
    catchup=False,
    tags=["bitcoin", "mlops", "mlflow", "dvc"],
)
def bitcoin_pipeline():
    @task
    def refresh_config_task():
        from trading_pipeline.config_manager import refresh_config_from_google_sheet
        return refresh_config_from_google_sheet()

    @task
    def fetch_data_task(config):
        from trading_pipeline.pipeline import fetch_data
        fetch_data(config)
        return "data/raw/btc_candles.csv"

    @task
    def build_features_task(_raw_path):
        from trading_pipeline.pipeline import build_features
        build_features()
        return "data/processed/features.csv"

    @task
    def backtest_task(config, _features_path):
        from trading_pipeline.pipeline import run_backtest
        return run_backtest(config)

    cfg = refresh_config_task()
    raw = fetch_data_task(cfg)
    features = build_features_task(raw)
    backtest_task(cfg, features)


bitcoin_pipeline()
