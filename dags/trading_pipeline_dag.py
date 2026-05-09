from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    dag_id="bitcoin_trading_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 5, 5),
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
) as dag:

    run_pipeline = BashOperator(
        task_id="run_pipeline",
        bash_command="cd /app && PYTHONPATH=/app/src python -m trading_pipeline.pipeline",
    )