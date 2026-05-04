from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="bitcoin_trading_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@hourly",
    catchup=False,
) as dag:
    run_pipeline = BashOperator(
        task_id="run_pipeline",
        bash_command="cd /app && PYTHONPATH=/app/src python -m trading_pipeline.pipeline",
    )
