from __future__ import annotations

import pendulum
from airflow.decorators import dag, task


@dag(
    dag_id="bitcoin_weekly_gmail_summary",
    start_date=pendulum.datetime(2026, 5, 4, 9, 0, tz="America/Toronto"),
    schedule="0 9 * * 1",
    catchup=False,
    tags=["bitcoin", "gmail", "summary"],
)
def weekly_gmail_summary():
    @task
    def send_summary_task():
        from trading_pipeline.notifications import send_weekly_summary

        return {"sent": send_weekly_summary("reports/latest_run")}

    send_summary_task()


weekly_gmail_summary()
