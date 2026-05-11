# Patched Bitcoin Trading Agent

This patched version repairs corrupted one-line Python/config files, isolates the Airflow Flask dependency conflict, and makes the project locally runnable.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venvScriptsactivate
pip install -r requirements.txt
```

## Docker

```bash
docker compose up --build pipeline
docker compose up --build api monitoring mlflow
```

FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs)  
Monitoring: [http://localhost:5000](http://localhost:5000)  
MLflow: [http://localhost:5000](http://localhost:5001)

## Docker run instructions

Start every service from the repo root:

```bash
docker compose down -v --remove-orphans
docker compose up --build
```

Open:

-   FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
-   FastAPI root: [http://localhost:8000/](http://localhost:8000/)
-   MLflow: [http://localhost:5001](http://localhost:5000)
-   Monitoring dashboard: [http://localhost:8050](http://localhost:8050)
-   Airflow: [http://localhost:8080](http://localhost:8080)

Run a fresh pipeline execution and log results to MLflow:

```bash
docker compose run --rm app python -m trading_pipeline.pipeline
```

Or trigger it through FastAPI:

```bash
curl -X POST "http://localhost:8000/run?use_live_data=false"

```
In Powershell:

```bash
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/run?use_live_data=false"

```

Use `use_live_data=true` only when you want Coinbase candles instead of the built-in synthetic demo data.

## Notifications

Telegram trade alerts are sent after each pipeline/backtest run for new buy or sell rows. Configure:

```bash
TELEGRAM_NOTIFICATIONS_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Already-sent trades are tracked in `reports/telegram_notified.json` by default so reruns do not resend old backtest trades. Override with `TELEGRAM_NOTIFICATIONS_STATE_PATH` if needed.

The weekly Gmail summary is scheduled by Airflow in `bitcoin_weekly_gmail_summary` every Monday at 9:00 AM America/Toronto time. Configure:

```bash
GMAIL_SENDER=your_gmail_address
GMAIL_APP_PASSWORD=your_gmail_app_password
GMAIL_RECIPIENTS=recipient@example.com
```

You can also send it manually:

```bash
PYTHONPATH=src python scripts/send_weekly_summary.py
```
