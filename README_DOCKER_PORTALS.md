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
MLflow: [http://localhost:5001](http://localhost:5001)

## Docker run instructions

Start every service from the repo root:

```bash
docker compose down -v --remove-orphans
docker compose up --build
```

Open:

-   FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
-   FastAPI root: [http://localhost:8000/](http://localhost:8000/)
-   MLflow: [http://localhost:5000](http://localhost:5000)
-   Flask Monitoring dashboard: [http://localhost:8050](http://localhost:8050)
-   Airflow: [http://localhost:8080](http://localhost:8080)

Run a fresh pipeline execution and log results to MLflow:

```bash
docker compose run --rm app python -m trading_pipeline.pipeline
```

Or trigger it through FastAPI:

```bash
curl -X POST "http://localhost:8000/run?use_live_data=false"
```

Use `use_live_data=true` only when you want Coinbase candles instead of the built-in synthetic demo data.