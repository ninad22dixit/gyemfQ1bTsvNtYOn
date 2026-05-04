# Running the Docker portals

Start from a clean state:

```bash
docker compose down -v --remove-orphans
docker compose up --build
```

Open these URLs:

- FastAPI: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Flask monitoring: http://localhost:8050
- Airflow: http://localhost:8080

Airflow login:

- Username: `admin`
- Password: `admin`

Trigger a fresh pipeline run:

```bash
docker compose run --rm app python -m trading_pipeline.pipeline
```

Check service status:

```bash
docker compose ps
```

View logs if a portal does not open:

```bash
docker compose logs -f mlflow
docker compose logs -f airflow-webserver
docker compose logs -f airflow-scheduler
```
