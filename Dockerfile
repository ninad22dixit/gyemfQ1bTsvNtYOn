FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY api ./api
COPY monitoring ./monitoring
COPY dags ./dags
COPY tests ./tests
COPY scripts ./scripts
COPY config ./config
COPY dvc.yaml params.yaml README.md ./

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

EXPOSE 8000 5000 5001 8080
CMD ["bash", "-lc", "python scripts/run_pipeline.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
