"""FastAPI service for exposing the latest strategy outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi

from trading_pipeline.pipeline import run_pipeline

OUTPUT_DIR = Path("outputs")

app = FastAPI(
    title="Bitcoin Trading Agent API",
    version="0.1.0",
    description=(
        "API for running the local Bitcoin trading pipeline and reading the latest "
        "summary/equity outputs."
    ),
)


def _clean_openapi_schema() -> dict[str, Any]:
    """Generate OpenAPI docs without FastAPI's default validation-error schema noise.

    FastAPI automatically adds HTTPValidationError and ValidationError schemas when
    an endpoint has query parameters. They are not application errors, but they can
    look alarming in Swagger UI. This custom schema removes those default 422 docs.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    components = schema.get("components", {}).get("schemas", {})
    components.pop("HTTPValidationError", None)
    components.pop("ValidationError", None)

    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.get("responses", {}).pop("422", None)

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _clean_openapi_schema  # type: ignore[method-assign]


@app.get("/", tags=["status"])
def root() -> dict[str, str]:
    return {
        "message": "Bitcoin Trading Agent API",
        "docs": "/docs",
        "health": "/health",
        "run_pipeline": "POST /run",
        "summary": "/summary",
        "equity": "/equity",
    }


@app.get("/health", tags=["status"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", tags=["pipeline"])
def run_backtest(use_live_data: bool = True) -> dict[str, Any]:
    """Run a fresh pipeline job.

    Set ``use_live_data=true`` only when the data-fetching code and internet access
    are available. The default demo mode uses local/sample data so Docker can run
    fully offline.
    """
    return run_pipeline(output_dir=str(OUTPUT_DIR), use_live_data=use_live_data)


@app.get("/summary", tags=["outputs"])
def summary() -> dict[str, Any]:
    path = OUTPUT_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No backtest summary found. Run the pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/equity", tags=["outputs"])
def equity(limit: int = 50) -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "equity_curve.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No equity curve found. Run the pipeline first.")
    limit = max(1, min(limit, 500))
    return pd.read_csv(path).tail(limit).to_dict(orient="records")
