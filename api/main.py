from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from trading_pipeline.config_manager import load_config, refresh_config_from_google_sheet
from trading_pipeline.pipeline import run_full_pipeline
from trading_pipeline.strategy_core import heuristic_decision, prepare_feature_table

app = FastAPI(title="Bitcoin Hybrid LLM + Quant MLOps API", version="1.0.0")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Candle(BaseModel):
    time: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float


class DecisionRequest(BaseModel):
    candles: List[Candle]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/config/refresh")
def refresh_config() -> Dict[str, Any]:
    return refresh_config_from_google_sheet()


@app.get("/summary")
def latest_summary() -> Dict[str, Any]:
    path = PROJECT_ROOT / "reports" / "latest_run" / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No backtest summary found. Run the pipeline first.")
    return pd.read_json(path, typ="series").to_dict()


@app.post("/backtest/run")
def run_backtest_now() -> Dict[str, Any]:
    return run_full_pipeline(refresh_config=True)


@app.post("/decision")
def decision(request: DecisionRequest) -> Dict[str, Any]:
    if not request.candles:
        raise HTTPException(status_code=400, detail="At least one candle is required.")
    df = pd.DataFrame([c.model_dump() for c in request.candles])
    if "time" not in df or df["time"].isna().all():
        df["time"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="D")
    features = prepare_feature_table(df)
    row = features.iloc[-1]
    return heuristic_decision(row)
