from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "latest_run"

TEMPLATE = """
<!doctype html>
<title>Bitcoin Strategy Monitoring</title>
<h1>Bitcoin Hybrid Strategy Monitoring Demo</h1>
<p>This lightweight Flask app reads the latest local pipeline artifacts.</p>
<h2>Summary</h2>
<pre>{{ summary }}</pre>
<h2>Recent equity points</h2>
{{ equity_table|safe }}
<h2>Recent buys</h2>
{{ buys_table|safe }}
<h2>Recent sells</h2>
{{ sells_table|safe }}
"""


def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_table(name: str):
    path = REPORT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).tail(10)


@app.get("/")
def dashboard():
    summary = read_json(REPORT_DIR / "summary.json")
    return render_template_string(
        TEMPLATE,
        summary=json.dumps(summary, indent=2),
        equity_table=read_table("equity_curve.csv").to_html(index=False),
        buys_table=read_table("buys.csv").to_html(index=False),
        sells_table=read_table("sells.csv").to_html(index=False),
    )


@app.get("/metrics")
def metrics():
    return jsonify(read_json(REPORT_DIR / "summary.json"))
