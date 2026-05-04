"""Simple Flask monitoring dashboard for the local demo."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string

app = Flask(__name__)
OUTPUT_DIR = Path("outputs")


@app.route("/")
def dashboard():
    summary_path = OUTPUT_DIR / "summary.json"
    equity_path = OUTPUT_DIR / "equity_curve.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    equity_tail = []
    if equity_path.exists():
        equity_tail = pd.read_csv(equity_path).tail(20).to_dict(orient="records")
    return render_template_string(
        """
        <!doctype html>
        <html>
        <head>
          <title>Bitcoin Strategy Monitoring</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 2rem; background: #111827; color: #f9fafb; }
            .card { background: #1f2937; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border-bottom: 1px solid #374151; padding: 0.4rem; text-align: left; }
          </style>
        </head>
        <body>
          <h1>Bitcoin Strategy Monitoring</h1>
          <div class="card">
            <h2>Summary</h2>
            <pre>{{ summary }}</pre>
          </div>
          <div class="card">
            <h2>Latest Equity Records</h2>
            <table>
              <tr><th>Time</th><th>Close</th><th>Cash</th><th>Total BTC</th><th>Total Equity</th></tr>
              {% for row in equity_tail %}
              <tr>
                <td>{{ row.get('time') }}</td>
                <td>{{ '%.2f'|format(row.get('close', 0)) }}</td>
                <td>{{ '%.2f'|format(row.get('remaining_cash', 0)) }}</td>
                <td>{{ '%.6f'|format(row.get('total_btc', 0)) }}</td>
                <td>{{ '%.2f'|format(row.get('total_equity', 0)) }}</td>
              </tr>
              {% endfor %}
            </table>
          </div>
        </body>
        </html>
        """,
        summary=json.dumps(summary, indent=2),
        equity_tail=equity_tail,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
