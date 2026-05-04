#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src"
python -m trading_pipeline.pipeline
pytest -q
