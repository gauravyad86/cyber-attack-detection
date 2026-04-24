#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore::DeprecationWarning"}
export CADS_LOG_LEVEL=${CADS_LOG_LEVEL:-"WARNING"}

mkdir -p artifacts/reports data/raw data/processed

# Auto-heal runtime prerequisites on first boot or empty deploys.
if [[ ! -f "data/processed/test.csv" ]]; then
  echo "[boot] missing processed data -> generating synthetic + prepare-data"
  uv run cads generate-sample-data --rows "${CADS_BOOT_ROWS:-5000}" --seed "${CADS_BOOT_SEED:-42}"
  uv run cads prepare-data
fi

if [[ ! -f "artifacts/models/random_forest.joblib" || ! -f "artifacts/models/label_encoder.joblib" ]]; then
  echo "[boot] missing model artifacts -> training models"
  uv run cads train-models
fi

# Optional clean start for alerts DB.
if [[ "${CADS_RESET_ALERT_DB:-0}" == "1" ]]; then
  rm -f artifacts/reports/alerts.db
fi

# Start continuous alert simulation in the background.
echo "[boot] starting live simulator in background"
uv run cads simulate-live-alerts \
  --mode "${CADS_LIVE_MODE:-ensemble}" \
  --interval "${CADS_LIVE_INTERVAL:-5}" \
  --batch-size "${CADS_LIVE_BATCH_SIZE:-3}" \
  --cycles "${CADS_LIVE_CYCLES:-100000000}" \
  >/tmp/cads-live.log 2>&1 &

# Start Streamlit as foreground process (Render keeps service alive).
echo "[boot] starting streamlit on port ${PORT:-8501}"
exec uv run streamlit run src/cads/dashboard/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --browser.gatherUsageStats false \
  --logger.level "${STREAMLIT_LOG_LEVEL:-warning}"

