#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACKING_URI="${MLFLOW_TRACKING_URI:-$ROOT_DIR/mlruns}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_LOCATION:-$ROOT_DIR/mlartifacts}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"

cd "$ROOT_DIR"
mkdir -p "$ARTIFACT_ROOT"
exec uv run mlflow server \
  --host "$HOST" \
  --port "$PORT" \
  --backend-store-uri "$TRACKING_URI" \
  --artifacts-destination "$ARTIFACT_ROOT"
