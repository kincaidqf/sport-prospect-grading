#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACKING_URI="${MLFLOW_TRACKING_URI:-$ROOT_DIR/mlruns}"

cd "$ROOT_DIR"
exec uv run mlflow ui --backend-store-uri "$TRACKING_URI"
