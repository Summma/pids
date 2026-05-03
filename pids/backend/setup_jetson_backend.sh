#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PIDS_BOOTSTRAP_PYTHON:-python3}"
VENV_DIR="${PIDS_VENV_DIR:-$ROOT_DIR/.venv-jetson}"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/pids/backend/requirements.txt"

echo "Jetson backend environment ready: $VENV_DIR"
echo "Edit pids/backend/jetson.env, then run: pids/backend/run_jetson_backend.sh"
