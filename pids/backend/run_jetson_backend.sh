#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${PIDS_JETSON_ENV:-$ROOT_DIR/pids/backend/jetson.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PIDS_PYTHON:-python3}"
if [[ -x "$ROOT_DIR/.venv-jetson/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv-jetson/bin/python"
fi

ARGS=(
  "$ROOT_DIR/pids/backend/server.py"
  --host "${PIDS_BACKEND_HOST:-0.0.0.0}"
  --port "${PIDS_BACKEND_PORT:-9090}"
  --lidar-host "${PIDS_LIDAR_HOST:-169.254.62.165}"
  --lidar-fps "${PIDS_LIDAR_FPS:-5}"
  --lidar-max-points "${PIDS_LIDAR_MAX_POINTS:-60000}"
  --camera-device "${PIDS_CAMERA_DEVICE:-/dev/video0}"
  --camera-fps "${PIDS_CAMERA_FPS:-15}"
  --camera-width "${PIDS_CAMERA_WIDTH:-640}"
  --camera-height "${PIDS_CAMERA_HEIGHT:-480}"
  --camera-jpeg-quality "${PIDS_CAMERA_JPEG_QUALITY:-75}"
  --detection-mode "${PIDS_DETECTION_MODE:-indoor_human}"
  --detection-fps "${PIDS_DETECTION_FPS:-2}"
  --gemini-model "${PIDS_GEMINI_MODEL:-${GEMINI_MODEL:-gemini-2.5-flash}}"
  --gemini-api-mode "${PIDS_GEMINI_API_MODE:-${GEMINI_API_MODE:-auto}}"
  --gemini-location "${PIDS_GEMINI_LOCATION:-${GOOGLE_CLOUD_LOCATION:-global}}"
)

if [[ -n "${PIDS_GEMINI_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}" ]]; then
  ARGS+=(--gemini-project "${PIDS_GEMINI_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}")
fi

if [[ -n "${PIDS_THERMAL_DEVICE:-}" && "${PIDS_THERMAL_DEVICE}" != "-1" ]]; then
  ARGS+=(--device "${PIDS_THERMAL_DEVICE}" --fps "${PIDS_THERMAL_FPS:-10}")
else
  ARGS+=(--disable-thermal)
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
