#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-cask-02@10.1.63.30}"
REMOTE_DIR="${2:-~/narya/pids}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv*" \
  --exclude "node_modules" \
  --exclude "pids/backend/jetson.env" \
  "$ROOT_DIR/" "$TARGET:$REMOTE_DIR/"

ssh "$TARGET" "cd $REMOTE_DIR && if [[ -f pids/backend/jetson.env ]]; then echo 'Existing pids/backend/jetson.env found; skipping environment bootstrap'; else pids/backend/setup_jetson_backend.sh; fi"

cat <<EOF
Deployed Jetson backend to $TARGET:$REMOTE_DIR

Next on the Jetson:
  cd $REMOTE_DIR
  cp -n pids/backend/jetson.env.example pids/backend/jetson.env
  nano pids/backend/jetson.env
  pids/backend/run_jetson_backend.sh
EOF
