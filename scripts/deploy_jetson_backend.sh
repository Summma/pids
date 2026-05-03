#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-cask-02@10.1.63.30}"
REMOTE_DIR="${2:-~/narya/pids}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pull origin/main first so this deploy can't accidentally stomp commits
# that landed on origin while the local branch lagged. Skip with
# DEPLOY_SKIP_PULL=1 if you really intend to deploy your local-only state.
if [[ "${DEPLOY_SKIP_PULL:-0}" != "1" ]]; then
  if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo ">> git pull --ff-only origin main (set DEPLOY_SKIP_PULL=1 to bypass)"
    if ! git -C "$ROOT_DIR" pull --ff-only origin main; then
      echo "ERROR: pull failed (likely a divergent local main). Rebase or set DEPLOY_SKIP_PULL=1." >&2
      exit 1
    fi
  fi
fi

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
