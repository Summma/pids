#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-root@216.243.220.226}"
DEST="${2:-/root/ns-hackathon}"

rsync -az --delete \
  --exclude '.git/' \
  --exclude 'pids-frontend/node_modules/' \
  --exclude 'ml/runs/' \
  --exclude 'ml/checkpoints/' \
  --exclude 'ml/outputs/' \
  --exclude '__pycache__/' \
  ./ "${TARGET}:${DEST}/"

echo "Synced to ${TARGET}:${DEST}"
