#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-talmog@watgpu.cs.uwaterloo.ca}"
DEST="${2:-/u401/talmog/ns-hackathon}"

rsync -az --delete \
  --exclude '.git/' \
  --exclude 'pids-frontend/node_modules/' \
  --exclude 'backend/static/' \
  --exclude 'ml/data/' \
  --exclude 'ml/runs/' \
  --exclude 'ml/checkpoints/' \
  --exclude 'ml/outputs/' \
  --exclude '__pycache__/' \
  ./ "${TARGET}:${DEST}/"

echo "Synced to ${TARGET}:${DEST}"
