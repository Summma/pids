#!/usr/bin/env bash
set -euo pipefail

OPENPCDET_DIR="${OPENPCDET_DIR:-$HOME/OpenPCDet}"
POINTPILLAR_ID="${POINTPILLAR_ID:-1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm}"
CKPT_PATH="$OPENPCDET_DIR/checkpoints/pointpillar_7728.pth"

python3 -m pip install --user -r "$(dirname "$0")/requirements.txt"

if [ ! -d "$OPENPCDET_DIR/.git" ]; then
  git clone https://github.com/open-mmlab/OpenPCDet.git "$OPENPCDET_DIR"
fi

cd "$OPENPCDET_DIR"
python3 -m pip install --user -r requirements.txt
python3 setup.py develop --user

mkdir -p checkpoints
if [ ! -f "$CKPT_PATH" ]; then
  python3 -m gdown "$POINTPILLAR_ID" -O "$CKPT_PATH"
fi

python3 - <<'PY'
import torch
import pcdet
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("pcdet", getattr(pcdet, "__file__", "ok"))
PY

echo "OpenPCDet ready at $OPENPCDET_DIR"
echo "Checkpoint: $CKPT_PATH"
