#!/usr/bin/env bash
set -euo pipefail

TMP_DIR="${1:-/private/tmp/narya_smoke_scene_packs}"

python3 ml/scripts/render_synthetic_scene_packs.py --out "${TMP_DIR}" --count 4
python3 ml/scripts/heuristic_teacher_labels.py \
  --scene-packs "${TMP_DIR}" \
  --out /private/tmp/narya_teacher_labels.smoke.jsonl
python3 ml/scripts/build_sft_jsonl.py \
  --teacher-labels /private/tmp/narya_teacher_labels.smoke.jsonl \
  --out /private/tmp/narya_sft_train.smoke.jsonl

python3 - <<'PY'
import json
from pathlib import Path

path = Path("/private/tmp/narya_sft_train.smoke.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert len(rows) == 4
assert rows[0]["messages"][0]["role"] == "system"
assert rows[0]["messages"][-1]["role"] == "assistant"
print(f"smoke ok: {len(rows)} SFT rows at {path}")
PY
