#!/usr/bin/env bash
set -euo pipefail

SCENE_PACKS="${1:-ml/data/scene_packs}"
COUNT="${2:-32}"

python3 ml/scripts/render_synthetic_scene_packs.py --out "${SCENE_PACKS}" --count "${COUNT}"
python3 ml/scripts/generate_teacher_labels.py \
  --scene-packs "${SCENE_PACKS}" \
  --out ml/data/teacher_labels.jsonl
python3 ml/scripts/build_sft_jsonl.py \
  --teacher-labels ml/data/teacher_labels.jsonl \
  --out ml/data/sft_train.jsonl

echo "Ready for H100 training: ml/data/sft_train.jsonl"
