#!/bin/bash
#SBATCH --job-name=narya_h200_1b5
#SBATCH --time=08:00:00
#SBATCH --mem=360GB
#SBATCH --cpus-per-task=16
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu808
#SBATCH --gres=gpu:2
#SBATCH -o logs/narya_h200_1b5_%j.out
#SBATCH -e logs/narya_h200_1b5_%j.err

set -euo pipefail

mkdir -p logs ml/data/h200_scene_packs ml/runs

echo "=========================================="
echo "Narya H200 Tactical Scene LoRA Training"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo ""
echo "GPU info"
nvidia-smi || true

echo ""
echo "Python setup"
VENV_DIR="${VENV_DIR:-.venv-watgpu}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r ml/requirements-h100.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda device count", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"gpu {idx}", torch.cuda.get_device_name(idx))
PY

TEACHER_COUNT="${TEACHER_COUNT:-48}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SCENE_PACK_DIR="${SCENE_PACK_DIR:-ml/data/h200_scene_packs}"
TEACHER_LABELS="${TEACHER_LABELS:-ml/data/h200_teacher_labels.jsonl}"
SFT_JSONL="${SFT_JSONL:-ml/data/h200_sft_train.jsonl}"
OUT_DIR="${OUT_DIR:-ml/runs/qwen1p5b-tactical-lora-${SLURM_JOB_ID:-manual}}"
EPOCHS="${EPOCHS:-1}"

echo ""
echo "Generating ${TEACHER_COUNT} scene packs"
python3 ml/scripts/render_synthetic_scene_packs.py \
  --out "${SCENE_PACK_DIR}" \
  --count "${TEACHER_COUNT}"

if [[ -f "$HOME/.gemini_key" ]]; then
  export GEMINI_API_KEY
  GEMINI_API_KEY="$(tr -d '\n\r ' < "$HOME/.gemini_key")"
  echo ""
  echo "Generating Gemini teacher labels with configured cluster key"
  python3 ml/scripts/generate_teacher_labels.py \
    --scene-packs "${SCENE_PACK_DIR}" \
    --out "${TEACHER_LABELS}" \
    --model "${GEMINI_MODEL:-gemini-2.5-flash}"
else
  echo ""
  echo "WARNING: ~/.gemini_key not found; using heuristic labels for smoke training"
  python3 ml/scripts/heuristic_teacher_labels.py \
    --scene-packs "${SCENE_PACK_DIR}" \
    --out "${TEACHER_LABELS}"
fi

echo ""
echo "Building SFT dataset"
python3 ml/scripts/build_sft_jsonl.py \
  --teacher-labels "${TEACHER_LABELS}" \
  --out "${SFT_JSONL}"

echo ""
echo "Training LoRA adapter across the allocated GPUs"
python3 ml/scripts/train_qwen_lora.py \
  --train-jsonl "${SFT_JSONL}" \
  --output-dir "${OUT_DIR}" \
  --model "${BASE_MODEL}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-16}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-4096}" \
  --save-steps "${SAVE_STEPS:-10}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}"

echo ""
echo "Running sample inference"
python3 ml/scripts/infer_scene_model.py \
  --adapter "${OUT_DIR}" \
  --scene-pack "${SCENE_PACK_DIR}/scene_0000" \
  > "${OUT_DIR}/sample_inference.json"

echo ""
echo "=========================================="
echo "H200 training complete: ${OUT_DIR}"
echo "End: $(date)"
echo "=========================================="
