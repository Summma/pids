#!/bin/bash
#SBATCH --job-name=narya_scene_lora
#SBATCH --time=06:00:00
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o logs/narya_scene_lora_%j.out
#SBATCH -e logs/narya_scene_lora_%j.err

set -euo pipefail

mkdir -p logs ml/data/scene_packs ml/runs

echo "=========================================="
echo "Narya Tactical Scene LoRA Training"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo ""
echo "GPU info"
nvidia-smi || true

echo ""
echo "Python setup"
python3 -m venv .venv-watgpu
source .venv-watgpu/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r ml/requirements-h100.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

TEACHER_COUNT="${TEACHER_COUNT:-32}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT_DIR="${OUT_DIR:-ml/runs/qwen-tactical-lora-${SLURM_JOB_ID:-manual}}"
EPOCHS="${EPOCHS:-2}"

echo ""
echo "Generating ${TEACHER_COUNT} scene packs"
python3 ml/scripts/render_synthetic_scene_packs.py \
  --out ml/data/scene_packs \
  --count "${TEACHER_COUNT}"

if [[ -f "$HOME/.gemini_key" ]]; then
  export GEMINI_API_KEY
  GEMINI_API_KEY="$(tr -d '\n\r ' < "$HOME/.gemini_key")"
  echo ""
  echo "Generating Gemini teacher labels with configured cluster key"
  python3 ml/scripts/generate_teacher_labels.py \
    --scene-packs ml/data/scene_packs \
    --out ml/data/teacher_labels.jsonl \
    --model "${GEMINI_MODEL:-gemini-2.5-flash}"
else
  echo ""
  echo "WARNING: ~/.gemini_key not found; using heuristic labels for smoke training"
  python3 ml/scripts/heuristic_teacher_labels.py \
    --scene-packs ml/data/scene_packs \
    --out ml/data/teacher_labels.jsonl
fi

echo ""
echo "Building SFT dataset"
python3 ml/scripts/build_sft_jsonl.py \
  --teacher-labels ml/data/teacher_labels.jsonl \
  --out ml/data/sft_train.jsonl

echo ""
echo "Training LoRA adapter"
python3 ml/scripts/train_qwen_lora.py \
  --train-jsonl ml/data/sft_train.jsonl \
  --output-dir "${OUT_DIR}" \
  --model "${BASE_MODEL}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-4096}" \
  --save-steps "${SAVE_STEPS:-25}"

echo ""
echo "Running sample inference"
python3 ml/scripts/infer_scene_model.py \
  --adapter "${OUT_DIR}" \
  --scene-pack ml/data/scene_packs/scene_0000 \
  > "${OUT_DIR}/sample_inference.json"

echo ""
echo "=========================================="
echo "Training complete: ${OUT_DIR}"
echo "End: $(date)"
echo "=========================================="
