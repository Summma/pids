# H100 Runbook

Use this once the SSH key issue is resolved.

## 1. Connect

```bash
ssh root@216.243.220.226
```

If using the sponsor attachment:

```bash
chmod 600 /path/to/narya_key
ssh -i /path/to/narya_key root@216.243.220.226
```

## 2. Sync Repo From This Laptop

From `/Users/tomalmog/projects/ns-hackathon`:

```bash
bash ml/scripts/h100_sync.sh root@216.243.220.226
```

## 3. Bootstrap H100

On the H100:

```bash
cd /root/ns-hackathon
bash ml/scripts/h100_bootstrap.sh
source .venv-h100/bin/activate
```

Check GPU:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## 4. Generate Training Data

If Gemini labels have already been generated locally, sync `ml/data/`.

Otherwise, on the H100:

```bash
export GEMINI_API_KEY=...
bash ml/scripts/run_teacher_pipeline.sh ml/data/scene_packs 64
```

For a no-API smoke test only:

```bash
bash ml/scripts/run_smoke_pipeline.sh /tmp/narya_smoke_scene_packs
```

## 5. Train

```bash
python3 ml/scripts/train_qwen_lora.py \
  --train-jsonl ml/data/sft_train.jsonl \
  --output-dir ml/runs/qwen-tactical-lora \
  --model Qwen/Qwen2.5-7B-Instruct \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-seq-length 4096
```

Expected output:

```text
ml/runs/qwen-tactical-lora/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
  narya_training_manifest.json
```

## 6. Inference

```bash
python3 ml/scripts/infer_scene_model.py \
  --adapter ml/runs/qwen-tactical-lora \
  --scene-pack ml/data/scene_packs/scene_0000
```

## 7. Demo Claim

Use this wording:

> We use Gemini as a teacher analyst over rendered LiDAR, thermal, RF, and
> camera evidence, then distill that reasoning into an H100-trained local
> tactical scene model for edge deployment. The local model emits conservative
> scene descriptions, threat levels, and Palantir-ready entity rationale.

## Current Access Blocker

Local SSH attempts to `root@216.243.220.226` failed with:

```text
Permission denied (publickey,password)
```

Resolution needed:

- Download/decrypt the email key attachment, or
- Ask sponsor/team to add this laptop's public key to root's
  `~/.ssh/authorized_keys`.
