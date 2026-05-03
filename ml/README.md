# Tactical Scene Model

This folder is the H100-ready training path for the demo.

The goal is not to train a foundation model from scratch. The goal is to make
the project maximally credible:

1. Build multimodal scene packs from the edge system.
2. Use Gemini as a teacher analyst to label each scene.
3. Fine-tune a compact local tactical scene model on the H100/H200.
4. Run local inference that turns sensor evidence into scene descriptions,
   threat rankings, and Palantir-ready entity rationale.

## Demo Story

> We use Gemini to synthesize expert analyst labels over LiDAR, thermal, RF,
> and camera evidence, then distill that behavior into a compact locally
> trained tactical scene model for edge operation.

That gives us both:

- High-quality scene reasoning from Gemini.
- A concrete trained model artifact from the H100/H200 that is small enough to
  run for demo inference.

## Scene Pack Format

Each scene pack is a directory:

```text
scene_pack/
  scene.json
  images/
    topdown_lidar.png
    thermal_overlay.png
    rgb_frame.png
    annotated_scene.png
```

`scene.json` should contain the structured sensor output:

```json
{
  "scene_id": "demo_001",
  "timestamp": "2026-05-03T06:30:00Z",
  "sensor_node": {
    "id": "narya-edge-01",
    "lat": 37.7955,
    "lon": -122.3937
  },
  "entities": [
    {
      "track_id": "trk_001",
      "classification": "unknown",
      "position_m": {"x": 8.2, "y": 0.0, "z": 12.4},
      "size_m": {"height": 1.78, "width": 0.55, "depth": 0.42},
      "thermal": {"mean_c": 31.2, "max_c": 35.8, "hot_pixel_fraction": 0.41},
      "lidar": {"points": 442, "upright_score": 0.91},
      "rf": {"drone_candidate": false, "emitters": []},
      "confidence": 0.74
    }
  ]
}
```

## Quickstart

Install local generation dependencies:

```bash
python3 -m pip install -r ml/requirements-local.txt
```

Create synthetic scene packs for the teacher pipeline:

```bash
python3 ml/scripts/render_synthetic_scene_packs.py --out ml/data/scene_packs --count 24
```

Generate Gemini teacher labels:

```bash
export GEMINI_API_KEY=...
python3 ml/scripts/generate_teacher_labels.py \
  --scene-packs ml/data/scene_packs \
  --out ml/data/teacher_labels.jsonl
```

Build supervised fine-tuning data:

```bash
python3 ml/scripts/build_sft_jsonl.py \
  --teacher-labels ml/data/teacher_labels.jsonl \
  --out ml/data/sft_train.jsonl
```

On the H100:

```bash
bash ml/scripts/h100_bootstrap.sh
source .venv-h100/bin/activate
python3 ml/scripts/train_qwen_lora.py \
  --train-jsonl ml/data/sft_train.jsonl \
  --output-dir ml/runs/qwen-tactical-lora
```

Run inference:

```bash
python3 ml/scripts/infer_scene_model.py \
  --adapter ml/runs/qwen-tactical-lora \
  --scene-pack ml/data/scene_packs/scene_0000
```

Export a Palantir-ready ontology/action payload:

```bash
python3 ml/scripts/export_palantir_payload.py \
  --scene-pack ml/data/scene_packs/scene_0000 \
  --analysis-json ml/runs/qwen-tactical-lora/sample_inference.json \
  --out ml/outputs/palantir_payload.json
```

For a local no-API smoke test:

```bash
bash ml/scripts/run_smoke_pipeline.sh
```

For the exact H100 sequence, see [H100_RUNBOOK.md](H100_RUNBOOK.md). For the
current hackathon demo path, see [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md).

## H100 Access Note

SSH to the sponsor machine currently rejects the local keys available on this
Mac. Once the encrypted key attachment is downloaded/decrypted or the sponsor
adds the right public key, use:

```bash
ssh root@216.243.220.226
```

Then sync:

```bash
bash ml/scripts/h100_sync.sh root@216.243.220.226
```

## Sources Used For Implementation Choices

- Google Gemini API supports multimodal `generate_content` with text and
  images through the `google-genai` Python SDK.
- Hugging Face TRL supports VLM SFT, and Qwen2.5-VL-7B-Instruct is available
  as a current multimodal baseline. The default here uses text-only Qwen2.5
  over structured scene evidence because it is faster and more reliable under
  hackathon time pressure. The intended default is
  `Qwen/Qwen2.5-1.5B-Instruct`; larger models are optional experiments, not the
  core demo path.
