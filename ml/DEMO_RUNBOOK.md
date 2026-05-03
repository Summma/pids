# Narya Demo Runbook

This is the practical path for the hackathon demo.

## Current Training Lanes

Fallback adapter:

```bash
ml/runs/qwen-tactical-lora-1433707
```

This run completed on WATGPU with Gemini teacher labels and produced:

```bash
ml/runs/qwen-tactical-lora-1433707/sample_inference.json
ml/outputs/palantir_payload_1433707.json
```

Small trained adapter:

Use WATGPU/H200 as fast training hardware for the compact adapter. The default
base model is `Qwen/Qwen2.5-1.5B-Instruct`, because the trained component is a
scene-to-JSON tactical analyst head over structured sensor evidence, not a full
multimodal foundation model.

```bash
TEACHER_COUNT=48 EPOCHS=1 sbatch ml/scripts/watgpu_train_h200_scene_model.sbatch
```

Check it from the local machine:

```bash
ssh talmog@watgpu.cs.uwaterloo.ca 'squeue -u talmog -o "%.18i %.9P %.30j %.8T %.10M %.10l %.6D %R"'
```

Tail the H200 logs:

```bash
ssh talmog@watgpu.cs.uwaterloo.ca 'cd /u401/talmog/ns-hackathon && tail -120 logs/narya_h200_1b5_JOBID.out && tail -120 logs/narya_h200_1b5_JOBID.err'
```

## Judge-Facing Story

Narya is not just a webcam demo. The system fuses LiDAR, thermal, RF, and
camera-derived evidence into a tactical scene pack. Gemini acts as an analyst
teacher over that multimodal evidence. We then distill the analyst behavior
into a small local LoRA adapter trained on WATGPU/H200-class hardware, so the
demo has both frontier-model reasoning and a concrete edge-deployable trained
artifact.

The Palantir-facing layer is a plain JSON contract with ontology-shaped
objects, relationships, actions, and GeoJSON. That keeps integration credible
without needing to block on Foundry credentials during the hackathon.

## Demo Flow

1. Open the local Narya web UI and show live sensor health. If thermal/LiDAR are
   not attached, keep the UI honest and show offline instead of fake streams.
2. Capture or load a scene pack containing `scene.json` plus rendered sensor
   views under `images/`.
3. Ask Gemini for the full tactical scene description when API/network access is
   available.
4. Run the trained local adapter as the edge model fallback.
5. Export the result as Palantir-ready objects/actions/GeoJSON.

## Palantir Payload Export

From WATGPU, export the completed fallback adapter sample:

```bash
cd /u401/talmog/ns-hackathon
python3 ml/scripts/export_palantir_payload.py \
  --scene-pack ml/data/scene_packs/scene_0000 \
  --analysis-json ml/runs/qwen-tactical-lora-1433707/sample_inference.json \
  --out ml/outputs/palantir_payload_1433707.json
```

For a real capture, first package the live output:

```bash
python3 ml/scripts/make_scene_pack_from_snapshot.py \
  --entities-json path/to/entities.json \
  --topdown-lidar path/to/topdown_lidar.png \
  --thermal-overlay path/to/thermal_overlay.png \
  --rgb-frame path/to/rgb_frame.png \
  --annotated-scene path/to/annotated_scene.png \
  --out ml/data/live_scene_pack
```

Then run Gemini or the local adapter and export:

```bash
python3 ml/scripts/export_palantir_payload.py \
  --scene-pack ml/data/live_scene_pack \
  --analysis-json path/to/analysis.json \
  --out ml/outputs/live_palantir_payload.json
```

## What To Do While Training Runs

- Keep the local web app and backend available for hardware-connected live data.
- Preserve offline status honestly when a physical sensor is missing.
- Use `ml/outputs/palantir_payload_1433707.json` as the fallback Palantir demo artifact.
- If a small adapter run completes, repeat the export with its `sample_inference.json`.
