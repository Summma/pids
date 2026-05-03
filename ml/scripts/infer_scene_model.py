#!/usr/bin/env python3
"""Run local inference with the trained tactical scene LoRA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are Narya Tactical Scene Model, an edge-deployable sensor-fusion assistant.
Given structured evidence from LiDAR, thermal, RF, camera summaries, and tracked entities,
produce conservative tactical scene descriptions and threat assessments as JSON."""


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = load_manifest(args)
    base_model = args.model or manifest.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompt = build_prompt(Path(args.scene_pack), args.question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=0.9,
        )
    generated = output_ids[0][inputs.input_ids.shape[-1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--scene-pack", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--question", default="Describe the scene and identify any possible threats.")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.adapter) / "narya_training_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_prompt(scene_pack: Path, question: str) -> str:
    scene = json.loads((scene_pack / "scene.json").read_text())
    image_manifest = sorted(str(path) for path in (scene_pack / "images").glob("*.png"))
    payload = {
        "question": question,
        "image_manifest": image_manifest,
        "structured_sensor_evidence": scene,
    }
    return (
        "Analyze this multimodal sensor scene. Use the image_manifest as "
        "descriptions of accompanying rendered views, and use the structured "
        "evidence as authoritative data.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


if __name__ == "__main__":
    main()
