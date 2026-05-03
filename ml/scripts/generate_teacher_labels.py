#!/usr/bin/env python3
"""Generate Gemini teacher labels for tactical scene packs."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


IMAGE_ORDER = [
    "annotated_scene.png",
    "topdown_lidar.png",
    "thermal_overlay.png",
    "rgb_frame.png",
]


def main() -> None:
    args = parse_args()
    prompt = Path(args.prompt).read_text()
    packs = sorted(Path(args.scene_packs).glob("scene_*"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = loaded_scene_ids(out_path) if out_path.exists() and args.resume else set()
    client, types = make_client(args)

    with out_path.open("a" if args.resume else "w") as out:
        for pack in tqdm(packs, desc="teacher labels"):
            scene = json.loads((pack / "scene.json").read_text())
            scene_id = scene.get("scene_id", pack.name)
            if scene_id in done:
                continue
            record = label_one_pack(client, types, args.model, prompt, pack, scene)
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            out.flush()
            time.sleep(args.sleep_s)

    print(f"teacher labels written to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-packs", default="ml/data/scene_packs")
    parser.add_argument("--out", default="ml/data/teacher_labels.jsonl")
    parser.add_argument("--prompt", default="ml/prompts/tactical_scene_teacher.md")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vertexai", action="store_true")
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    return parser.parse_args()


def make_client(args: argparse.Namespace):
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise SystemExit("Install google-genai: python3 -m pip install google-genai") from exc

    if args.vertexai or os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "true":
        client = genai.Client(vertexai=True, project=args.project, location=args.location)
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before calling Gemini.")
        client = genai.Client(api_key=api_key)
    return client, types


def label_one_pack(
    client: Any,
    types: Any,
    model: str,
    prompt: str,
    pack: Path,
    scene: dict[str, Any],
) -> dict[str, Any]:
    scene_text = json.dumps(scene, indent=2)
    parts: list[Any] = [
        types.Part.from_text(
            text=(
                f"{prompt}\n\n"
                f"User scene question: {scene.get('question')}\n\n"
                f"Structured sensor evidence:\n```json\n{scene_text}\n```"
            )
        )
    ]

    for image_name in IMAGE_ORDER:
        image_path = pack / "images" / image_name
        if image_path.exists():
            parts.append(
                types.Part.from_bytes(
                    data=image_path.read_bytes(),
                    mime_type="image/png",
                )
            )

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.15,
            max_output_tokens=1800,
            response_mime_type="application/json",
        ),
    )
    raw_text = response.text or ""
    parse_error = None
    try:
        parsed = parse_jsonish(raw_text)
    except Exception as exc:
        parse_error = str(exc)
        parsed = fallback_teacher(scene, parse_error)

    record = {
        "scene_id": scene.get("scene_id", pack.name),
        "scene_pack": str(pack),
        "question": scene.get("question"),
        "scene": scene,
        "teacher_model": model,
        "teacher_raw": raw_text,
        "teacher": parsed,
    }
    if parse_error:
        record["teacher_parse_error"] = parse_error
    return record


def loaded_scene_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["scene_id"])
            except Exception:
                continue
    return ids


def parse_jsonish(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        return json.loads(match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Gemini did not return JSON: {text[:300]}")


def fallback_teacher(scene: dict[str, Any], parse_error: str) -> dict[str, Any]:
    entities = [fallback_entity(ent) for ent in scene.get("entities", [])]
    risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    global_risk = "none"
    for ent in entities:
        if risk_order[ent["threat_level"]] > risk_order[global_risk]:
            global_risk = ent["threat_level"]

    return {
        "scene_description": (
            "Gemini returned malformed JSON for this scene, so Narya used a "
            "conservative structured fallback based on the sensor evidence."
        ),
        "tactical_assessment": (
            f"Highest assessed risk is {global_risk}. Original teacher parse error: "
            f"{parse_error[:240]}"
        ),
        "entities": entities,
        "global_risk_level": global_risk,
        "answer": "Conservative fallback assessment generated from LiDAR, thermal, and RF fields.",
    }


def fallback_entity(ent: dict[str, Any]) -> dict[str, Any]:
    rf_drone = ent.get("rf", {}).get("drone_candidate", False)
    height = float(ent.get("size_m", {}).get("height") or 0)
    mean_c = float(ent.get("thermal", {}).get("mean_c") or 0)
    upright = float(ent.get("lidar", {}).get("upright_score") or 0)

    if rf_drone:
        classification = "drone"
        threat_level = "high"
        action = "intercept"
        rationale = "RF drone-control evidence is present."
    elif height > 1.4 and mean_c > 26 and upright > 0.7:
        classification = "human"
        threat_level = "medium"
        action = "investigate"
        rationale = "Human-scale upright geometry is supported by a warm thermal signature."
    else:
        classification = ent.get("classification") or "unknown"
        threat_level = "low"
        action = "monitor"
        rationale = "Evidence is insufficient for a stronger tactical classification."

    return {
        "track_id": ent.get("track_id", "unknown"),
        "best_classification": classification,
        "threat_level": threat_level,
        "confidence": ent.get("confidence", 0.55),
        "rationale": rationale,
        "recommended_action": action,
    }


if __name__ == "__main__":
    main()
