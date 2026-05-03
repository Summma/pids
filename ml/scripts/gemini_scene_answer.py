#!/usr/bin/env python3
"""Ask Gemini directly about one scene pack for the live demo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


IMAGE_ORDER = [
    "annotated_scene.png",
    "topdown_lidar.png",
    "thermal_overlay.png",
    "rgb_frame.png",
]


def main() -> None:
    args = parse_args()
    prompt = Path(args.prompt).read_text()
    scene_pack = Path(args.scene_pack)
    scene = json.loads((scene_pack / "scene.json").read_text())

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise SystemExit("Install google-genai: python3 -m pip install google-genai") from exc

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    client = genai.Client(api_key=api_key)

    parts = [
        types.Part.from_text(
            text=(
                f"{prompt}\n\n"
                f"User scene question: {args.question}\n\n"
                f"Structured sensor evidence:\n```json\n"
                f"{json.dumps(scene, indent=2)}\n```"
            )
        )
    ]
    for image_name in IMAGE_ORDER:
        image_path = scene_pack / "images" / image_name
        if image_path.exists():
            parts.append(types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"))

    response = client.models.generate_content(
        model=args.model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            response_mime_type="application/json" if args.json else None,
        ),
    )
    print(response.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-pack", required=True)
    parser.add_argument("--question", default="Describe the scene and identify any possible threats.")
    parser.add_argument("--prompt", default="ml/prompts/tactical_scene_teacher.md")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--json", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
