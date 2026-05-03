#!/usr/bin/env python3
"""Build chat-style SFT JSONL from Gemini teacher labels."""
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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with Path(args.teacher_labels).open() as src, out_path.open("w") as out:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            example = make_example(record)
            out.write(json.dumps(example, separators=(",", ":")) + "\n")
            n += 1

    print(f"wrote {n} SFT examples to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-labels", default="ml/data/teacher_labels.jsonl")
    parser.add_argument("--out", default="ml/data/sft_train.jsonl")
    return parser.parse_args()


def make_example(record: dict[str, Any]) -> dict[str, Any]:
    scene = record["scene"]
    teacher = record["teacher"]
    user = {
        "question": record.get("question") or scene.get("question"),
        "image_manifest": sorted(str(p) for p in Path(record["scene_pack"]).glob("images/*.png")),
        "structured_sensor_evidence": scene,
    }
    return {
        "scene_id": record["scene_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze this multimodal sensor scene. Use the image_manifest "
                    "as descriptions of accompanying rendered views, and use the "
                    "structured evidence as the authoritative data.\n\n"
                    f"{json.dumps(user, indent=2)}"
                ),
            },
            {"role": "assistant", "content": json.dumps(teacher, indent=2)},
        ],
    }


if __name__ == "__main__":
    main()
