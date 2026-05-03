#!/usr/bin/env python3
"""Package already-rendered scene assets and entity JSON into a scene pack.

This is for real demo captures. It does not generate fake data; it simply wraps
files produced by the edge pipeline into the format expected by the Gemini and
H100 scripts.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    entities = json.loads(Path(args.entities_json).read_text())
    if isinstance(entities, list):
        entities = {"entities": entities}

    scene = {
        "scene_id": args.scene_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": args.question,
        "sensor_node": {
            "id": args.sensor_id,
            "lat": args.lat,
            "lon": args.lon,
        },
        **entities,
    }

    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    copy_if(args.topdown_lidar, images_dir / "topdown_lidar.png")
    copy_if(args.thermal_overlay, images_dir / "thermal_overlay.png")
    copy_if(args.rgb_frame, images_dir / "rgb_frame.png")
    copy_if(args.annotated_scene, images_dir / "annotated_scene.png")
    print(f"wrote scene pack to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--entities-json", required=True)
    parser.add_argument("--scene-id", default=f"capture_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--question", default="Describe the scene and identify any possible threats.")
    parser.add_argument("--sensor-id", default="narya-edge-01")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--topdown-lidar", default=None)
    parser.add_argument("--thermal-overlay", default=None)
    parser.add_argument("--rgb-frame", default=None)
    parser.add_argument("--annotated-scene", default=None)
    return parser.parse_args()


def copy_if(src: str | None, dst: Path) -> None:
    if src:
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()
