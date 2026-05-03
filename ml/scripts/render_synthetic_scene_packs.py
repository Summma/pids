#!/usr/bin/env python3
"""Create small synthetic scene packs for pipeline smoke tests.

These are not fake live sensor streams. They are seed examples for validating
the Gemini-teacher and H100-training pipeline before real captures exist.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


W = 768
H = 768
WORLD_M = 40.0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for idx in range(args.count):
        pack_dir = out_dir / f"scene_{idx:04d}"
        (pack_dir / "images").mkdir(parents=True, exist_ok=True)
        scene = make_scene(idx, rng)
        write_scene_pack(pack_dir, scene)

    print(f"wrote {args.count} scene packs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ml/data/scene_packs")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260503)
    return parser.parse_args()


def make_scene(idx: int, rng: random.Random) -> dict[str, Any]:
    templates = [
        ("single_human", ["human"]),
        ("animal_false_alarm", ["animal"]),
        ("human_plus_animal", ["human", "animal"]),
        ("warm_static_object", ["hot_object"]),
        ("drone_rf", ["drone"]),
        ("vehicle_heat", ["vehicle"]),
        ("mixed_perimeter", ["human", "animal", "drone"]),
        ("empty", []),
    ]
    scenario, kinds = templates[idx % len(templates)]
    entities = []

    for ent_idx, kind in enumerate(kinds):
        x = rng.uniform(-14, 14)
        z = rng.uniform(5, 18)
        entities.append(make_entity(kind, ent_idx, x, z, rng))

    return {
        "scene_id": f"scene_{idx:04d}",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": "Describe the scene and identify any possible threats.",
        "sensor_node": {
            "id": "narya-edge-01",
            "lat": 37.7955,
            "lon": -122.3937,
            "heading_deg": 18.0,
        },
        "environment": {
            "ambient_temp_c": round(rng.uniform(17.0, 22.0), 1),
            "visibility": "indoor demo / line of sight",
            "area": "temporary perimeter sector",
        },
        "entities": entities,
        "rf_events": [
            ent["rf"] for ent in entities if ent.get("rf", {}).get("drone_candidate")
        ],
    }


def make_entity(kind: str, idx: int, x: float, z: float, rng: random.Random) -> dict[str, Any]:
    if kind == "human":
        height = rng.uniform(1.62, 1.92)
        width = rng.uniform(0.42, 0.68)
        depth = rng.uniform(0.32, 0.55)
        mean_c = rng.uniform(28, 33)
        max_c = rng.uniform(34, 37)
        upright = rng.uniform(0.84, 0.98)
        speed = rng.uniform(0.1, 1.4)
    elif kind == "animal":
        height = rng.uniform(0.38, 0.85)
        width = rng.uniform(0.35, 0.65)
        depth = rng.uniform(0.8, 1.5)
        mean_c = rng.uniform(27, 32)
        max_c = rng.uniform(33, 37)
        upright = rng.uniform(0.08, 0.35)
        speed = rng.uniform(0.2, 2.2)
    elif kind == "drone":
        height = rng.uniform(0.18, 0.45)
        width = rng.uniform(0.45, 0.9)
        depth = rng.uniform(0.45, 0.9)
        mean_c = rng.uniform(20, 28)
        max_c = rng.uniform(26, 42)
        upright = rng.uniform(0.15, 0.45)
        speed = rng.uniform(0.5, 6.0)
    elif kind == "vehicle":
        height = rng.uniform(1.2, 2.4)
        width = rng.uniform(1.5, 2.8)
        depth = rng.uniform(2.4, 5.5)
        mean_c = rng.uniform(24, 38)
        max_c = rng.uniform(45, 75)
        upright = rng.uniform(0.35, 0.65)
        speed = rng.uniform(0.0, 3.0)
    else:
        height = rng.uniform(0.2, 0.9)
        width = rng.uniform(0.3, 1.2)
        depth = rng.uniform(0.3, 1.2)
        mean_c = rng.uniform(30, 45)
        max_c = rng.uniform(38, 65)
        upright = rng.uniform(0.1, 0.7)
        speed = 0.0

    rf = {"drone_candidate": False, "emitters": []}
    if kind == "drone":
        rf = {
            "drone_candidate": True,
            "emitters": [
                {
                    "band": "ism915",
                    "freq_mhz": round(915 + rng.uniform(-0.35, 0.35), 3),
                    "confidence": round(rng.uniform(0.72, 0.94), 2),
                }
            ],
        }

    return {
        "track_id": f"trk_{idx:03d}",
        "ground_truth_hint": kind,
        "classification": "unknown",
        "position_m": {"x": round(x, 2), "y": 0.0, "z": round(z, 2)},
        "size_m": {
            "height": round(height, 2),
            "width": round(width, 2),
            "depth": round(depth, 2),
        },
        "velocity_mps": round(speed, 2),
        "thermal": {
            "mean_c": round(mean_c, 1),
            "max_c": round(max_c, 1),
            "hot_pixel_fraction": round(rng.uniform(0.18, 0.72), 2),
            "cooling_rate_cpm": round(rng.uniform(-0.1, 1.4), 2),
        },
        "lidar": {
            "points": rng.randint(80, 1200),
            "upright_score": round(upright, 2),
            "range_m": round(math.hypot(x, z), 2),
        },
        "rf": rf,
        "confidence": round(rng.uniform(0.42, 0.86), 2),
    }


def write_scene_pack(pack_dir: Path, scene: dict[str, Any]) -> None:
    (pack_dir / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    render_topdown(scene).save(pack_dir / "images" / "topdown_lidar.png")
    render_thermal(scene).save(pack_dir / "images" / "thermal_overlay.png")
    render_rgb(scene).save(pack_dir / "images" / "rgb_frame.png")
    render_annotated(scene).save(pack_dir / "images" / "annotated_scene.png")


def render_topdown(scene: dict[str, Any]) -> Image.Image:
    img = base_canvas("LiDAR top-down geometry")
    draw = ImageDraw.Draw(img)
    draw_grid(draw)
    for ent in scene["entities"]:
        x, z = world_to_px(ent["position_m"]["x"], ent["position_m"]["z"])
        size = ent["size_m"]
        color = (0, 212, 255)
        radius = max(5, int(size["width"] * 14))
        draw.ellipse((x - radius, z - radius, x + radius, z + radius), outline=color, width=3)
        draw.line((x, z, x, z - int(size["height"] * 24)), fill=color, width=3)
    return img


def render_thermal(scene: dict[str, Any]) -> Image.Image:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[:, :, :] = (5, 9, 16)
    img = Image.fromarray(arr)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    for ent in scene["entities"]:
        x, z = world_to_px(ent["position_m"]["x"], ent["position_m"]["z"])
        temp = ent["thermal"]["max_c"]
        intensity = min(255, max(60, int((temp - 18) / 55 * 255)))
        color = (255, int(max(30, 190 - intensity * 0.35)), 20, 190)
        sx = int(max(10, ent["size_m"]["width"] * 30))
        sy = int(max(10, ent["size_m"]["height"] * 22))
        draw.ellipse((x - sx, z - sy, x + sx, z + sy), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(12))
    return Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")


def render_rgb(scene: dict[str, Any]) -> Image.Image:
    img = base_canvas("RGB context approximation")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, H * 0.64, W, H), fill=(24, 32, 36))
    draw.rectangle((0, 0, W, H * 0.64), fill=(15, 22, 28))
    for ent in scene["entities"]:
        x, z = world_to_px(ent["position_m"]["x"], ent["position_m"]["z"])
        y = int(H * 0.72 - ent["size_m"]["height"] * 36)
        h = int(ent["size_m"]["height"] * 70)
        w = int(max(12, ent["size_m"]["width"] * 45))
        draw.rectangle((x - w, y - h, x + w, y), outline=(220, 230, 240), width=3)
    return img


def render_annotated(scene: dict[str, Any]) -> Image.Image:
    img = render_thermal(scene).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw_grid(draw, alpha=False)
    for ent in scene["entities"]:
        x, z = world_to_px(ent["position_m"]["x"], ent["position_m"]["z"])
        label = f"{ent['track_id']} {ent['thermal']['max_c']:.1f}C {ent['lidar']['range_m']:.1f}m"
        draw.rectangle((x - 42, z - 26, x + 42, z + 26), outline=(255, 245, 80), width=3)
        draw.text((x + 48, z - 18), label, fill=(255, 245, 80))
    draw.text((20, H - 36), scene["scenario"], fill=(255, 255, 255))
    return img


def base_canvas(title: str) -> Image.Image:
    img = Image.new("RGB", (W, H), (8, 12, 18))
    draw = ImageDraw.Draw(img)
    draw.text((20, 18), title, fill=(210, 230, 245))
    return img


def draw_grid(draw: ImageDraw.ImageDraw, alpha: bool = True) -> None:
    color = (35, 58, 78) if not alpha else (28, 48, 68)
    for px in range(64, W, 64):
        draw.line((px, 64, px, H - 64), fill=color)
    for py in range(64, H, 64):
        draw.line((64, py, W - 64, py), fill=color)
    cx, cy = world_to_px(0, 0)
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(0, 230, 180))


def world_to_px(x_m: float, z_m: float) -> tuple[int, int]:
    scale = (W - 128) / WORLD_M
    x = int(W / 2 + x_m * scale)
    y = int(H - 96 - z_m * scale)
    return x, y


if __name__ == "__main__":
    main()
