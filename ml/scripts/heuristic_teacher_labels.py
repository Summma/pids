#!/usr/bin/env python3
"""Create deterministic teacher-label-shaped records for local smoke tests.

Use Gemini labels for the real demo. This script only lets us validate the
pipeline when API keys are not present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    packs = sorted(Path(args.scene_packs).glob("scene_*"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out:
        for pack in packs:
            scene = json.loads((pack / "scene.json").read_text())
            record = {
                "scene_id": scene["scene_id"],
                "scene_pack": str(pack),
                "question": scene.get("question"),
                "scene": scene,
                "teacher_model": "heuristic-smoke-test",
                "teacher_raw": "",
                "teacher": assess(scene),
            }
            out.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"wrote {len(packs)} smoke-test labels to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-packs", default="ml/data/scene_packs")
    parser.add_argument("--out", default="ml/data/teacher_labels.smoke.jsonl")
    return parser.parse_args()


def assess(scene: dict[str, Any]) -> dict[str, Any]:
    entities = [assess_entity(ent) for ent in scene.get("entities", [])]
    risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    global_risk = "none"
    for ent in entities:
        if risk_order[ent["threat_level"]] > risk_order[global_risk]:
            global_risk = ent["threat_level"]

    if not entities:
        description = "No tracked entities are currently present in the sensor scene."
        assessment = "No tactical threat is indicated by the available evidence."
    else:
        description = f"The scene contains {len(entities)} tracked entity or entities with LiDAR, thermal, and RF evidence."
        assessment = f"The highest assessed risk is {global_risk} based on modality agreement and RF cues."

    return {
        "scene_description": description,
        "tactical_assessment": assessment,
        "entities": entities,
        "global_risk_level": global_risk,
        "answer": assessment,
    }


def assess_entity(ent: dict[str, Any]) -> dict[str, Any]:
    height = ent.get("size_m", {}).get("height", 0)
    depth = ent.get("size_m", {}).get("depth", 0)
    mean_c = ent.get("thermal", {}).get("mean_c", 0)
    upright = ent.get("lidar", {}).get("upright_score", 0)
    rf_drone = ent.get("rf", {}).get("drone_candidate", False)
    hint = ent.get("ground_truth_hint")

    if rf_drone:
        cls = "drone"
        level = "high"
        action = "intercept"
        rationale = "RF drone-control evidence is present, with a compact moving LiDAR track."
    elif height > 1.4 and upright > 0.75 and mean_c > 26:
        cls = "human"
        level = "medium"
        action = "investigate"
        rationale = "Human-scale upright LiDAR geometry is supported by a warm thermal signature."
    elif depth > height and mean_c > 24:
        cls = "animal"
        level = "low"
        action = "monitor"
        rationale = "Low elongated LiDAR geometry with warmth is more consistent with an animal."
    elif hint == "vehicle":
        cls = "vehicle"
        level = "medium"
        action = "monitor"
        rationale = "Large geometry and elevated thermal maximum are consistent with a vehicle or engine heat."
    elif mean_c > 35:
        cls = "hot_object"
        level = "low"
        action = "monitor"
        rationale = "Thermal evidence is warm but shape and motion do not support a human classification."
    else:
        cls = "unknown"
        level = "low"
        action = "monitor"
        rationale = "Evidence is insufficient for a stronger classification."

    return {
        "track_id": ent.get("track_id", "unknown"),
        "best_classification": cls,
        "threat_level": level,
        "confidence": 0.78 if cls == hint or rf_drone else 0.62,
        "rationale": rationale,
        "recommended_action": action,
    }


if __name__ == "__main__":
    main()
