#!/usr/bin/env python3
"""Export a Narya scene analysis as Palantir Ontology-ready JSON.

This intentionally avoids depending on a Palantir SDK. The output is a stable,
plain JSON contract that can be mapped into Foundry object/action imports or
used directly by a demo adapter.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THREAT_SCORE = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def main() -> None:
    args = parse_args()
    scene_pack = Path(args.scene_pack)
    scene = json.loads((scene_pack / "scene.json").read_text())
    analysis = load_analysis(Path(args.analysis_json)) if args.analysis_json else {}

    payload = build_payload(scene_pack, scene, analysis)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote Palantir payload to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-pack", required=True)
    parser.add_argument(
        "--analysis-json",
        default=None,
        help="Gemini/local-model JSON output, or a teacher-label JSONL record.",
    )
    parser.add_argument("--out", default="ml/outputs/palantir_payload.json")
    return parser.parse_args()


def load_analysis(path: Path) -> dict[str, Any]:
    text = path.read_text().strip()
    if not text:
        return {}

    parsed = parse_jsonish(text)
    if isinstance(parsed, dict) and "teacher" in parsed and isinstance(parsed["teacher"], dict):
        return {
            **parsed["teacher"],
            "_analysis_source": parsed.get("teacher_model") or parsed.get("model"),
        }
    if isinstance(parsed, dict):
        return parsed
    return {"raw_analysis": parsed}


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

    # Accept a single teacher-label JSONL record pasted with trailing progress text.
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    return {"raw_text": text}


def build_payload(scene_pack: Path, scene: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    scene_id = str(scene.get("scene_id") or scene_pack.name)
    sensor_node = scene.get("sensor_node") or {}
    global_threat = normalize_threat(
        analysis.get("global_risk_level")
        or nested_get(analysis, ("threat_assessment", "threat_level"))
    )
    analysis_by_track = {
        str(ent.get("track_id")): ent
        for ent in analysis_entities(analysis)
        if isinstance(ent, dict) and ent.get("track_id") is not None
    }

    objects: list[dict[str, Any]] = [
        {
            "objectType": "TacticalScene",
            "primaryKey": scene_id,
            "properties": compact_dict(
                {
                    "sceneId": scene_id,
                    "timestamp": scene.get("timestamp"),
                    "question": scene.get("question"),
                    "scenePackPath": str(scene_pack),
                    "globalRiskLevel": global_threat,
                    "sceneDescription": stringify_analysis_value(analysis.get("scene_description")),
                    "tacticalAssessment": analysis.get("tactical_assessment")
                    or analysis.get("answer")
                    or nested_get(analysis, ("threat_assessment", "reasoning")),
                    "exportedAt": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }
    ]

    sensor_id = sensor_node.get("id")
    if sensor_id:
        objects.append(
            {
                "objectType": "SensorNode",
                "primaryKey": str(sensor_id),
                "properties": compact_dict(
                    {
                        "sensorId": sensor_id,
                        "latitude": sensor_node.get("lat"),
                        "longitude": sensor_node.get("lon"),
                    }
                ),
            }
        )

    relations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []

    if sensor_id:
        relations.append(
            {
                "relationType": "SCENE_OBSERVED_BY_SENSOR",
                "fromType": "TacticalScene",
                "fromId": scene_id,
                "toType": "SensorNode",
                "toId": str(sensor_id),
            }
        )

    for entity in scene.get("entities", []):
        track_id = str(entity.get("track_id", "unknown"))
        entity_id = f"{scene_id}:{track_id}"
        assessed = analysis_by_track.get(track_id, {})
        threat_level = normalize_threat(assessed.get("threat_level") or global_threat)
        classification = (
            assessed.get("best_classification")
            or assessed.get("classification")
            or entity.get("classification")
            or "unknown"
        )

        objects.append(
            {
                "objectType": "TacticalEntity",
                "primaryKey": entity_id,
                "properties": compact_dict(
                    {
                        "sceneId": scene_id,
                        "trackId": track_id,
                        "classification": classification,
                        "threatLevel": threat_level,
                        "threatScore": THREAT_SCORE.get(threat_level, 0),
                        "confidence": assessed.get("confidence", entity.get("confidence")),
                        "rationale": assessed.get("rationale"),
                        "recommendedAction": assessed.get("recommended_action")
                        or recommend_action(threat_level),
                        "positionM": entity.get("position_m"),
                        "sizeM": entity.get("size_m"),
                        "thermal": entity.get("thermal"),
                        "lidar": entity.get("lidar"),
                        "rf": entity.get("rf"),
                    }
                ),
            }
        )
        relations.append(
            {
                "relationType": "SCENE_CONTAINS_ENTITY",
                "fromType": "TacticalScene",
                "fromId": scene_id,
                "toType": "TacticalEntity",
                "toId": entity_id,
            }
        )
        actions.append(make_action(entity_id, track_id, threat_level, assessed))
        feature = make_feature(sensor_node, entity, entity_id, classification, threat_level)
        if feature:
            features.append(feature)

        for emitter in entity.get("rf", {}).get("emitters", []):
            emitter_id = f"{entity_id}:rf:{emitter.get('band_mhz', 'unknown')}"
            objects.append(
                {
                    "objectType": "RFEmitter",
                    "primaryKey": emitter_id,
                    "properties": compact_dict(
                        {
                            "sceneId": scene_id,
                            "trackId": track_id,
                            "bandMhz": emitter.get("band_mhz"),
                            "frequencyHz": emitter.get("frequency_hz"),
                            "snrDb": emitter.get("snr_db"),
                            "persistenceS": emitter.get("persistence_s"),
                        }
                    ),
                }
            )
            relations.append(
                {
                    "relationType": "ENTITY_HAS_RF_EMITTER",
                    "fromType": "TacticalEntity",
                    "fromId": entity_id,
                    "toType": "RFEmitter",
                    "toId": emitter_id,
                }
            )

    return {
        "ontologyVersion": "narya.demo.v1",
        "source": {
            "scenePack": str(scene_pack),
            "analysisSource": analysis.get("_analysis_source")
            or analysis.get("teacher_model")
            or analysis.get("model")
            or "narya-analysis-json",
        },
        "objects": objects,
        "relations": relations,
        "actions": actions,
        "geojson": {
            "type": "FeatureCollection",
            "features": features,
        },
    }


def compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v is not None}


def nested_get(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def stringify_analysis_value(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def analysis_entities(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    entities = analysis.get("entities")
    if isinstance(entities, list):
        return [ent for ent in entities if isinstance(ent, dict)]

    nested_entities = nested_get(analysis, ("scene_description", "entities"))
    if isinstance(nested_entities, list):
        return [ent for ent in nested_entities if isinstance(ent, dict)]

    return []


def normalize_threat(value: Any) -> str:
    text = str(value or "low").strip().lower()
    return text if text in THREAT_SCORE else "low"


def recommend_action(threat_level: str) -> str:
    if threat_level in {"critical", "high"}:
        return "dispatch_or_intercept"
    if threat_level == "medium":
        return "investigate"
    if threat_level == "low":
        return "monitor"
    return "no_action"


def make_action(
    entity_id: str,
    track_id: str,
    threat_level: str,
    assessed: dict[str, Any],
) -> dict[str, Any]:
    action_type = assessed.get("recommended_action") or recommend_action(threat_level)
    return {
        "actionType": "NaryaTriageTrack",
        "targetObjectType": "TacticalEntity",
        "targetObjectId": entity_id,
        "parameters": compact_dict(
            {
                "trackId": track_id,
                "priority": threat_level,
                "recommendedAction": action_type,
                "rationale": assessed.get("rationale"),
            }
        ),
    }


def make_feature(
    sensor_node: dict[str, Any],
    entity: dict[str, Any],
    entity_id: str,
    classification: str,
    threat_level: str,
) -> dict[str, Any] | None:
    lat = sensor_node.get("lat")
    lon = sensor_node.get("lon")
    position = entity.get("position_m") or {}
    if lat is None or lon is None or position.get("x") is None or position.get("y") is None:
        return None

    entity_lat, entity_lon = offset_lat_lon(float(lat), float(lon), float(position["x"]), float(position["y"]))
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [entity_lon, entity_lat],
        },
        "properties": {
            "id": entity_id,
            "trackId": entity.get("track_id"),
            "classification": classification,
            "threatLevel": threat_level,
            "altitudeM": position.get("z"),
        },
    }


def offset_lat_lon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    meters_per_degree_lat = 111_111.0
    meters_per_degree_lon = meters_per_degree_lat * max(math.cos(math.radians(lat)), 0.01)
    return lat + north_m / meters_per_degree_lat, lon + east_m / meters_per_degree_lon


if __name__ == "__main__":
    main()
