#!/usr/bin/env python3
"""Offline evaluator for indoor seated/standing human detection captures.

Input is a directory of ``.npz`` files with ``xyz`` and optional labels. The
GUI debug capture already writes the needed metadata:

  - ``scene_label``: unknown/no_human/seated_human_present/empty_chairs/...
  - ``humans``: saved indoor ROI detections
  - ``pointpillars``: saved PointPillars detections, for baseline comparison
  - ``candidate_debug`` + ``candidate_features``: ROI classifier diagnostics

For labelled datasets, files may also contain:

  - ``boxes``: axis-aligned boxes ``cx,cy,cz,dx,dy,dz``
  - ``box_labels``: labels such as standing_person, seated_person,
    occupied_chair, empty_chair, chair_with_bag_or_coat, clutter.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from indoor_human import (
    IndoorHumanParams,
    HumanTrackTracker,
    detect_human_candidates,
    _estimate_floor_z,
)

POSITIVE_LABELS = {"standing_person", "seated_person", "occupied_chair", "human", "person"}
NEGATIVE_LABELS = {"empty_chair", "chair_with_bag_or_coat", "clutter", "no_human"}
SEATED_LABELS = {"seated_person", "occupied_chair"}
EMPTY_CHAIR_LABELS = {"empty_chair", "chair_with_bag_or_coat"}


def _scalar_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data:
        return default
    return str(np.asarray(data[key]).item())


def _truth_labels(data: np.lib.npyio.NpzFile) -> list[str]:
    labels: list[str] = []
    if "box_labels" in data:
        labels.extend(str(x) for x in np.asarray(data["box_labels"]).reshape(-1))
    scene = _scalar_str(data, "scene_label", "")
    if scene and scene != "unknown":
        labels.append(scene)
    if "scenario" in data:
        labels.append(_scalar_str(data, "scenario"))
    for key in [
        "standing_person",
        "seated_person",
        "occupied_chair",
        "empty_chair",
        "chair_with_bag_or_coat",
        "clutter",
        "near_table",
        "near_wall",
        "multiple_people",
    ]:
        if key in data and bool(np.asarray(data[key]).item()):
            labels.append(key)
    if "has_human" in data:
        labels.append("human" if bool(np.asarray(data["has_human"]).item()) else "no_human")
    if "boxes" in data and len(np.asarray(data["boxes"])) > 0 and not any(l in POSITIVE_LABELS for l in labels):
        labels.append("human")
    return labels


def _truth_has_human(labels: Iterable[str]) -> bool | None:
    labels = set(labels)
    if labels & POSITIVE_LABELS:
        return True
    if labels & NEGATIVE_LABELS:
        return False
    return None


def _scenario_tags(labels: Iterable[str]) -> list[str]:
    tags = [l for l in labels if l not in {"human", "person", "no_human"}]
    return tags or ["all_labelled"]


def _saved_detection_rows(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        return np.empty((0, 13), dtype=np.float32)
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 9:
        return np.empty((0, 13), dtype=np.float32)
    return arr


def _candidate_rows(candidates) -> np.ndarray:
    rows = []
    for i, cand in enumerate(candidates, start=1):
        rows.append([
            *cand.center.tolist(),
            *cand.size.tolist(),
            float(cand.yaw),
            float(cand.score),
            float(cand.score),
            1.0,
            float(i),
            float(cand.points),
            float(cand.z_span),
        ])
    return np.asarray(rows, dtype=np.float32)


def _run_pipeline(
    data: np.lib.npyio.NpzFile,
    pipeline: str,
    tracker: HumanTrackTracker,
    fps: float,
    use_tracking: bool,
) -> tuple[np.ndarray, float, int]:
    t0 = time.monotonic()
    if pipeline == "indoor":
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        candidates, _detail, debug = detect_human_candidates(xyz, tracker.params)
        if not use_tracking:
            return _candidate_rows(candidates), (time.monotonic() - t0) * 1000.0, len(debug)
        detections = tracker.update(candidates, 1.0 / max(fps, 1e-6))
        rows = []
        for det in detections:
            rows.append([
                *det.center.tolist(),
                *det.size.tolist(),
                float(det.yaw),
                float(det.score),
                float(det.model_score),
                float(det.class_id),
                float(det.track_id),
                float(det.support_points),
                float(det.support_z_span),
            ])
        return np.asarray(rows, dtype=np.float32), (time.monotonic() - t0) * 1000.0, len(debug)
    if pipeline == "pointpillars_saved":
        return _saved_detection_rows(data, "pointpillars"), (time.monotonic() - t0) * 1000.0, 0
    if pipeline == "indoor_saved":
        return _saved_detection_rows(data, "humans"), (time.monotonic() - t0) * 1000.0, 0
    raise ValueError(f"unknown pipeline {pipeline}")


def _axis_box_bounds(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = rows[:, :3]
    sizes = np.maximum(rows[:, 3:6], 1e-3)
    return centers - sizes * 0.5, centers + sizes * 0.5


def _iou_matrix(pred_rows: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(pred_rows) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_rows), len(gt_boxes)), dtype=np.float32)
    p_mn, p_mx = _axis_box_bounds(pred_rows)
    g_mn, g_mx = _axis_box_bounds(gt_boxes)
    out = np.zeros((len(pred_rows), len(gt_boxes)), dtype=np.float32)
    for i in range(len(pred_rows)):
        for j in range(len(gt_boxes)):
            inter = np.maximum(0.0, np.minimum(p_mx[i], g_mx[j]) - np.maximum(p_mn[i], g_mn[j]))
            inter_vol = float(np.prod(inter))
            if inter_vol <= 0:
                continue
            p_vol = float(np.prod(p_mx[i] - p_mn[i]))
            g_vol = float(np.prod(g_mx[j] - g_mn[j]))
            out[i, j] = inter_vol / max(p_vol + g_vol - inter_vol, 1e-6)
    return out


def _match_boxes(pred_rows: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> tuple[int, list[float], list[float]]:
    if len(pred_rows) == 0 or len(gt_boxes) == 0:
        return 0, [], []
    iou = _iou_matrix(pred_rows, gt_boxes)
    pairs = sorted(
        ((float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])),
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    center_errors: list[float] = []
    ious: list[float] = []
    for val, i, j in pairs:
        if val < iou_threshold:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(val)
        center_errors.append(float(np.linalg.norm(pred_rows[i, :3] - gt_boxes[j, :3])))
    return len(used_g), center_errors, ious


def _points_in_axis_box(xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    pts = xyz.reshape(-1, 3)
    mn = box[:3] - box[3:6] * 0.5
    mx = box[:3] + box[3:6] * 0.5
    return (
        (pts[:, 0] >= mn[0]) & (pts[:, 0] <= mx[0])
        & (pts[:, 1] >= mn[1]) & (pts[:, 1] <= mx[1])
        & (pts[:, 2] >= mn[2]) & (pts[:, 2] <= mx[2])
    )


def _missed_roi_stats(data: np.lib.npyio.NpzFile, matched_gt: int) -> tuple[int, int]:
    if "boxes" not in data or matched_gt > 0:
        return 0, 0
    labels = [str(x) for x in np.asarray(data["box_labels"], dtype=str).reshape(-1)] if "box_labels" in data else []
    boxes = np.asarray(data["boxes"], dtype=np.float32)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    floor_z = _estimate_floor_z(xyz.reshape(-1, 3))
    total_points = 0
    above_seat_points = 0
    for i, box in enumerate(boxes):
        label = labels[i] if i < len(labels) else "human"
        if label not in SEATED_LABELS:
            continue
        mask = _points_in_axis_box(xyz, box)
        pts = xyz.reshape(-1, 3)[mask]
        total_points += int(len(pts))
        above_seat_points += int(np.count_nonzero((pts[:, 2] - floor_z) > 0.55)) if len(pts) else 0
    return total_points, above_seat_points


def _update_counts(counts: dict[str, int], truth: bool, pred: bool) -> None:
    if truth and pred:
        counts["tp"] += 1
    elif truth and not pred:
        counts["fn"] += 1
    elif not truth and pred:
        counts["fp"] += 1
    else:
        counts["tn"] += 1


def _prf(counts: dict[str, int]) -> tuple[float, float, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def evaluate(path: Path, fps: float, pipeline: str, iou_threshold: float, use_tracking: bool) -> None:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz frames found in {path}")

    params = IndoorHumanParams()
    tracker = HumanTrackTracker(params)
    labelled = 0
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    seated = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    empty_chair = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    by_tag: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    total_ms = 0.0
    center_errors: list[float] = []
    ious: list[float] = []
    missed_seated_roi_points: list[int] = []
    missed_seated_above_seat_points: list[int] = []
    detection_delays: list[int] = []
    open_positive_run: int | None = None
    unlabelled_positive = 0

    for frame_i, file in enumerate(files):
        data = np.load(file, allow_pickle=False)
        if "xyz" not in data and pipeline == "indoor":
            print(f"skip {file.name}: missing xyz")
            continue

        labels = _truth_labels(data)
        truth = _truth_has_human(labels)
        rows, elapsed_ms, debug_count = _run_pipeline(data, pipeline, tracker, fps, use_tracking)
        total_ms += elapsed_ms
        pred = len(rows) > 0

        gt_boxes = np.asarray(data["boxes"], dtype=np.float32) if "boxes" in data else np.empty((0, 6), dtype=np.float32)
        matched_gt, errs, frame_ious = _match_boxes(rows, gt_boxes, iou_threshold)
        center_errors.extend(errs)
        ious.extend(frame_ious)

        if truth is None:
            unlabelled_positive += int(pred)
            print(f"{file.name}: unlabelled pred={pred} det={len(rows)} {elapsed_ms:.0f}ms")
            continue

        labelled += 1
        _update_counts(counts, truth, pred)
        has_seated = bool(set(labels) & SEATED_LABELS)
        has_empty_chair = bool(set(labels) & EMPTY_CHAIR_LABELS)
        if has_seated:
            _update_counts(seated, True, pred)
            if not pred:
                pts, above = _missed_roi_stats(data, matched_gt)
                if pts:
                    missed_seated_roi_points.append(pts)
                    missed_seated_above_seat_points.append(above)
        elif truth is not None:
            _update_counts(seated, False, pred)

        if has_empty_chair:
            # For empty-chair precision, a detection is the bad event.
            _update_counts(empty_chair, False, pred)
        elif truth is not None:
            _update_counts(empty_chair, True, pred)

        for tag in _scenario_tags(labels):
            _update_counts(by_tag[tag], truth, pred)

        if truth:
            if open_positive_run is None:
                open_positive_run = frame_i
            if pred and open_positive_run is not None:
                detection_delays.append(frame_i - open_positive_run)
                open_positive_run = None
        else:
            open_positive_run = None

        print(
            f"{file.name}: labels={','.join(labels) or 'unknown'} pred={pred} det={len(rows)} "
            f"matched_gt={matched_gt}/{len(gt_boxes)} debug={debug_count} {elapsed_ms:.0f}ms"
        )

    precision, recall, f1 = _prf(counts)
    seated_p, seated_r, seated_f1 = _prf(seated)
    empty_p, empty_r, _empty_f1 = _prf(empty_chair)
    minutes = len(files) / max(fps, 1e-6) / 60.0
    fp_per_min = counts["fp"] / max(minutes, 1e-9)

    print()
    print(f"pipeline={pipeline} tracking={'on' if use_tracking else 'off'}")
    print(f"frames={len(files)} labelled={labelled}")
    print(f"TP={counts['tp']} FP={counts['fp']} TN={counts['tn']} FN={counts['fn']}")
    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print(f"false_positives_per_min={fp_per_min:.3f}")
    print(f"avg_latency_ms={total_ms / max(len(files), 1):.1f}")
    print(f"seated_person_recall={seated_r:.3f} seated_precision={seated_p:.3f} seated_f1={seated_f1:.3f}")
    print(f"occupied_or_empty_chair_precision_proxy={empty_p:.3f}")
    print(f"empty_chair_false_positive_rate={empty_chair['fp'] / max(empty_chair['fp'] + empty_chair['tn'], 1):.3f}")
    if center_errors:
        print(f"localization_error_m_mean={np.mean(center_errors):.3f} median={np.median(center_errors):.3f}")
        print(f"matched_iou_mean={np.mean(ious):.3f}")
    if missed_seated_roi_points:
        print(
            "missed_seated_points_per_roi_mean="
            f"{np.mean(missed_seated_roi_points):.1f}"
            " above_seat_mean="
            f"{np.mean(missed_seated_above_seat_points):.1f}"
        )
    if detection_delays:
        print(f"detection_delay_frames_mean={np.mean(detection_delays):.2f}")
    if by_tag:
        print()
        print("by_scenario:")
        for tag, cur in sorted(by_tag.items()):
            p, r, tf1 = _prf(cur)
            print(
                f"  {tag}: TP={cur['tp']} FP={cur['fp']} TN={cur['tn']} FN={cur['fn']} "
                f"precision={p:.3f} recall={r:.3f} f1={tf1:.3f}"
            )
    if labelled == 0:
        print(f"unlabelled_positive_frames={unlabelled_positive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument(
        "--pipeline",
        choices=["indoor", "indoor_saved", "pointpillars_saved"],
        default="indoor",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    parser.add_argument("--no-tracking", action="store_true", help="evaluate raw candidates instead of confirmed tracks")
    args = parser.parse_args()
    evaluate(args.path, args.fps, args.pipeline, args.iou_threshold, not args.no_tracking)


if __name__ == "__main__":
    main()
