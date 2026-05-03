"""Indoor seated/standing human detection from ROI crops + tracking.

This is intentionally separate from PointPillars. It is a conservative
room-scale pipeline:

    raw cloud -> floor/large-plane filtering -> DBSCAN/ROI proposals
    -> crop-level occupied-chair/person classifier -> temporal confirmation

The crop classifier starts as a deterministic geometric baseline. It is shaped
like a learned classifier on purpose: every ROI gets a fixed feature vector and
class score vector that can be captured and used to train/replace the baseline
with a RandomForest, PointNet, sparse CNN, or another crop classifier later.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from clustering import DBSCAN, HAS_SKLEARN, voxel_downsample
from detection import Detection

ROI_CLASS_NAMES = (
    "occupied_chair",
    "empty_chair",
    "standing_person",
    "seated_person",
    "non_human_clutter",
    "uncertain",
)
ROI_HUMAN_LABELS = {"occupied_chair", "standing_person", "seated_person"}
ROI_FEATURE_NAMES = (
    "points",
    "dx",
    "dy",
    "dz",
    "min_xy",
    "max_xy",
    "bottom_gap",
    "top_rel",
    "range_m",
    "density_log",
    "upper_count",
    "upper_ratio",
    "torso_count",
    "torso_ratio",
    "head_count",
    "head_ratio",
    "seat_count",
    "seat_ratio",
    "above_seat_count",
    "above_seat_ratio",
    "linearity",
    "planarity",
    "scattering",
    "compactness",
    "floor_contact",
    "vertical_score",
    "human_visible_score",
    "chair_context_score",
    "upper_min_xy",
    "upper_max_xy",
    "upper_thickness_score",
)


@dataclass
class IndoorHumanParams:
    range_min: float = 0.7
    range_max: float = 28.0
    z_min: float = -2.8
    z_max: float = 3.0
    floor_margin: float = 0.16
    ceiling_height: float = 2.6
    voxel_size: float = 0.08
    max_points: int = 18_000
    eps: float = 0.42
    min_samples: int = 8
    min_cluster_points: int = 18
    association_distance: float = 1.2
    confirm_hits: int = 2
    confirm_window: int = 5
    max_misses: int = 5


@dataclass
class HumanCandidate:
    center: np.ndarray
    size: np.ndarray
    yaw: float
    score: float
    points: int
    z_span: float
    floor_z: float
    kind: str
    features: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    class_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.center))


@dataclass
class CandidateDebug:
    center: np.ndarray
    size: np.ndarray
    yaw: float
    score: float
    points: int
    z_span: float
    kind: str
    reason: str
    accepted: bool
    cluster_id: int = -1
    source: str = "cluster"
    features: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    class_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class RoiProposal:
    points: np.ndarray
    seed_points: np.ndarray
    source: str
    cluster_id: int


@dataclass
class RoiClassification:
    label: str
    score: float
    scores: np.ndarray
    reason: str


@dataclass
class HumanTrack:
    track_id: int
    candidate: HumanCandidate
    history: deque[bool]
    scores: deque[float]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    age: int = 1
    misses: int = 0

    @property
    def confirmed_hits(self) -> int:
        return int(sum(self.history))


def _estimate_floor_z(points: np.ndarray) -> float:
    if len(points) == 0:
        return -1.5
    near = points[
        (points[:, 0] > 1.0)
        & (points[:, 0] < 20.0)
        & (np.abs(points[:, 1]) < 12.0)
    ]
    if len(near) < 256:
        near = points
    return float(np.percentile(near[:, 2], 4.0))


def _fit_plane(sample: np.ndarray, rng: np.random.Generator) -> Optional[tuple[np.ndarray, float]]:
    if len(sample) < 3:
        return None
    ids = rng.choice(len(sample), 3, replace=False)
    a, b, c = sample[ids]
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return None
    normal = normal / norm
    d = -float(np.dot(normal, a))
    return normal.astype(np.float32), d


def _remove_large_planes(points: np.ndarray, floor_z: float) -> tuple[np.ndarray, int]:
    """Remove large wall/table-like planes without depending on a room map."""
    if len(points) < 1500:
        return points, 0

    rng = np.random.default_rng(7)
    keep = np.ones(len(points), dtype=bool)
    removed_planes = 0

    for _ in range(3):
        cur = points[keep]
        if len(cur) < 1500:
            break
        sample = cur
        if len(sample) > 5000:
            sample = sample[rng.choice(len(sample), 5000, replace=False)]

        best = None
        best_count = 0
        for _ in range(64):
            plane = _fit_plane(sample, rng)
            if plane is None:
                continue
            n, d = plane
            dist = np.abs(sample @ n + d)
            count = int(np.count_nonzero(dist < 0.045))
            if count > best_count:
                best = plane
                best_count = count

        if best is None:
            break

        n, d = best
        dist_full = np.abs(points @ n + d)
        inlier = keep & (dist_full < 0.055)
        count_full = int(np.count_nonzero(inlier))
        if count_full < max(650, int(0.10 * np.count_nonzero(keep))):
            break

        plane_pts = points[inlier]
        extent = plane_pts.max(axis=0) - plane_pts.min(axis=0)
        is_wall = abs(float(n[2])) < 0.25 and extent[2] > 1.0 and max(extent[0], extent[1]) > 1.5
        is_table_or_ceiling = (
            abs(float(n[2])) > 0.85
            and float(np.median(plane_pts[:, 2])) > floor_z + 0.45
            and extent[0] > 0.8
            and extent[1] > 0.8
        )
        if not (is_wall or is_table_or_ceiling):
            break

        keep[inlier] = False
        removed_planes += 1

    return points[keep], removed_planes


def _preprocess(xyz: np.ndarray, params: IndoorHumanParams) -> tuple[np.ndarray, float, int]:
    pts = xyz.reshape(-1, 3).astype(np.float32, copy=False)
    finite = np.isfinite(pts).all(axis=1)
    rng = np.linalg.norm(pts, axis=1)
    keep = (
        finite
        & (rng >= params.range_min)
        & (rng <= params.range_max)
        & (pts[:, 2] >= params.z_min)
        & (pts[:, 2] <= params.z_max)
    )
    pts = pts[keep]
    if len(pts) == 0:
        return pts, -1.5, 0

    floor_z = _estimate_floor_z(pts)
    vertical_keep = (
        (pts[:, 2] > floor_z + params.floor_margin)
        & (pts[:, 2] < floor_z + params.ceiling_height)
    )
    pts = pts[vertical_keep]
    pts, plane_count = _remove_large_planes(pts, floor_z)
    pts = voxel_downsample(pts, params.voxel_size)
    if len(pts) > params.max_points:
        idx = np.linspace(0, len(pts) - 1, params.max_points, dtype=np.int64)
        pts = pts[idx]
    return pts, floor_z, plane_count


def _pca_yaw(points: np.ndarray) -> tuple[float, float]:
    xy = points[:, :2]
    if len(xy) < 3:
        return 0.0, 0.0
    centered = xy - xy.mean(axis=0)
    cov = centered.T @ centered / max(len(xy) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    yaw = float(np.arctan2(vec[1], vec[0]))
    linearity = float((vals[0] - vals[1]) / max(vals[0], 1e-6))
    return yaw, linearity


def _oriented_bbox(points: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a PCA-yaw box with size ordered as local dx, dy, dz.

    The GL renderer interprets size in the same local frame and rotates it by
    yaw around +Z, so using an axis-aligned size with a non-zero yaw makes boxes
    look mislocalized. This keeps the classifier output and visualization in the
    same lidar sensor frame: X forward, Y left, Z up, yaw in radians around +Z.
    """
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32) * 0.08
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    local = np.empty_like(points, dtype=np.float32)
    local[:, 0] = c * points[:, 0] + s * points[:, 1]
    local[:, 1] = -s * points[:, 0] + c * points[:, 1]
    local[:, 2] = points[:, 2]
    mn = local.min(axis=0)
    mx = local.max(axis=0)
    center_local = (mn + mx) * 0.5
    size = np.maximum(mx - mn, np.array([0.08, 0.08, 0.08], dtype=np.float32))
    center = np.array(
        [
            c * center_local[0] - s * center_local[1],
            s * center_local[0] + c * center_local[1],
            center_local[2],
        ],
        dtype=np.float32,
    )
    return center, size.astype(np.float32, copy=False)


def _shape_features(points: np.ndarray) -> tuple[float, float, float]:
    if len(points) < 4:
        return 1.0, 0.0, 0.0
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    vals = np.maximum(vals, 1e-9)
    linearity = float((vals[0] - vals[1]) / vals[0])
    planarity = float((vals[1] - vals[2]) / vals[0])
    scattering = float(vals[2] / vals[0])
    return linearity, planarity, scattering


def _tri_score(value: float, lo: float, mid: float, hi: float) -> float:
    if value <= lo or value >= hi:
        return 0.0
    if value == mid:
        return 1.0
    if value < mid:
        return float((value - lo) / max(mid - lo, 1e-6))
    return float((hi - value) / max(hi - mid, 1e-6))


def _band_score(value: float, lo: float, hi: float, soft: float = 0.15) -> float:
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return float(np.clip(1.0 - (lo - value) / max(soft, 1e-6), 0.0, 1.0))
    return float(np.clip(1.0 - (value - hi) / max(soft, 1e-6), 0.0, 1.0))


def _softmax(scores: np.ndarray) -> np.ndarray:
    scaled = np.asarray(scores, dtype=np.float32) * 4.0
    scaled = scaled - float(np.max(scaled))
    ex = np.exp(scaled)
    total = float(np.sum(ex))
    if total <= 1e-9:
        return np.full(len(scores), 1.0 / max(len(scores), 1), dtype=np.float32)
    return (ex / total).astype(np.float32, copy=False)


def _roi_features(points: np.ndarray, floor_z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    yaw, pca_linearity = _pca_yaw(points)
    center, size = _oriented_bbox(points, yaw)
    n = int(len(points))
    dx, dy, dz = [float(v) for v in size]
    min_xy = min(dx, dy)
    max_xy = max(dx, dy)
    mn_z = float(np.min(points[:, 2]))
    mx_z = float(np.max(points[:, 2]))
    bottom_gap = mn_z - floor_z
    top_rel = mx_z - floor_z
    rel_z = points[:, 2] - floor_z
    volume = float(np.prod(np.maximum(size, 0.12)))
    density_log = float(np.clip(np.log1p(n / max(volume, 1e-6)) / 5.0, 0.0, 1.0))
    linearity, planarity, scattering = _shape_features(points)
    linearity = max(linearity, pca_linearity)

    upper_count = int(np.count_nonzero((rel_z > 0.65) & (rel_z < 1.90)))
    torso_count = int(np.count_nonzero((rel_z > 0.45) & (rel_z < 1.50)))
    head_count = int(np.count_nonzero((rel_z > 1.05) & (rel_z < 1.95)))
    seat_count = int(np.count_nonzero((rel_z > 0.20) & (rel_z < 0.95)))
    above_seat_count = int(np.count_nonzero((rel_z > 0.55) & (rel_z < 1.95)))

    upper_ratio = upper_count / max(n, 1)
    torso_ratio = torso_count / max(n, 1)
    head_ratio = head_count / max(n, 1)
    seat_ratio = seat_count / max(n, 1)
    above_seat_ratio = above_seat_count / max(n, 1)

    # Use the torso/head band for thickness. Including the seat band makes an
    # empty chair look artificially body-like because the horizontal seat and
    # vertical chair back together fill a 3D box.
    upper_pts = points[(rel_z > 0.82) & (rel_z < 1.95)]
    if len(upper_pts) >= 4:
        upper_yaw, _ = _pca_yaw(upper_pts)
        _upper_center, upper_size = _oriented_bbox(upper_pts, upper_yaw)
        upper_min_xy = float(np.min(upper_size[:2]))
        upper_max_xy = float(np.max(upper_size[:2]))
    else:
        upper_min_xy = 0.0
        upper_max_xy = 0.0
    upper_thickness_score = _band_score(upper_min_xy, 0.18, 1.20, 0.12)

    compactness = 1.0 - np.clip((max_xy - 1.35) / 1.15, 0.0, 1.0)
    floor_contact = 1.0 - np.clip(abs(bottom_gap - 0.18) / 0.85, 0.0, 1.0)
    vertical_score = np.clip(dz / 1.55, 0.0, 1.0)
    torso_score = float(np.clip(torso_ratio / 0.34, 0.0, 1.0))
    head_score = float(np.clip(head_ratio / 0.16, 0.0, 1.0))
    above_score = float(np.clip(above_seat_ratio / 0.45, 0.0, 1.0))
    human_visible_score = float(
        np.clip(
            (0.50 * torso_score + 0.35 * head_score + 0.15 * above_score)
            * (0.20 + 0.80 * upper_thickness_score),
            0.0,
            1.0,
        )
    )
    chair_context_score = float(
        np.clip(0.45 * (seat_ratio / 0.45) + 0.35 * _band_score(max_xy, 0.45, 2.20, 0.35) + 0.20 * vertical_score, 0.0, 1.0)
    )

    feats = np.array(
        [
            n,
            dx,
            dy,
            dz,
            min_xy,
            max_xy,
            bottom_gap,
            top_rel,
            float(np.linalg.norm(center[:2])),
            density_log,
            upper_count,
            upper_ratio,
            torso_count,
            torso_ratio,
            head_count,
            head_ratio,
            seat_count,
            seat_ratio,
            above_seat_count,
            above_seat_ratio,
            linearity,
            planarity,
            scattering,
            compactness,
            floor_contact,
            vertical_score,
            human_visible_score,
            chair_context_score,
            upper_min_xy,
            upper_max_xy,
            upper_thickness_score,
        ],
        dtype=np.float32,
    )
    return feats, center, size, yaw


class CropClassifier:
    """Prototype crop classifier for seated people and occupied chairs.

    This is the baseline until there is enough captured data to train a real
    crop model. The output is deliberately multi-class so empty chairs and
    clutter are measured instead of being silently folded into "not human".
    """

    def classify(self, features: np.ndarray) -> RoiClassification:
        f = {name: float(features[i]) for i, name in enumerate(ROI_FEATURE_NAMES)}
        n = f["points"]
        min_xy = f["min_xy"]
        max_xy = f["max_xy"]
        height = f["dz"]
        bottom_gap = f["bottom_gap"]
        density = f["density_log"]
        upper_ratio = f["upper_ratio"]
        torso_ratio = f["torso_ratio"]
        head_ratio = f["head_ratio"]
        seat_ratio = f["seat_ratio"]
        planarity = f["planarity"]
        scattering = f["scattering"]
        compact = f["compactness"]
        floor_contact = f["floor_contact"]
        visible = f["human_visible_score"]
        chair_context = f["chair_context_score"]
        upper_thickness = f["upper_thickness_score"]

        enough_points = float(np.clip((n - 12.0) / 45.0, 0.0, 1.0))
        nonplanar = float(np.clip(scattering / 0.050, 0.0, 1.0))
        not_wall_sheet = 1.0 - float(np.clip((planarity - 0.72) / 0.23, 0.0, 1.0))
        upper_evidence = float(np.clip(0.65 * (torso_ratio / 0.28) + 0.35 * (head_ratio / 0.10), 0.0, 1.0))

        standing_dims = (
            _band_score(height, 1.05, 2.25, 0.25)
            * _band_score(min_xy, 0.15, 0.95, 0.20)
            * _band_score(max_xy, 0.25, 1.45, 0.30)
        )
        seated_dims = (
            _band_score(height, 0.42, 1.55, 0.22)
            * _band_score(min_xy, 0.18, 1.15, 0.20)
            * _band_score(max_xy, 0.35, 1.95, 0.35)
        )
        occupied_dims = (
            _band_score(height, 0.50, 1.95, 0.25)
            * _band_score(min_xy, 0.22, 1.65, 0.25)
            * _band_score(max_xy, 0.45, 2.55, 0.40)
        )
        empty_chair_dims = (
            _band_score(height, 0.35, 1.45, 0.20)
            * _band_score(min_xy, 0.22, 1.35, 0.20)
            * _band_score(max_xy, 0.35, 1.85, 0.30)
        )
        too_large = 1.0 - _band_score(max_xy, 0.0, 2.65, 0.45)
        too_low = 1.0 - _band_score(height, 0.38, 2.35, 0.20)

        standing = (
            0.25 * standing_dims
            + 0.22 * floor_contact
            + 0.22 * upper_evidence
            + 0.18 * density
            + 0.13 * compact
        ) * enough_points

        seated = (
            0.27 * seated_dims
            + 0.30 * visible
            + 0.18 * nonplanar
            + 0.15 * density
            + 0.10 * compact
        ) * enough_points

        occupied = (
            0.25 * occupied_dims
            + 0.28 * visible
            + 0.18 * chair_context
            + 0.10 * nonplanar
            + 0.15 * density
            + 0.04 * upper_thickness
        ) * enough_points

        empty_chair = (
            0.34 * empty_chair_dims
            + 0.24 * chair_context
            + 0.20 * (1.0 - min(visible, 1.0))
            + 0.12 * planarity
            + 0.10 * density
            + 0.12 * (1.0 - upper_thickness)
        ) * enough_points

        clutter = float(
            np.clip(
                0.30 * too_large
                + 0.22 * too_low
                + 0.18 * (1.0 - not_wall_sheet)
                + 0.15 * (1.0 - upper_evidence)
                + 0.15 * (1.0 - density),
                0.0,
                1.0,
            )
        )

        # Tie-break common indoor ambiguities before selecting a label.
        if standing > 0.62 and standing >= occupied - 0.08 and height > 1.25 and floor_contact > 0.45:
            standing += 0.10
        if empty_chair >= occupied - 0.05 and visible < 0.55:
            empty_chair += 0.10

        raw = np.array([occupied, empty_chair, standing, seated, clutter, 0.28], dtype=np.float32)
        probs = _softmax(raw)
        best_i = int(np.argmax(raw))
        label = ROI_CLASS_NAMES[best_i]
        score = float(raw[best_i])

        # Ambiguous chair/person crops are safer as uncertain than false human.
        human_best = max(float(raw[0]), float(raw[2]), float(raw[3]))
        empty_or_clutter = max(float(raw[1]), float(raw[4]))
        if score < 0.42 or (human_best > 0.40 and abs(human_best - empty_or_clutter) < 0.08):
            label = "uncertain"
            score = max(score, float(probs[-1]))

        reason = "accepted" if label in ROI_HUMAN_LABELS else "failed_classifier"
        if label == "empty_chair":
            reason = "empty_chair"
        elif label == "non_human_clutter":
            reason = "non_human_clutter"
        elif label == "uncertain":
            reason = "uncertain"
        return RoiClassification(label=label, score=score, scores=probs, reason=reason)


def _debug_item(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
    *,
    score: float,
    kind: str,
    reason: str,
    accepted: bool,
    cluster_id: int = -1,
    source: str = "cluster",
    features: Optional[np.ndarray] = None,
    class_scores: Optional[np.ndarray] = None,
) -> CandidateDebug:
    return CandidateDebug(
        center=center.astype(np.float32),
        size=size.astype(np.float32),
        yaw=yaw,
        score=float(score),
        points=int(len(points)),
        z_span=float(size[2]),
        kind=kind,
        reason=reason,
        accepted=accepted,
        cluster_id=cluster_id,
        source=source,
        features=np.asarray(features, dtype=np.float32) if features is not None else np.empty(0, dtype=np.float32),
        class_scores=np.asarray(class_scores, dtype=np.float32) if class_scores is not None else np.empty(0, dtype=np.float32),
    )


def _classify_roi(
    proposal: RoiProposal,
    floor_z: float,
    params: IndoorHumanParams,
    classifier: CropClassifier,
) -> tuple[Optional[HumanCandidate], CandidateDebug]:
    points = proposal.points
    features, center, size, yaw = _roi_features(points, floor_z)
    f = {name: float(features[i]) for i, name in enumerate(ROI_FEATURE_NAMES)}
    height = f["dz"]
    min_xy = f["min_xy"]
    max_xy = f["max_xy"]
    bottom_gap = f["bottom_gap"]
    top_rel = f["top_rel"]

    def reject(reason: str, score: float = 0.0, kind: str = "unknown", scores: Optional[np.ndarray] = None):
        return None, _debug_item(
            points,
            center,
            size,
            yaw,
            score=score,
            kind=kind,
            reason=reason,
            accepted=False,
            cluster_id=proposal.cluster_id,
            source=proposal.source,
            features=features,
            class_scores=scores,
        )

    if len(points) < params.min_cluster_points:
        return reject("too_few_points")
    if height > 2.55:
        return reject("too_tall")
    if height < 0.32:
        return reject("too_short")
    if max_xy > 3.25 or min_xy > 2.45:
        return reject("too_wide")
    if bottom_gap < -0.22 or bottom_gap > 1.55 or top_rel < 0.35:
        return reject("bad_floor_relation")

    cls = classifier.classify(features)
    if cls.label in ROI_HUMAN_LABELS:
        # Label-specific confidence floors are internal safeguards, not user
        # knobs. Seated/occupied targets are allowed to be lower and wider than
        # standing pedestrians, but empty-chair ambiguity still blocks output.
        min_score = {
            "standing_person": 0.54,
            "seated_person": 0.46,
            "occupied_chair": 0.50,
        }[cls.label]
        if cls.score < min_score:
            return reject("below_threshold", score=cls.score, kind=cls.label, scores=cls.scores)
        candidate = HumanCandidate(
            center=center.astype(np.float32),
            size=size.astype(np.float32),
            yaw=yaw,
            score=min(cls.score, 0.98),
            points=int(len(points)),
            z_span=height,
            floor_z=floor_z,
            kind=cls.label,
            features=features,
            class_scores=cls.scores,
        )
        return candidate, _debug_item(
            points,
            center,
            size,
            yaw,
            score=candidate.score,
            kind=cls.label,
            reason="accepted",
            accepted=True,
            cluster_id=proposal.cluster_id,
            source=proposal.source,
            features=features,
            class_scores=cls.scores,
        )

    return reject(cls.reason, score=cls.score, kind=cls.label, scores=cls.scores)


def _context_crop(
    seed_points: np.ndarray,
    all_points: np.ndarray,
    floor_z: float,
    *,
    xy_margin: float = 0.35,
    z_below: float = 0.18,
    z_above: float = 0.25,
) -> np.ndarray:
    mn = seed_points.min(axis=0)
    mx = seed_points.max(axis=0)
    keep = (
        (all_points[:, 0] >= mn[0] - xy_margin)
        & (all_points[:, 0] <= mx[0] + xy_margin)
        & (all_points[:, 1] >= mn[1] - xy_margin)
        & (all_points[:, 1] <= mx[1] + xy_margin)
        & (all_points[:, 2] >= max(floor_z + 0.05, mn[2] - z_below))
        & (all_points[:, 2] <= mx[2] + z_above)
    )
    crop = all_points[keep]
    return crop if len(crop) >= len(seed_points) else seed_points


def _candidate_proposals(
    cluster: np.ndarray,
    all_points: np.ndarray,
    floor_z: float,
    params: IndoorHumanParams,
    cluster_id: int,
) -> list[RoiProposal]:
    proposals = [
        RoiProposal(
            points=_context_crop(cluster, all_points, floor_z, xy_margin=0.28),
            seed_points=cluster,
            source="cluster_context",
            cluster_id=cluster_id,
        )
    ]

    mn = cluster.min(axis=0)
    mx = cluster.max(axis=0)
    size = mx - mn
    rel_z = cluster[:, 2] - floor_z

    # Seated people often merge with chair/table geometry. Re-cluster the
    # torso/head band and crop context around those seeds instead of requiring
    # the whole cluster to look like a standing body.
    upper = cluster[(rel_z > 0.38) & (rel_z < 1.95)]
    if len(upper) >= params.min_cluster_points * 2:
        try:
            labels = DBSCAN(
                eps=max(0.24, params.eps * 0.62),
                min_samples=max(5, params.min_samples - 2),
                algorithm="kd_tree",
                n_jobs=-1,
            ).fit_predict(upper)
            for label in np.unique(labels):
                if label == -1:
                    continue
                sub = upper[labels == label]
                if len(sub) < params.min_cluster_points:
                    continue
                proposals.append(
                    RoiProposal(
                        points=_context_crop(sub, all_points, floor_z, xy_margin=0.45, z_below=0.35, z_above=0.22),
                        seed_points=sub,
                        source="upper_body_context",
                        cluster_id=cluster_id,
                    )
                )
        except Exception:
            pass

    # Chair/seat proposals keep more local context and explicitly allow a wider
    # footprint. Empty chairs are classified as negatives here instead of being
    # removed before they can be measured.
    chair_like = (
        0.35 <= float(size[2]) <= 1.65
        and 0.25 <= float(min(size[0], size[1])) <= 1.55
        and 0.35 <= float(max(size[0], size[1])) <= 2.10
    )
    if chair_like:
        proposals.append(
            RoiProposal(
                points=_context_crop(cluster, all_points, floor_z, xy_margin=0.55, z_below=0.25, z_above=0.35),
                seed_points=cluster,
                source="seat_roi",
                cluster_id=cluster_id,
            )
        )

    return proposals


def _candidate_iou(a: HumanCandidate, b: HumanCandidate) -> float:
    # Conservative axis-aligned overlap used only for duplicate suppression.
    a_mn, a_mx = a.center - a.size * 0.5, a.center + a.size * 0.5
    b_mn, b_mx = b.center - b.size * 0.5, b.center + b.size * 0.5
    inter = np.maximum(0.0, np.minimum(a_mx, b_mx) - np.maximum(a_mn, b_mn))
    inter_vol = float(np.prod(inter))
    if inter_vol <= 0:
        return 0.0
    a_vol = float(np.prod(np.maximum(a.size, 1e-3)))
    b_vol = float(np.prod(np.maximum(b.size, 1e-3)))
    return inter_vol / max(a_vol + b_vol - inter_vol, 1e-6)


def _dedupe_candidates(candidates: list[HumanCandidate]) -> list[HumanCandidate]:
    kept: list[HumanCandidate] = []
    priority = {"occupied_chair": 0.04, "seated_person": 0.03, "standing_person": 0.0}
    ordered = sorted(candidates, key=lambda c: c.score + priority.get(c.kind, 0.0), reverse=True)
    for cand in ordered:
        duplicate = False
        for old in kept:
            if _candidate_iou(cand, old) > 0.28:
                duplicate = True
                break
            if float(np.linalg.norm(cand.center - old.center)) < 0.32:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    kept.sort(key=lambda c: c.score, reverse=True)
    return kept


def detect_human_candidates(
    xyz: np.ndarray,
    params: IndoorHumanParams,
    classifier: Optional[CropClassifier] = None,
) -> tuple[list[HumanCandidate], str, list[CandidateDebug]]:
    if not HAS_SKLEARN:
        return [], "sklearn missing", []
    classifier = classifier or CropClassifier()
    pts, floor_z, plane_count = _preprocess(xyz, params)
    if len(pts) < params.min_samples:
        return [], f"{len(pts):,} fg pts | floor {floor_z:.2f}m | {plane_count} planes", []

    labels = DBSCAN(
        eps=params.eps,
        min_samples=params.min_samples,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit_predict(pts)

    candidates: list[HumanCandidate] = []
    debug: list[CandidateDebug] = []
    cluster_count = 0
    roi_count = 0
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_count += 1
        cluster = pts[labels == label]
        for proposal in _candidate_proposals(cluster, pts, floor_z, params, int(label)):
            roi_count += 1
            cand, item = _classify_roi(proposal, floor_z, params, classifier)
            debug.append(item)
            if cand is not None:
                candidates.append(cand)
    candidates = _dedupe_candidates(candidates)
    status = (
        f"{len(pts):,} fg pts | {cluster_count} clusters / {roi_count} ROIs"
        f" -> {len(candidates)} human/seat candidates"
        f" | floor {floor_z:.2f}m | {plane_count} planes"
    )
    return candidates, status, debug


class HumanTrackTracker:
    def __init__(self, params: IndoorHumanParams) -> None:
        self.params = params
        self.next_id = 1
        self.tracks: dict[int, HumanTrack] = {}

    def update(self, candidates: list[HumanCandidate], dt: float) -> list[Detection]:
        if not self.tracks:
            for cand in candidates:
                self._spawn(cand)
            return self._confirmed()

        tids = list(self.tracks.keys())
        pairs: list[tuple[float, int, int]] = []
        for ci, cand in enumerate(candidates):
            for ti, tid in enumerate(tids):
                old = self.tracks[tid].candidate
                dist = float(np.linalg.norm(cand.center - old.center))
                pairs.append((dist, ci, ti))
        pairs.sort(key=lambda p: p[0])

        used_cands: set[int] = set()
        used_tracks: set[int] = set()
        for dist, ci, ti in pairs:
            if dist > self.params.association_distance:
                break
            if ci in used_cands or ti in used_tracks:
                continue
            tid = tids[ti]
            trk = self.tracks[tid]
            cand = candidates[ci]
            if dt > 0:
                trk.velocity = (cand.center - trk.candidate.center) / dt
            trk.candidate = cand
            trk.age += 1
            trk.misses = 0
            trk.history.append(True)
            trk.scores.append(cand.score)
            used_cands.add(ci)
            used_tracks.add(ti)

        for ci, cand in enumerate(candidates):
            if ci not in used_cands:
                self._spawn(cand)

        matched = {tids[ti] for ti in used_tracks}
        for tid in list(self.tracks):
            if tid in matched:
                continue
            trk = self.tracks[tid]
            trk.misses += 1
            trk.age += 1
            trk.history.append(False)
            if trk.misses > self.params.max_misses:
                self.tracks.pop(tid, None)

        return self._confirmed()

    def _spawn(self, cand: HumanCandidate) -> None:
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = HumanTrack(
            track_id=tid,
            candidate=cand,
            history=deque([True], maxlen=self.params.confirm_window),
            scores=deque([cand.score], maxlen=self.params.confirm_window),
        )

    def _confirmed(self) -> list[Detection]:
        out: list[Detection] = []
        for trk in self.tracks.values():
            if trk.confirmed_hits < self.params.confirm_hits or trk.misses > 1:
                continue
            cand = trk.candidate
            score = float(np.mean(trk.scores)) if trk.scores else cand.score
            out.append(
                Detection(
                    track_id=trk.track_id,
                    center=cand.center,
                    size=np.maximum(cand.size, np.array([0.25, 0.25, 0.55], dtype=np.float32)),
                    yaw=cand.yaw,
                    class_id=1,
                    class_name=cand.kind,
                    score=score,
                    model_score=cand.score,
                    velocity=trk.velocity.astype(np.float32),
                    age=trk.age,
                    misses=trk.misses,
                    support_points=cand.points,
                    support_z_span=cand.z_span,
                )
            )
        out.sort(key=lambda d: d.track_id)
        return out


class IndoorHumanWorker(QThread):
    result = Signal(object, float, str, object)  # (detections, elapsed_ms, status, debug)
    error = Signal(str)

    def __init__(self, params: Optional[IndoorHumanParams] = None, parent=None) -> None:
        super().__init__(parent)
        self.params = params or IndoorHumanParams()
        self.classifier = CropClassifier()
        self.tracker = HumanTrackTracker(self.params)
        self._lock = Lock()
        self._pending: Optional[np.ndarray] = None
        self._running = False
        self._last_scan_t = time.monotonic()

    def submit(self, xyz: np.ndarray) -> None:
        with self._lock:
            self._pending = xyz

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        if not HAS_SKLEARN:
            self.error.emit("sklearn missing")
            return

        self._running = True
        while self._running:
            with self._lock:
                xyz = self._pending
                self._pending = None
            if xyz is None:
                self.msleep(15)
                continue

            t0 = time.monotonic()
            candidates, detail, debug = detect_human_candidates(xyz, self.params, self.classifier)
            now = time.monotonic()
            dt = now - self._last_scan_t
            self._last_scan_t = now
            tracked = self.tracker.update(candidates, dt)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            status = f"indoor {elapsed_ms:.0f}ms | {detail} -> {len(tracked)} confirmed"
            self.result.emit(tracked, elapsed_ms, status, debug)
