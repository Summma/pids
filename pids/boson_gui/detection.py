"""Jetson PointPillars detector client.

The desktop GUI keeps lidar capture and rendering local, but ships each scan's
XYZ + intensity points to a Jetson-side ZMQ server for neural inference.  This
module mirrors ``ClusterWorker``: callers submit the newest frame, the worker
drops stale frames while busy, and results arrive on a Qt signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from lidar import LidarFrame

try:
    import msgpack

    try:
        import msgpack_numpy as _msgpack_numpy

        _msgpack_numpy.patch()
        HAS_MSGPACK_NUMPY = True
    except Exception:
        HAS_MSGPACK_NUMPY = False
    HAS_MSGPACK = True
except Exception:
    msgpack = None  # type: ignore
    HAS_MSGPACK = False
    HAS_MSGPACK_NUMPY = False

try:
    import zmq

    HAS_ZMQ = True
except Exception:
    zmq = None  # type: ignore
    HAS_ZMQ = False


KITTI_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
DEFAULT_ENDPOINT = "tcp://10.1.63.30:5555"
N_CLASSES = len(KITTI_CLASS_NAMES)
KITTI_POINT_CLOUD_RANGE = np.array([0.0, -39.68, -3.0, 69.12, 39.68, 1.0], dtype=np.float32)
KITTI_GROUND_Z_M = -1.60
DETECTOR_SCORE_THRESHOLD = 0.12
TRACK_CONFIRM_SCORE = 0.58
TRACK_CONFIRM_HITS = 2
TRACK_IMMEDIATE_SCORE = 0.82


@dataclass
class PreparedPointPillarsFrame:
    points: np.ndarray  # (N, 4), z-shifted into the KITTI training height distribution
    original_xyz: np.ndarray  # (N, 3), same selected points in the native lidar frame
    z_shift: float


def _uniform_class_probs() -> np.ndarray:
    return np.full(N_CLASSES, 1.0 / N_CLASSES, dtype=np.float32)


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.shape != (N_CLASSES,):
        return _uniform_class_probs()
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(probs))
    if total <= 1e-6:
        return _uniform_class_probs()
    return (probs / total).astype(np.float32, copy=False)


@dataclass
class Detection:
    track_id: int
    center: np.ndarray  # (3,) lidar frame: X forward, Y left, Z up
    size: np.ndarray  # (3,) dx, dy, dz in metres
    yaw: float  # radians around +Z
    class_id: int
    class_name: str
    score: float  # temporal posterior after tracking; raw model score before tracking
    model_score: float = 0.0
    class_probs: np.ndarray = field(default_factory=_uniform_class_probs)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    age: int = 1
    misses: int = 0
    support_points: int = 0
    support_z_span: float = 0.0

    def __post_init__(self) -> None:
        if self.model_score <= 0.0:
            self.model_score = self.score
        self.class_probs = _normalize_probs(self.class_probs)

    @property
    def n_points(self) -> int:
        return 0

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.center))

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def label(self) -> str:
        return self.class_name or class_name(self.class_id)


class DetectionTracker:
    """Nearest-centre tracking with Bayesian class posteriors.

    The neural detector is still the per-frame measurement source. This tracker
    prevents each frame from becoming a fresh UI truth by integrating class
    evidence over time and only returning confirmed tracks.
    """

    def __init__(
        self,
        max_distance: float = 2.0,
        max_age: int = 5,
        prior_decay: float = 0.03,
        miss_decay: float = 0.12,
    ) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self.confirm_score = TRACK_CONFIRM_SCORE
        self.confirm_hits = TRACK_CONFIRM_HITS
        self.immediate_score = TRACK_IMMEDIATE_SCORE
        self.prior_decay = prior_decay
        self.miss_decay = miss_decay
        self.next_id = 1
        self.tracks: dict[int, Detection] = {}
        self.miss_count: dict[int, int] = {}

    def update(self, raw: list[Detection], dt: float) -> list[Detection]:
        if not raw:
            self._age_unmatched(set())
            return self._confirmed_tracks()

        if not self.tracks:
            for det in raw:
                self._spawn(det)
            return self._confirmed_tracks()

        track_ids = list(self.tracks.keys())
        pairs: list[tuple[float, int, int]] = []
        for di, det in enumerate(raw):
            for ti, tid in enumerate(track_ids):
                old = self.tracks[tid]
                dist = float(np.linalg.norm(det.center - old.center))
                pairs.append((dist, di, ti))
        pairs.sort(key=lambda x: x[0])

        used_det: set[int] = set()
        used_track_idx: set[int] = set()
        for dist, di, ti in pairs:
            if dist > self.max_distance:
                break
            if di in used_det or ti in used_track_idx:
                continue
            tid = track_ids[ti]
            old = self.tracks[tid]
            det = raw[di]
            det.track_id = tid
            det.age = old.age + 1
            if dt > 0:
                det.velocity = (det.center - old.center) / dt
            det.misses = 0
            det.model_score = det.model_score or det.score
            det.class_probs = self._bayes_update(old.class_probs, det)
            det.class_id = int(np.argmax(det.class_probs))
            det.class_name = class_name(det.class_id)
            det.score = float(det.class_probs[det.class_id])
            self.tracks[tid] = det
            self.miss_count[tid] = 0
            used_det.add(di)
            used_track_idx.add(ti)

        for di, det in enumerate(raw):
            if di not in used_det:
                self._spawn(det)

        matched_ids = {track_ids[ti] for ti in used_track_idx}
        self._age_unmatched(matched_ids)
        return self._confirmed_tracks()

    def _spawn(self, det: Detection) -> None:
        det.track_id = self.next_id
        self.next_id += 1
        det.model_score = det.model_score or det.score
        det.class_probs = self._measurement_likelihood(det)
        det.class_id = int(np.argmax(det.class_probs))
        det.class_name = class_name(det.class_id)
        det.score = float(det.class_probs[det.class_id])
        det.misses = 0
        self.tracks[det.track_id] = det
        self.miss_count[det.track_id] = 0

    def _age_unmatched(self, matched_ids: set[int]) -> None:
        for tid in list(self.tracks):
            if tid in matched_ids:
                continue
            self.miss_count[tid] = self.miss_count.get(tid, 0) + 1
            det = self.tracks[tid]
            det.misses = self.miss_count[tid]
            det.class_probs = _normalize_probs(
                (1.0 - self.miss_decay) * det.class_probs
                + self.miss_decay * _uniform_class_probs()
            )
            det.class_id = int(np.argmax(det.class_probs))
            det.class_name = class_name(det.class_id)
            det.score = float(det.class_probs[det.class_id])
            if self.miss_count[tid] > self.max_age:
                self.tracks.pop(tid, None)
                self.miss_count.pop(tid, None)

    def _confirmed_tracks(self) -> list[Detection]:
        out: list[Detection] = []
        for det in self.tracks.values():
            if det.score >= self.immediate_score or (
                det.age >= self.confirm_hits and det.score >= self.confirm_score
            ):
                out.append(det)
        out.sort(key=lambda d: d.track_id)
        return out

    def _bayes_update(self, prior: np.ndarray, det: Detection) -> np.ndarray:
        prior = _normalize_probs(
            (1.0 - self.prior_decay) * prior + self.prior_decay * _uniform_class_probs()
        )
        likelihood = self._measurement_likelihood(det)
        return _normalize_probs(prior * likelihood)

    @staticmethod
    def _measurement_likelihood(det: Detection) -> np.ndarray:
        probs = _uniform_class_probs()
        if 0 <= det.class_id < N_CLASSES:
            strength = float(np.clip(det.model_score, 0.01, 0.95))
            probs = probs * (1.0 - strength)
            probs[det.class_id] += strength
        return _normalize_probs(probs)


def class_name(class_id: int) -> str:
    if 0 <= class_id < len(KITTI_CLASS_NAMES):
        return KITTI_CLASS_NAMES[class_id]
    return f"class {class_id}"


def frame_to_kitti_points(
    frame: LidarFrame,
    *,
    range_min: float = 0.3,
    range_max: float = 80.0,
    max_points: int = 30_000,
) -> np.ndarray:
    return prepare_pointpillars_frame(
        frame,
        range_min=range_min,
        range_max=range_max,
        max_points=max_points,
    ).points


def prepare_pointpillars_frame(
    frame: LidarFrame,
    *,
    range_min: float = 0.3,
    range_max: float = 80.0,
    max_points: int = 80_000,
) -> PreparedPointPillarsFrame:
    """Convert an Ouster frame to KITTI-style ``(x, y, z, intensity)`` points.

    Ouster and KITTI lidar axes are both X-forward/Y-left/Z-up, but the model
    was trained with KITTI's LiDAR height above the road. Estimate the local
    ground height and shift Z before inference; returned detections are shifted
    back to the native lidar frame before rendering.
    """

    if frame.xyz is None:
        empty = np.empty((0, 4), dtype=np.float32)
        return PreparedPointPillarsFrame(empty, np.empty((0, 3), dtype=np.float32), 0.0)

    xyz = frame.xyz.reshape(-1, 3).astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    rng = np.linalg.norm(xyz, axis=1)
    base_keep = (
        finite
        & (rng >= range_min)
        & (rng <= range_max)
        & (xyz[:, 0] >= KITTI_POINT_CLOUD_RANGE[0])
        & (xyz[:, 0] < KITTI_POINT_CLOUD_RANGE[3])
        & (xyz[:, 1] >= KITTI_POINT_CLOUD_RANGE[1])
        & (xyz[:, 1] < KITTI_POINT_CLOUD_RANGE[4])
    )
    z_shift = _estimate_kitti_z_shift(xyz[base_keep])
    shifted_z = xyz[:, 2] + z_shift
    keep = (
        base_keep
        & (shifted_z >= KITTI_POINT_CLOUD_RANGE[2])
        & (shifted_z < KITTI_POINT_CLOUD_RANGE[5])
    )
    if not np.any(keep):
        empty = np.empty((0, 4), dtype=np.float32)
        return PreparedPointPillarsFrame(empty, np.empty((0, 3), dtype=np.float32), z_shift)

    original_xyz = xyz[keep].copy()
    pts = original_xyz.copy()
    pts[:, 2] += z_shift
    intensity = _normalized_intensity(frame, keep)
    if len(pts) > max_points:
        idx = _balanced_sample_indices(pts, max_points)
        pts = pts[idx]
        original_xyz = original_xyz[idx]
        intensity = intensity[idx]

    points = np.column_stack((pts, intensity)).astype(np.float32, copy=False)
    return PreparedPointPillarsFrame(points, original_xyz, z_shift)


def _estimate_kitti_z_shift(xyz: np.ndarray) -> float:
    if len(xyz) < 256:
        return 0.0
    near = xyz[(xyz[:, 0] > 2.0) & (xyz[:, 0] < 35.0) & (np.abs(xyz[:, 1]) < 15.0)]
    if len(near) < 256:
        near = xyz
    ground_z = float(np.percentile(near[:, 2], 5.0))
    shift = KITTI_GROUND_Z_M - ground_z
    return float(np.clip(shift, -2.5, 1.0))


def _balanced_sample_indices(points: np.ndarray, max_points: int) -> np.ndarray:
    """Keep dense near/front returns while sampling far background deterministically."""
    if len(points) <= max_points:
        return np.arange(len(points), dtype=np.int64)

    near = np.flatnonzero((points[:, 0] < 35.0) & (np.abs(points[:, 1]) < 18.0))
    far = np.setdiff1d(np.arange(len(points), dtype=np.int64), near, assume_unique=True)
    near_budget = min(len(near), int(max_points * 0.75))
    far_budget = max_points - near_budget

    parts = []
    if near_budget > 0:
        parts.append(near[np.linspace(0, len(near) - 1, near_budget, dtype=np.int64)])
    if far_budget > 0 and len(far) > 0:
        parts.append(far[np.linspace(0, len(far) - 1, min(far_budget, len(far)), dtype=np.int64)])
    return np.sort(np.concatenate(parts)).astype(np.int64, copy=False)


def _normalized_intensity(frame: LidarFrame, keep: np.ndarray) -> np.ndarray:
    img = frame.signal_img
    if img is None:
        img = frame.reflectivity_img
    if img is None:
        return np.zeros(int(np.count_nonzero(keep)), dtype=np.float32)

    vals = img.reshape(-1)[keep].astype(np.float32, copy=False)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    if vals.size == 0:
        return vals
    if float(np.nanmax(vals)) <= 1.0:
        return np.clip(vals, 0.0, 1.0)

    positive = vals[vals > 0]
    hi = float(np.percentile(positive, 99.5)) if positive.size else float(np.max(vals))
    if hi <= 1e-6:
        return np.zeros_like(vals, dtype=np.float32)
    return np.clip(vals / hi, 0.0, 1.0).astype(np.float32, copy=False)


def encode_request(points: np.ndarray, seq: int) -> bytes:
    if not HAS_MSGPACK:
        raise RuntimeError("msgpack not installed")
    payload = {
        "type": "scan",
        "seq": seq,
        "points": points if HAS_MSGPACK_NUMPY else points.tobytes(),
        "shape": points.shape,
        "dtype": "float32",
        "encoding": "msgpack-numpy" if HAS_MSGPACK_NUMPY else "raw",
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_response(blob: bytes) -> dict:
    if not HAS_MSGPACK:
        raise RuntimeError("msgpack not installed")
    return msgpack.unpackb(blob, raw=False)


def _score_floor(class_id: int, range_m: float) -> float:
    base = {
        0: 0.40,  # Car
        1: 0.24,  # Pedestrian
        2: 0.32,  # Cyclist
    }.get(class_id, DETECTOR_SCORE_THRESHOLD)
    if range_m > 35.0:
        base += 0.08
    return base


def _passes_geometry(class_id: int, size: np.ndarray) -> bool:
    dx, dy, dz = [float(v) for v in size]
    footprint = max(dx, dy)
    if class_id == 1:  # Pedestrian
        return 0.6 <= dz <= 2.5 and 0.15 <= min(dx, dy) and footprint <= 1.6
    if class_id == 2:  # Cyclist
        return 0.8 <= dz <= 2.4 and 0.25 <= min(dx, dy) and footprint <= 2.5
    if class_id == 0:  # Car
        return 0.8 <= dz <= 3.2 and 1.0 <= footprint <= 7.0
    return True


def _box_support(
    points_xyz: Optional[np.ndarray],
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> tuple[int, float]:
    if points_xyz is None or len(points_xyz) == 0:
        return -1, 0.0
    delta = points_xyz - center
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    local_x = c * delta[:, 0] + s * delta[:, 1]
    local_y = -s * delta[:, 0] + c * delta[:, 1]
    local_z = delta[:, 2]
    half = np.maximum(size * 0.5, np.array([0.12, 0.12, 0.25], dtype=np.float32))
    half += np.array([0.25, 0.25, 0.25], dtype=np.float32)
    inside = (
        (np.abs(local_x) <= half[0])
        & (np.abs(local_y) <= half[1])
        & (np.abs(local_z) <= half[2])
    )
    z_vals = local_z[inside]
    if z_vals.size == 0:
        return 0, 0.0
    return int(z_vals.size), float(np.percentile(z_vals, 95) - np.percentile(z_vals, 5))


def _passes_support(class_id: int, score: float, support_points: int, z_span: float) -> bool:
    if support_points < 0:
        return True
    if class_id == 1:  # Pedestrian
        if score >= 0.55:
            return support_points >= 3 and z_span >= 0.20
        if score >= 0.32:
            return support_points >= 5 and z_span >= 0.25
        return support_points >= 8 and z_span >= 0.32
    if class_id == 2:  # Cyclist
        return support_points >= 8 and z_span >= 0.35
    if class_id == 0:  # Car
        return support_points >= 18 and z_span >= 0.35
    return support_points >= 4


def detections_from_payload(
    payload: dict,
    score_threshold: float,
    *,
    points_xyz: Optional[np.ndarray] = None,
    z_shift: float = 0.0,
) -> list[Detection]:
    out: list[Detection] = []
    for item in payload.get("detections", []):
        score = float(item.get("score", 0.0))
        center = np.asarray(item.get("center", item.get("translation", [0, 0, 0])), dtype=np.float32)
        size = np.asarray(item.get("size", item.get("dimensions", [0, 0, 0])), dtype=np.float32)
        if center.shape != (3,) or size.shape != (3,):
            continue
        center = center.copy()
        center[2] -= float(z_shift)
        class_id = int(item.get("class_id", item.get("label", -1)))
        if score < max(score_threshold, _score_floor(class_id, float(np.linalg.norm(center[:2])))):
            continue
        if not _passes_geometry(class_id, size):
            continue
        support_points, support_z_span = _box_support(points_xyz, center, size, float(item.get("yaw", item.get("heading", 0.0))))
        if not _passes_support(class_id, score, support_points, support_z_span):
            continue
        name = str(item.get("class_name", item.get("name", class_name(class_id))))
        out.append(
            Detection(
                track_id=0,
                center=center,
                size=size,
                yaw=float(item.get("yaw", item.get("heading", 0.0))),
                class_id=class_id,
                class_name=name,
                score=score,
                model_score=score,
                support_points=support_points,
                support_z_span=support_z_span,
            )
        )
    return out


class DetectorWorker(QThread):
    """Background ZMQ client for Jetson PointPillars inference."""

    result = Signal(object, float, str)  # (list[Detection], elapsed_ms, status)
    error = Signal(str)

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout_ms: int = 1500,
        max_points: int = 80_000,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.score_threshold = DETECTOR_SCORE_THRESHOLD
        self.max_points = max_points
        self.tracker = DetectionTracker(max_distance=3.0, max_age=5)
        self._lock = Lock()
        self._pending: Optional[LidarFrame] = None
        self._running = False
        self._seq = 0
        self._last_scan_t = time.monotonic()

    def submit(self, frame: LidarFrame) -> None:
        with self._lock:
            self._pending = frame

    def update_endpoint(self, endpoint: str) -> None:
        with self._lock:
            self.endpoint = endpoint

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        if not HAS_ZMQ:
            self.error.emit("pyzmq not installed")
            return
        if not HAS_MSGPACK:
            self.error.emit("msgpack not installed")
            return

        ctx = zmq.Context.instance()
        sock = None
        sock_endpoint = None
        self._running = True

        while self._running:
            with self._lock:
                frame = self._pending
                self._pending = None
                endpoint = self.endpoint
                score_threshold = self.score_threshold
            if frame is None:
                self.msleep(10)
                continue

            prepared = prepare_pointpillars_frame(frame, max_points=self.max_points)
            if len(prepared.points) == 0:
                continue

            try:
                if sock is None or sock_endpoint != endpoint:
                    if sock is not None:
                        sock.close(0)
                    sock = ctx.socket(zmq.REQ)
                    sock.setsockopt(zmq.LINGER, 0)
                    sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                    sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                    sock.connect(endpoint)
                    sock_endpoint = endpoint

                self._seq += 1
                req = encode_request(prepared.points, self._seq)
                t0 = time.monotonic()
                sock.send(req)
                reply = sock.recv()
                now = time.monotonic()
                payload = decode_response(reply)
                if payload.get("type") == "error":
                    raise RuntimeError(str(payload.get("message", "detector error")))
                raw = detections_from_payload(
                    payload,
                    score_threshold,
                    points_xyz=prepared.original_xyz,
                    z_shift=prepared.z_shift,
                )
                dt = now - self._last_scan_t
                self._last_scan_t = now
                tracked = self.tracker.update(raw, dt)
                elapsed_ms = (now - t0) * 1000.0
                model_ms = payload.get("model_ms")
                if model_ms is not None:
                    status = (
                        f"PointPillars {float(model_ms):.0f}ms model / {elapsed_ms:.0f}ms RTT"
                        f"  |  {len(raw)} raw -> {len(tracked)} confirmed"
                    )
                else:
                    status = (
                        f"PointPillars {elapsed_ms:.0f}ms RTT"
                        f"  |  {len(raw)} raw -> {len(tracked)} confirmed"
                    )
                self.result.emit(tracked, elapsed_ms, status)
            except Exception as exc:
                if sock is not None:
                    sock.close(0)
                    sock = None
                    sock_endpoint = None
                self.error.emit(str(exc))

        if sock is not None:
            sock.close(0)
