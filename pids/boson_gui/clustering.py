"""DBSCAN clustering of a lidar point cloud + greedy track association.

Includes a `ClusterWorker` QThread that runs DBSCAN off the GUI thread so the
viewer stays responsive at the lidar's full frame rate. The worker holds at
most one pending scan and drops older ones — clustering output rate is
whatever the worker can sustain (typically 5–10 Hz with voxel downsampling).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except Exception:
    DBSCAN = None  # type: ignore
    HAS_SKLEARN = False


@dataclass
class Cluster:
    track_id: int
    points: np.ndarray  # (N, 3)
    centroid: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    age: int = 1

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.centroid))

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass
class ClusterParams:
    eps: float = 0.5         # m
    min_samples: int = 10
    z_min: float = -1.0      # ground filter (m, lidar frame: Z up)
    z_max: float = 3.0
    range_min: float = 1.0   # ignore points too close (sensor housing)
    range_max: float = 30.0
    voxel_size: float = 0.10  # m; 0 disables voxel downsampling
    max_points: int = 8000   # hard cap after voxel downsample


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one representative point per occupied voxel (its centroid)."""
    if voxel <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    # Pack 3 ints into one for unique
    k = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, inv = np.unique(k, return_inverse=True)
    n = inv.max() + 1
    sums = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(sums, inv, pts)
    np.add.at(counts, inv, 1)
    return (sums / counts[:, None]).astype(np.float32)


def cluster_xyz(xyz: np.ndarray, params: ClusterParams) -> list[Cluster]:
    """Run DBSCAN on a (H, W, 3) or (N, 3) point cloud, return clusters."""
    if not HAS_SKLEARN:
        return []
    pts = xyz.reshape(-1, 3).astype(np.float32)
    rng = np.linalg.norm(pts, axis=1)
    keep = (
        (rng > params.range_min)
        & (rng < params.range_max)
        & (pts[:, 2] > params.z_min)
        & (pts[:, 2] < params.z_max)
    )
    pts = pts[keep]
    if len(pts) < params.min_samples:
        return []

    pts = voxel_downsample(pts, params.voxel_size)
    if len(pts) > params.max_points:
        idx = np.random.default_rng(0).choice(len(pts), params.max_points, replace=False)
        pts = pts[idx]

    labels = DBSCAN(
        eps=params.eps, min_samples=params.min_samples,
        algorithm="kd_tree", n_jobs=-1,
    ).fit_predict(pts)
    clusters: list[Cluster] = []
    for cid in np.unique(labels):
        if cid == -1:  # noise
            continue
        cpts = pts[labels == cid]
        clusters.append(
            Cluster(
                track_id=0,
                points=cpts,
                centroid=cpts.mean(axis=0),
                bbox_min=cpts.min(axis=0),
                bbox_max=cpts.max(axis=0),
            )
        )
    return clusters


class Tracker:
    """Greedy nearest-centroid association across frames.

    Stable IDs survive temporary misses up to `max_age` frames; velocity is
    estimated from centroid delta / dt.
    """

    def __init__(self, max_distance: float = 1.5, max_age: int = 5) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, Cluster] = {}
        self.miss_count: dict[int, int] = {}

    def update(self, raw: list[Cluster], dt: float) -> list[Cluster]:
        if not raw:
            for tid in list(self.tracks):
                self.miss_count[tid] = self.miss_count.get(tid, 0) + 1
                if self.miss_count[tid] > self.max_age:
                    self.tracks.pop(tid, None)
                    self.miss_count.pop(tid, None)
            return []

        if not self.tracks:
            for c in raw:
                c.track_id = self.next_id
                self.next_id += 1
                self.tracks[c.track_id] = c
                self.miss_count[c.track_id] = 0
            return list(raw)

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
        new_centroids = np.array([c.centroid for c in raw])

        # All pairwise (new, track) distances; take greedy nearest under threshold.
        d = np.linalg.norm(new_centroids[:, None] - track_centroids[None, :], axis=2)
        pairs = sorted(
            ((d[i, j], i, j) for i in range(len(raw)) for j in range(len(track_ids))),
            key=lambda t: t[0],
        )
        used_new: set[int] = set()
        used_trk: set[int] = set()
        for dist, ci, ti in pairs:
            if dist > self.max_distance:
                break
            if ci in used_new or ti in used_trk:
                continue
            tid = track_ids[ti]
            old = self.tracks[tid]
            new = raw[ci]
            new.track_id = tid
            new.age = old.age + 1
            if dt > 0:
                new.velocity = (new.centroid - old.centroid) / dt
            self.tracks[tid] = new
            self.miss_count[tid] = 0
            used_new.add(ci)
            used_trk.add(ti)

        # Unmatched new clusters → spawn new tracks.
        for ci, c in enumerate(raw):
            if ci in used_new:
                continue
            c.track_id = self.next_id
            self.next_id += 1
            self.tracks[c.track_id] = c
            self.miss_count[c.track_id] = 0

        # Unmatched tracks → age out.
        for ti, tid in enumerate(track_ids):
            if ti in used_trk:
                continue
            self.miss_count[tid] = self.miss_count.get(tid, 0) + 1
            if self.miss_count[tid] > self.max_age:
                self.tracks.pop(tid, None)
                self.miss_count.pop(tid, None)

        return list(raw)


class ClusterWorker(QThread):
    """Background DBSCAN worker.

    Caller pushes the latest XYZ via `submit(xyz)`; the worker keeps only the
    most recent submission and drops anything that piled up while it was busy.
    Emits `result(clusters, dt_ms)` after each cluster pass.
    """

    result = Signal(object, float)  # (list[Cluster], elapsed_ms)

    def __init__(self, params: ClusterParams, parent=None) -> None:
        super().__init__(parent)
        self.params = params
        self.tracker = Tracker(max_distance=1.5, max_age=5)
        self._lock = Lock()
        self._pending: Optional[np.ndarray] = None
        self._running = False
        self._last_scan_t = time.monotonic()

    def submit(self, xyz: np.ndarray) -> None:
        with self._lock:
            self._pending = xyz  # newest wins; older drops on the floor

    def update_params(self, params: ClusterParams) -> None:
        self.params = params

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        self._running = True
        while self._running:
            with self._lock:
                xyz = self._pending
                self._pending = None
            if xyz is None:
                self.msleep(15)
                continue
            t0 = time.monotonic()
            raw = cluster_xyz(xyz, self.params)
            now = time.monotonic()
            dt = now - self._last_scan_t
            self._last_scan_t = now
            tracked = self.tracker.update(raw, dt)
            elapsed_ms = (now - t0) * 1000.0
            self.result.emit(tracked, elapsed_ms)
