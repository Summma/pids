"""DBSCAN clustering of a lidar point cloud + greedy track association."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

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
    range_max: float = 50.0
    max_points: int = 30000  # subsample cap for DBSCAN speed


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
    if len(pts) > params.max_points:
        idx = np.random.default_rng(0).choice(len(pts), params.max_points, replace=False)
        pts = pts[idx]

    labels = DBSCAN(eps=params.eps, min_samples=params.min_samples, n_jobs=-1).fit_predict(pts)
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
