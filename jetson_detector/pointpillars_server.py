#!/usr/bin/env python3
"""ZMQ PointPillars inference server for the Jetson.

Protocol:
  request: msgpack {type:"scan", seq:int, points:(N,4) float32}
  reply:   msgpack {type:"detections", detections:[{center,size,yaw,class_id,class_name,score}], model_ms:float}

Run with OpenPCDet:
  python3 pointpillars_server.py \
      --cfg ~/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
      --ckpt ~/OpenPCDet/checkpoints/pointpillar_7728.pth

Run without OpenPCDet for laptop/client protocol testing:
  python3 pointpillars_server.py --mock
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import msgpack
    import msgpack_numpy as msgpack_numpy

    msgpack_numpy.patch()
except Exception as exc:  # pragma: no cover - startup failure path
    raise SystemExit(f"msgpack and msgpack-numpy are required: {exc}")

try:
    import zmq
except Exception as exc:  # pragma: no cover - startup failure path
    raise SystemExit(f"pyzmq is required: {exc}")


KITTI_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
MIN_SCORE_BY_CLASS = {
    0: 0.40,  # Car
    1: 0.24,  # Pedestrian
    2: 0.32,  # Cyclist
}
MAX_RETURNED_DETECTIONS = 80


def ensure_spconv_importable() -> None:
    """Use real spconv when present, otherwise fall back to the PointPillars shim.

    OpenPCDet imports spconv unconditionally even for PointPillars, although the
    KITTI PointPillars config only needs voxelization and dense 2D convolutions.
    Jetson/aarch64 CUDA wheels for spconv are not available on PyPI, so the
    local shim keeps this server deployable for the PointPillars-only path.
    """

    try:
        import spconv  # noqa: F401

        return
    except Exception:
        pass

    vendor = Path(__file__).resolve().parent / "vendor"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))

    try:
        import spconv  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "spconv is not installed and the bundled PointPillars shim could not be loaded"
        ) from exc


def decode_points(req: dict[str, Any]) -> np.ndarray:
    points = req.get("points")
    if isinstance(points, np.ndarray):
        return np.asarray(points, dtype=np.float32)

    if isinstance(points, (bytes, bytearray)):
        shape = tuple(req.get("shape", (0, 4)))
        dtype = np.dtype(req.get("dtype", "float32"))
        arr = np.frombuffer(points, dtype=dtype).reshape(shape)
        return np.asarray(arr, dtype=np.float32)

    raise ValueError("request missing points array")


def score_floor(class_id: int, center: np.ndarray) -> float:
    floor = MIN_SCORE_BY_CLASS.get(class_id, 0.12)
    if float(np.linalg.norm(center[:2])) > 35.0:
        floor += 0.08
    return floor


def passes_geometry(class_id: int, size: np.ndarray) -> bool:
    dx, dy, dz = [float(v) for v in size]
    footprint = max(dx, dy)
    if class_id == 1:
        return 0.6 <= dz <= 2.5 and 0.15 <= min(dx, dy) and footprint <= 1.6
    if class_id == 2:
        return 0.8 <= dz <= 2.4 and 0.25 <= min(dx, dy) and footprint <= 2.5
    if class_id == 0:
        return 0.8 <= dz <= 3.2 and 1.0 <= footprint <= 7.0
    return True


def box_support(points_xyz: np.ndarray, center: np.ndarray, size: np.ndarray, yaw: float) -> tuple[int, float]:
    if len(points_xyz) == 0:
        return 0, 0.0
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


def passes_support(class_id: int, score: float, support_points: int, z_span: float) -> bool:
    if class_id == 1:
        if score >= 0.55:
            return support_points >= 3 and z_span >= 0.20
        if score >= 0.32:
            return support_points >= 5 and z_span >= 0.25
        return support_points >= 8 and z_span >= 0.32
    if class_id == 2:
        return support_points >= 8 and z_span >= 0.35
    if class_id == 0:
        return support_points >= 18 and z_span >= 0.35
    return support_points >= 4


def filter_detections(detections: list[dict[str, Any]], points: np.ndarray) -> list[dict[str, Any]]:
    points_xyz = points[:, :3]
    kept: list[dict[str, Any]] = []
    for det in detections:
        center = np.asarray(det.get("center", [0, 0, 0]), dtype=np.float32)
        size = np.asarray(det.get("size", [0, 0, 0]), dtype=np.float32)
        if center.shape != (3,) or size.shape != (3,):
            continue
        class_id = int(det.get("class_id", -1))
        score = float(det.get("score", 0.0))
        if score < score_floor(class_id, center):
            continue
        if not passes_geometry(class_id, size):
            continue
        support_points, support_z_span = box_support(
            points_xyz, center, size, float(det.get("yaw", 0.0))
        )
        if not passes_support(class_id, score, support_points, support_z_span):
            continue
        det = dict(det)
        det["support_points"] = support_points
        det["support_z_span"] = support_z_span
        kept.append(det)

    kept.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    return kept[:MAX_RETURNED_DETECTIONS]


class MockDetector:
    """Cheap protocol tester when OpenPCDet is not installed yet."""

    def infer(self, points: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        t0 = time.monotonic()
        if len(points) == 0:
            return [], 0.0
        # Pick a stable centre in front of the sensor so the renderer path can
        # be tested without model weights.
        fwd = points[(points[:, 0] > 2.0) & (np.abs(points[:, 1]) < 4.0)]
        if len(fwd) == 0:
            return [], (time.monotonic() - t0) * 1000.0
        centre = np.median(fwd[:, :3], axis=0)
        centre[2] = max(float(centre[2]), 0.8)
        det = {
            "center": centre.astype(np.float32),
            "size": np.array([0.7, 0.7, 1.7], dtype=np.float32),
            "yaw": 0.0,
            "class_id": 1,
            "class_name": "Pedestrian",
            "score": 0.55,
        }
        return [det], (time.monotonic() - t0) * 1000.0


class LivePointPillarDataset:
    """Minimal OpenPCDet dataset interface for single-frame PointPillars inference."""

    def __init__(self, dataset_cfg: Any, class_names: Iterable[str]) -> None:
        import spconv

        self.dataset_cfg = dataset_cfg
        self.class_names = tuple(class_names)
        self.point_cloud_range = np.asarray(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.depth_downsample_factor = None

        encoding = dataset_cfg.POINT_FEATURE_ENCODING
        self.used_feature_list = list(encoding.used_feature_list)
        self.src_feature_list = list(encoding.src_feature_list)

        voxel_cfg = None
        for cur_cfg in dataset_cfg.DATA_PROCESSOR:
            if cur_cfg.NAME == "transform_points_to_voxels":
                voxel_cfg = cur_cfg
                break
        if voxel_cfg is None:
            raise ValueError("DATA_PROCESSOR missing transform_points_to_voxels")

        self.voxel_size = np.asarray(voxel_cfg.VOXEL_SIZE, dtype=np.float32)
        xyz_range = self.point_cloud_range[3:6] - self.point_cloud_range[:3]
        self.grid_size = np.round(xyz_range / self.voxel_size).astype(np.int64)
        self.max_num_points = int(voxel_cfg.MAX_POINTS_PER_VOXEL)
        self.max_voxels = int(voxel_cfg.MAX_NUMBER_OF_VOXELS["test"])
        self.voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )
        self.point_feature_encoder = self

    @property
    def num_point_features(self) -> int:
        return len(self.used_feature_list)

    def prepare_points(self, points: np.ndarray) -> dict[str, Any]:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < len(self.src_feature_list):
            raise ValueError(f"expected points shape (N,{len(self.src_feature_list)}+), got {points.shape}")

        points = self._select_features(points)
        keep = np.all(
            (points[:, :3] >= self.point_cloud_range[:3])
            & (points[:, :3] < self.point_cloud_range[3:6]),
            axis=1,
        )
        points = points[keep]
        voxels, coords, num_points = self.voxel_generator.generate(points)
        return {
            "frame_id": np.array(0),
            "points": points,
            "voxels": voxels,
            "voxel_coords": coords,
            "voxel_num_points": num_points,
            "use_lead_xyz": np.array(True),
        }

    def _select_features(self, points: np.ndarray) -> np.ndarray:
        if self.used_feature_list == self.src_feature_list[: len(self.used_feature_list)]:
            return points[:, : len(self.used_feature_list)].astype(np.float32, copy=False)

        cols = []
        for name in self.used_feature_list:
            if name not in self.src_feature_list:
                raise ValueError(f"feature {name!r} not present in source feature list")
            cols.append(points[:, self.src_feature_list.index(name)])
        return np.stack(cols, axis=1).astype(np.float32, copy=False)

    @staticmethod
    def collate_batch(batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch_list) != 1:
            raise ValueError("LivePointPillarDataset only supports batch size 1")
        sample = batch_list[0]
        coords = np.pad(sample["voxel_coords"], ((0, 0), (1, 0)), mode="constant", constant_values=0)
        return {
            "frame_id": np.array([sample["frame_id"]]),
            "points": np.pad(sample["points"], ((0, 0), (1, 0)), mode="constant", constant_values=0),
            "voxels": sample["voxels"],
            "voxel_coords": coords,
            "voxel_num_points": sample["voxel_num_points"],
            "use_lead_xyz": np.array([sample["use_lead_xyz"]]),
            "batch_size": 1,
        }


class OpenPCDetPointPillars:
    def __init__(
        self,
        cfg_file: Path,
        ckpt: Path,
        openpcdet_dir: Path | None = None,
        warmup_iters: int = 1,
    ) -> None:
        if openpcdet_dir is not None:
            sys.path.insert(0, str(openpcdet_dir))
            sys.path.insert(0, str(openpcdet_dir / "tools"))
        ensure_spconv_importable()

        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils

        self.torch = torch
        self.load_data_to_gpu = load_data_to_gpu
        cfg_file = cfg_file.expanduser().resolve()
        old_cwd = Path.cwd()
        try:
            os.chdir(cfg_file.parent.parent.parent)
            self.cfg = cfg_from_yaml_file(str(cfg_file), cfg)
        finally:
            os.chdir(old_cwd)
        self.logger = common_utils.create_logger()
        self.dataset = LivePointPillarDataset(self.cfg.DATA_CONFIG, self.cfg.CLASS_NAMES)
        self.class_names = tuple(self.cfg.CLASS_NAMES)
        self.model = build_network(
            model_cfg=self.cfg.MODEL,
            num_class=len(self.class_names),
            dataset=self.dataset,
        )
        self.model.load_params_from_file(filename=str(ckpt), logger=self.logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()
        self._warmup(max(0, int(warmup_iters)))

    def _warmup(self, iters: int) -> None:
        if iters <= 0:
            return
        points = np.zeros((2048, 4), dtype=np.float32)
        points[:, 0] = np.linspace(2.0, 30.0, len(points), dtype=np.float32)
        points[:, 1] = np.sin(points[:, 0]) * 5.0
        points[:, 2] = -1.0
        points[:, 3] = 0.5
        for _ in range(iters):
            self.infer(points)

    def infer(self, points: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        data = self.dataset.prepare_points(points)
        batch = self.dataset.collate_batch([data])
        self.load_data_to_gpu(batch)

        start_evt = self.torch.cuda.Event(enable_timing=True)
        end_evt = self.torch.cuda.Event(enable_timing=True)
        with self.torch.no_grad():
            start_evt.record()
            pred_dicts, _ = self.model.forward(batch)
            end_evt.record()
            self.torch.cuda.synchronize()
        model_ms = float(start_evt.elapsed_time(end_evt))

        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].detach().cpu().numpy()
        scores = pred["pred_scores"].detach().cpu().numpy()
        labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int64)
        detections: list[dict[str, Any]] = []
        for box, score, label in zip(boxes, scores, labels):
            class_id = int(label) - 1  # OpenPCDet labels are 1-based.
            name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class {class_id}"
            detections.append(
                {
                    "center": box[:3].astype(np.float32),
                    "size": box[3:6].astype(np.float32),
                    "yaw": float(box[6]),
                    "class_id": class_id,
                    "class_name": name,
                    "score": float(score),
                }
            )
        return detections, model_ms


def run_server(args: argparse.Namespace) -> None:
    if args.mock:
        detector = MockDetector()
        print("detector: mock mode")
    else:
        detector = OpenPCDetPointPillars(
            cfg_file=Path(args.cfg).expanduser(),
            ckpt=Path(args.ckpt).expanduser(),
            openpcdet_dir=Path(args.openpcdet_dir).expanduser() if args.openpcdet_dir else None,
            warmup_iters=args.warmup_iters,
        )
        print(f"detector: OpenPCDet {args.cfg}")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    bind = f"tcp://{args.host}:{args.port}"
    sock.bind(bind)
    print(f"listening on {bind}")

    while True:
        try:
            req = msgpack.unpackb(sock.recv(), raw=False)
            if req.get("type") != "scan":
                raise ValueError("unsupported request type")
            points = decode_points(req)
            if points.ndim != 2 or points.shape[1] != 4:
                raise ValueError(f"expected points shape (N,4), got {points.shape}")
            if len(points) > args.max_points:
                idx = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
                points = points[idx]
            detections, model_ms = detector.infer(points)
            detections = filter_detections(detections, points)
            reply = {
                "type": "detections",
                "seq": req.get("seq", 0),
                "model_ms": model_ms,
                "detections": detections,
            }
        except KeyboardInterrupt:
            break
        except Exception as exc:
            reply = {"type": "error", "message": str(exc)}
        sock.send(msgpack.packb(reply, use_bin_type=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson ZMQ PointPillars detector")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--cfg", default="~/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--ckpt", default="~/OpenPCDet/checkpoints/pointpillar_7728.pth")
    parser.add_argument("--openpcdet-dir", default="~/OpenPCDet")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_server(parse_args())


if __name__ == "__main__":
    main()
