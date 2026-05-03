from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import importlib
import json
import os
import re
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

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

try:
    from flirpy.camera.boson import Boson
except Exception:
    Boson = None

try:
    from ouster.sdk import open_source
    from ouster.sdk.core import ChanField, XYZLut
except Exception:
    open_source = None
    ChanField = None
    XYZLut = None


GEMINI_DEV_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_VERTEX_ENDPOINT = "https://aiplatform.googleapis.com/v1"
GEMINI_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language.retriever",
)
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_GEMINI_IMAGE_BYTES = 6 * 1024 * 1024
MAX_GEMINI_IMAGES = 4
DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.+/]+);base64,(?P<data>.*)$", re.S)
ADVANCED_ROI_DETECTION_MODES = (
    "pointnet_roi",
    "pointnext_roi",
    "dgcnn_roi",
    "kpconv_roi",
    "sparse_cnn_roi",
    "point_transformer_roi",
)
DETECTION_MODES = (
    "off",
    "indoor_human",
    "roi_thermal",
    "seated_roi",
    *ADVANCED_ROI_DETECTION_MODES,
    "pointpillars",
    "auto",
)
DETECTION_MODE_LABELS = {
    "off": "Off",
    "indoor_human": "Indoor ROI",
    "roi_thermal": "Thermal ROI",
    "seated_roi": "Seated ROI",
    "pointnet_roi": "PointNet++ ROI + thermal",
    "pointnext_roi": "PointNeXt ROI + thermal",
    "dgcnn_roi": "DGCNN ROI + thermal",
    "kpconv_roi": "KPConv ROI + thermal",
    "sparse_cnn_roi": "Sparse CNN ROI + thermal",
    "point_transformer_roi": "Point Transformer ROI + thermal",
    "pointpillars": "PointPillars",
    "auto": "Auto fusion",
}
INDOOR_DETECTION_MODES = {"indoor_human", "auto", "roi_thermal", "seated_roi", *ADVANCED_ROI_DETECTION_MODES}
POINTPILLARS_DETECTION_MODES = {"pointpillars", "auto", "roi_thermal", "seated_roi", *ADVANCED_ROI_DETECTION_MODES}
LIDAR_MIN_POINTS = 1_000
LIDAR_MAX_POINTS = 131_072

SCENE_ANALYST_INSTRUCTION = """You are Narya's scene analyst. Use the current lidar/thermal 3D render, visible camera frame, and structured detector metadata to answer the operator's question.

Be direct and evidence-based. Count people or objects only when they are visible in the images or present in structured detections. Use track IDs and local sensor coordinates when provided. Do not invent GPS coordinates; if GPS is not present, say that only local sensor coordinates are available. Describe observable entities as people, objects, vehicles, or unknown subjects; do not label a person as an enemy unless structured sensor metadata explicitly says so."""


@dataclass
class ThermalPacket:
    pixels: np.ndarray
    t_min: float
    t_max: float
    width: int
    height: int
    radiometric: bool


@dataclass
class BackendDetection:
    track_id: int
    center: np.ndarray
    size: np.ndarray
    yaw: float
    class_id: int
    class_name: str
    score: float
    model_score: float = 0.0
    velocity: np.ndarray = None
    age: int = 1
    misses: int = 0
    support_points: int = 0
    support_z_span: float = 0.0
    source: str = "detector"
    thermal_score: float = 0.0
    thermal_coverage: float = 0.0
    thermal_mean: float = 0.0
    thermal_max: float = 0.0
    thermal_hot_fraction: float = 0.0
    fusion_score: float = 0.0
    fusion_note: str = ""
    pointpillars_support: float = 0.0

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32)
        self.size = np.asarray(self.size, dtype=np.float32)
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float32)
        else:
            self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.model_score <= 0.0:
            self.model_score = self.score
        if self.fusion_score <= 0.0:
            self.fusion_score = self.score


class ThermalCamera:
    def __init__(self, device: Optional[int], fps: float) -> None:
        self.device = device
        self.period = 1.0 / max(fps, 1.0)
        self.cap: Optional[cv2.VideoCapture] = None
        self.boson = None
        self.seq = 0
        self.last_packet: Optional[ThermalPacket] = None

    def open(self) -> None:
        if self.device is None or self.device < 0:
            raise RuntimeError("thermal camera disabled; omit --disable-thermal to enable it")
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video device /dev/video{self.device}")

        y16 = cv2.VideoWriter_fourcc("Y", "1", "6", " ")
        self.cap.set(cv2.CAP_PROP_FOURCC, y16)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        if Boson is not None:
            try:
                self.boson = Boson()
            except Exception:
                self.boson = None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.boson is not None:
            try:
                self.boson.close()
            except Exception:
                pass
            self.boson = None

    def read_packet(self) -> ThermalPacket:
        if self.cap is None:
            self.open()

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("thermal frame grab failed")

        packet = normalize_thermal(frame)
        self.seq += 1
        self.last_packet = packet
        return packet

    def trigger_ffc(self) -> bool:
        if self.boson is None:
            return False
        try:
            self.boson.do_ffc()
            return True
        except Exception:
            return False


class LidarStreamer:
    def __init__(self, host: Optional[str], fps: float, max_points: int) -> None:
        self.host = host
        self.period = 1.0 / max(fps, 1.0)
        self.max_points = max_points
        self.source = None
        self.scans = None
        self.xyz_lut = None
        self.seq = 0

    def open(self) -> None:
        if not self.host:
            raise RuntimeError("lidar host not configured; pass --lidar-host LIDAR_IP")
        if open_source is None or ChanField is None or XYZLut is None:
            raise RuntimeError("ouster-sdk not installed; run: python3 -m pip install ouster-sdk")

        self.source = open_source(self.host)
        meta = self.source.sensor_info
        if isinstance(meta, list):
            meta = meta[0]
        self.xyz_lut = XYZLut(meta)
        self.scans = iter(self.source)

    def close(self) -> None:
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
            self.source = None
        self.scans = None
        self.xyz_lut = None

    def read_json(self) -> str:
        if self.scans is None or self.xyz_lut is None:
            self.open()

        scan = self._next_scan()
        points, intensities = self._extract_points(scan)
        payload = pack_lidar(points, intensities, self.max_points)
        self.seq += 1
        return json.dumps(
            {
                "type": "frame",
                "n": payload["n"],
                "data": payload["data"],
                "clusters": [],
                "seq": self.seq,
                "ts": time.time(),
            },
            separators=(",", ":"),
        )

    def read_binary(self) -> bytes:
        frame_bytes, _points, _intensities = self.read_binary_frame()
        return frame_bytes

    def read_binary_frame(self, max_points: Optional[int] = None) -> tuple[bytes, np.ndarray, np.ndarray]:
        if self.scans is None or self.xyz_lut is None:
            self.open()

        scan = self._next_scan()
        points, intensities = self._extract_points(scan)
        payload, n = pack_lidar_binary(points, intensities, clamp_lidar_max_points(max_points, self.max_points))
        self.seq += 1
        header = struct.pack("<4sIIId", b"PCLD", 1, self.seq, n, time.time())
        return header + payload, points, intensities

    def _next_scan(self):
        while True:
            scan_set = next(self.scans)
            if scan_set is None:
                continue

            # Ouster SDK 0.16+ yields a LidarScanSet, even for one sensor.
            # Unwrap it before passing the scan into XYZLut.
            try:
                scan = next((s for s in scan_set if s is not None), None)
            except TypeError:
                scan = None
            if scan is not None:
                return scan

            # Older SDKs may yield a LidarScan directly.
            if hasattr(scan_set, "field"):
                return scan_set

    def _extract_points(self, scan) -> Tuple[np.ndarray, np.ndarray]:
        xyz = self.xyz_lut(scan).reshape(-1, 3).astype(np.float32)

        intensity_img = field_or_none(scan, "REFLECTIVITY")
        if intensity_img is None:
            intensity_img = field_or_none(scan, "SIGNAL")
        if intensity_img is None:
            intensity = np.ones(len(xyz), dtype=np.float32) * 0.5
        else:
            intensity = intensity_img.reshape(-1).astype(np.float32)

        finite = np.isfinite(xyz).all(axis=1)
        rng = np.linalg.norm(xyz, axis=1)
        keep = finite & (rng > 0.1)
        xyz = xyz[keep]
        intensity = intensity[keep]

        if len(intensity):
            hi = float(np.percentile(intensity, 99))
            if hi > 1e-6:
                intensity = np.clip(intensity / hi, 0, 1)
            else:
                intensity = np.zeros_like(intensity)

        return xyz, intensity


class CameraStreamer:
    def __init__(
        self,
        device: Optional[str],
        fps: float,
        width: int,
        height: int,
        jpeg_quality: int,
    ) -> None:
        self.device = device
        self.period = 1.0 / max(fps, 1.0)
        self.width = width
        self.height = height
        self.jpeg_quality = int(np.clip(jpeg_quality, 35, 95))
        self.cap: Optional[cv2.VideoCapture] = None
        self.seq = 0

    def open(self) -> None:
        if self.device is None or str(self.device).strip() == "":
            raise RuntimeError("camera device not configured; pass --camera-device 0 or /dev/videoX")

        dev = str(self.device)
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        source = int(dev) if dev.isdigit() else dev
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera device {dev}")

        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if sys.platform.startswith("linux"):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_packet(self) -> str:
        if self.cap is None:
            self.open()

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("camera frame grab failed")

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("camera JPEG encode failed")

        h, w = frame.shape[:2]
        self.seq += 1
        return json.dumps(
            {
                "type": "frame",
                "w": w,
                "h": h,
                "mime": "image/jpeg",
                "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "seq": self.seq,
                "ts": time.time(),
            },
            separators=(",", ":"),
        )


class DetectionEngine:
    def __init__(
        self,
        mode: str,
        fps: float,
        *,
        pointpillars_endpoint: str = "",
        pointpillars_timeout_ms: int = 250,
        pointpillars_max_points: int = 80_000,
    ) -> None:
        self.mode = mode
        self.period = 1.0 / max(fps, 0.1)
        self._last_run_t = 0.0
        self._last_scan_t = time.monotonic()
        self._last_result = self._empty("idle")
        self._indoor = None
        self._params = None
        self._classifier = None
        self._tracker = None
        self._pointpillars = None
        self._pointpillars_tracker = SimpleDetectionTracker(max_distance=3.0, max_misses=5)
        self._pointpillars_error = ""
        self._load_error = ""

        if mode in INDOOR_DETECTION_MODES:
            self._configure_indoor_human()
        if mode in POINTPILLARS_DETECTION_MODES:
            self._configure_pointpillars(pointpillars_endpoint, pointpillars_timeout_ms, pointpillars_max_points)
        if mode not in DETECTION_MODES and mode != "":
            self._load_error = f"unknown detection mode: {mode}"

    def maybe_process(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        thermal_packet: Optional[ThermalPacket] = None,
    ) -> dict:
        now = time.monotonic()
        if self.mode in ("off", ""):
            return self._empty("detector off")
        if self._load_error:
            return self._empty(self._load_error)
        if now - self._last_run_t < self.period:
            return self._last_result

        self._last_run_t = now
        t0 = time.monotonic()
        boxes: list = []
        debug_count = 0
        status_parts: list[str] = []
        pointpillars_tracked: list[BackendDetection] = []
        try:
            scan_now = time.monotonic()
            dt = scan_now - self._last_scan_t
            self._last_scan_t = scan_now

            if self.mode in POINTPILLARS_DETECTION_MODES:
                pointpillars_raw, pp_status = self._run_pointpillars(points, intensities, dt)
                pointpillars_tracked = pointpillars_raw
                status_parts.append(pp_status)

            if self.mode in INDOOR_DETECTION_MODES:
                if self._indoor is None:
                    status_parts.append(self._load_error or "indoor detector unavailable")
                else:
                    candidates, detail, debug = self._indoor.detect_human_candidates(
                        points,
                        self._params,
                        self._classifier,
                    )
                    tracked = self._tracker.update(candidates, dt)
                    annotate_pointpillars_support(tracked, pointpillars_tracked)
                    apply_thermal_fusion(tracked, points, thermal_packet)
                    boxes.extend(tracked)
                    debug_count = len(debug)
                    status_parts.append(f"indoor | {detail} -> {len(tracked)} confirmed")

            if self.mode == "pointpillars":
                apply_thermal_fusion(pointpillars_tracked, points, thermal_packet)
                boxes = pointpillars_tracked
            elif self.mode == "auto":
                extra = [
                    det for det in pointpillars_tracked
                    if det.class_id in (0, 2) and not overlaps_existing(det, boxes)
                ]
                apply_thermal_fusion(extra, points, thermal_packet)
                boxes.extend(extra)
            elif self.mode == "roi_thermal":
                before = len(boxes)
                boxes = select_thermal_roi_detections(boxes)
                status_parts.append(f"thermal ROI gate kept {len(boxes)}/{before}")
            elif self.mode == "seated_roi":
                before = len(boxes)
                boxes = select_seated_roi_detections(boxes)
                status_parts.append(f"seated ROI gate kept {len(boxes)}/{before}")
            elif self.mode in ADVANCED_ROI_DETECTION_MODES:
                before = len(boxes)
                boxes, profile_status = select_advanced_roi_detections(boxes, self.mode)
                status_parts.append(f"{profile_status} kept {len(boxes)}/{before}")

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._last_result = {
                "mode": self.mode,
                "source": self.mode if self.mode != "indoor_human" else "indoor_human",
                "status": f"{elapsed_ms:.0f}ms | {' | '.join(status_parts) or 'no detector output'}",
                "elapsed_ms": round(elapsed_ms, 1),
                "boxes": [detection_to_json(det) for det in boxes],
                "debug_count": debug_count,
                "ts": time.time(),
            }
        except Exception as exc:
            self._last_result = self._empty(f"detector error: {exc}")
        return self._last_result

    def _configure_indoor_human(self) -> None:
        indoor, error = load_indoor_human_module()
        if indoor is None:
            self._load_error = error
            return
        self._indoor = indoor
        self._params = indoor.IndoorHumanParams()
        configure_indoor_profile(self._params, self.mode)
        self._classifier = indoor.CropClassifier()
        self._tracker = indoor.HumanTrackTracker(self._params)
        self._last_result = self._empty("indoor human ready")

    def _configure_pointpillars(self, endpoint: str, timeout_ms: int, max_points: int) -> None:
        self._pointpillars = PointPillarsClient(endpoint, timeout_ms, max_points)
        if not self._pointpillars.available:
            self._pointpillars_error = self._pointpillars.unavailable_reason

    def _run_pointpillars(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray],
        dt: float,
    ) -> tuple[list[BackendDetection], str]:
        if self._pointpillars is None:
            return [], "pointpillars disabled"
        if not self._pointpillars.available:
            return [], self._pointpillars.unavailable_reason

        try:
            vals = intensities if intensities is not None else np.zeros(len(points), dtype=np.float32)
            raw, _elapsed_ms, status = self._pointpillars.infer(points, vals)
            self._pointpillars_error = ""
            tracked = self._pointpillars_tracker.update(raw, dt)
            return tracked, f"{status} -> {len(tracked)} confirmed"
        except Exception as exc:
            self._pointpillars_error = str(exc)
            return [], f"pointpillars unavailable: {exc}"

    def _empty(self, status: str) -> dict:
        return {
            "mode": self.mode,
            "source": "indoor_human" if self.mode == "indoor_human" else self.mode,
            "status": status,
            "elapsed_ms": 0.0,
            "boxes": [],
            "debug_count": 0,
            "ts": time.time(),
        }

    def close(self) -> None:
        if self._pointpillars is not None:
            self._pointpillars.close()


def normalize_detection_mode(mode: Optional[str], default: str = "auto") -> str:
    value = (mode or "").strip()
    if value in DETECTION_MODES:
        return value
    return default if default in DETECTION_MODES else "auto"


def get_detection_engine(app: web.Application, mode: Optional[str] = None) -> DetectionEngine:
    default_mode = normalize_detection_mode(app.get("detection_engine_default_mode", "auto"))
    selected_mode = normalize_detection_mode(mode, default_mode)
    engines: Dict[str, DetectionEngine] = app["detection_engines"]
    engine = engines.get(selected_mode)
    if engine is None:
        config = app["detection_engine_config"]
        engine = DetectionEngine(
            mode=selected_mode,
            fps=config["fps"],
            pointpillars_endpoint=config["pointpillars_endpoint"],
            pointpillars_timeout_ms=config["pointpillars_timeout_ms"],
            pointpillars_max_points=config["pointpillars_max_points"],
        )
        engines[selected_mode] = engine
    return engine


def configure_indoor_profile(params: Any, mode: str) -> None:
    """Tune the reusable ROI pipeline for the selected web model profile."""
    if mode == "seated_roi":
        params.voxel_size = 0.06
        params.max_points = 26_000
        params.eps = 0.36
        params.min_samples = 6
        params.min_cluster_points = 12
        params.association_distance = 1.45
        params.confirm_hits = 2
        params.max_misses = 6
    elif mode in ADVANCED_ROI_DETECTION_MODES:
        params.voxel_size = 0.065 if mode in {"pointnext_roi", "point_transformer_roi"} else 0.075
        params.max_points = 18_000
        params.eps = 0.36 if mode in {"dgcnn_roi", "kpconv_roi"} else 0.40
        params.min_samples = 7
        params.min_cluster_points = 14
        params.association_distance = 1.50
        params.confirm_hits = 2
        params.max_misses = 6
        params.include_ambiguous_candidates = True
        params.ambiguous_score_floor = 0.32 if mode in {"pointnext_roi", "point_transformer_roi"} else 0.36
        params.max_clusters = 48
        params.max_roi_proposals = 110
    elif mode == "roi_thermal":
        params.confirm_hits = 3
        params.max_misses = 4
        params.min_cluster_points = 16


class GeminiSceneClient:
    def __init__(
        self,
        model: str,
        project: Optional[str],
        location: str,
        api_mode: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> None:
        self.model = model
        self.project = project
        self.location = location
        self.api_mode = api_mode
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self._credentials = None
        self._request = None
        self._lock = threading.Lock()

    @property
    def endpoint_name(self) -> str:
        if self.api_mode == "vertex":
            return "vertex"
        if self.api_mode == "developer":
            return "developer"
        return "vertex" if env_truthy("GOOGLE_GENAI_USE_VERTEXAI") or self.project else "developer"

    async def ask(
        self,
        question: str,
        images: list[dict[str, str]],
        scene: dict[str, Any],
    ) -> dict[str, Any]:
        token = await asyncio.to_thread(self._access_token)
        endpoint_name = self.endpoint_name
        url = self._generate_url(endpoint_name)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if self.project:
            headers["x-goog-user-project"] = self.project

        body = self._request_body(question, images, scene, endpoint_name)
        timeout = ClientTimeout(total=self.timeout_s)
        async with ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"Gemini request failed ({resp.status}): {compact_error(raw)}")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Gemini returned a non-JSON response") from exc

        return {
            "text": extract_gemini_text(data),
            "model": self.model,
            "endpoint": endpoint_name,
            "raw": data,
        }

    def _request_body(
        self,
        question: str,
        images: list[dict[str, str]],
        scene: dict[str, Any],
        endpoint_name: str,
    ) -> dict[str, Any]:
        inline_key = "inlineData" if endpoint_name == "vertex" else "inline_data"
        mime_key = "mimeType" if endpoint_name == "vertex" else "mime_type"
        system_key = "systemInstruction" if endpoint_name == "vertex" else "system_instruction"
        generation_key = "generationConfig" if endpoint_name == "vertex" else "generation_config"
        max_tokens_key = "maxOutputTokens" if endpoint_name == "vertex" else "max_output_tokens"
        scene_text = json.dumps(scene, indent=2, sort_keys=True)[:12000]
        parts: list[dict[str, Any]] = [
            {
                "text": (
                    f"Operator question: {question}\n\n"
                    f"Structured sensor snapshot:\n```json\n{scene_text}\n```"
                )
            }
        ]
        for image in images:
            label = image.get("label", "scene image")
            parts.append({"text": f"Image: {label}"})
            parts.append(
                {
                    inline_key: {
                        mime_key: image["mime_type"],
                        "data": image["data"],
                    }
                }
            )

        return {
            system_key: {
                "parts": [{"text": SCENE_ANALYST_INSTRUCTION}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            generation_key: {
                "temperature": self.temperature,
                max_tokens_key: self.max_output_tokens,
            },
        }

    def _generate_url(self, endpoint_name: str) -> str:
        if endpoint_name == "vertex":
            if not self.project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT or --gemini-project is required for Vertex AI mode")
            return (
                f"{GEMINI_VERTEX_ENDPOINT}/projects/{self.project}/locations/{self.location}"
                f"/publishers/google/models/{self.model}:generateContent"
            )
        return f"{GEMINI_DEV_API_ENDPOINT}/models/{self.model}:generateContent"

    def _access_token(self) -> str:
        credentials = self._ensure_credentials()
        with self._lock:
            if not credentials.valid or credentials.expired:
                credentials.refresh(self._request)
            token = getattr(credentials, "token", None)
            if not token:
                raise RuntimeError("ADC did not provide an access token")
            return token

    def _ensure_credentials(self):
        if self._credentials is not None:
            return self._credentials

        try:
            import google.auth
            from google.auth.transport.requests import Request
        except Exception as exc:
            raise RuntimeError(
                "google-auth is required for Gemini ADC; install backend requirements"
            ) from exc

        credentials, adc_project = google.auth.default(scopes=GEMINI_OAUTH_SCOPES)
        quota_project = self.project or getattr(credentials, "quota_project_id", None) or adc_project
        if quota_project and hasattr(credentials, "with_quota_project"):
            credentials = credentials.with_quota_project(quota_project)
        self.project = self.project or quota_project
        self._credentials = credentials
        self._request = Request()
        return credentials


def normalize_thermal(frame: np.ndarray) -> ThermalPacket:
    if frame.ndim == 3 and frame.shape[2] == 2:
        gray = frame.view(np.uint16).reshape(frame.shape[0], frame.shape[1])
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3:
        gray = frame[:, :, 0]
    else:
        gray = frame

    radiometric = gray.dtype == np.uint16
    if radiometric:
        celsius = gray.astype(np.float32) * 0.01 - 273.15
        t_min = float(np.percentile(celsius, 1))
        t_max = float(np.percentile(celsius, 99))
        span = max(t_max - t_min, 0.1)
        pixels = np.clip((celsius - t_min) * (255.0 / span), 0, 255).astype(np.uint8)
    else:
        pixels = gray.astype(np.uint8, copy=False)
        t_min = 0.0
        t_max = 100.0

    h, w = pixels.shape[:2]
    return ThermalPacket(
        pixels=np.ascontiguousarray(pixels),
        t_min=t_min,
        t_max=t_max,
        width=w,
        height=h,
        radiometric=radiometric,
    )


def field_or_none(scan, attr: str) -> Optional[np.ndarray]:
    try:
        return scan.field(getattr(ChanField, attr))
    except Exception:
        return None


def pack_lidar(points: np.ndarray, intensities: np.ndarray, max_points: int) -> Dict[str, object]:
    n = min(len(points), max_points)
    if n <= 0:
        return {"n": 0, "data": ""}

    if len(points) > n:
        idx = np.linspace(0, len(points) - 1, n, dtype=np.int32)
        points = points[idx]
        intensities = intensities[idx]

    out = np.empty((n, 4), dtype="<f4")
    out[:, 0:3] = points[:n]
    out[:, 3] = intensities[:n]
    return {
        "n": n,
        "data": base64.b64encode(out.tobytes()).decode("ascii"),
    }


def pack_lidar_binary(points: np.ndarray, intensities: np.ndarray, max_points: int) -> Tuple[bytes, int]:
    n = min(len(points), max_points)
    if n <= 0:
        return b"", 0

    if len(points) > n:
        idx = np.linspace(0, len(points) - 1, n, dtype=np.int32)
        points = points[idx]
        intensities = intensities[idx]

    out = np.empty((n, 4), dtype="<f4")
    out[:, 0:3] = points[:n]
    out[:, 3] = intensities[:n]
    return out.tobytes(), n


def clamp_lidar_max_points(value: Optional[int], default: int) -> int:
    try:
        points = int(value if value is not None else default)
    except (TypeError, ValueError):
        points = int(default)
    return max(LIDAR_MIN_POINTS, min(points, LIDAR_MAX_POINTS))


KITTI_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
KITTI_POINT_CLOUD_RANGE = np.array([0.0, -39.68, -3.0, 69.12, 39.68, 1.0], dtype=np.float32)
KITTI_GROUND_Z_M = -1.60
POINTPILLARS_SCORE_THRESHOLD = 0.12


@dataclass
class PreparedPointPillarsPoints:
    points: np.ndarray
    original_xyz: np.ndarray
    z_shift: float


class SimpleDetectionTracker:
    def __init__(
        self,
        *,
        max_distance: float = 2.4,
        max_misses: int = 5,
        confirm_hits: int = 2,
        immediate_score: float = 0.82,
    ) -> None:
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.confirm_hits = confirm_hits
        self.immediate_score = immediate_score
        self.next_id = 1
        self.tracks: dict[int, BackendDetection] = {}
        self.hits: dict[int, int] = {}
        self.misses: dict[int, int] = {}

    def update(self, raw: list[BackendDetection], dt: float) -> list[BackendDetection]:
        if not raw:
            for tid in list(self.tracks):
                self.misses[tid] = self.misses.get(tid, 0) + 1
                self.tracks[tid].misses = self.misses[tid]
                self.tracks[tid].age += 1
                if self.misses[tid] > self.max_misses:
                    self.tracks.pop(tid, None)
                    self.hits.pop(tid, None)
                    self.misses.pop(tid, None)
            return self._confirmed()

        if not self.tracks:
            for det in raw:
                self._spawn(det)
            return self._confirmed()

        tids = list(self.tracks.keys())
        pairs: list[tuple[float, int, int]] = []
        for di, det in enumerate(raw):
            for ti, tid in enumerate(tids):
                old = self.tracks[tid]
                pairs.append((float(np.linalg.norm(det.center - old.center)), di, ti))
        pairs.sort(key=lambda item: item[0])

        used_det: set[int] = set()
        used_track_idx: set[int] = set()
        for dist, di, ti in pairs:
            if dist > self.max_distance:
                break
            if di in used_det or ti in used_track_idx:
                continue
            tid = tids[ti]
            old = self.tracks[tid]
            det = raw[di]
            det.track_id = tid
            det.age = old.age + 1
            det.misses = 0
            if dt > 0:
                det.velocity = (det.center - old.center) / dt
            self.tracks[tid] = det
            self.hits[tid] = self.hits.get(tid, 1) + 1
            self.misses[tid] = 0
            used_det.add(di)
            used_track_idx.add(ti)

        for di, det in enumerate(raw):
            if di not in used_det:
                self._spawn(det)

        matched = {tids[ti] for ti in used_track_idx}
        for tid in list(self.tracks):
            if tid in matched:
                continue
            self.misses[tid] = self.misses.get(tid, 0) + 1
            self.tracks[tid].misses = self.misses[tid]
            self.tracks[tid].age += 1
            if self.misses[tid] > self.max_misses:
                self.tracks.pop(tid, None)
                self.hits.pop(tid, None)
                self.misses.pop(tid, None)

        return self._confirmed()

    def _spawn(self, det: BackendDetection) -> None:
        tid = self.next_id
        self.next_id += 1
        det.track_id = tid
        det.age = 1
        det.misses = 0
        self.tracks[tid] = det
        self.hits[tid] = 1
        self.misses[tid] = 0

    def _confirmed(self) -> list[BackendDetection]:
        out = [
            det
            for tid, det in self.tracks.items()
            if self.misses.get(tid, 0) <= 1
            and (self.hits.get(tid, 0) >= self.confirm_hits or det.model_score >= self.immediate_score)
        ]
        out.sort(key=lambda det: det.track_id)
        return out


class PointPillarsClient:
    def __init__(self, endpoint: str, timeout_ms: int, max_points: int) -> None:
        self.endpoint = endpoint.strip()
        self.timeout_ms = max(50, int(timeout_ms))
        self.max_points = max(1024, int(max_points))
        self._ctx = None
        self._sock = None
        self._sock_endpoint = ""
        self._seq = 0

    @property
    def available(self) -> bool:
        return bool(self.endpoint) and HAS_ZMQ and HAS_MSGPACK

    @property
    def unavailable_reason(self) -> str:
        if not self.endpoint:
            return "pointpillars disabled"
        if not HAS_ZMQ:
            return "pyzmq missing"
        if not HAS_MSGPACK:
            return "msgpack missing"
        return ""

    def infer(self, xyz: np.ndarray, intensities: np.ndarray) -> tuple[list[BackendDetection], float, str]:
        if not self.available:
            raise RuntimeError(self.unavailable_reason)

        prepared = prepare_pointpillars_points(
            xyz,
            intensities,
            max_points=self.max_points,
        )
        if len(prepared.points) == 0:
            return [], 0.0, "pointpillars no points"

        sock = self._socket()
        self._seq += 1
        request = encode_pointpillars_request(prepared.points, self._seq)
        t0 = time.monotonic()
        try:
            sock.send(request)
            reply = sock.recv()
        except Exception:
            self.close()
            raise
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        payload = decode_pointpillars_response(reply)
        if payload.get("type") == "error":
            raise RuntimeError(str(payload.get("message", "pointpillars error")))

        raw = pointpillars_detections_from_payload(
            payload,
            POINTPILLARS_SCORE_THRESHOLD,
            points_xyz=prepared.original_xyz,
            z_shift=prepared.z_shift,
        )
        model_ms = payload.get("model_ms")
        if model_ms is not None:
            status = f"pointpillars {float(model_ms):.0f}ms model / {elapsed_ms:.0f}ms rtt | {len(raw)} raw"
        else:
            status = f"pointpillars {elapsed_ms:.0f}ms rtt | {len(raw)} raw"
        return raw, elapsed_ms, status

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
        self._sock = None
        self._sock_endpoint = ""

    def _socket(self):
        if self._sock is not None and self._sock_endpoint == self.endpoint:
            return self._sock
        self.close()
        if self._ctx is None:
            self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.connect(self.endpoint)
        self._sock = sock
        self._sock_endpoint = self.endpoint
        return sock


def prepare_pointpillars_points(
    xyz: np.ndarray,
    intensities: np.ndarray,
    *,
    range_min: float = 0.3,
    range_max: float = 80.0,
    max_points: int = 80_000,
) -> PreparedPointPillarsPoints:
    pts_xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    vals = np.asarray(intensities, dtype=np.float32).reshape(-1)
    if len(vals) != len(pts_xyz):
        vals = np.zeros(len(pts_xyz), dtype=np.float32)

    finite = np.isfinite(pts_xyz).all(axis=1)
    rng = np.linalg.norm(pts_xyz, axis=1)
    base_keep = (
        finite
        & (rng >= range_min)
        & (rng <= range_max)
        & (pts_xyz[:, 0] >= KITTI_POINT_CLOUD_RANGE[0])
        & (pts_xyz[:, 0] < KITTI_POINT_CLOUD_RANGE[3])
        & (pts_xyz[:, 1] >= KITTI_POINT_CLOUD_RANGE[1])
        & (pts_xyz[:, 1] < KITTI_POINT_CLOUD_RANGE[4])
    )
    z_shift = estimate_kitti_z_shift(pts_xyz[base_keep])
    shifted_z = pts_xyz[:, 2] + z_shift
    keep = (
        base_keep
        & (shifted_z >= KITTI_POINT_CLOUD_RANGE[2])
        & (shifted_z < KITTI_POINT_CLOUD_RANGE[5])
    )
    if not np.any(keep):
        empty = np.empty((0, 4), dtype=np.float32)
        return PreparedPointPillarsPoints(empty, np.empty((0, 3), dtype=np.float32), z_shift)

    original_xyz = pts_xyz[keep].copy()
    shifted_xyz = original_xyz.copy()
    shifted_xyz[:, 2] += z_shift
    intensity = np.nan_to_num(vals[keep], nan=0.0, posinf=0.0, neginf=0.0)
    intensity = np.clip(intensity, 0.0, 1.0).astype(np.float32, copy=False)
    if len(shifted_xyz) > max_points:
        idx = balanced_sample_indices(shifted_xyz, max_points)
        shifted_xyz = shifted_xyz[idx]
        original_xyz = original_xyz[idx]
        intensity = intensity[idx]

    points = np.column_stack((shifted_xyz, intensity)).astype(np.float32, copy=False)
    return PreparedPointPillarsPoints(points, original_xyz, z_shift)


def estimate_kitti_z_shift(xyz: np.ndarray) -> float:
    if len(xyz) < 256:
        return 0.0
    near = xyz[(xyz[:, 0] > 2.0) & (xyz[:, 0] < 35.0) & (np.abs(xyz[:, 1]) < 15.0)]
    if len(near) < 256:
        near = xyz
    ground_z = float(np.percentile(near[:, 2], 5.0))
    return float(np.clip(KITTI_GROUND_Z_M - ground_z, -2.5, 1.0))


def balanced_sample_indices(points: np.ndarray, max_points: int) -> np.ndarray:
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


def encode_pointpillars_request(points: np.ndarray, seq: int) -> bytes:
    if not HAS_MSGPACK:
        raise RuntimeError("msgpack missing")
    payload = {
        "type": "scan",
        "seq": seq,
        "points": points if HAS_MSGPACK_NUMPY else points.tobytes(),
        "shape": points.shape,
        "dtype": "float32",
        "encoding": "msgpack-numpy" if HAS_MSGPACK_NUMPY else "raw",
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_pointpillars_response(blob: bytes) -> dict:
    if not HAS_MSGPACK:
        raise RuntimeError("msgpack missing")
    return msgpack.unpackb(blob, raw=False)


def pointpillars_score_floor(class_id: int, range_m: float) -> float:
    base = {
        0: 0.40,
        1: 0.24,
        2: 0.32,
    }.get(class_id, POINTPILLARS_SCORE_THRESHOLD)
    if range_m > 35.0:
        base += 0.08
    return base


def pointpillars_class_name(class_id: int) -> str:
    if 0 <= class_id < len(KITTI_CLASS_NAMES):
        return KITTI_CLASS_NAMES[class_id]
    return f"class {class_id}"


def pointpillars_passes_geometry(class_id: int, size: np.ndarray) -> bool:
    dx, dy, dz = [float(v) for v in size]
    footprint = max(dx, dy)
    if class_id == 1:
        return 0.6 <= dz <= 2.5 and 0.15 <= min(dx, dy) and footprint <= 1.6
    if class_id == 2:
        return 0.8 <= dz <= 2.4 and 0.25 <= min(dx, dy) and footprint <= 2.5
    if class_id == 0:
        return 0.8 <= dz <= 3.2 and 1.0 <= footprint <= 7.0
    return True


def pointpillars_box_support(
    points_xyz: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> tuple[int, float]:
    if len(points_xyz) == 0:
        return 0, 0.0
    inside = points_in_oriented_box(points_xyz, center, size, yaw, margin=(0.25, 0.25, 0.25))
    if not np.any(inside):
        return 0, 0.0
    z_vals = points_xyz[inside, 2] - float(center[2])
    return int(len(z_vals)), float(np.percentile(z_vals, 95) - np.percentile(z_vals, 5))


def pointpillars_passes_support(class_id: int, score: float, support_points: int, z_span: float) -> bool:
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


def pointpillars_detections_from_payload(
    payload: dict,
    score_threshold: float,
    *,
    points_xyz: np.ndarray,
    z_shift: float,
) -> list[BackendDetection]:
    out: list[BackendDetection] = []
    for item in payload.get("detections", []):
        score = float(item.get("score", 0.0))
        center = np.asarray(item.get("center", item.get("translation", [0, 0, 0])), dtype=np.float32)
        size = np.asarray(item.get("size", item.get("dimensions", [0, 0, 0])), dtype=np.float32)
        if center.shape != (3,) or size.shape != (3,):
            continue
        center = center.copy()
        center[2] -= float(z_shift)
        class_id = int(item.get("class_id", item.get("label", -1)))
        range_m = float(np.linalg.norm(center[:2]))
        if score < max(score_threshold, pointpillars_score_floor(class_id, range_m)):
            continue
        if not pointpillars_passes_geometry(class_id, size):
            continue
        yaw = float(item.get("yaw", item.get("heading", 0.0)))
        support_points, support_z_span = pointpillars_box_support(points_xyz, center, size, yaw)
        if not pointpillars_passes_support(class_id, score, support_points, support_z_span):
            continue
        out.append(
            BackendDetection(
                track_id=0,
                center=center,
                size=size,
                yaw=yaw,
                class_id=class_id,
                class_name=str(item.get("class_name", item.get("name", pointpillars_class_name(class_id)))),
                score=score,
                model_score=score,
                support_points=support_points,
                support_z_span=support_z_span,
                source="pointpillars",
                pointpillars_support=score,
            )
        )
    return out


def annotate_pointpillars_support(detections: list, pointpillars: list[BackendDetection]) -> None:
    if not detections or not pointpillars:
        return
    for det in detections:
        best = 0.0
        for pp in pointpillars:
            dist = float(np.linalg.norm(np.asarray(pp.center) - np.asarray(getattr(det, "center", [0, 0, 0]))))
            if dist > 1.25:
                continue
            overlap = axis_aligned_iou(
                np.asarray(getattr(det, "center", [0, 0, 0]), dtype=np.float32),
                np.asarray(getattr(det, "size", [0, 0, 0]), dtype=np.float32),
                pp.center,
                pp.size,
            )
            proximity = 1.0 - float(np.clip(dist / 1.25, 0.0, 1.0))
            best = max(best, float(pp.model_score or pp.score) * max(overlap, 0.35 * proximity))
        setattr(det, "pointpillars_support", float(best))


def overlaps_existing(det: BackendDetection, existing: list) -> bool:
    for old in existing:
        old_center = np.asarray(getattr(old, "center", [0, 0, 0]), dtype=np.float32)
        old_size = np.asarray(getattr(old, "size", [0, 0, 0]), dtype=np.float32)
        if float(np.linalg.norm(det.center - old_center)) < 0.45:
            return True
        if axis_aligned_iou(det.center, det.size, old_center, old_size) > 0.25:
            return True
    return False


def axis_aligned_iou(a_center: np.ndarray, a_size: np.ndarray, b_center: np.ndarray, b_size: np.ndarray) -> float:
    a_mn, a_mx = a_center - a_size * 0.5, a_center + a_size * 0.5
    b_mn, b_mx = b_center - b_size * 0.5, b_center + b_size * 0.5
    inter = np.maximum(0.0, np.minimum(a_mx, b_mx) - np.maximum(a_mn, b_mn))
    inter_vol = float(np.prod(inter))
    if inter_vol <= 0.0:
        return 0.0
    a_vol = float(np.prod(np.maximum(a_size, 1e-3)))
    b_vol = float(np.prod(np.maximum(b_size, 1e-3)))
    return inter_vol / max(a_vol + b_vol - inter_vol, 1e-6)


GUI_DEFAULT_THERMAL_INTR = {
    "width": 640,
    "height": 512,
    "fx": 686.0,
    "fy": 686.0,
    "cx": 320.0,
    "cy": 256.0,
}
GUI_DEFAULT_THERMAL_EXTR = {
    "tx": 0.0,
    "ty": 0.0,
    "tz": 0.0,
    "roll_deg": 0.0,
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
}


def _numeric(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if np.isfinite(out) else fallback


def _normalize_thermal_intrinsics(value: Any) -> dict[str, float]:
    src = value if isinstance(value, dict) else {}
    intr = dict(GUI_DEFAULT_THERMAL_INTR)
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        intr[key] = _numeric(src.get(key), float(intr[key]))
    intr["width"] = int(max(1, round(intr["width"])))
    intr["height"] = int(max(1, round(intr["height"])))
    return intr


def _normalize_thermal_extrinsics(value: Any) -> dict[str, float]:
    src = value if isinstance(value, dict) else {}
    extr = dict(GUI_DEFAULT_THERMAL_EXTR)
    for key in ("tx", "ty", "tz", "roll_deg", "pitch_deg", "yaw_deg"):
        camel = {
            "roll_deg": "rollDeg",
            "pitch_deg": "pitchDeg",
            "yaw_deg": "yawDeg",
        }.get(key, key)
        extr[key] = _numeric(src.get(key, src.get(camel)), float(extr[key]))
    return extr


def resolve_thermal_calibration_path() -> Optional[Path]:
    override = os.environ.get("PIDS_THERMAL_CALIBRATION", "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "pids" / "boson_gui" / "calibration.json")
        candidates.append(parent / "boson_gui" / "calibration.json")

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def load_thermal_calibration() -> dict[str, Any]:
    path = resolve_thermal_calibration_path()
    data: dict[str, Any] = {}
    error = ""
    ok = False
    source = "gui_default"

    if path is not None:
        source = str(path)
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
            ok = True
        except Exception as exc:
            error = str(exc)
            data = {}

    return {
        "intrinsics": _normalize_thermal_intrinsics(data.get("intrinsics")),
        "extrinsics": _normalize_thermal_extrinsics(data.get("extrinsics")),
        "source": source,
        "ok": ok,
        "error": error,
    }


def refresh_thermal_calibration() -> dict[str, Any]:
    global THERMAL_CALIBRATION, THERMAL_INTR, THERMAL_EXTR
    THERMAL_CALIBRATION = load_thermal_calibration()
    THERMAL_INTR = THERMAL_CALIBRATION["intrinsics"]
    THERMAL_EXTR = THERMAL_CALIBRATION["extrinsics"]
    return THERMAL_CALIBRATION


THERMAL_CALIBRATION = load_thermal_calibration()
THERMAL_INTR = THERMAL_CALIBRATION["intrinsics"]
THERMAL_EXTR = THERMAL_CALIBRATION["extrinsics"]
R_LIDAR_TO_CAM_BASE = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def apply_thermal_fusion(detections: list, points: np.ndarray, thermal_packet: Optional[ThermalPacket]) -> None:
    if thermal_packet is None or thermal_packet.pixels is None or thermal_packet.pixels.size == 0:
        for det in detections:
            base_score = float(getattr(det, "score", 0.0))
            pp_support = float(getattr(det, "pointpillars_support", 0.0))
            fused = min(0.99, base_score + 0.06 * np.clip(pp_support, 0.0, 1.0))
            setattr(det, "fusion_score", float(fused))
            setattr(det, "fusion_note", "thermal_unavailable+pointpillars_support" if pp_support > 0 else "thermal_unavailable")
            setattr(det, "score", float(fused))
        return

    pixels = thermal_packet.pixels.astype(np.float32) / 255.0
    sample = pixels.reshape(-1)
    if sample.size > 20000:
        sample = sample[:: max(1, sample.size // 20000)]
    global_mean = float(np.mean(sample))
    global_std = max(float(np.std(sample)), 1e-6)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    for det in detections:
        base_score = float(getattr(det, "score", 0.0))
        metrics = thermal_metrics_for_detection(det, pts, pixels, global_mean, global_std)
        if metrics is None:
            pp_support = float(getattr(det, "pointpillars_support", 0.0))
            fused = min(0.99, base_score + 0.06 * np.clip(pp_support, 0.0, 1.0))
            setattr(det, "thermal_score", 0.0)
            setattr(det, "thermal_coverage", 0.0)
            setattr(det, "thermal_mean", 0.0)
            setattr(det, "thermal_max", 0.0)
            setattr(det, "thermal_hot_fraction", 0.0)
            setattr(det, "fusion_score", float(fused))
            setattr(det, "fusion_note", "thermal_unavailable+pointpillars_support" if pp_support > 0 else "thermal_unavailable")
            setattr(det, "score", float(fused))
            continue

        for key, value in metrics.items():
            setattr(det, f"thermal_{key}" if key in {"score", "coverage", "mean", "max"} else key, value)
        setattr(det, "thermal_hot_fraction", metrics["hot_fraction"])

        fused = base_score
        if metrics["coverage"] >= 0.12:
            if metrics["score"] >= 0.62:
                boost = 0.10 + 0.12 * np.clip((metrics["score"] - 0.62) / 0.38, 0.0, 1.0)
                fused = min(0.99, fused + boost)
                note = "thermal_boost"
            elif metrics["score"] <= 0.28 and metrics["coverage"] >= 0.35:
                fused = max(0.05, fused * 0.78)
                note = "thermal_cool"
            else:
                note = "thermal_neutral"
        else:
            note = "thermal_low_coverage"
        pp_support = float(getattr(det, "pointpillars_support", 0.0))
        if pp_support > 0:
            fused = min(0.99, fused + 0.06 * np.clip(pp_support, 0.0, 1.0))
            note = f"{note}+pointpillars_support"
        setattr(det, "fusion_score", float(fused))
        setattr(det, "fusion_note", note)
        setattr(det, "score", float(fused))


def select_thermal_roi_detections(detections: list) -> list:
    """Keep detections with real heat evidence or strong independent support."""
    kept = []
    for det in detections:
        pp_support = float(getattr(det, "pointpillars_support", 0.0))
        model_score = float(getattr(det, "model_score", getattr(det, "score", 0.0)))
        support_points = int(getattr(det, "support_points", getattr(det, "n_points", 0)))
        coverage = float(getattr(det, "thermal_coverage", 0.0))
        thermal_score = float(getattr(det, "thermal_score", 0.0))
        hot_fraction = float(getattr(det, "thermal_hot_fraction", 0.0))
        note = str(getattr(det, "fusion_note", ""))

        thermal_confirms = coverage >= 0.14 and (
            thermal_score >= 0.46 or hot_fraction >= 0.035
        )
        thermal_rejects = coverage >= 0.35 and thermal_score <= 0.26 and hot_fraction < 0.015
        independent_support = (
            pp_support >= 0.34
            or (model_score >= 0.82 and support_points >= 120)
        )

        if thermal_confirms or (not thermal_rejects and independent_support):
            setattr(det, "fusion_note", append_fusion_note(note, "thermal_roi"))
            setattr(det, "source", "roi_thermal")
            kept.append(det)
    return kept


def select_seated_roi_detections(detections: list) -> list:
    """Prefer seated-person / occupied-chair hypotheses with heat or strong geometry."""
    kept = []
    for det in detections:
        label = str(getattr(det, "class_name", getattr(det, "label", ""))).lower()
        pp_support = float(getattr(det, "pointpillars_support", 0.0))
        model_score = float(getattr(det, "model_score", getattr(det, "score", 0.0)))
        support_points = int(getattr(det, "support_points", getattr(det, "n_points", 0)))
        coverage = float(getattr(det, "thermal_coverage", 0.0))
        thermal_score = float(getattr(det, "thermal_score", 0.0))
        hot_fraction = float(getattr(det, "thermal_hot_fraction", 0.0))
        note = str(getattr(det, "fusion_note", ""))

        seated_like = "seated" in label or "occupied" in label or "chair" in label
        standing_like = "standing" in label or "person" in label or "pedestrian" in label
        thermal_confirms = coverage >= 0.12 and (
            thermal_score >= 0.42 or hot_fraction >= 0.03
        )
        strong_geometry = model_score >= 0.74 and support_points >= 75
        cool_covered_roi = coverage >= 0.38 and thermal_score <= 0.25 and hot_fraction < 0.012

        keep = False
        if seated_like:
            keep = thermal_confirms or pp_support >= 0.28 or (strong_geometry and not cool_covered_roi)
        elif standing_like:
            keep = thermal_confirms or pp_support >= 0.42

        if keep:
            setattr(det, "fusion_note", append_fusion_note(note, "seated_roi"))
            setattr(det, "source", "seated_roi")
            kept.append(det)
    return kept


ADVANCED_ROI_PROFILES = {
    "pointnet_roi": {
        "note": "pointnet++_roi_thermal_features",
        "weights": (0.42, 0.34, 0.16, 0.08),
        "threshold": 0.56,
        "thermal_confirm": 0.48,
        "min_support": 45,
    },
    "pointnext_roi": {
        "note": "pointnext_roi_thermal_features",
        "weights": (0.36, 0.36, 0.18, 0.10),
        "threshold": 0.54,
        "thermal_confirm": 0.45,
        "min_support": 38,
    },
    "dgcnn_roi": {
        "note": "dgcnn_edge_roi_thermal_features",
        "weights": (0.38, 0.26, 0.28, 0.08),
        "threshold": 0.58,
        "thermal_confirm": 0.50,
        "min_support": 55,
    },
    "kpconv_roi": {
        "note": "kpconv_geometry_roi_thermal_features",
        "weights": (0.34, 0.28, 0.30, 0.08),
        "threshold": 0.57,
        "thermal_confirm": 0.49,
        "min_support": 55,
    },
    "sparse_cnn_roi": {
        "note": "sparse_cnn_voxel_roi_thermal_features",
        "weights": (0.30, 0.30, 0.32, 0.08),
        "threshold": 0.58,
        "thermal_confirm": 0.50,
        "min_support": 70,
    },
    "point_transformer_roi": {
        "note": "point_transformer_roi_thermal_features",
        "weights": (0.30, 0.42, 0.18, 0.10),
        "threshold": 0.60,
        "thermal_confirm": 0.44,
        "min_support": 38,
    },
}


def select_advanced_roi_detections(detections: list, mode: str) -> tuple[list, str]:
    """Thermal-aware ROI model profiles for advanced crop-model options.

    These profiles keep the existing ROI generator but score each tracked crop
    with geometry, point support, PointPillars support, and projected thermal
    evidence. They are model-shaped feature priors until trained crop-model
    checkpoints are added; importantly, thermal can promote ambiguous
    seated/empty-chair ROIs that the geometry-only baseline would discard.
    """
    profile = ADVANCED_ROI_PROFILES.get(mode, ADVANCED_ROI_PROFILES["pointnext_roi"])
    geom_w, thermal_w, support_w, pp_w = profile["weights"]
    kept = []
    promoted = 0
    cooled = 0

    for det in detections:
        label = str(getattr(det, "class_name", getattr(det, "label", ""))).lower()
        note = str(getattr(det, "fusion_note", ""))
        geometry = float(np.clip(getattr(det, "model_score", getattr(det, "score", 0.0)), 0.0, 1.0))
        coverage = float(np.clip(getattr(det, "thermal_coverage", 0.0), 0.0, 1.0))
        thermal_score = float(np.clip(getattr(det, "thermal_score", 0.0), 0.0, 1.0))
        hot_fraction = float(np.clip(getattr(det, "thermal_hot_fraction", 0.0), 0.0, 1.0))
        thermal_max = float(np.clip(getattr(det, "thermal_max", 0.0), 0.0, 1.0))
        pp_support = float(np.clip(getattr(det, "pointpillars_support", 0.0), 0.0, 1.0))
        support_points = int(getattr(det, "support_points", getattr(det, "n_points", 0)))
        z_span = float(getattr(det, "support_z_span", 0.0))

        support_signal = float(
            np.clip(
                0.55 * (support_points / max(float(profile["min_support"]), 1.0))
                + 0.45 * (z_span / 1.15),
                0.0,
                1.0,
            )
        )
        thermal_signal = float(
            np.clip(
                coverage
                * (
                    0.52 * thermal_score
                    + 0.28 * thermal_max
                    + 0.20 * np.clip(hot_fraction / 0.055, 0.0, 1.0)
                ),
                0.0,
                1.0,
            )
        )
        profile_score = float(
            np.clip(
                geom_w * geometry
                + thermal_w * thermal_signal
                + support_w * support_signal
                + pp_w * pp_support,
                0.0,
                0.99,
            )
        )

        seated_like = "seated" in label or "occupied" in label or "chair" in label
        human_like = seated_like or "standing" in label or "person" in label or "pedestrian" in label or "human" in label
        ambiguous = "empty_chair" in label or "uncertain" in label
        thermal_confirms = thermal_signal >= float(profile["thermal_confirm"]) or (
            coverage >= 0.18 and (thermal_score >= 0.50 or hot_fraction >= 0.035)
        )
        thermal_rejects = coverage >= 0.34 and thermal_score <= 0.24 and hot_fraction < 0.012 and pp_support < 0.34

        if thermal_rejects and not ambiguous:
            cooled += 1
            continue

        if ambiguous:
            if not thermal_confirms:
                continue
            det.class_name = "occupied_chair"
            det.class_id = 1
            label = "occupied_chair"
            human_like = True
            promoted += 1

        if not human_like:
            continue

        if seated_like and thermal_confirms:
            profile_score = min(0.99, profile_score + 0.10)
        elif pp_support >= 0.42:
            profile_score = min(0.99, profile_score + 0.06)

        if profile_score < float(profile["threshold"]):
            continue

        setattr(det, "source", mode)
        setattr(det, "score", profile_score)
        setattr(det, "fusion_score", profile_score)
        setattr(det, "fusion_note", append_fusion_note(note, profile["note"]))
        kept.append(det)

    status = f"{profile['note']}, promoted {promoted}, cool-rejected {cooled}"
    return kept, status


def append_fusion_note(note: str, suffix: str) -> str:
    if not note:
        return suffix
    if suffix in note.split("+"):
        return note
    return f"{note}+{suffix}"


def thermal_metrics_for_detection(
    det,
    points: np.ndarray,
    pixels: np.ndarray,
    global_mean: float,
    global_std: float,
) -> Optional[dict[str, float]]:
    if points.size == 0:
        return None
    center = np.asarray(getattr(det, "center", [0, 0, 0]), dtype=np.float32)
    size = np.asarray(getattr(det, "size", [0, 0, 0]), dtype=np.float32)
    yaw = float(getattr(det, "yaw", 0.0))
    inside = points_in_oriented_box(points, center, size, yaw, margin=(0.25, 0.25, 0.15))
    if not np.any(inside):
        return None
    roi = points[inside]
    if len(roi) > 4096:
        roi = roi[np.linspace(0, len(roi) - 1, 4096, dtype=np.int64)]

    uv, valid = project_lidar_to_thermal(roi)
    coverage = float(np.count_nonzero(valid) / max(len(roi), 1))
    if not np.any(valid):
        return {"score": 0.0, "coverage": coverage, "mean": 0.0, "max": 0.0, "hot_fraction": 0.0}

    h, w = pixels.shape[:2]
    scale_u = w / max(float(THERMAL_INTR["width"]), 1.0)
    scale_v = h / max(float(THERMAL_INTR["height"]), 1.0)
    u = np.clip((uv[valid, 0].astype(np.float32) * scale_u).astype(np.int32), 0, w - 1)
    v = np.clip((uv[valid, 1].astype(np.float32) * scale_v).astype(np.int32), 0, h - 1)
    vals = pixels[v, u].astype(np.float32)
    mean = float(np.mean(vals))
    max_val = float(np.max(vals))
    hot_threshold = max(0.62, global_mean + 0.65 * global_std)
    hot_fraction = float(np.count_nonzero(vals >= hot_threshold) / max(len(vals), 1))
    contrast = float(np.clip((mean - global_mean) / max(global_std * 1.8, 0.12), -1.0, 1.0))
    score = float(np.clip(0.46 * max_val + 0.34 * hot_fraction + 0.20 * ((contrast + 1.0) * 0.5), 0.0, 1.0))
    return {"score": score, "coverage": coverage, "mean": mean, "max": max_val, "hot_fraction": hot_fraction}


def points_in_oriented_box(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
    margin: tuple[float, float, float],
) -> np.ndarray:
    delta = points - center
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    local_x = c * delta[:, 0] + s * delta[:, 1]
    local_y = -s * delta[:, 0] + c * delta[:, 1]
    local_z = delta[:, 2]
    half = np.maximum(size * 0.5, 0.05) + np.asarray(margin, dtype=np.float32)
    return (
        (np.abs(local_x) <= half[0])
        & (np.abs(local_y) <= half[1])
        & (np.abs(local_z) <= half[2])
    )


def project_lidar_to_thermal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_user = euler_to_r(
        np.deg2rad(THERMAL_EXTR["roll_deg"]),
        np.deg2rad(THERMAL_EXTR["pitch_deg"]),
        np.deg2rad(THERMAL_EXTR["yaw_deg"]),
    )
    r = r_user @ R_LIDAR_TO_CAM_BASE
    t = np.array([THERMAL_EXTR["tx"], THERMAL_EXTR["ty"], THERMAL_EXTR["tz"]], dtype=np.float64)
    cam = (r @ points.astype(np.float64).T).T + t
    z = cam[:, 2]
    in_front = z > 0.05
    safe_z = np.where(in_front, z, 1.0)
    u = (THERMAL_INTR["fx"] * cam[:, 0] / safe_z + THERMAL_INTR["cx"]).astype(np.int32)
    v = (THERMAL_INTR["fy"] * cam[:, 1] / safe_z + THERMAL_INTR["cy"]).astype(np.int32)
    valid = (
        in_front
        & (u >= 0)
        & (u < int(THERMAL_INTR["width"]))
        & (v >= 0)
        & (v < int(THERMAL_INTR["height"]))
    )
    uv = np.full((len(points), 2), -1, dtype=np.int32)
    uv[valid, 0] = u[valid]
    uv[valid, 1] = v[valid]
    return uv, valid


def euler_to_r(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def detection_to_json(det) -> dict:
    def vec3(value) -> list[float]:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return [round(float(v), 4) for v in arr[:3]]

    data = {
        "track_id": int(getattr(det, "track_id", 0)),
        "center": vec3(getattr(det, "center", [0, 0, 0])),
        "size": vec3(getattr(det, "size", [0.2, 0.2, 0.2])),
        "yaw": round(float(getattr(det, "yaw", 0.0)), 5),
        "class_id": int(getattr(det, "class_id", -1)),
        "class_name": str(getattr(det, "class_name", getattr(det, "label", "object"))),
        "score": round(float(getattr(det, "score", 0.0)), 4),
        "model_score": round(float(getattr(det, "model_score", getattr(det, "score", 0.0))), 4),
        "age": int(getattr(det, "age", 1)),
        "misses": int(getattr(det, "misses", 0)),
        "support_points": int(getattr(det, "support_points", getattr(det, "n_points", 0))),
        "support_z_span": round(float(getattr(det, "support_z_span", 0.0)), 4),
        "thermal_score": round(float(getattr(det, "thermal_score", 0.0)), 4),
        "thermal_coverage": round(float(getattr(det, "thermal_coverage", 0.0)), 4),
        "thermal_mean": round(float(getattr(det, "thermal_mean", 0.0)), 4),
        "thermal_max": round(float(getattr(det, "thermal_max", 0.0)), 4),
        "thermal_hot_fraction": round(float(getattr(det, "thermal_hot_fraction", 0.0)), 4),
        "fusion_score": round(float(getattr(det, "fusion_score", getattr(det, "score", 0.0))), 4),
        "fusion_note": str(getattr(det, "fusion_note", "")),
        "pointpillars_support": round(float(getattr(det, "pointpillars_support", 0.0)), 4),
    }
    source = str(getattr(det, "source", "")).strip()
    if source:
        data["source"] = source
    return data


def load_indoor_human_module():
    """Load the GUI detector in a headless backend process.

    The detector commit put the classifier/tracker next to Qt GUI workers.
    The web backend only needs the pure numpy/sklearn functions, so if PySide6
    is absent we install a tiny QtCore stub before importing those modules.
    """
    try:
        install_qtcore_stub_if_needed()
        here = Path(__file__).resolve()
        candidates = []
        for parent in here.parents:
            candidates.extend([parent / "boson_gui", parent / "pids" / "boson_gui"])
        gui_dir = next((path for path in candidates if (path / "indoor_human.py").exists()), None)
        if gui_dir is None:
            tried = ", ".join(str(path) for path in candidates)
            return None, f"boson_gui detector path missing; tried: {tried}"
        gui_path = str(gui_dir)
        if gui_path not in sys.path:
            sys.path.insert(0, gui_path)
        indoor = importlib.import_module("indoor_human")
        if not getattr(indoor, "HAS_SKLEARN", False):
            return None, "sklearn missing; install scikit-learn for indoor human/chair boxes"
        return indoor, ""
    except Exception as exc:
        return None, f"indoor detector import failed: {exc}"


def install_qtcore_stub_if_needed() -> None:
    try:
        import PySide6.QtCore  # noqa: F401
        return
    except Exception:
        pass

    if "PySide6.QtCore" in sys.modules:
        return

    pyside = ModuleType("PySide6")
    qtcore = ModuleType("PySide6.QtCore")

    class Signal:
        def __init__(self, *_args, **_kwargs) -> None:
            self._callbacks = []

        def connect(self, cb) -> None:
            self._callbacks.append(cb)

        def emit(self, *args, **kwargs) -> None:
            for cb in list(self._callbacks):
                cb(*args, **kwargs)

    class QThread:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def wait(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def msleep(ms: int) -> None:
            time.sleep(ms / 1000.0)

    qtcore.Signal = Signal
    qtcore.QThread = QThread
    pyside.QtCore = qtcore
    sys.modules.setdefault("PySide6", pyside)
    sys.modules.setdefault("PySide6.QtCore", qtcore)


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def compact_error(raw: str) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return raw[:500]
    message = data.get("error", {}).get("message") if isinstance(data, dict) else None
    return str(message or data)[:500]


def extract_gemini_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                chunks.append(text)
    text = "\n".join(chunks).strip()
    if text:
        return text

    prompt_feedback = data.get("promptFeedback") or data.get("prompt_feedback")
    if prompt_feedback:
        return f"Gemini did not return text. Prompt feedback: {prompt_feedback}"
    return "Gemini did not return text for this scene."


def parse_gemini_images(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    parsed: list[dict[str, str]] = []
    for item in items[:MAX_GEMINI_IMAGES]:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label") or "scene image")[:80]
        mime_type = str(item.get("mime_type") or item.get("mime") or "").strip()
        data = str(item.get("data") or item.get("dataUrl") or item.get("data_url") or "").strip()

        match = DATA_URL_RE.match(data)
        if match:
            mime_type = match.group("mime")
            data = match.group("data")

        if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError(f"unsupported image MIME type: {mime_type or 'missing'}")

        try:
            raw = base64.b64decode(data, validate=True)
        except binascii.Error as exc:
            raise ValueError(f"invalid base64 image data for {label}") from exc

        if len(raw) > MAX_GEMINI_IMAGE_BYTES:
            raise ValueError(f"{label} is too large for Gemini upload")

        parsed.append(
            {
                "label": label,
                "mime_type": mime_type,
                "data": base64.b64encode(raw).decode("ascii"),
            }
        )
    return parsed


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


def packet_json(packet: ThermalPacket, seq: int, calibration: Optional[dict[str, Any]] = None) -> str:
    return json.dumps(
        {
            "type": "frame",
            "w": packet.width,
            "h": packet.height,
            "data": base64.b64encode(packet.pixels.tobytes()).decode("ascii"),
            "t_min": packet.t_min,
            "t_max": packet.t_max,
            "radiometric": packet.radiometric,
            "seq": seq,
            "ts": time.time(),
            "calibration": calibration or THERMAL_CALIBRATION,
        },
        separators=(",", ":"),
    )


async def thermal_ws(request: web.Request) -> web.WebSocketResponse:
    camera: ThermalCamera = request.app["thermal_camera"]
    calibration: dict[str, Any] = request.app.get("thermal_calibration", THERMAL_CALIBRATION)
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)
    await ws.send_json({"type": "calibration", "calibration": calibration, "seq": camera.seq, "ts": time.time()})

    async def read_controls() -> None:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                cmd = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if cmd.get("type") == "ffc":
                await ws.send_json({"type": "ack", "cmd": "ffc", "ok": camera.trigger_ffc()})

    controls = asyncio.create_task(read_controls())
    try:
        while not ws.closed:
            packet = await asyncio.to_thread(camera.read_packet)
            await ws.send_str(packet_json(packet, camera.seq, calibration))
            await asyncio.sleep(camera.period)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"thermal websocket error: {exc}", file=sys.stderr)
        if not ws.closed:
            await ws.send_json({"type": "error", "message": str(exc), "calibration": calibration})
    finally:
        controls.cancel()
    return ws


async def lidar_ws(request: web.Request) -> web.WebSocketResponse:
    lidar: LidarStreamer = request.app["lidar_streamer"]
    detector = get_detection_engine(request.app, request.query.get("mode"))
    thermal: ThermalCamera = request.app["thermal_camera"]
    max_points = clamp_lidar_max_points(request.query.get("max_points"), lidar.max_points)
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)

    try:
        while not ws.closed:
            frame_bytes, points, intensities = await asyncio.to_thread(lidar.read_binary_frame, max_points)
            await ws.send_bytes(frame_bytes)
            detections = await asyncio.to_thread(detector.maybe_process, points, intensities, thermal.last_packet)
            await ws.send_json({"type": "detections", **detections})
            await asyncio.sleep(lidar.period)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"lidar websocket error: {exc}", file=sys.stderr)
        if not ws.closed:
            await ws.send_json({"type": "error", "message": str(exc)})
    return ws


async def camera_ws(request: web.Request) -> web.WebSocketResponse:
    camera: CameraStreamer = request.app["camera_streamer"]
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)

    try:
        while not ws.closed:
            packet = await asyncio.to_thread(camera.read_packet)
            await ws.send_str(packet)
            await asyncio.sleep(camera.period)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"camera websocket error: {exc}", file=sys.stderr)
        if not ws.closed:
            await ws.send_json({"type": "error", "message": str(exc)})
    return ws


async def gemini_chat(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"ok": False, "error": "request body must be JSON"}, status=400)

    question = str(payload.get("message") or payload.get("text") or "").strip()
    if not question:
        return web.json_response({"ok": False, "error": "message is required"}, status=400)
    if len(question) > 4000:
        return web.json_response({"ok": False, "error": "message is too long"}, status=400)

    try:
        images = parse_gemini_images(payload.get("images"))
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    scene = payload.get("scene")
    if not isinstance(scene, dict):
        scene = {}

    client: GeminiSceneClient = request.app["gemini_client"]
    try:
        result = await client.ask(question, images, scene)
    except Exception as exc:
        print(f"gemini chat error: {exc}", file=sys.stderr)
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    return web.json_response(
        {
            "ok": True,
            "text": result["text"],
            "model": result["model"],
            "endpoint": result["endpoint"],
        }
    )


async def health(request: web.Request) -> web.Response:
    thermal: ThermalCamera = request.app["thermal_camera"]
    lidar: LidarStreamer = request.app["lidar_streamer"]
    camera: CameraStreamer = request.app["camera_streamer"]
    detector = get_detection_engine(request.app)
    gemini: GeminiSceneClient = request.app["gemini_client"]
    static_dir: Optional[Path] = request.app.get("static_dir")
    engines: Dict[str, DetectionEngine] = request.app.get("detection_engines", {})
    detector_config: dict = request.app.get("detection_engine_config", {})
    calibration: dict[str, Any] = request.app.get("thermal_calibration", THERMAL_CALIBRATION)
    pp_endpoint = str(detector_config.get("pointpillars_endpoint", ""))
    return web.json_response(
        {
            "ok": True,
            "service": "pids-backend",
            "routes": ["/thermal", "/lidar", "/camera", "/gemini/chat"],
            "serves_frontend": bool(static_dir),
            "frontend_static_dir": str(static_dir) if static_dir else "",
            "thermal_configured": thermal.device is not None and thermal.device >= 0,
            "thermal_device": thermal.device,
            "thermal_calibration": calibration,
            "thermal_calibration_source": calibration.get("source", ""),
            "thermal_calibration_configured": bool(calibration.get("ok")),
            "lidar_configured": bool(lidar.host),
            "lidar_host": lidar.host,
            "lidar_default_max_points": lidar.max_points,
            "lidar_max_points_limit": LIDAR_MAX_POINTS,
            "camera_configured": camera.device is not None,
            "camera_device": camera.device,
            "detection_mode": detector.mode,
            "detection_modes": [
                {"value": mode, "label": DETECTION_MODE_LABELS[mode]}
                for mode in DETECTION_MODES
            ],
            "active_detection_modes": sorted(engines.keys()),
            "pointpillars_configured": bool(pp_endpoint),
            "pointpillars_endpoint": pp_endpoint,
            "pointpillars_available": bool(detector._pointpillars and detector._pointpillars.available),
            "pointpillars_error": detector._pointpillars_error,
            "gemini_model": gemini.model,
            "gemini_endpoint": gemini.endpoint_name,
            "gemini_project": gemini.project,
        }
    )


async def frontend_index(request: web.Request) -> web.FileResponse:
    static_dir: Optional[Path] = request.app.get("static_dir")
    if not static_dir:
        raise web.HTTPNotFound(text="frontend static build not found")
    return web.FileResponse(static_dir / "index.html")


def resolve_frontend_static_dir() -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates = [here.parent / "static"]
    for parent in here.parents:
        candidates.append(parent / "backend" / "static")
    for path in candidates:
        if (path / "index.html").is_file():
            return path
    return None


def build_app(
    device: Optional[int],
    fps: float,
    lidar_host: Optional[str],
    lidar_fps: float,
    lidar_max_points: int,
    camera_device: Optional[str],
    camera_fps: float,
    camera_width: int,
    camera_height: int,
    camera_jpeg_quality: int,
    detection_mode: str,
    detection_fps: float,
    pointpillars_endpoint: str,
    pointpillars_timeout_ms: int,
    pointpillars_max_points: int,
    gemini_model: str,
    gemini_project: Optional[str],
    gemini_location: str,
    gemini_api_mode: str,
    gemini_temperature: float,
    gemini_max_output_tokens: int,
    gemini_timeout_s: float,
) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["thermal_calibration"] = refresh_thermal_calibration()
    app["thermal_camera"] = ThermalCamera(device=device, fps=fps)
    app["lidar_streamer"] = LidarStreamer(
        host=lidar_host,
        fps=lidar_fps,
        max_points=lidar_max_points,
    )
    app["camera_streamer"] = CameraStreamer(
        device=camera_device,
        fps=camera_fps,
        width=camera_width,
        height=camera_height,
        jpeg_quality=camera_jpeg_quality,
    )
    app["detection_engine_default_mode"] = normalize_detection_mode(detection_mode, "indoor_human")
    app["detection_engine_config"] = {
        "fps": detection_fps,
        "pointpillars_endpoint": pointpillars_endpoint,
        "pointpillars_timeout_ms": pointpillars_timeout_ms,
        "pointpillars_max_points": pointpillars_max_points,
    }
    app["detection_engines"] = {}
    app["detection_engine"] = get_detection_engine(app)
    app["gemini_client"] = GeminiSceneClient(
        model=gemini_model,
        project=gemini_project,
        location=gemini_location,
        api_mode=gemini_api_mode,
        temperature=gemini_temperature,
        max_output_tokens=gemini_max_output_tokens,
        timeout_s=gemini_timeout_s,
    )
    app.router.add_get("/health", health)
    app.router.add_get("/thermal", thermal_ws)
    app.router.add_get("/lidar", lidar_ws)
    app.router.add_get("/camera", camera_ws)
    app.router.add_post("/gemini/chat", gemini_chat)
    static_dir = resolve_frontend_static_dir()
    app["static_dir"] = static_dir
    if static_dir is not None:
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.router.add_static("/assets", assets_dir, name="assets")
        app.router.add_get("/", frontend_index)
        app.router.add_get("/{tail:.*}", frontend_index)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIDS Jetson WebSocket backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--disable-thermal", action="store_true", help="Do not open a thermal video device")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--lidar-host", default=None)
    parser.add_argument("--lidar-fps", type=float, default=5.0)
    parser.add_argument("--lidar-max-points", type=int, default=60000)
    parser.add_argument("--camera-device", default=None, help="Visible camera source, e.g. 0 or /dev/video2")
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-jpeg-quality", type=int, default=75)
    parser.add_argument("--detection-mode", default="indoor_human", choices=DETECTION_MODES)
    parser.add_argument("--detection-fps", type=float, default=2.0)
    parser.add_argument("--pointpillars-endpoint", default=os.getenv("PIDS_POINTPILLARS_ENDPOINT", ""))
    parser.add_argument("--pointpillars-timeout-ms", type=int, default=int(os.getenv("PIDS_POINTPILLARS_TIMEOUT_MS", "250")))
    parser.add_argument("--pointpillars-max-points", type=int, default=int(os.getenv("PIDS_POINTPILLARS_MAX_POINTS", "80000")))
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--gemini-project", default=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT"))
    parser.add_argument("--gemini-location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--gemini-api-mode", default=os.getenv("GEMINI_API_MODE", "auto"), choices=["auto", "developer", "vertex"])
    parser.add_argument("--gemini-temperature", type=float, default=float(os.getenv("GEMINI_TEMPERATURE", "0.15")))
    parser.add_argument("--gemini-max-output-tokens", type=int, default=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1200")))
    parser.add_argument("--gemini-timeout-s", type=float, default=float(os.getenv("GEMINI_TIMEOUT_S", "45")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        device=None if args.disable_thermal else args.device,
        fps=args.fps,
        lidar_host=args.lidar_host,
        lidar_fps=args.lidar_fps,
        lidar_max_points=args.lidar_max_points,
        camera_device=args.camera_device,
        camera_fps=args.camera_fps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_jpeg_quality=args.camera_jpeg_quality,
        detection_mode=args.detection_mode,
        detection_fps=args.detection_fps,
        pointpillars_endpoint=args.pointpillars_endpoint,
        pointpillars_timeout_ms=args.pointpillars_timeout_ms,
        pointpillars_max_points=args.pointpillars_max_points,
        gemini_model=args.gemini_model,
        gemini_project=args.gemini_project,
        gemini_location=args.gemini_location,
        gemini_api_mode=args.gemini_api_mode,
        gemini_temperature=args.gemini_temperature,
        gemini_max_output_tokens=args.gemini_max_output_tokens,
        gemini_timeout_s=args.gemini_timeout_s,
    )

    def shutdown(*_: object) -> None:
        camera: ThermalCamera = app["thermal_camera"]
        lidar: LidarStreamer = app["lidar_streamer"]
        visible_camera: CameraStreamer = app["camera_streamer"]
        camera.close()
        lidar.close()
        visible_camera.close()
        for detector in app.get("detection_engines", {}).values():
            detector.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    web.run_app(app, host=args.host, port=args.port, shutdown_timeout=2.0)


if __name__ == "__main__":
    main()
