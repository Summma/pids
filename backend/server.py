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


class ThermalCamera:
    def __init__(self, device: Optional[int], fps: float) -> None:
        self.device = device
        self.period = 1.0 / max(fps, 1.0)
        self.cap: Optional[cv2.VideoCapture] = None
        self.boson = None
        self.seq = 0

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
        frame_bytes, _points = self.read_binary_frame()
        return frame_bytes

    def read_binary_frame(self) -> tuple[bytes, np.ndarray]:
        if self.scans is None or self.xyz_lut is None:
            self.open()

        scan = self._next_scan()
        points, intensities = self._extract_points(scan)
        payload, n = pack_lidar_binary(points, intensities, self.max_points)
        self.seq += 1
        header = struct.pack("<4sIIId", b"PCLD", 1, self.seq, n, time.time())
        return header + payload, points

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
    def __init__(self, mode: str, fps: float) -> None:
        self.mode = mode
        self.period = 1.0 / max(fps, 0.1)
        self._last_run_t = 0.0
        self._last_scan_t = time.monotonic()
        self._last_result = self._empty("idle")
        self._indoor = None
        self._params = None
        self._classifier = None
        self._tracker = None
        self._load_error = ""

        if mode == "indoor_human":
            self._configure_indoor_human()
        elif mode not in ("off", ""):
            self._load_error = f"unknown detection mode: {mode}"

    def maybe_process(self, points: np.ndarray) -> dict:
        now = time.monotonic()
        if self.mode in ("off", ""):
            return self._empty("detector off")
        if self._indoor is None:
            return self._empty(self._load_error or "detector unavailable")
        if now - self._last_run_t < self.period:
            return self._last_result

        self._last_run_t = now
        t0 = time.monotonic()
        try:
            candidates, detail, debug = self._indoor.detect_human_candidates(
                points,
                self._params,
                self._classifier,
            )
            scan_now = time.monotonic()
            dt = scan_now - self._last_scan_t
            self._last_scan_t = scan_now
            tracked = self._tracker.update(candidates, dt)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._last_result = {
                "mode": self.mode,
                "source": "indoor_human",
                "status": f"indoor {elapsed_ms:.0f}ms | {detail} -> {len(tracked)} confirmed",
                "elapsed_ms": round(elapsed_ms, 1),
                "boxes": [detection_to_json(det) for det in tracked],
                "debug_count": len(debug),
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
        self._classifier = indoor.CropClassifier()
        self._tracker = indoor.HumanTrackTracker(self._params)
        self._last_result = self._empty("indoor human ready")

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


def detection_to_json(det) -> dict:
    def vec3(value) -> list[float]:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return [round(float(v), 4) for v in arr[:3]]

    return {
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
    }


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


def packet_json(packet: ThermalPacket, seq: int) -> str:
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
        },
        separators=(",", ":"),
    )


async def thermal_ws(request: web.Request) -> web.WebSocketResponse:
    camera: ThermalCamera = request.app["thermal_camera"]
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)

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
            await ws.send_str(packet_json(packet, camera.seq))
            await asyncio.sleep(camera.period)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"thermal websocket error: {exc}", file=sys.stderr)
        if not ws.closed:
            await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        controls.cancel()
    return ws


async def lidar_ws(request: web.Request) -> web.WebSocketResponse:
    lidar: LidarStreamer = request.app["lidar_streamer"]
    detector: DetectionEngine = request.app["detection_engine"]
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)

    try:
        while not ws.closed:
            frame_bytes, points = await asyncio.to_thread(lidar.read_binary_frame)
            await ws.send_bytes(frame_bytes)
            detections = await asyncio.to_thread(detector.maybe_process, points)
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
    lidar: LidarStreamer = request.app["lidar_streamer"]
    camera: CameraStreamer = request.app["camera_streamer"]
    detector: DetectionEngine = request.app["detection_engine"]
    gemini: GeminiSceneClient = request.app["gemini_client"]
    return web.json_response(
        {
            "ok": True,
            "service": "pids-backend",
            "routes": ["/thermal", "/lidar", "/camera", "/gemini/chat"],
            "lidar_configured": bool(lidar.host),
            "lidar_host": lidar.host,
            "camera_configured": camera.device is not None,
            "camera_device": camera.device,
            "detection_mode": detector.mode,
            "gemini_model": gemini.model,
            "gemini_endpoint": gemini.endpoint_name,
            "gemini_project": gemini.project,
        }
    )


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
    gemini_model: str,
    gemini_project: Optional[str],
    gemini_location: str,
    gemini_api_mode: str,
    gemini_temperature: float,
    gemini_max_output_tokens: int,
    gemini_timeout_s: float,
) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
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
    app["detection_engine"] = DetectionEngine(mode=detection_mode, fps=detection_fps)
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
    parser.add_argument("--detection-mode", default="indoor_human", choices=["off", "indoor_human"])
    parser.add_argument("--detection-fps", type=float, default=2.0)
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

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
