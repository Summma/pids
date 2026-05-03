"""Local WebSocket bridge from the Boson GUI to the React frontend.

The desktop GUI already owns the Ouster connection and the indoor-human
detector state. This bridge publishes that same in-process data on the
frontend backend contract, avoiding a second Ouster client fighting over UDP.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from lidar import LidarFrame

try:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
    HAS_AIOHTTP = True
except Exception:
    ClientSession = None  # type: ignore
    ClientTimeout = None  # type: ignore
    WSMsgType = None  # type: ignore
    web = None  # type: ignore
    HAS_AIOHTTP = False


def first_nonempty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def auto_detect_flir_avfoundation() -> Optional[str]:
    """Return avfoundation:<index> for a connected FLIR UVC camera on macOS."""
    if sys.platform != "darwin" or not shutil.which("ffmpeg"):
        return None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            text=True,
            capture_output=True,
            timeout=4,
            check=False,
        )
    except Exception:
        return None
    listing = f"{proc.stdout}\n{proc.stderr}"
    for line in listing.splitlines():
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match and "flir" in match.group(2).lower():
            return f"avfoundation:{match.group(1)}"
    return None


def is_avfoundation_source(source: Optional[str]) -> bool:
    return bool(source and str(source).startswith("avfoundation:"))


class WebBridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        max_points: int = 60_000,
        camera_source: Optional[str] = None,
        camera_ws_url: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.max_points = max_points
        self.camera_source = first_nonempty(
            camera_source,
            os.getenv("PIDS_CAMERA_SOURCE"),
            os.getenv("PIDS_CAMERA_DEVICE"),
            auto_detect_flir_avfoundation(),
        )
        self.camera_ws_url = first_nonempty(
            camera_ws_url,
            os.getenv("PIDS_CAMERA_WS"),
            os.getenv("PIDS_CAMERA_WS_URL"),
            os.getenv("PIDS_VISIBLE_CAMERA_WS"),
        )
        self.camera_width = env_int("PIDS_CAMERA_WIDTH", 640 if is_avfoundation_source(self.camera_source) else 1280)
        self.camera_height = env_int("PIDS_CAMERA_HEIGHT", 512 if is_avfoundation_source(self.camera_source) else 720)
        self.camera_fps = env_float("PIDS_CAMERA_FPS", 15.0)
        self.camera_jpeg_quality = int(np.clip(env_int("PIDS_CAMERA_JPEG_QUALITY", 75), 35, 95))
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._runner = None
        self._loop = None
        self._started = False
        self.status = "stopped"

        self._lidar_packet: Optional[bytes] = None
        self._lidar_seq = 0
        self._lidar_ts = 0.0
        self._lidar_n = 0

        self._thermal_json: Optional[str] = None
        self._thermal_seq = 0

        self._detections = self._empty_detections("waiting for Boson GUI detector")
        self._detection_seq = 0
        if self.camera_ws_url:
            self._camera_status = f"configured upstream websocket {self.camera_ws_url}"
        elif self.camera_source:
            self._camera_status = f"configured source {self.camera_source}"
        else:
            self._camera_status = "not configured"

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not HAS_AIOHTTP:
            self.status = "aiohttp missing; pip install aiohttp"
            print(f"web bridge disabled: {self.status}", file=sys.stderr)
            return
        self._thread = threading.Thread(target=self._run, name="BosonWebBridge", daemon=True)
        self._thread.start()

    def update_lidar(self, frame: LidarFrame) -> None:
        packet, n = self._pack_lidar(frame)
        if packet is None:
            return
        with self._lock:
            self._lidar_seq += 1
            self._lidar_ts = time.time()
            self._lidar_n = n
            header = struct.pack("<4sIIId", b"PCLD", 1, self._lidar_seq, n, self._lidar_ts)
            self._lidar_packet = header + packet

    def update_thermal(self, frame: np.ndarray) -> None:
        pixels, t_min, t_max, radiometric = self._normalize_thermal(frame)
        if pixels is None:
            return
        h, w = pixels.shape[:2]
        with self._lock:
            self._thermal_seq += 1
            self._thermal_json = json.dumps(
                {
                    "type": "frame",
                    "w": int(w),
                    "h": int(h),
                    "data": base64.b64encode(pixels.tobytes()).decode("ascii"),
                    "t_min": float(t_min),
                    "t_max": float(t_max),
                    "radiometric": bool(radiometric),
                    "seq": self._thermal_seq,
                    "ts": time.time(),
                },
                separators=(",", ":"),
            )

    def update_detections(
        self,
        boxes: list,
        *,
        mode: str,
        source: str,
        status: str,
        elapsed_ms: float = 0.0,
        debug_count: int = 0,
    ) -> None:
        with self._lock:
            self._detection_seq += 1
            self._detections = {
                "mode": mode,
                "source": source,
                "status": status,
                "elapsed_ms": round(float(elapsed_ms), 1),
                "boxes": [detection_to_json(det) for det in boxes],
                "debug_count": int(debug_count),
                "ts": time.time(),
            }

    def update_detection_status(self, status: str, *, mode: str = "indoor_human") -> None:
        with self._lock:
            self._detection_seq += 1
            self._detections = self._empty_detections(status, mode=mode)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/lidar", self._lidar_ws)
        app.router.add_get("/thermal", self._thermal_ws)
        app.router.add_get("/camera", self._camera_ws)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        try:
            await site.start()
            self.status = f"web bridge live on {self.host}:{self.port}"
            print(self.status)
            await asyncio.Event().wait()
        except OSError as exc:
            self.status = f"web bridge port unavailable: {exc}"
            print(self.status, file=sys.stderr)

    async def _health(self, _request) -> web.Response:
        with self._lock:
            payload = {
                "ok": True,
                "service": "boson-gui-web-bridge",
                "routes": ["/thermal", "/lidar", "/camera"],
                "lidar_configured": self._lidar_packet is not None,
                "lidar_host": "boson_gui",
                "camera_configured": self._camera_configured(),
                "camera_device": self.camera_source,
                "camera_ws_url": self.camera_ws_url,
                "camera_status": self._camera_status,
                "detection_mode": self._detections.get("mode", "indoor_human"),
                "lidar_points": self._lidar_n,
                "status": self.status,
            }
        return web.json_response(payload)

    async def _lidar_ws(self, request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        last_lidar = -1
        last_detection = -1
        try:
            while not ws.closed:
                with self._lock:
                    lidar_seq = self._lidar_seq
                    packet = self._lidar_packet
                    detection_seq = self._detection_seq
                    detections = dict(self._detections)
                if packet is not None and lidar_seq != last_lidar:
                    await ws.send_bytes(packet)
                    last_lidar = lidar_seq
                if detection_seq != last_detection:
                    await ws.send_json({"type": "detections", **detections})
                    last_detection = detection_seq
                await asyncio.sleep(0.03)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"web bridge lidar websocket error: {exc}", file=sys.stderr)
        return ws

    async def _thermal_ws(self, request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        last_seq = -1
        try:
            while not ws.closed:
                with self._lock:
                    seq = self._thermal_seq
                    payload = self._thermal_json
                if payload is not None and seq != last_seq:
                    await ws.send_str(payload)
                    last_seq = seq
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"web bridge thermal websocket error: {exc}", file=sys.stderr)
        return ws

    async def _camera_ws(self, _request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(_request)
        if self.camera_ws_url:
            await self._proxy_camera_ws(ws)
            return ws
        if self.camera_source:
            if is_avfoundation_source(self.camera_source):
                await self._capture_avfoundation_ws(ws)
                return ws
            await self._capture_camera_ws(ws)
            return ws

        await self._send_camera_error(
            ws,
            "actual camera source is not configured; set PIDS_CAMERA_WS to the Jetson camera websocket "
            "or PIDS_CAMERA_SOURCE to a real camera/RTSP/device source",
        )
        await ws.close()
        return ws

    def _camera_configured(self) -> bool:
        return bool(self.camera_ws_url or self.camera_source)

    async def _proxy_camera_ws(self, ws: web.WebSocketResponse) -> None:
        assert ClientSession is not None and ClientTimeout is not None and WSMsgType is not None
        url = self.camera_ws_url or ""
        seq = 0
        timeout = ClientTimeout(total=None, sock_connect=5)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.ws_connect(url, heartbeat=15, max_msg_size=16 * 1024 * 1024) as upstream:
                    self._camera_status = f"proxying {url}"
                    async for msg in upstream:
                        if ws.closed:
                            break
                        if msg.type == WSMsgType.TEXT:
                            await ws.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            seq += 1
                            await ws.send_json(self._jpeg_bytes_to_camera_payload(msg.data, seq))
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                            break
                        elif msg.type == WSMsgType.ERROR:
                            raise RuntimeError(str(upstream.exception()))
        except Exception as exc:
            self._camera_status = f"camera upstream unavailable: {exc}"
            if not ws.closed:
                await self._send_camera_error(ws, self._camera_status)
                await ws.close()

    async def _capture_camera_ws(self, ws: web.WebSocketResponse) -> None:
        source_label = str(self.camera_source or "").strip()
        source = int(source_label) if source_label.isdigit() else source_label
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened():
            self._camera_status = f"could not open actual camera source {source_label}"
            await self._send_camera_error(ws, self._camera_status)
            await ws.close()
            return

        try:
            if self.camera_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            if self.camera_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            if sys.platform.startswith("linux"):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            period = 1.0 / max(float(self.camera_fps), 1.0)
            seq = 0
            self._camera_status = f"streaming actual camera source {source_label}"
            while not ws.closed:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok or frame is None:
                    self._camera_status = f"camera frame grab failed from {source_label}"
                    await self._send_camera_error(ws, self._camera_status)
                    await asyncio.sleep(0.25)
                    continue

                ok, encoded = await asyncio.to_thread(
                    cv2.imencode,
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.camera_jpeg_quality],
                )
                if not ok:
                    await self._send_camera_error(ws, "camera JPEG encode failed")
                    await asyncio.sleep(0.25)
                    continue

                h, w = frame.shape[:2]
                seq += 1
                await ws.send_json(
                    {
                        "type": "frame",
                        "w": int(w),
                        "h": int(h),
                        "mime": "image/jpeg",
                        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                        "seq": seq,
                        "ts": time.time(),
                    }
                )
                await asyncio.sleep(period)
        except Exception as exc:
            self._camera_status = f"camera source error: {exc}"
            if not ws.closed:
                await self._send_camera_error(ws, self._camera_status)
                await ws.close()
        finally:
            cap.release()

    async def _capture_avfoundation_ws(self, ws: web.WebSocketResponse) -> None:
        if not shutil.which("ffmpeg"):
            self._camera_status = "ffmpeg is required for FLIR AVFoundation streaming"
            await self._send_camera_error(ws, self._camera_status)
            await ws.close()
            return

        index = str(self.camera_source).split(":", 1)[1]
        input_name = f"{index}:none"
        vf_fps = max(float(self.camera_fps), 1.0)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "avfoundation",
            "-framerate",
            "30",
            "-video_size",
            f"{self.camera_width}x{self.camera_height}",
            "-pixel_format",
            "uyvy422",
            "-i",
            input_name,
            "-an",
            "-vf",
            f"fps={vf_fps:g}",
            "-q:v",
            "5",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stderr_task = asyncio.create_task(self._collect_camera_stderr(proc))
            seq = 0
            buffer = bytearray()
            self._camera_status = f"streaming FLIR Camera via AVFoundation index {index}"
            while not ws.closed:
                assert proc.stdout is not None
                try:
                    chunk = await asyncio.wait_for(proc.stdout.read(64 * 1024), timeout=6)
                except asyncio.TimeoutError:
                    if proc.returncode is not None:
                        break
                    self._camera_status = (
                        f"FLIR Camera index {index} opened but has not delivered frames; "
                        "close QuickTime/other camera apps and grant Camera permission to Codex/Python"
                    )
                    await self._send_camera_error(ws, self._camera_status)
                    continue
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    jpg, consumed = self._pop_jpeg(buffer)
                    if jpg is None:
                        if consumed:
                            del buffer[:consumed]
                        break
                    if consumed:
                        del buffer[:consumed]
                    seq += 1
                    await ws.send_json(self._jpeg_bytes_to_camera_payload(jpg, seq))
                    if ws.closed:
                        break
            stderr_task.cancel()
        except Exception as exc:
            self._camera_status = f"FLIR camera stream error: {exc}"
            if not ws.closed:
                await self._send_camera_error(ws, self._camera_status)
                await ws.close()
        finally:
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except asyncio.TimeoutError:
                    proc.kill()

    async def _collect_camera_stderr(self, proc) -> None:
        assert proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self._camera_status = text
        except asyncio.CancelledError:
            return

    @staticmethod
    def _pop_jpeg(buffer: bytearray) -> tuple[Optional[bytes], int]:
        start = buffer.find(b"\xff\xd8")
        if start < 0:
            return None, max(len(buffer) - 1, 0)
        end = buffer.find(b"\xff\xd9", start + 2)
        if end < 0:
            return None, start
        end += 2
        return bytes(buffer[start:end]), end

    def _jpeg_bytes_to_camera_payload(self, data: bytes, seq: int) -> dict:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is not None:
            h, w = image.shape[:2]
        else:
            h, w = 0, 0
        return {
            "type": "frame",
            "w": int(w),
            "h": int(h),
            "mime": "image/jpeg",
            "data": base64.b64encode(data).decode("ascii"),
            "seq": seq,
            "ts": time.time(),
        }

    async def _send_camera_error(self, ws: web.WebSocketResponse, message: str) -> None:
        if not ws.closed:
            await ws.send_json({"type": "error", "message": message})

    def _empty_detections(self, status: str, *, mode: str = "indoor_human") -> dict:
        return {
            "mode": mode,
            "source": mode,
            "status": status,
            "elapsed_ms": 0.0,
            "boxes": [],
            "debug_count": 0,
            "ts": time.time(),
        }

    def _pack_lidar(self, frame: LidarFrame) -> tuple[Optional[bytes], int]:
        if frame.xyz is None:
            return None, 0
        xyz = frame.xyz.reshape(-1, 3).astype(np.float32, copy=False)
        rng = np.linalg.norm(xyz, axis=1)
        keep = np.isfinite(xyz).all(axis=1) & np.isfinite(rng) & (rng > 0.1)
        points = xyz[keep]
        if len(points) == 0:
            return None, 0

        intensities = self._lidar_intensity(frame, keep)
        n = min(len(points), self.max_points)
        if len(points) > n:
            idx = np.linspace(0, len(points) - 1, n, dtype=np.int32)
            points = points[idx]
            intensities = intensities[idx]

        out = np.empty((n, 4), dtype="<f4")
        out[:, 0:3] = points[:n]
        out[:, 3] = intensities[:n]
        return out.tobytes(), n

    @staticmethod
    def _lidar_intensity(frame: LidarFrame, keep: np.ndarray) -> np.ndarray:
        img = frame.reflectivity_img if frame.reflectivity_img is not None else frame.signal_img
        if img is None:
            return np.ones(int(np.count_nonzero(keep)), dtype=np.float32) * 0.5
        values = img.reshape(-1).astype(np.float32, copy=False)[keep]
        positive = values[np.isfinite(values) & (values > 0)]
        hi = float(np.percentile(positive, 99)) if positive.size else float(np.max(values)) if values.size else 1.0
        if hi <= 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip(values / hi, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_thermal(frame: np.ndarray) -> tuple[Optional[np.ndarray], float, float, bool]:
        if frame is None:
            return None, 0.0, 100.0, False
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
        elif gray.dtype != np.uint8:
            f = gray.astype(np.float32)
            t_min = float(np.percentile(f, 1))
            t_max = float(np.percentile(f, 99))
            span = max(t_max - t_min, 0.1)
            pixels = np.clip((f - t_min) * (255.0 / span), 0, 255).astype(np.uint8)
        else:
            pixels = gray.astype(np.uint8, copy=False)
            t_min = 0.0
            t_max = 100.0
        return np.ascontiguousarray(pixels), t_min, t_max, radiometric


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
