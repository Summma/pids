from __future__ import annotations

import argparse
import asyncio
import base64
import json
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from aiohttp import WSMsgType, web

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


@dataclass
class ThermalPacket:
    pixels: np.ndarray
    t_min: float
    t_max: float
    width: int
    height: int
    radiometric: bool


class ThermalCamera:
    def __init__(self, device: int, fps: float) -> None:
        self.device = device
        self.period = 1.0 / max(fps, 1.0)
        self.cap: Optional[cv2.VideoCapture] = None
        self.boson = None
        self.seq = 0

    def open(self) -> None:
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

    def _next_scan(self):
        while True:
            scan_set = next(self.scans)
            if scan_set is None:
                continue
            if hasattr(scan_set, "field"):
                return scan_set
            try:
                scan = next((s for s in scan_set if s is not None), None)
            except TypeError:
                scan = None
            if scan is not None:
                return scan

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


def normalize_thermal(frame: np.ndarray) -> ThermalPacket:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)

    try:
        while not ws.closed:
            frame_json = await asyncio.to_thread(lidar.read_json)
            await ws.send_str(frame_json)
            await asyncio.sleep(lidar.period)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"lidar websocket error: {exc}", file=sys.stderr)
        if not ws.closed:
            await ws.send_json({"type": "error", "message": str(exc)})
    return ws


async def health(request: web.Request) -> web.Response:
    lidar: LidarStreamer = request.app["lidar_streamer"]
    return web.json_response(
        {
            "ok": True,
            "service": "pids-backend",
            "routes": ["/thermal", "/lidar"],
            "lidar_configured": bool(lidar.host),
            "lidar_host": lidar.host,
        }
    )


def build_app(
    device: int,
    fps: float,
    lidar_host: Optional[str],
    lidar_fps: float,
    lidar_max_points: int,
) -> web.Application:
    app = web.Application()
    app["thermal_camera"] = ThermalCamera(device=device, fps=fps)
    app["lidar_streamer"] = LidarStreamer(
        host=lidar_host,
        fps=lidar_fps,
        max_points=lidar_max_points,
    )
    app.router.add_get("/health", health)
    app.router.add_get("/thermal", thermal_ws)
    app.router.add_get("/lidar", lidar_ws)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIDS Jetson WebSocket backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--lidar-host", default=None)
    parser.add_argument("--lidar-fps", type=float, default=5.0)
    parser.add_argument("--lidar-max-points", type=int, default=60000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        device=args.device,
        fps=args.fps,
        lidar_host=args.lidar_host,
        lidar_fps=args.lidar_fps,
        lidar_max_points=args.lidar_max_points,
    )

    def shutdown(*_: object) -> None:
        camera: ThermalCamera = app["thermal_camera"]
        lidar: LidarStreamer = app["lidar_streamer"]
        camera.close()
        lidar.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
