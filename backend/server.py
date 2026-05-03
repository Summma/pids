from __future__ import annotations

import argparse
import asyncio
import base64
import json
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from aiohttp import WSMsgType, web

try:
    from flirpy.camera.boson import Boson
except Exception:
    Boson = None


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
        if not ws.closed:
            await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        controls.cancel()
    return ws


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "pids-backend"})


def build_app(device: int, fps: float) -> web.Application:
    app = web.Application()
    app["thermal_camera"] = ThermalCamera(device=device, fps=fps)
    app.router.add_get("/health", health)
    app.router.add_get("/thermal", thermal_ws)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIDS Jetson WebSocket backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(device=args.device, fps=args.fps)

    def shutdown(*_: object) -> None:
        camera: ThermalCamera = app["thermal_camera"]
        camera.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
