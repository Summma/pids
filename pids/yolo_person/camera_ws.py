#!/usr/bin/env python3
"""Real Jetson camera websocket for the PIDS frontend.

Streams frames from a V4L2 camera such as /dev/video0 using the same JSON
contract as the frontend /camera panel:

    {"type":"frame","w":640,"h":480,"mime":"image/jpeg","data":"..."}

This intentionally has no mock or laptop-camera fallback. If the configured
device cannot be opened, clients receive an error and the stream closes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
from aiohttp import web

PERSON_CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V4L2 camera -> frontend websocket")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9091)
    p.add_argument("--device", default="0", help="V4L2 index or path, e.g. 0 or /dev/video0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--overlay-yolo", action="store_true", help="Draw YOLO person boxes on streamed frames")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def open_capture(device_arg: str, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    if device_arg.isdigit():
        source = int(device_arg)
        label = f"/dev/video{device_arg}"
    else:
        source = device_arg
        label = device_arg

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"could not open actual camera device {label}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap, label


class CameraServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.period = 1.0 / max(args.fps, 1.0)
        self.jpeg_quality = max(35, min(int(args.jpeg_quality), 95))
        self.model = None
        self.status = "idle"
        if args.overlay_yolo:
            from ultralytics import YOLO

            self.model = YOLO(args.model)

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "service": "pids-jetson-camera-ws",
                "routes": ["/camera"],
                "camera_configured": bool(str(self.args.device).strip()),
                "camera_device": self.args.device,
                "status": self.status,
                "overlay_yolo": self.model is not None,
            }
        )

    async def camera_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15, max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)

        cap: Optional[cv2.VideoCapture] = None
        try:
            cap, device_label = open_capture(self.args.device, self.args.width, self.args.height)
            self.status = f"streaming {device_label}"
            seq = 0
            while not ws.closed:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok or frame is None:
                    await self._send_error(ws, f"frame grab failed from {device_label}")
                    await asyncio.sleep(0.25)
                    continue

                frame = await asyncio.to_thread(self._annotate_if_needed, frame)
                ok, encoded = await asyncio.to_thread(
                    cv2.imencode,
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    await self._send_error(ws, "camera JPEG encode failed")
                    await asyncio.sleep(0.25)
                    continue

                h, w = frame.shape[:2]
                seq += 1
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "frame",
                            "w": int(w),
                            "h": int(h),
                            "mime": "image/jpeg",
                            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                            "seq": seq,
                            "ts": time.time(),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        separators=(",", ":"),
                    )
                )
                await asyncio.sleep(self.period)
        except Exception as exc:
            self.status = f"camera unavailable: {exc}"
            await self._send_error(ws, self.status)
            await ws.close()
        finally:
            if cap is not None:
                cap.release()
        return ws

    def _annotate_if_needed(self, frame):
        if self.model is None:
            return frame
        results = self.model.predict(
            source=frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        return results[0].plot()

    async def _send_error(self, ws: web.WebSocketResponse, message: str) -> None:
        if not ws.closed:
            await ws.send_json({"type": "error", "message": message})


def build_app(args: argparse.Namespace) -> web.Application:
    server = CameraServer(args)
    app = web.Application()
    app.router.add_get("/health", server.health)
    app.router.add_get("/camera", server.camera_ws)
    return app


def main() -> int:
    args = parse_args()
    app = build_app(args)
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
