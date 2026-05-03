#!/usr/bin/env python3
"""Basic YOLO person detection from a USB webcam.

Reads frames from a V4L2 device, runs a pretrained YOLO model, filters for
the COCO `person` class, and prints one JSON event per frame to stdout.

The output schema is intentionally close to what a Foundry/Lohi action would
take, so this can later be wired to a push step:

    {
      "timestamp": "2026-05-03T01:55:12.345678+00:00",
      "frame_idx": 42,
      "device": "/dev/video0",
      "model": "yolov8n.pt",
      "inference_ms": 18.4,
      "frame_w": 640, "frame_h": 480,
      "num_persons": 2,
      "persons": [
        {"bbox_xyxy": [x1, y1, x2, y2], "confidence": 0.91},
        ...
      ]
    }

Usage:
    python3 detect_person.py
    python3 detect_person.py --device 0 --conf 0.5 --max-frames 100
    python3 detect_person.py --save-dir snapshots --save-every 30
    python3 detect_person.py --quiet > events.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam YOLO person detector")
    p.add_argument("--device", default="0", help="V4L2 index (e.g. 0) or path (e.g. /dev/video0)")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights to load")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run forever)")
    p.add_argument("--save-dir", default="", help="If set, save annotated frames here")
    p.add_argument("--save-every", type=int, default=30, help="Save every Nth frame to --save-dir")
    p.add_argument("--save-only-with-person", action="store_true", help="Skip saves when no person found")
    p.add_argument("--display", action="store_true", help="Show window (needs a display)")
    p.add_argument("--quiet", action="store_true", help="Suppress per-frame human-readable line on stderr")
    return p.parse_args()


def open_capture(device_arg: str, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    if device_arg.isdigit():
        cap = cv2.VideoCapture(int(device_arg), cv2.CAP_V4L2)
        device_str = f"/dev/video{device_arg}"
    else:
        cap = cv2.VideoCapture(device_arg, cv2.CAP_V4L2)
        device_str = device_arg
    if not cap.isOpened():
        raise SystemExit(f"could not open video device {device_str}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap, device_str


def main() -> int:
    args = parse_args()

    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    cap, device_str = open_capture(args.device, args.width, args.height)

    model = YOLO(args.model)
    if not args.quiet:
        print(f"[+] device={device_str} model={args.model} conf={args.conf} imgsz={args.imgsz}", file=sys.stderr)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[!] frame grab failed; stopping", file=sys.stderr)
                break

            t0 = time.monotonic()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
            dt_ms = (time.monotonic() - t0) * 1000.0

            r = results[0]
            persons = []
            for b in r.boxes:
                if int(b.cls.item()) != PERSON_CLASS_ID:
                    continue
                xyxy = [float(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf.item())
                persons.append({"bbox_xyxy": xyxy, "confidence": conf})

            h, w = frame.shape[:2]
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "frame_idx": frame_idx,
                "device": device_str,
                "model": args.model,
                "inference_ms": round(dt_ms, 2),
                "frame_w": w,
                "frame_h": h,
                "num_persons": len(persons),
                "persons": persons,
            }
            print(json.dumps(event), flush=True)

            if not args.quiet:
                print(
                    f"[frame {frame_idx:5d}] persons={len(persons)} "
                    f"infer={dt_ms:5.1f}ms",
                    file=sys.stderr,
                )

            should_save = (
                save_dir is not None
                and args.save_every > 0
                and frame_idx % args.save_every == 0
                and (not args.save_only_with_person or persons)
            )
            if should_save or args.display:
                annotated = r.plot()
                if should_save:
                    out_path = save_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), annotated)
                if args.display:
                    cv2.imshow("yolo person", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
