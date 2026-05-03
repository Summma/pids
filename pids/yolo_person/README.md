# yolo_person

Edge-side person detection pipeline for the CASK Jetson. YOLOv8 runs
locally on a USB webcam; detections are pushed through Lohi (the
Palantir edge sync service) which buffers them in a local SQLite store
and syncs to Foundry every 5 seconds. If the network drops, detections
keep accumulating locally and replay when the link returns.

Three scripts:

- **`detect_person.py`** — webcam → YOLOv8n → one JSON event per frame on stdout.
- **`cam_uplink.py`** — reads stdin, enriches with live GPS, rate-limits, posts
  `create-cam-detected-person` actions to Lohi at `https://localhost:18380`.
- **`camera_ws.py`** — webcam → JPEG frames on `ws://<jetson>:9091/camera`
  for the frontend's real camera panel.

## Hardware

- Webcam: Logitech C270 HD on `/dev/video0`
- GPS: USB NMEA receiver on `/dev/ttyACM0` @ 57600 baud
- Jetson Orin (aarch64), CUDA torch 2.11, Ultralytics 8.4.x
- Lohi server running locally on port 18380 (mirrors the Foundry ontology;
  `CamDetectedPerson` must be in `sync_config.object_types` of `install.yml`)

## Setup

```bash
python3 -m pip install aiohttp ultralytics opencv-python
export FOUNDRY_TOKEN="<your foundry bearer JWT>"
```

The same token Lohi uses for its upstream sync. `cam_uplink.py` will exit
with an error if `FOUNDRY_TOKEN` is unset.

## Run

```bash
# detection only
python3 detect_person.py                              # default: /dev/video0, conf=0.4
python3 detect_person.py --max-frames 100             # short test run

# full pipeline (detect → GPS-enrich → push to Lohi)
python3 detect_person.py --quiet | python3 cam_uplink.py --push-interval 2.0

# real camera websocket for the web UI (no mock/fallback)
python3 camera_ws.py --device 0 --host 0.0.0.0 --port 9091

# optional: stream frames with YOLO person boxes drawn on top
python3 camera_ws.py --device 0 --overlay-yolo --host 0.0.0.0 --port 9091
```

The model weights (`yolov8n.pt`, ~6 MB) auto-download on first run.

## Detection event schema (`detect_person.py` stdout, one per frame)

```json
{
  "timestamp": "2026-05-03T01:55:12.345678+00:00",
  "frame_idx": 42,
  "device": "/dev/video0",
  "model": "yolov8n.pt",
  "inference_ms": 18.4,
  "frame_w": 640,
  "frame_h": 480,
  "num_persons": 2,
  "persons": [
    {"bbox_xyxy": [120.5, 80.1, 310.8, 470.2], "confidence": 0.91}
  ]
}
```

`bbox_xyxy` is original-frame pixel coordinates, top-left + bottom-right.

## Action parameters (`cam_uplink.py` → Lohi)

One `create-cam-detected-person` action per detected person:

| field | source |
|---|---|
| `detection-id` | UUID v4 |
| `timestamp` | from detection event |
| `latitude` / `longitude` | live GPS (NMEA GGA/RMC) |
| `confidence` | YOLO confidence |
| `device-id` | hardcoded `cask-02` |

Rate-limited via `--push-interval` (default 1.0 s). Frames between ticks
are dropped — only the freshest frame's persons are sent. Within a tick,
up to `--max-persons-per-push` (default 5) of the highest-confidence
persons are sent.

## Why Lohi (and not direct-to-cloud)

Pushing straight to `https://nshackathon.palantirfoundry.com` works while
the network is up but loses every detection during an outage. Posting to
Lohi at `localhost:18380` writes to the local SQLite store first;
Lohi's snapshot sync drains it to Foundry as soon as the link is healthy.
This is the CASK pattern — orchestrate across two tiers when the link is
unreliable.
