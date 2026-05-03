#!/usr/bin/env python3
"""cam_uplink.py — GPS-enriched person detections → Foundry ontology.

Reads JSON events from detect_person.py on stdin, enriches each detection
with live GPS from /dev/ttyACM0, and pushes them as `create-cam-detected-person`
action calls against the Foundry ontology.

Rate-limited: at most one push every --push-interval seconds (default 1.0).
Detection events arriving faster than that are dropped — only the most
recent frame's persons are pushed at each tick. This keeps a 30 FPS
detector from saturating the Foundry API.

Usage:
    python3 detect_person.py --quiet | python3 cam_uplink.py
    python3 detect_person.py --quiet | python3 cam_uplink.py --push-interval 2.0
    python3 detect_person.py --quiet | python3 cam_uplink.py --skip-no-gps
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Optional

import pynmea2
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import serial

# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI)
# ---------------------------------------------------------------------------

FOUNDRY_URL = "https://localhost:18380"
TOKEN = os.environ.get("FOUNDRY_TOKEN", "")
ONTOLOGY_RID = "ri.ontology.main.ontology.41fccd0c-2180-4c1d-841d-8a488d1abb46"
ACTION_API_NAME = "create-cam-detected-person"
DEVICE_ID = "cask-02"
GPS_DEVICE = "/dev/ttyACM0"
GPS_BAUD = 57600
GPS_STALE_WARN_S = 30


# ---------------------------------------------------------------------------
# GPS state (shared between GPS thread and main thread)
# ---------------------------------------------------------------------------

@dataclass
class GpsState:
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0
    fix_quality: int = 0
    updated_at: float = 0.0

    def is_fresh(self) -> bool:
        return self.fix_quality > 0 and (time.time() - self.updated_at) < GPS_STALE_WARN_S

    def snapshot(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
        }


class GpsThread(Thread):
    """Background thread: continuously reads NMEA from serial, updates GpsState."""

    def __init__(self, device: str, baud: int, state: GpsState, lock: Lock, stop: Event) -> None:
        super().__init__(daemon=True, name="gps-reader")
        self.device = device
        self.baud = baud
        self.state = state
        self.lock = lock
        self.stop = stop

    def run(self) -> None:
        backoff = 1
        while not self.stop.is_set():
            try:
                with serial.Serial(self.device, self.baud, timeout=2) as ser:
                    _log(f"gps connected: {self.device} @ {self.baud}")
                    backoff = 1
                    self._read_loop(ser)
            except serial.SerialException as exc:
                _log(f"gps serial error: {exc} — retry in {backoff}s")
            except Exception as exc:
                _log(f"gps unexpected error: {exc} — retry in {backoff}s")
            self.stop.wait(backoff)
            backoff = min(backoff * 2, 60)

    def _read_loop(self, ser: serial.Serial) -> None:
        while not self.stop.is_set():
            try:
                raw = ser.readline().decode("ascii", errors="replace").strip()
            except serial.SerialException:
                return
            if not raw.startswith("$"):
                continue
            try:
                msg = pynmea2.parse(raw)
            except pynmea2.ParseError:
                continue

            with self.lock:
                if isinstance(msg, pynmea2.types.talker.GGA):
                    if msg.latitude:
                        self.state.latitude = float(msg.latitude)
                        self.state.longitude = float(msg.longitude)
                        self.state.fix_quality = int(msg.gps_qual or 0)
                    if msg.altitude:
                        self.state.altitude_m = float(msg.altitude)
                    self.state.updated_at = time.time()
                elif isinstance(msg, pynmea2.types.talker.RMC):
                    if msg.latitude and self.state.latitude == 0.0:
                        self.state.latitude = float(msg.latitude)
                        self.state.longitude = float(msg.longitude)
                        self.state.fix_quality = 1
                        self.state.updated_at = time.time()


# ---------------------------------------------------------------------------
# Foundry Ontology action push
# ---------------------------------------------------------------------------

def push_action(parameters: dict, foundry_url: str, token: str, ontology_rid: str, action_api_name: str) -> bool:
    """POST one action apply to Foundry. Returns True on 2xx."""
    endpoint = f"{foundry_url.rstrip('/')}/api/v2/ontologies/{ontology_rid}/actions/{action_api_name}/apply"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        resp = requests.post(endpoint, headers=headers, json={"parameters": parameters}, timeout=10, verify=False)
        if resp.status_code >= 400:
            _log(f"action failed HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except requests.RequestException as exc:
        _log(f"action exception: {exc}")
        return False


def build_action_params(event: dict, person: dict, gps_snap: dict) -> dict:
    """One person's detection → action parameters (kebab-case keys)."""
    return {
        "detection-id": str(uuid.uuid4()),
        "timestamp": event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "latitude": gps_snap["latitude"],
        "longitude": gps_snap["longitude"],
        "confidence": round(float(person["confidence"]), 4),
        "device-id": DEVICE_ID,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cam_uplink: GPS-enriched detections → Foundry ontology")
    p.add_argument("--gps-device", default=GPS_DEVICE)
    p.add_argument("--gps-baud", type=int, default=GPS_BAUD)
    p.add_argument("--ontology-rid", default=ONTOLOGY_RID)
    p.add_argument("--action", default=ACTION_API_NAME, help="Action API name")
    p.add_argument("--foundry-url", default=FOUNDRY_URL)
    p.add_argument("--token", default=TOKEN)
    p.add_argument("--push-interval", type=float, default=1.0,
                   help="Min seconds between pushes — frames between intervals are dropped")
    p.add_argument("--max-persons-per-push", type=int, default=5,
                   help="Cap on action calls per push (drops lowest-confidence first)")
    p.add_argument("--skip-no-gps", action="store_true",
                   help="Drop detections if GPS has no fix (otherwise push with last-known coords)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.token:
        print("error: no token provided. Set FOUNDRY_TOKEN env var or pass --token", file=sys.stderr)
        return 2

    gps_state = GpsState()
    gps_lock = Lock()
    stop_event = Event()

    def _shutdown(*_):
        _log("shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    gps_thread = GpsThread(args.gps_device, args.gps_baud, gps_state, gps_lock, stop_event)
    gps_thread.start()

    _log(
        f"started — ontology={args.ontology_rid} action={args.action} "
        f"push-interval={args.push_interval}s max-persons={args.max_persons_per_push}"
    )

    latest_event: Optional[dict] = None  # most recent frame with persons (drop older ones)
    frames_seen = 0
    frames_dropped = 0
    last_push = 0.0
    pushed_total = 0
    failed_total = 0
    last_gps_warn = 0.0

    def do_push() -> None:
        """Push the latest event's persons (capped). Mutates counters."""
        nonlocal latest_event, last_push, pushed_total, failed_total, frames_dropped
        nonlocal last_gps_warn

        if latest_event is None:
            last_push = time.time()
            return

        # GPS snapshot
        with gps_lock:
            gps_snap = gps_state.snapshot()
            gps_fresh = gps_state.is_fresh()

        if not gps_fresh:
            now = time.time()
            if now - last_gps_warn > 10:
                _log("warning: GPS has no fresh fix — coordinates may be 0,0")
                last_gps_warn = now
            if args.skip_no_gps:
                latest_event = None
                last_push = time.time()
                return

        persons = sorted(
            latest_event.get("persons", []),
            key=lambda p: p.get("confidence", 0),
            reverse=True,
        )[: args.max_persons_per_push]

        ok = 0
        for person in persons:
            params = build_action_params(latest_event, person, gps_snap)
            if push_action(params, args.foundry_url, args.token,
                           args.ontology_rid, args.action):
                ok += 1
                _log(
                    f">>> PERSON DETECTED -> Foundry  "
                    f"confidence={params['confidence']:.2f}  "
                    f"lat={params['latitude']:.5f} lon={params['longitude']:.5f}  "
                    f"id={params['detection-id'][:8]}"
                )
            else:
                failed_total += 1

        pushed_total += ok
        _log(
            f"  (push tick: {ok}/{len(persons)} ok | "
            f"total: ok={pushed_total} fail={failed_total} dropped_frames={frames_dropped})"
        )
        latest_event = None
        last_push = time.time()

    for line in sys.stdin:
        if stop_event.is_set():
            break

        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        frames_seen += 1
        if event.get("persons"):
            if latest_event is not None:
                # Replacing an unfinished latest event = dropping a frame
                frames_dropped += 1
            latest_event = event

        # Time-based push trigger
        if time.time() - last_push >= args.push_interval:
            do_push()

    # Final push before exit
    do_push()
    stop_event.set()
    _log(
        f"done — frames_seen={frames_seen} pushed={pushed_total} "
        f"failed={failed_total} dropped_frames={frames_dropped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
