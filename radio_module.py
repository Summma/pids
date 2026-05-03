"""Standalone passive RF module for drone-class emitter detection.

This file is intentionally UI-free. It can be imported by the rest of the
system as a sensor input, or run directly to print JSON events.

Typical library use:

    from radio_module import RadioModule

    radio = RadioModule()
    radio.start()
    ...
    for event in radio.get_events():
        if event["type"] == "drone_candidate":
            ...
    radio.stop()

Direct test:

    DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 radio_module.py

The module scans 433 MHz and 915 MHz ISM bands with an RTL-SDR, tracks
persistent emitters, and classifies strong pulsed signals as drone-control
candidates. It does not detect 2.4 GHz or 5.8 GHz drone links because those
are outside the RTL-SDR v5 tuning range.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


Event = dict[str, Any]
Peak = tuple[float, float, float]  # freq_hz, power_db, snr_db
EventCallback = Callable[[Event], None]
SdrFactory = Callable[[], Any]


SDR_RATE_HZ = 1.024e6
FFT_SIZE = 4096
AVERAGES = 4

BANDS: dict[str, dict[str, float | str]] = {
    "ism433": {
        "center_hz": 433.92e6,
        "label": "ISM 433 (FrSky / generic RC)",
    },
    "ism915": {
        "center_hz": 915.0e6,
        "label": "ISM 915 (Crossfire / ELRS / long-range)",
    },
}

DEFAULT_DWELL_S = 3.0
DEFAULT_THRESHOLD_DB = 15.0
DEFAULT_BASELINE_S = 6.0


def _load_rtlsdr() -> type[Any]:
    try:
        from rtlsdr import RtlSdr
    except Exception as exc:  # pragma: no cover - depends on host hardware setup
        raise RuntimeError(
            "radio_module requires pyrtlsdr to start hardware scanning. "
            "Install with `pip install pyrtlsdr` and ensure librtlsdr is on "
            "the dynamic library path."
        ) from exc
    return RtlSdr


@dataclass
class EmitterTrack:
    band: str
    freq_hz: float
    first_seen: float
    last_seen: float
    snapshot_count: int = 1
    powers_db: list[float] = field(default_factory=list)
    snrs_db: list[float] = field(default_factory=list)
    freqs_hz: list[float] = field(default_factory=list)
    classified_drone: bool = False

    @property
    def duration_s(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def mean_snr_db(self) -> float:
        return float(np.mean(self.snrs_db)) if self.snrs_db else 0.0

    @property
    def power_variance_db(self) -> float:
        return float(np.std(self.powers_db)) if len(self.powers_db) > 1 else 0.0

    @property
    def freq_drift_hz(self) -> float:
        return float(max(self.freqs_hz) - min(self.freqs_hz)) if self.freqs_hz else 0.0

    def estimate_pulse_rate_hz(self) -> float:
        """Rough pulse-rate bucket inferred from snapshot-to-snapshot variance."""
        variance = self.power_variance_db
        if variance < 1.0:
            return 0.0
        if variance < 3.0:
            return 30.0
        if variance < 6.0:
            return 80.0
        return 200.0

    def to_event(self) -> Event:
        return {
            "band": self.band,
            "freq_mhz": round(self.freq_hz / 1e6, 3),
            "mean_snr_db": round(self.mean_snr_db, 1),
            "duration_s": round(self.duration_s, 1),
            "pulse_rate_hz": self.estimate_pulse_rate_hz(),
        }


class _DroneClassifier:
    MIN_DURATION_S = 3.0
    MIN_SNAPSHOTS = 12
    MIN_SNR_DB = 20.0
    MIN_POWER_VARIANCE_DB = 1.5
    MAX_POWER_VARIANCE_DB = 12.0
    MAX_FREQ_DRIFT_HZ = 30_000
    LOST_AFTER_S = 2.0
    DEDUP_HZ = 80_000
    MAX_HISTORY = 60

    def __init__(self) -> None:
        self.tracks: dict[tuple[str, int], EmitterTrack] = {}

    def update(self, band: str, peaks: list[Peak], now: float) -> list[Event]:
        events: list[Event] = []
        seen_keys: set[tuple[str, int]] = set()

        for freq_hz, power_db, snr_db in peaks:
            key = self._key(band, freq_hz)
            seen_keys.add(key)
            track = self.tracks.get(key)

            if track is None:
                self.tracks[key] = EmitterTrack(
                    band=band,
                    freq_hz=freq_hz,
                    first_seen=now,
                    last_seen=now,
                    powers_db=[power_db],
                    snrs_db=[snr_db],
                    freqs_hz=[freq_hz],
                )
                continue

            track.last_seen = now
            track.snapshot_count += 1
            track.powers_db.append(power_db)
            track.snrs_db.append(snr_db)
            track.freqs_hz.append(freq_hz)

            if track.snapshot_count == 2:
                events.append(
                    {
                        "type": "emitter_seen",
                        "band": band,
                        "freq_mhz": round(track.freq_hz / 1e6, 3),
                        "power_db": round(power_db, 1),
                        "snr_db": round(snr_db, 1),
                    }
                )

            self._trim(track)

            if self._should_classify(track):
                track.classified_drone = True
                confidence = min(
                    0.99,
                    0.4 + 0.05 * track.snapshot_count + 0.02 * track.mean_snr_db,
                )
                events.append(
                    {
                        "type": "drone_candidate",
                        "band": band,
                        "freq_mhz": round(track.freq_hz / 1e6, 3),
                        "pulse_rate_hz": track.estimate_pulse_rate_hz(),
                        "duration_s": round(track.duration_s, 1),
                        "mean_snr_db": round(track.mean_snr_db, 1),
                        "power_variance_db": round(track.power_variance_db, 1),
                        "confidence": round(confidence, 2),
                    }
                )

        events.extend(self._expire_missing_tracks(band, seen_keys, now))
        return events

    def active_drones(self) -> list[EmitterTrack]:
        return [track for track in self.tracks.values() if track.classified_drone]

    def reset(self) -> None:
        self.tracks.clear()

    def _key(self, band: str, freq_hz: float) -> tuple[str, int]:
        return band, int(freq_hz / self.DEDUP_HZ)

    def _trim(self, track: EmitterTrack) -> None:
        if len(track.powers_db) <= self.MAX_HISTORY:
            return
        track.powers_db = track.powers_db[-self.MAX_HISTORY :]
        track.snrs_db = track.snrs_db[-self.MAX_HISTORY :]
        track.freqs_hz = track.freqs_hz[-self.MAX_HISTORY :]

    def _should_classify(self, track: EmitterTrack) -> bool:
        if track.classified_drone:
            return False
        return (
            track.duration_s >= self.MIN_DURATION_S
            and track.snapshot_count >= self.MIN_SNAPSHOTS
            and track.mean_snr_db >= self.MIN_SNR_DB
            and track.freq_drift_hz <= self.MAX_FREQ_DRIFT_HZ
            and self.MIN_POWER_VARIANCE_DB
            <= track.power_variance_db
            <= self.MAX_POWER_VARIANCE_DB
        )

    def _expire_missing_tracks(
        self,
        band: str,
        seen_keys: set[tuple[str, int]],
        now: float,
    ) -> list[Event]:
        events: list[Event] = []

        for key in list(self.tracks.keys()):
            track = self.tracks[key]
            if track.band != band or key in seen_keys:
                continue
            if now - track.last_seen <= self.LOST_AFTER_S:
                continue

            if track.classified_drone:
                events.append(
                    {
                        "type": "drone_lost",
                        "band": track.band,
                        "freq_mhz": round(track.freq_hz / 1e6, 3),
                        "duration_total_s": round(track.duration_s, 1),
                    }
                )
            del self.tracks[key]

        return events


def _take_snapshot(sdr: Any) -> tuple[np.ndarray, np.ndarray]:
    n_samples = FFT_SIZE * AVERAGES
    iq = np.asarray(sdr.read_samples(n_samples), dtype=np.complex64)
    if iq.size < n_samples:
        return np.array([]), np.array([])

    window = np.hanning(FFT_SIZE).astype(np.float32)
    window_norm = float(np.sum(window**2))
    spectrum = np.zeros(FFT_SIZE, dtype=np.float32)

    for idx in range(AVERAGES):
        block = iq[idx * FFT_SIZE : (idx + 1) * FFT_SIZE] * window
        fft = np.fft.fftshift(np.fft.fft(block))
        spectrum += fft.real * fft.real + fft.imag * fft.imag

    power = spectrum / (AVERAGES * window_norm)
    power_db = 10.0 * np.log10(power + 1e-12)
    freqs = sdr.center_freq + np.fft.fftshift(
        np.fft.fftfreq(FFT_SIZE, 1.0 / SDR_RATE_HZ)
    )
    return freqs, power_db


def _find_peaks(
    freqs: np.ndarray,
    power_db: np.ndarray,
    snr_threshold_db: float,
) -> list[Peak]:
    if power_db.size == 0:
        return []

    edge = max(8, FFT_SIZE // 32)
    view = power_db[edge:-edge].copy()
    view_freqs = freqs[edge:-edge]

    midpoint = view.size // 2
    dc_skip = max(2, FFT_SIZE // 256)
    view[midpoint - dc_skip : midpoint + dc_skip] = -1e9

    sorted_power = np.sort(view[view > -1e8])
    if sorted_power.size == 0:
        return []

    noise_db = float(np.mean(sorted_power[: int(0.6 * sorted_power.size)]))
    snr_db = view - noise_db
    is_peak = (
        (snr_db > snr_threshold_db)
        & (view > np.roll(view, 1))
        & (view > np.roll(view, -1))
    )

    candidate_idxs = np.where(is_peak)[0]
    bin_hz = SDR_RATE_HZ / FFT_SIZE
    dedup_bins = max(1, int(50_000 / bin_hz))
    keep: list[int] = []

    for idx in sorted(candidate_idxs, key=lambda candidate: -view[candidate]):
        if all(abs(idx - kept) > dedup_bins for kept in keep):
            keep.append(idx)
        if len(keep) >= 16:
            break

    return [
        (float(view_freqs[idx]), float(view[idx]), float(snr_db[idx]))
        for idx in keep
    ]


class RadioModule:
    """Passive SDR drone-RF detector.

    The detector can run in two modes:
    - Active SDR scanning with start()/stop().
    - Data-input mode via ingest_peaks(), useful when another component already
      extracted RF peaks or when testing without hardware.
    """

    def __init__(
        self,
        bands: Optional[list[str]] = None,
        dwell_s: float = DEFAULT_DWELL_S,
        threshold_db: float = DEFAULT_THRESHOLD_DB,
        baseline_s: float = DEFAULT_BASELINE_S,
        on_event: Optional[EventCallback] = None,
        sdr_factory: Optional[SdrFactory] = None,
    ) -> None:
        if bands is None:
            bands = list(BANDS.keys())

        unknown_bands = [band for band in bands if band not in BANDS]
        if unknown_bands:
            raise ValueError(f"unknown radio bands: {unknown_bands}")
        if dwell_s <= 0:
            raise ValueError("dwell_s must be positive")
        if threshold_db <= 0:
            raise ValueError("threshold_db must be positive")
        if baseline_s < 0:
            raise ValueError("baseline_s must be non-negative")

        self._bands = list(bands)
        self._dwell_s = float(dwell_s)
        self._threshold_db = float(threshold_db)
        self._baseline_s = float(baseline_s)
        self._sdr_factory = sdr_factory

        self._callbacks: list[EventCallback] = []
        if on_event is not None:
            self._callbacks.append(on_event)

        self._sdr: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._classifier = _DroneClassifier()
        self._event_queue: queue.Queue[Event] = queue.Queue()
        self._lock = threading.Lock()
        self._current_band: str | None = None

    @property
    def current_band(self) -> str | None:
        return self._current_band

    def on_event(self, callback: EventCallback) -> None:
        """Register an event callback.

        The callback runs on the SDR thread. Keep it lightweight and hand work
        off to another queue if needed.
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="RadioModule",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def drain_events(self) -> list[Event]:
        """Return and clear all pending events without blocking."""
        events: list[Event] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def get_events(self) -> list[Event]:
        """Compatibility alias for drain_events()."""
        return self.drain_events()

    def active_drones(self) -> list[Event]:
        with self._lock:
            return [track.to_event() for track in self._classifier.active_drones()]

    def get_active_drones(self) -> list[Event]:
        """Compatibility alias for active_drones()."""
        return self.active_drones()

    def ingest_peaks(
        self,
        band: str,
        peaks: list[Peak],
        timestamp_s: float | None = None,
    ) -> list[Event]:
        """Feed precomputed RF peaks into the classifier.

        Args:
            band: One of "ism433" or "ism915".
            peaks: Tuples of (freq_hz, power_db, snr_db).
            timestamp_s: Monotonic timestamp. Defaults to time.monotonic().

        Returns:
            Events produced by this update. They are also queued and passed to
            registered callbacks.
        """
        if band not in BANDS:
            raise ValueError(f"unknown radio band: {band}")

        now = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            events = self._classifier.update(band, peaks, now)

        for event in events:
            self._emit(event)
        return events

    def reset_tracks(self) -> None:
        with self._lock:
            self._classifier.reset()

    def _run(self) -> None:
        try:
            self._open_sdr()
        except Exception as exc:
            self._emit({"type": "error", "stage": "open", "msg": str(exc)})
            return

        try:
            self._emit(
                {
                    "type": "started",
                    "tuner": self._safe_tuner_type(),
                    "rate_mhz": SDR_RATE_HZ / 1e6,
                }
            )
            self._baseline()
            self._monitor()
        finally:
            self._close_sdr()
            self._emit({"type": "stopped"})

    def _open_sdr(self) -> None:
        factory = self._sdr_factory
        if factory is None:
            factory = _load_rtlsdr()

        self._sdr = factory()
        self._sdr.sample_rate = SDR_RATE_HZ
        try:
            self._sdr.set_gain(49.6)
        except Exception:
            self._sdr.set_gain("auto")

    def _close_sdr(self) -> None:
        try:
            if self._sdr is not None:
                self._sdr.close()
        except Exception:
            pass
        finally:
            self._sdr = None

    def _safe_tuner_type(self) -> Any:
        try:
            if self._sdr is not None:
                return self._sdr.get_tuner_type()
        except Exception:
            pass
        return None

    def _baseline(self) -> None:
        if self._baseline_s == 0:
            self.reset_tracks()
            self._emit_baseline_done()
            return

        baseline_end = time.monotonic() + self._baseline_s
        while time.monotonic() < baseline_end and not self._stop_event.is_set():
            for band in self._bands:
                if self._stop_event.is_set() or time.monotonic() >= baseline_end:
                    break
                self._scan_once(band)

        self.reset_tracks()
        self._emit_baseline_done()

    def _emit_baseline_done(self) -> None:
        for band in self._bands:
            self._emit(
                {
                    "type": "baseline_done",
                    "band": band,
                    "label": BANDS[band]["label"],
                }
            )

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            for band in self._bands:
                if self._stop_event.is_set():
                    break
                dwell_until = time.monotonic() + self._dwell_s
                while not self._stop_event.is_set() and time.monotonic() < dwell_until:
                    self._scan_once(band)

    def _scan_once(self, band: str) -> None:
        if self._sdr is None or not self._tune(band):
            return

        freqs, power_db = _take_snapshot(self._sdr)
        peaks = _find_peaks(freqs, power_db, self._threshold_db)
        self.ingest_peaks(band, peaks, time.monotonic())

    def _tune(self, band: str) -> bool:
        if self._sdr is None:
            return False

        try:
            self._sdr.center_freq = BANDS[band]["center_hz"]
        except Exception as exc:
            self._emit(
                {
                    "type": "error",
                    "stage": "tune",
                    "band": band,
                    "msg": str(exc),
                }
            )
            return False

        self._current_band = band
        time.sleep(0.05)
        return True

    def _emit(self, event: Event) -> None:
        self._event_queue.put(event)
        for callback in list(self._callbacks):
            try:
                callback(event)
            except Exception:
                # Consumer callbacks should never kill SDR acquisition.
                pass


def _cli() -> None:
    radio = RadioModule(on_event=lambda event: print(json.dumps(event), flush=True))
    radio.start()
    try:
        while radio.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        radio.stop()


if __name__ == "__main__":
    try:
        _cli()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
