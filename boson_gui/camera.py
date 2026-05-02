"""Camera I/O for the FLIR Boson 640.

Two channels:
  - Video: UVC stream via OpenCV. On Linux this can deliver Y16 (16-bit
    radiometric); on macOS the UVC driver usually only exposes 8-bit YUYV.
  - Control: FSLP over the Boson's USB CDC serial port (FFC, gain mode,
    part number, serial). Wrapped by flirpy when available.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    from flirpy.camera.boson import Boson
    HAS_FLIRPY = True
except Exception:
    Boson = None  # type: ignore
    HAS_FLIRPY = False


@dataclass
class FrameInfo:
    width: int
    height: int
    dtype: str
    radiometric: bool


def list_video_devices(max_index: int = 6) -> list[int]:
    """Probe device indices and return those that successfully open a frame."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    found: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
            cap.release()
    return found


class CameraThread(QThread):
    """Grabs frames in a background thread and emits them to the GUI."""

    frame_ready = Signal(np.ndarray, bool)  # frame, is_radiometric
    error = Signal(str)
    opened = Signal(object)  # FrameInfo

    def __init__(self, device_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.device_index = device_index
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.device_index, backend)
        if not cap.isOpened():
            self.error.emit(f"Could not open video device index {self.device_index}")
            return

        # Try to coax the driver into delivering raw Y16 (radiometric).
        # On macOS this is typically ignored; we detect post-hoc from dtype.
        y16 = cv2.VideoWriter_fourcc("Y", "1", "6", " ")
        cap.set(cv2.CAP_PROP_FOURCC, y16)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        self._cap = cap
        self._running = True

        # Emit one info packet after first successful read.
        info_emitted = False

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.error.emit("Frame grab failed")
                break

            radiometric = frame.dtype == np.uint16
            if not info_emitted:
                h, w = frame.shape[:2]
                self.opened.emit(
                    FrameInfo(w, h, str(frame.dtype), radiometric)
                )
                info_emitted = True

            self.frame_ready.emit(frame, radiometric)

        cap.release()
        self._cap = None

    def stop(self) -> None:
        self._running = False
        self.wait(1500)


class BosonControl:
    """Thin wrapper over flirpy's Boson serial control.

    Methods are no-ops (with a logged reason) when flirpy isn't installed
    or the serial port can't be opened, so the GUI stays usable as a
    plain video viewer.
    """

    def __init__(self) -> None:
        self._cam = None
        self._reason: str | None = None
        self._connect()

    def _connect(self) -> None:
        if not HAS_FLIRPY:
            self._reason = "flirpy not installed (pip install flirpy)"
            return
        try:
            self._cam = Boson()  # autodetects the Boson serial port
        except Exception as e:
            self._reason = f"serial connect failed: {e}"
            self._cam = None

    @property
    def connected(self) -> bool:
        return self._cam is not None

    @property
    def status(self) -> str:
        if self._cam is None:
            return self._reason or "not connected"
        return "connected"

    def part_number(self) -> str | None:
        if not self._cam:
            return None
        try:
            return self._cam.get_camera_part_number()
        except Exception:
            return None

    def serial_number(self) -> str | None:
        if not self._cam:
            return None
        try:
            return str(self._cam.get_camera_serial())
        except Exception:
            return None

    def trigger_ffc(self) -> bool:
        if not self._cam:
            return False
        try:
            self._cam.do_ffc()
            return True
        except Exception:
            return False

    def set_gain_mode(self, mode: str) -> bool:
        """mode in {"high", "low", "auto"}."""
        if not self._cam:
            return False
        mapping = {"high": 0, "low": 1, "auto": 2}
        if mode not in mapping:
            return False
        try:
            self._cam.set_gain_mode(mapping[mode])
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None


# --- radiometric helpers ---------------------------------------------------

def y16_to_celsius(frame: np.ndarray) -> np.ndarray:
    """Boson Y16 radiometric: pixel value is centi-Kelvin (cK)."""
    return frame.astype(np.float32) * 0.01 - 273.15
