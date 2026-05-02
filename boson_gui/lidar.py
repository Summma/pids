"""Ouster lidar I/O.

Wraps `ouster.sdk.open_source` so the GUI gets a Qt thread that emits
LidarFrame objects containing the standard 2D channels (range, signal,
reflectivity, near-IR) plus the projected XYZ point cloud for the 3D view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    from ouster.sdk import open_source
    from ouster.sdk.core import ChanField, XYZLut
    HAS_OUSTER = True
except Exception:
    open_source = None  # type: ignore
    ChanField = None  # type: ignore
    XYZLut = None  # type: ignore
    HAS_OUSTER = False


CHANNELS = ["range", "signal", "reflectivity", "near_ir"]


@dataclass
class LidarFrame:
    range_img: Optional[np.ndarray]
    signal_img: Optional[np.ndarray]
    reflectivity_img: Optional[np.ndarray]
    nearir_img: Optional[np.ndarray]
    xyz: Optional[np.ndarray]  # (H, W, 3) in meters
    beam_altitudes: Optional[np.ndarray] = None  # (H,) per-row altitude in degrees

    def channel(self, name: str) -> Optional[np.ndarray]:
        return {
            "range": self.range_img,
            "signal": self.signal_img,
            "reflectivity": self.reflectivity_img,
            "near_ir": self.nearir_img,
        }.get(name)


class OusterThread(QThread):
    """Background thread that connects to a sensor and emits LidarFrame."""

    frame_ready = Signal(object)  # LidarFrame
    error = Signal(str)
    info = Signal(str)

    def __init__(self, hostname: str, parent=None) -> None:
        super().__init__(parent)
        self.hostname = hostname
        self._running = False
        self._source = None

    def run(self) -> None:
        if not HAS_OUSTER:
            self.error.emit("ouster-sdk not installed (pip install ouster-sdk)")
            return

        try:
            self._source = open_source(self.hostname)
        except Exception as e:
            self.error.emit(f"connect failed: {e}")
            return

        try:
            meta = self._source.sensor_info
            if isinstance(meta, list):
                meta = meta[0]
            prod = getattr(meta, "prod_line", "ouster")
            sn = getattr(meta, "sn", "?")
            self.info.emit(f"{prod} sn={sn}")
            xyz_lut = XYZLut(meta)
            beam_alts = getattr(getattr(meta, "beam_intrinsics", None),
                                "beam_altitude_angles", None)
            if beam_alts is not None:
                beam_alts = np.asarray(beam_alts, dtype=np.float32)
        except Exception as e:
            self.error.emit(f"metadata error: {e}")
            return

        self._running = True
        try:
            for scan_set in self._source:
                if not self._running:
                    break
                if scan_set is None:
                    continue
                # ouster-sdk 0.16 yields a LidarScanSet (iterable of LidarScan,
                # one per sensor); for our single-sensor case take the first valid one.
                scan = next((s for s in scan_set if s is not None), None)
                if scan is None:
                    continue
                frame = self._extract(scan, xyz_lut)
                frame.beam_altitudes = beam_alts
                self.frame_ready.emit(frame)
        except Exception as e:
            if self._running:
                self.error.emit(f"stream error: {e}")
        finally:
            try:
                self._source.close()
            except Exception:
                pass
            self._source = None

    @staticmethod
    def _extract(scan, xyz_lut) -> LidarFrame:
        def field_or_none(attr: str) -> Optional[np.ndarray]:
            try:
                return scan.field(getattr(ChanField, attr))
            except Exception:
                return None

        try:
            xyz = xyz_lut(scan)
        except Exception:
            xyz = None

        return LidarFrame(
            range_img=field_or_none("RANGE"),
            signal_img=field_or_none("SIGNAL"),
            reflectivity_img=field_or_none("REFLECTIVITY"),
            nearir_img=field_or_none("NEAR_IR"),
            xyz=xyz,
        )

    def stop(self) -> None:
        self._running = False
        try:
            if self._source is not None:
                self._source.close()
        except Exception:
            pass
        self.wait(2000)
