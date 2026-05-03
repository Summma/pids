"""Shared 2D image viewer widget used by both thermal and lidar panels."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


PALETTES: dict[str, int | None] = {
    "White Hot": None,
    "Black Hot": -1,  # sentinel — handled in render
    "Iron (Inferno)": cv2.COLORMAP_INFERNO,
    "Hot": cv2.COLORMAP_HOT,
    "Rainbow (Jet)": cv2.COLORMAP_JET,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Turbo": cv2.COLORMAP_TURBO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
}


def autoscale_8bit(frame: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    """Linearly stretch a >8-bit frame into 8-bit using percentile bounds."""
    f = frame.astype(np.float32)
    lo, hi = np.percentile(f, (lo_pct, hi_pct))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    out = np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255)
    return out.astype(np.uint8)


def captures_dir() -> Path:
    d = Path(__file__).parent / "captures"
    d.mkdir(exist_ok=True)
    return d


class VideoLabel(QLabel):
    """QLabel that tracks the cursor (used for radiometric pixel readout)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;color:#888;")
        self.setText("no signal")
        self.setMouseTracking(True)
        self.cursor_xy: Optional[QPoint] = None

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        self.cursor_xy = ev.position().toPoint()
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev) -> None:
        self.cursor_xy = None
        super().leaveEvent(ev)


class Viewer2D(QWidget):
    """A reusable 2D image viewer: video label + palette/invert/autoscale + snapshot.

    Caller pushes frames via show_frame(); an optional overlay_callback runs
    after colormap so panels can draw modality-specific overlays
    (radiometric stats, lidar range labels, etc.).
    """

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.last_frame: Optional[np.ndarray] = None
        self.last_display_bgr: Optional[np.ndarray] = None  # last rendered BGR frame
        self.overlay_cb: Optional[Callable[[np.ndarray, np.ndarray], None]] = None
        self.max_fps = 15.0
        self._last_show_t = 0.0

        self.video = VideoLabel()

        self.palette_combo = QComboBox()
        self.palette_combo.addItems(PALETTES.keys())
        self.palette_combo.setCurrentText("Iron (Inferno)")
        self.invert_check = QCheckBox("Invert")
        self.autoscale_check = QCheckBox("Auto-scale")
        self.autoscale_check.setChecked(True)
        self.snap_btn = QPushButton("Snapshot")
        self.snap_btn.clicked.connect(self._snapshot)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Palette:"))
        controls.addWidget(self.palette_combo)
        controls.addWidget(self.invert_check)
        controls.addWidget(self.autoscale_check)
        controls.addStretch(1)
        controls.addWidget(self.snap_btn)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight:bold;font-size:14px;padding:4px;")

        root = QVBoxLayout()
        root.addWidget(title_lbl)
        root.addWidget(self.video, 1)
        root.addLayout(controls)
        self.setLayout(root)

    def set_overlay_callback(
        self, cb: Optional[Callable[[np.ndarray, np.ndarray], None]]
    ) -> None:
        self.overlay_cb = cb

    def show_frame(self, frame: np.ndarray) -> None:
        self.last_frame = frame
        now = time.monotonic()
        min_interval = 1.0 / max(self.max_fps, 1.0)
        if now - self._last_show_t < min_interval:
            return
        self._last_show_t = now

        display = self._render(frame)
        if self.overlay_cb is not None:
            self.overlay_cb(display, frame)
        self.last_display_bgr = display
        self._show(display)

    def _render(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if gray.dtype != np.uint8:
            if self.autoscale_check.isChecked():
                gray = autoscale_8bit(gray)
            else:
                gray = np.clip(gray >> 8, 0, 255).astype(np.uint8)
        if self.invert_check.isChecked():
            gray = 255 - gray
        cmap = PALETTES[self.palette_combo.currentText()]
        if cmap is None:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if cmap == -1:
            return cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)
        return cv2.applyColorMap(gray, cmap)

    def _show(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video.width(),
            self.video.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video.setPixmap(pix)

    def _snapshot(self) -> None:
        if self.last_frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self.title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out = captures_dir() / f"{slug}_{ts}.png"
        display = self._render(self.last_frame)
        if self.overlay_cb is not None:
            self.overlay_cb(display, self.last_frame)
        cv2.imwrite(str(out), display)
        if self.last_frame.dtype != np.uint8:
            cv2.imwrite(str(out.with_suffix(".tiff")), self.last_frame)
