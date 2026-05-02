"""Boson 640 operator GUI.

Run:
    python -m boson_gui.main
or from inside the folder:
    python main.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from camera import (
    BosonControl,
    CameraThread,
    FrameInfo,
    autoscale_8bit,
    list_video_devices,
    y16_to_celsius,
)


PALETTES: dict[str, int | None] = {
    "White Hot": None,
    "Black Hot": -1,  # sentinel — handled specially
    "Iron (Inferno)": cv2.COLORMAP_INFERNO,
    "Hot": cv2.COLORMAP_HOT,
    "Rainbow (Jet)": cv2.COLORMAP_JET,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Turbo": cv2.COLORMAP_TURBO,
}


class VideoLabel(QLabel):
    """QLabel that tracks the cursor for radiometric pixel readout."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 512)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;color:#888;")
        self.setText("no signal")
        self.setMouseTracking(True)
        self.cursor_xy: QPoint | None = None

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        self.cursor_xy = ev.position().toPoint()
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev) -> None:
        self.cursor_xy = None
        super().leaveEvent(ev)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Boson 640 — Operator GUI")

        self.cam_thread: CameraThread | None = None
        self.boson = BosonControl()
        self.last_frame: np.ndarray | None = None
        self.last_radiometric = False
        self.frame_info: FrameInfo | None = None

        self.frame_count = 0
        self.fps_t0 = time.monotonic()
        self.fps = 0.0

        self.recording = False
        self.writer: cv2.VideoWriter | None = None

        self._build_ui()
        self._refresh_devices()

        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._tick_status)
        self.fps_timer.start(500)

    # --- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        self.video = VideoLabel()

        # Device controls
        self.device_combo = QComboBox()
        self.refresh_btn = QPushButton("Rescan")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.toggled.connect(self._toggle_connect)

        device_box = QGroupBox("Video device")
        dl = QHBoxLayout()
        dl.addWidget(self.device_combo, 1)
        dl.addWidget(self.refresh_btn)
        dl.addWidget(self.connect_btn)
        device_box.setLayout(dl)

        # Camera control (serial / flirpy)
        self.ffc_btn = QPushButton("Trigger FFC")
        self.ffc_btn.clicked.connect(self._do_ffc)
        self.gain_combo = QComboBox()
        self.gain_combo.addItems(["auto", "high", "low"])
        self.gain_combo.currentTextChanged.connect(self._set_gain)
        self.serial_status = QLabel(self.boson.status)
        self.serial_status.setWordWrap(True)
        pn = self.boson.part_number() or "-"
        sn = self.boson.serial_number() or "-"
        self.part_label = QLabel(pn)
        self.serial_label = QLabel(sn)

        ctl_box = QGroupBox("Camera control (serial)")
        cf = QFormLayout()
        cf.addRow("Status:", self.serial_status)
        cf.addRow("Part #:", self.part_label)
        cf.addRow("Serial #:", self.serial_label)
        cf.addRow("Gain mode:", self.gain_combo)
        cf.addRow(self.ffc_btn)
        ctl_box.setLayout(cf)

        # Display controls
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(PALETTES.keys())
        self.palette_combo.setCurrentText("Iron (Inferno)")
        self.invert_check = QCheckBox("Invert (black hot)")
        self.autoscale_check = QCheckBox("Auto-scale contrast")
        self.autoscale_check.setChecked(True)

        disp_box = QGroupBox("Display")
        df = QFormLayout()
        df.addRow("Palette:", self.palette_combo)
        df.addRow(self.invert_check)
        df.addRow(self.autoscale_check)
        disp_box.setLayout(df)

        # Capture
        self.snap_btn = QPushButton("Save snapshot")
        self.snap_btn.clicked.connect(self._save_snapshot)
        self.record_btn = QPushButton("Start recording")
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self._toggle_record)

        cap_box = QGroupBox("Capture")
        cl = QVBoxLayout()
        cl.addWidget(self.snap_btn)
        cl.addWidget(self.record_btn)
        cap_box.setLayout(cl)

        # Right column
        right = QVBoxLayout()
        right.addWidget(device_box)
        right.addWidget(ctl_box)
        right.addWidget(disp_box)
        right.addWidget(cap_box)
        right.addStretch(1)

        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(320)

        # Root
        root = QHBoxLayout()
        root.addWidget(self.video, 1)
        root.addWidget(right_w)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._update_status_text()

    # --- device handling ---------------------------------------------------

    def _refresh_devices(self) -> None:
        self.device_combo.clear()
        for idx in list_video_devices():
            self.device_combo.addItem(f"Device {idx}", idx)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("(none found)", -1)

    def _toggle_connect(self, on: bool) -> None:
        if on:
            idx = self.device_combo.currentData()
            if idx is None or idx < 0:
                self.connect_btn.setChecked(False)
                QMessageBox.warning(self, "No device", "No video device selected.")
                return
            self.cam_thread = CameraThread(device_index=int(idx))
            self.cam_thread.frame_ready.connect(self._on_frame)
            self.cam_thread.error.connect(self._on_cam_error)
            self.cam_thread.opened.connect(self._on_cam_opened)
            self.cam_thread.start()
            self.connect_btn.setText("Disconnect")
        else:
            self._stop_camera()
            self.connect_btn.setText("Connect")

    def _stop_camera(self) -> None:
        if self.cam_thread is not None:
            self.cam_thread.stop()
            self.cam_thread = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.recording = False
            self.record_btn.setChecked(False)
            self.record_btn.setText("Start recording")

    def _on_cam_opened(self, info: FrameInfo) -> None:
        self.frame_info = info
        mode = "RADIOMETRIC (Y16)" if info.radiometric else "8-bit AGC"
        self.status.showMessage(
            f"opened {info.width}x{info.height} {info.dtype} — {mode}"
        )

    def _on_cam_error(self, msg: str) -> None:
        self.status.showMessage(f"camera: {msg}", 5000)
        self.connect_btn.setChecked(False)

    # --- camera control ----------------------------------------------------

    def _do_ffc(self) -> None:
        if self.boson.trigger_ffc():
            self.status.showMessage("FFC triggered", 2000)
        else:
            self.status.showMessage(
                f"FFC unavailable ({self.boson.status})", 3000
            )

    def _set_gain(self, mode: str) -> None:
        if self.boson.set_gain_mode(mode):
            self.status.showMessage(f"gain mode → {mode}", 2000)
        else:
            self.status.showMessage(
                f"gain change unavailable ({self.boson.status})", 3000
            )

    # --- frames ------------------------------------------------------------

    def _on_frame(self, frame: np.ndarray, radiometric: bool) -> None:
        self.last_frame = frame
        self.last_radiometric = radiometric
        self.frame_count += 1

        display = self._render(frame, radiometric)
        if self.recording and self.writer is not None:
            self.writer.write(display)

        self._show(display)

    def _render(self, frame: np.ndarray, radiometric: bool) -> np.ndarray:
        # Normalize to 8-bit single-channel for colormapping.
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

        palette_name = self.palette_combo.currentText()
        cmap = PALETTES[palette_name]
        if cmap is None:
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif cmap == -1:  # Black Hot — invert then grayscale
            display = cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)
        else:
            display = cv2.applyColorMap(gray, cmap)

        if radiometric:
            self._overlay_radiometric(display, frame)
        return display

    def _overlay_radiometric(self, display: np.ndarray, raw: np.ndarray) -> None:
        c = y16_to_celsius(raw)
        tmin = float(np.min(c))
        tmax = float(np.max(c))
        tavg = float(np.mean(c))

        h, w = display.shape[:2]
        text = f"min {tmin:5.1f}C  avg {tavg:5.1f}C  max {tmax:5.1f}C"
        cv2.rectangle(display, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.putText(
            display, text, (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Cursor temperature readout
        if self.video.cursor_xy is not None:
            lw, lh = self.video.width(), self.video.height()
            x = int(self.video.cursor_xy.x() * w / lw)
            y = int(self.video.cursor_xy.y() * h / lh)
            if 0 <= x < w and 0 <= y < h:
                t = float(c[y, x])
                cv2.drawMarker(display, (x, y), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 1)
                label = f"{t:.1f}C"
                cv2.putText(display, label, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

    def _show(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video.width(), self.video.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.video.setPixmap(pix)

    # --- capture -----------------------------------------------------------

    def _captures_dir(self) -> Path:
        d = Path(__file__).parent / "captures"
        d.mkdir(exist_ok=True)
        return d

    def _save_snapshot(self) -> None:
        if self.last_frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._captures_dir()

        display = self._render(self.last_frame, self.last_radiometric)
        cv2.imwrite(str(out_dir / f"snap_{ts}.png"), display)

        # Save raw too — TIFF preserves 16-bit if radiometric
        if self.last_radiometric:
            cv2.imwrite(str(out_dir / f"snap_{ts}_raw.tiff"), self.last_frame)
            celsius = y16_to_celsius(self.last_frame)
            np.save(out_dir / f"snap_{ts}_celsius.npy", celsius)

        self.status.showMessage(f"saved snap_{ts}.*", 3000)

    def _toggle_record(self, on: bool) -> None:
        if on:
            if self.last_frame is None:
                self.record_btn.setChecked(False)
                return
            display = self._render(self.last_frame, self.last_radiometric)
            h, w = display.shape[:2]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._captures_dir() / f"rec_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
            self.recording = True
            self.record_btn.setText("Stop recording")
            self.status.showMessage(f"recording → {path.name}", 3000)
        else:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.recording = False
            self.record_btn.setText("Start recording")
            self.status.showMessage("recording stopped", 2000)

    # --- status ------------------------------------------------------------

    def _tick_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.fps_t0
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        self.frame_count = 0
        self.fps_t0 = now
        self._update_status_text()

    def _update_status_text(self) -> None:
        parts = [f"{self.fps:5.1f} fps"]
        if self.frame_info:
            mode = "Y16" if self.frame_info.radiometric else "8-bit"
            parts.append(f"{self.frame_info.width}x{self.frame_info.height} {mode}")
        parts.append(f"serial: {self.boson.status}")
        if self.recording:
            parts.append("REC")
        self.status.showMessage("  |  ".join(parts))

    def closeEvent(self, ev) -> None:
        self._stop_camera()
        self.boson.close()
        super().closeEvent(ev)


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 720)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
