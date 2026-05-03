"""Boson 640 + Ouster lidar operator GUI.

Side-by-side layout:
  [ Thermal panel ] | [ Lidar panel ]
The lidar panel has a channel selector (range/signal/reflectivity/near-IR)
and a "3D View" button that opens a separate point-cloud window.

Run:
    python main.py
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from calibration_window import CalibrationWindow, load_calibration
from camera import (
    BosonControl,
    CameraThread,
    FrameInfo,
    list_video_devices,
    y16_to_celsius,
)
from fusion import Extrinsics, ThermalIntrinsics, rasterize_lidar_to_camera
from lidar import CHANNELS, LidarFrame, OusterThread
from pointcloud_window import PointCloudWindow
from rf_window import RFWindow
from web_bridge import WebBridge
from viewer import Viewer2D

DEFAULT_LIDAR_HOST = "169.254.62.165"


# --------------------------------------------------------------------------
# Thermal panel
# --------------------------------------------------------------------------

class ThermalPanel(QWidget):
    def __init__(self, web_bridge: Optional[WebBridge] = None, parent=None) -> None:
        super().__init__(parent)
        self.web_bridge = web_bridge
        self.cam_thread: Optional[CameraThread] = None
        self.boson = BosonControl()
        self.frame_info: Optional[FrameInfo] = None
        self.last_radiometric = False
        self.last_display_bgr: Optional[np.ndarray] = None  # rendered (colormapped) frame

        self.frame_count = 0
        self.fps_t0 = time.monotonic()
        self.fps = 0.0

        self.viewer = Viewer2D("Thermal — Boson 640")
        self.viewer.max_fps = 8.0
        self.viewer.set_overlay_callback(self._overlay)

        # Connection
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
        self.gain_combo = QComboBox()
        self.gain_combo.addItems(["auto", "high", "low"])
        self.gain_combo.currentTextChanged.connect(self._set_gain)
        self.ffc_btn = QPushButton("Trigger FFC")
        self.ffc_btn.clicked.connect(self._do_ffc)
        self.serial_status = QLabel(self.boson.status)
        self.serial_status.setWordWrap(True)
        pn = self.boson.part_number() or "-"
        sn = self.boson.serial_number() or "-"
        self.part_label = QLabel(pn)
        self.serial_label = QLabel(sn)

        ctl_box = QGroupBox("Camera control")
        cf = QFormLayout()
        cf.addRow("Status:", self.serial_status)
        cf.addRow("Part #:", self.part_label)
        cf.addRow("Serial #:", self.serial_label)
        cf.addRow("Gain mode:", self.gain_combo)
        cf.addRow(self.ffc_btn)
        ctl_box.setLayout(cf)

        # Status bar (per-panel mini status)
        self.status_lbl = QLabel("disconnected")
        self.status_lbl.setStyleSheet("color:#888;padding:2px 6px;")

        root = QVBoxLayout()
        root.addWidget(self.viewer, 1)
        root.addWidget(device_box)
        root.addWidget(ctl_box)
        root.addWidget(self.status_lbl)
        self.setLayout(root)

        self._refresh_devices()

        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._tick_status)
        self.fps_timer.start(500)

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

    def _on_cam_opened(self, info: FrameInfo) -> None:
        self.frame_info = info

    def _on_cam_error(self, msg: str) -> None:
        self.status_lbl.setText(f"camera: {msg}")
        self.connect_btn.setChecked(False)

    def _on_frame(self, frame: np.ndarray, radiometric: bool) -> None:
        self.last_radiometric = radiometric
        self.frame_count += 1
        if self.web_bridge is not None:
            self.web_bridge.update_thermal(frame)
        self.viewer.show_frame(frame)

    def _do_ffc(self) -> None:
        if self.boson.trigger_ffc():
            self.status_lbl.setText("FFC triggered")
        else:
            self.status_lbl.setText(f"FFC unavailable ({self.boson.status})")

    def _set_gain(self, mode: str) -> None:
        if self.boson.set_gain_mode(mode):
            self.status_lbl.setText(f"gain → {mode}")
        else:
            self.status_lbl.setText(f"gain unavailable ({self.boson.status})")

    def _overlay(self, display: np.ndarray, raw: np.ndarray) -> None:
        if not self.last_radiometric:
            return
        c = y16_to_celsius(raw)
        h, w = display.shape[:2]
        text = (
            f"min {float(np.min(c)):5.1f}C  "
            f"avg {float(np.mean(c)):5.1f}C  "
            f"max {float(np.max(c)):5.1f}C"
        )
        cv2.rectangle(display, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.putText(
            display, text, (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cur = self.viewer.video.cursor_xy
        if cur is not None:
            lw, lh = self.viewer.video.width(), self.viewer.video.height()
            x = int(cur.x() * w / max(lw, 1))
            y = int(cur.y() * h / max(lh, 1))
            if 0 <= x < w and 0 <= y < h:
                t = float(c[y, x])
                cv2.drawMarker(display, (x, y), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 1)
                cv2.putText(display, f"{t:.1f}C", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

    def _tick_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.fps_t0
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        self.frame_count = 0
        self.fps_t0 = now
        parts = [f"{self.fps:5.1f} fps"]
        if self.frame_info:
            mode = "Y16" if self.frame_info.radiometric else "8-bit"
            parts.append(f"{self.frame_info.width}x{self.frame_info.height} {mode}")
        parts.append(f"serial: {self.boson.status}")
        self.status_lbl.setText("  |  ".join(parts))

    def shutdown(self) -> None:
        self._stop_camera()
        self.boson.close()


# --------------------------------------------------------------------------
# Lidar panel
# --------------------------------------------------------------------------

class LidarPanel(QWidget):
    def __init__(self, thermal_provider=None, web_bridge: Optional[WebBridge] = None, parent=None) -> None:
        super().__init__(parent)
        self.thread: Optional[OusterThread] = None
        self.last_frame: Optional[LidarFrame] = None
        self.cloud_window: Optional[PointCloudWindow] = None
        self.thermal_provider = thermal_provider  # callable: () -> Optional[np.ndarray]
        self.web_bridge = web_bridge

        self.frame_count = 0
        self.fps_t0 = time.monotonic()
        self.fps = 0.0
        self._auto_opened_cloud = False
        self._last_2d_draw_t = 0.0
        self._last_cloud_update_t = 0.0
        self._2d_draw_interval = 1.0 / 6.0
        self._cloud_update_interval = 1.0 / 3.0

        self.viewer = Viewer2D("Lidar — Ouster")
        self.viewer.max_fps = 6.0
        self.viewer.palette_combo.setCurrentText("Turbo")

        # Connection
        self.host_edit = QLineEdit()
        self.host_edit.setText(DEFAULT_LIDAR_HOST)
        self.host_edit.setPlaceholderText("hostname or IP (e.g. os-122xxxx.local)")
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.toggled.connect(self._toggle_connect)

        conn_box = QGroupBox("Sensor")
        cl = QHBoxLayout()
        cl.addWidget(self.host_edit, 1)
        cl.addWidget(self.connect_btn)
        conn_box.setLayout(cl)

        # Channel selector + 3D button + calibration button
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(CHANNELS)
        self.channel_combo.setCurrentText("range")
        self.channel_combo.currentTextChanged.connect(self._redraw)
        self.fov_check = QCheckBox("Match camera FOV")
        self.fov_check.setToolTip(
            "Reproject the lidar through the thermal camera's pinhole intrinsics so "
            "this panel shows only the camera's field of view, pixel-aligned with "
            "the thermal image. Set lens/extrinsics in the calibration window."
        )
        self.fov_check.toggled.connect(self._redraw)
        self.cloud_btn = QPushButton("Open 3D View")
        self.cloud_btn.clicked.connect(self._open_cloud)
        self.calib_btn = QPushButton("Open Calibration")
        self.calib_btn.clicked.connect(self._open_calibration)
        self.rf_btn = QPushButton("Open RF Spectrum")
        self.rf_btn.clicked.connect(self._open_rf)
        self.cal_window: Optional[CalibrationWindow] = None
        self.rf_window: Optional[RFWindow] = None
        # Persistent calibration state shared between windows
        self.extr = Extrinsics()
        self.intr = ThermalIntrinsics()
        loaded = load_calibration()
        if loaded is not None:
            self.extr, self.intr = loaded

        ctl_box = QGroupBox("Display")
        df = QFormLayout()
        df.addRow("Channel:", self.channel_combo)
        df.addRow(self.fov_check)
        df.addRow(self.cloud_btn)
        df.addRow(self.calib_btn)
        df.addRow(self.rf_btn)
        ctl_box.setLayout(df)

        self.status_lbl = QLabel("disconnected")
        self.status_lbl.setStyleSheet("color:#888;padding:2px 6px;")

        root = QVBoxLayout()
        root.addWidget(self.viewer, 1)
        root.addWidget(conn_box)
        root.addWidget(ctl_box)
        root.addWidget(self.status_lbl)
        self.setLayout(root)

        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._tick_status)
        self.fps_timer.start(500)
        QTimer.singleShot(500, lambda: self.connect_btn.setChecked(True))

    def _toggle_connect(self, on: bool) -> None:
        if on:
            host = self.host_edit.text().strip()
            if not host:
                self.connect_btn.setChecked(False)
                QMessageBox.warning(self, "No sensor",
                                    "Enter a hostname or IP for the Ouster sensor.")
                return
            self.thread = OusterThread(hostname=host)
            self.thread.frame_ready.connect(self._on_frame)
            self.thread.error.connect(self._on_error)
            self.thread.info.connect(self._on_info)
            self.thread.start()
            self.connect_btn.setText("Disconnect")
            self.status_lbl.setText(f"connecting to {host}...")
        else:
            self._stop()
            self.connect_btn.setText("Connect")

    def _stop(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread = None

    def _on_info(self, msg: str) -> None:
        self.status_lbl.setText(f"connected: {msg}")

    def _on_error(self, msg: str) -> None:
        self.status_lbl.setText(f"lidar: {msg}")
        self.connect_btn.setChecked(False)

    def _on_frame(self, frame: LidarFrame) -> None:
        self.last_frame = frame
        self.frame_count += 1
        if self.web_bridge is not None:
            self.web_bridge.update_lidar(frame)

        now = time.monotonic()
        if now - self._last_2d_draw_t >= self._2d_draw_interval:
            self._last_2d_draw_t = now
            self._redraw()

        if self.web_bridge is not None and self.cloud_window is None and not self._auto_opened_cloud:
            self._auto_opened_cloud = True
            self._open_cloud()

        should_update_cloud = (
            self.cloud_window is not None
            and (self.cloud_window.isVisible() or self.web_bridge is not None)
            and now - self._last_cloud_update_t >= self._cloud_update_interval
        )
        if should_update_cloud:
            self._last_cloud_update_t = now
            self.cloud_window.update_cloud(frame)

    def _redraw(self) -> None:
        if self.last_frame is None:
            return
        channel = self.channel_combo.currentText()
        img = self.last_frame.channel(channel)
        if img is None:
            self.status_lbl.setText(f"channel '{channel}' not available")
            return
        if self.fov_check.isChecked() and self.last_frame.xyz is not None:
            reproj = rasterize_lidar_to_camera(
                self.last_frame.xyz, img, self.intr, self.extr
            )
            if reproj is not None:
                img = reproj
        self.viewer.show_frame(img)

    def _open_cloud(self) -> None:
        if self.cloud_window is None:
            self.cloud_window = PointCloudWindow(
                thermal_provider=self.thermal_provider,
                web_bridge=self.web_bridge,
            )
            self.cloud_window.set_calibration(self.extr, self.intr)
        self.cloud_window.show()
        self.cloud_window.raise_()
        self.cloud_window.activateWindow()
        if self.last_frame is not None:
            self.cloud_window.update_cloud(self.last_frame)

    def _open_calibration(self) -> None:
        if self.cal_window is None:
            self.cal_window = CalibrationWindow(
                extrinsics=self.extr,
                intrinsics=self.intr,
                lidar_provider=lambda: self.last_frame,
                thermal_provider=self.thermal_provider,
                on_change=self._apply_calibration,
            )
        self.cal_window.show()
        self.cal_window.raise_()
        self.cal_window.activateWindow()

    def _apply_calibration(self, extr: Extrinsics, intr: ThermalIntrinsics) -> None:
        self.extr = extr
        self.intr = intr
        if self.cloud_window is not None:
            self.cloud_window.set_calibration(extr, intr)
        if self.fov_check.isChecked():
            self._redraw()

    def _open_rf(self) -> None:
        if self.rf_window is None:
            self.rf_window = RFWindow()
        self.rf_window.show()
        self.rf_window.raise_()
        self.rf_window.activateWindow()

    def _tick_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.fps_t0
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        self.frame_count = 0
        self.fps_t0 = now
        if self.thread is not None and self.last_frame is not None:
            shape = ""
            if self.last_frame.range_img is not None:
                h, w = self.last_frame.range_img.shape
                shape = f" {w}x{h}"
            base = self.status_lbl.text().split("  |  ")[0]
            self.status_lbl.setText(f"{base}  |  {fps:4.1f} Hz{shape}")

    def shutdown(self) -> None:
        self._stop()
        if self.cloud_window is not None:
            self.cloud_window.close()
        if self.cal_window is not None:
            self.cal_window.close()
        if self.rf_window is not None:
            self.rf_window.close()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PIDS — Thermal + Lidar Operator GUI")
        self.web_bridge = WebBridge()
        self.web_bridge.start()

        self.thermal = ThermalPanel(web_bridge=self.web_bridge)
        self.lidar = LidarPanel(
            thermal_provider=lambda: self.thermal.viewer.last_display_bgr,
            web_bridge=self.web_bridge,
        )

        root = QHBoxLayout()
        root.addWidget(self.thermal, 1)
        root.addWidget(self.lidar, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("ready")

    def closeEvent(self, ev) -> None:
        self.thermal.shutdown()
        self.lidar.shutdown()
        super().closeEvent(ev)


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 820)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
