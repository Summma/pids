"""Interactive thermal-lidar calibration window.

Renders the lidar panorama reprojected into the thermal camera's pinhole
view, lets the user nudge 6-DoF extrinsics + lens until features line up
with the live thermal frame. Modes: side-by-side, alpha blend, lidar-only,
thermal-only, depth-edge overlay (Canny on synthesized lidar range).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fusion import (
    LENS_PRESETS,
    Extrinsics,
    R_LIDAR_TO_CAM_BASE,
    ThermalIntrinsics,
    euler_to_R,
    intrinsics_from_hfov,
)
from lidar import LidarFrame
from viewer import autoscale_8bit


CHANNELS = ["signal", "near_ir", "reflectivity", "range"]
VIEW_MODES = ["Edges on thermal", "Blend", "Side-by-side", "Lidar only", "Thermal only"]

CALIB_PATH = Path(__file__).parent / "calibration.json"


def save_calibration(extr: Extrinsics, intr: ThermalIntrinsics, path: Path = CALIB_PATH) -> None:
    path.write_text(json.dumps({
        "intrinsics": {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
        },
        "extrinsics": {
            "tx": extr.tx, "ty": extr.ty, "tz": extr.tz,
            "roll_deg": extr.roll_deg, "pitch_deg": extr.pitch_deg, "yaw_deg": extr.yaw_deg,
        },
    }, indent=2))


def load_calibration(path: Path = CALIB_PATH) -> Optional[tuple[Extrinsics, ThermalIntrinsics]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return (
            Extrinsics(**data["extrinsics"]),
            ThermalIntrinsics(**data["intrinsics"]),
        )
    except Exception:
        return None


class CalibrationWindow(QWidget):
    def __init__(
        self,
        extrinsics: Extrinsics,
        intrinsics: ThermalIntrinsics,
        lidar_provider: Callable[[], Optional[LidarFrame]],
        thermal_provider: Callable[[], Optional[np.ndarray]],
        on_change: Callable[[Extrinsics, ThermalIntrinsics], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Thermal–Lidar Calibration")
        self.resize(1500, 820)

        # Working copies
        self.extr = Extrinsics(**extrinsics.__dict__)
        self.intr = ThermalIntrinsics(**intrinsics.__dict__)
        self.lidar_provider = lidar_provider
        self.thermal_provider = thermal_provider
        self.on_change = on_change

        # Cache for altitude → row LUT, keyed by id(beam_altitudes array)
        self._alt_lut_key: Optional[int] = None
        self._alt_lut: Optional[np.ndarray] = None
        self._alt_lo = 0.0
        self._alt_hi = 0.0

        self._build_ui()
        self._sync_widgets()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._render)
        self.timer.start(100)  # 10 Hz

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.image_lbl = QLabel("waiting for frames…")
        self.image_lbl.setAlignment(Qt.AlignCenter)
        self.image_lbl.setMinimumSize(900, 720)
        self.image_lbl.setStyleSheet("background:#111;color:#888;")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(VIEW_MODES)
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(CHANNELS)

        view = QFormLayout()
        view.addRow("Mode:", self.mode_combo)
        view.addRow("Blend α:", self.blend_slider)
        view.addRow("Lidar channel:", self.channel_combo)
        view_box = QGroupBox("View")
        view_box.setLayout(view)

        self.lens_combo = QComboBox()
        self.lens_combo.addItems(LENS_PRESETS.keys())
        self.lens_combo.setCurrentText("Boson 14mm (~50°)")
        self.lens_combo.currentTextChanged.connect(self._on_lens)

        self.tx = self._dspin(-2, 2, 0.005)
        self.ty = self._dspin(-2, 2, 0.005)
        self.tz = self._dspin(-2, 2, 0.005)
        self.roll = self._dspin(-180, 180, 0.5)
        self.pitch = self._dspin(-180, 180, 0.5)
        self.yaw = self._dspin(-180, 180, 0.5)

        ex = QFormLayout()
        ex.addRow("Lens:", self.lens_combo)
        ex.addRow("tx (m):", self.tx)
        ex.addRow("ty (m):", self.ty)
        ex.addRow("tz (m):", self.tz)
        ex.addRow("roll (°):", self.roll)
        ex.addRow("pitch (°):", self.pitch)
        ex.addRow("yaw (°):", self.yaw)
        ex_box = QGroupBox("Extrinsics")
        ex_box.setLayout(ex)

        self.save_btn = QPushButton("Save")
        self.load_btn = QPushButton("Load")
        self.reset_btn = QPushButton("Reset")
        self.save_btn.clicked.connect(self._save)
        self.load_btn.clicked.connect(self._load)
        self.reset_btn.clicked.connect(self._reset)
        sl = QHBoxLayout()
        sl.addWidget(self.save_btn)
        sl.addWidget(self.load_btn)
        sl.addWidget(self.reset_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#888;padding:2px;")

        right = QVBoxLayout()
        right.addWidget(view_box)
        right.addWidget(ex_box)
        right.addLayout(sl)
        right.addWidget(self.status_lbl)
        right.addStretch(1)
        right_w = QWidget()
        right_w.setFixedWidth(320)
        right_w.setLayout(right)

        root = QHBoxLayout()
        root.addWidget(self.image_lbl, 1)
        root.addWidget(right_w)
        self.setLayout(root)

    def _dspin(self, lo, hi, step) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(3)
        s.valueChanged.connect(self._on_extr)
        return s

    def _sync_widgets(self) -> None:
        for w, v in [
            (self.tx, self.extr.tx), (self.ty, self.extr.ty), (self.tz, self.extr.tz),
            (self.roll, self.extr.roll_deg),
            (self.pitch, self.extr.pitch_deg),
            (self.yaw, self.extr.yaw_deg),
        ]:
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    # ---------------- callbacks ----------------

    def _on_extr(self) -> None:
        self.extr.tx = self.tx.value()
        self.extr.ty = self.ty.value()
        self.extr.tz = self.tz.value()
        self.extr.roll_deg = self.roll.value()
        self.extr.pitch_deg = self.pitch.value()
        self.extr.yaw_deg = self.yaw.value()
        self.on_change(self.extr, self.intr)

    def _on_lens(self, name: str) -> None:
        self.intr = intrinsics_from_hfov(self.intr.width, self.intr.height, LENS_PRESETS[name])
        self.on_change(self.extr, self.intr)

    def _reset(self) -> None:
        self.extr = Extrinsics()
        self._sync_widgets()
        self._on_extr()

    def _save(self) -> None:
        save_calibration(self.extr, self.intr)
        self.status_lbl.setText(f"saved → {CALIB_PATH.name}")

    def _load(self) -> None:
        loaded = load_calibration()
        if loaded is None:
            self.status_lbl.setText("no calibration.json found")
            return
        self.extr, self.intr = loaded
        self._sync_widgets()
        # Sync lens combo to whatever HFOV the loaded intr corresponds to
        self.on_change(self.extr, self.intr)
        self.status_lbl.setText(f"loaded ← {CALIB_PATH.name}")

    # ---------------- rendering ----------------

    def _render(self) -> None:
        lf = self.lidar_provider()
        therm = self.thermal_provider()
        if lf is None and therm is None:
            return

        if therm is not None:
            therm = cv2.resize(therm, (self.intr.width, self.intr.height))

        mode = self.mode_combo.currentText()
        if mode == "Thermal only":
            display = therm if therm is not None else self._blank()
        elif mode == "Lidar only":
            display = self._synth(lf, self.channel_combo.currentText()) or self._blank()
        elif mode == "Side-by-side":
            l = self._synth(lf, self.channel_combo.currentText())
            display = self._side_by_side(l, therm)
        elif mode == "Blend":
            l = self._synth(lf, self.channel_combo.currentText())
            alpha = self.blend_slider.value() / 100.0
            display = self._blend(l, therm, alpha)
        else:  # Edges on thermal
            display = self._edges(lf, therm)

        self._show(display)

    def _blank(self) -> np.ndarray:
        return np.zeros((self.intr.height, self.intr.width, 3), dtype=np.uint8)

    def _altitudes_to_rows_lut(self, beam_alts: np.ndarray) -> tuple[np.ndarray, float, float]:
        key = id(beam_alts)
        if self._alt_lut is not None and key == self._alt_lut_key:
            return self._alt_lut, self._alt_lo, self._alt_hi
        lo, hi = float(beam_alts.min()), float(beam_alts.max())
        bins = 4096
        alts = np.linspace(lo, hi, bins, dtype=np.float32)
        lut = np.argmin(np.abs(alts[:, None] - beam_alts[None, :]), axis=1).astype(np.int32)
        self._alt_lut = lut
        self._alt_lut_key = key
        self._alt_lo = lo
        self._alt_hi = hi
        return lut, lo, hi

    def _synth(self, lf: Optional[LidarFrame], channel: str) -> Optional[np.ndarray]:
        """Project lidar XYZ points into the thermal pinhole view (BGR uint8).

        Uses each point's actual 3D position (XYZLut already handles per-beam
        azimuth offsets), projects it through (intr, extr), and splats the
        chosen channel value at the resulting thermal pixel. A Z-buffer keeps
        the closest point per pixel; a 3×3 dilation fills small gaps so
        sparse coverage at the edges of FOV is still visible.
        """
        if lf is None or lf.xyz is None:
            return None
        src = lf.channel(channel)
        if src is None:
            src = lf.range_img
        if src is None:
            return None

        xyz = lf.xyz.reshape(-1, 3).astype(np.float64)
        vals = src.reshape(-1).astype(np.float32)
        rng = np.linalg.norm(xyz, axis=1)
        keep = rng > 0.3
        xyz = xyz[keep]
        vals = vals[keep]
        rng = rng[keep]
        if xyz.size == 0:
            return None

        R_user = euler_to_R(
            np.deg2rad(self.extr.roll_deg),
            np.deg2rad(self.extr.pitch_deg),
            np.deg2rad(self.extr.yaw_deg),
        )
        R = R_user @ R_LIDAR_TO_CAM_BASE
        t = np.array([self.extr.tx, self.extr.ty, self.extr.tz])

        P_cam = (R @ xyz.T).T + t
        Z = P_cam[:, 2]
        in_front = Z > 0.05
        safe_z = np.where(in_front, Z, 1.0)
        u = (self.intr.fx * P_cam[:, 0] / safe_z + self.intr.cx).astype(np.int32)
        v = (self.intr.fy * P_cam[:, 1] / safe_z + self.intr.cy).astype(np.int32)
        in_img = (u >= 0) & (u < self.intr.width) & (v >= 0) & (v < self.intr.height)
        ok = in_front & in_img
        if not ok.any():
            return None

        u_ok = u[ok]
        v_ok = v[ok]
        vals_ok = vals[ok]
        Z_ok = Z[ok]

        # Z-buffer: sort by depth descending so the nearest point overwrites last.
        order = np.argsort(-Z_ok)
        u_ord = u_ok[order]
        v_ord = v_ok[order]
        vals_ord = vals_ok[order]

        H, W = self.intr.height, self.intr.width
        img = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)
        img[v_ord, u_ord] = vals_ord
        mask[v_ord, u_ord] = True

        # Autoscale only where we have data, then dilate to fill small gaps.
        if mask.any():
            v_lo, v_hi = np.percentile(img[mask], (2, 98))
        else:
            v_lo, v_hi = 0.0, 1.0
        if v_hi - v_lo < 1e-6:
            v_hi = v_lo + 1.0
        gray = np.zeros((H, W), dtype=np.uint8)
        gray[mask] = np.clip(
            (img[mask] - v_lo) * (255.0 / (v_hi - v_lo)), 0, 255
        ).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.dilate(gray, kernel)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    def _side_by_side(self, lidar, thermal) -> np.ndarray:
        if lidar is None:
            lidar = self._blank()
        if thermal is None:
            thermal = self._blank()
        H = max(lidar.shape[0], thermal.shape[0])
        W = lidar.shape[1] + thermal.shape[1]
        out = np.zeros((H, W, 3), dtype=np.uint8)
        out[: lidar.shape[0], : lidar.shape[1]] = lidar
        out[: thermal.shape[0], lidar.shape[1] :] = thermal
        cv2.putText(out, "LIDAR", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "THERMAL", (lidar.shape[1] + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _blend(self, lidar, thermal, alpha) -> np.ndarray:
        if lidar is None and thermal is None:
            return self._blank()
        if lidar is None:
            return thermal
        if thermal is None:
            return lidar
        return cv2.addWeighted(lidar, alpha, thermal, 1.0 - alpha, 0)

    def _edges(self, lf: Optional[LidarFrame], thermal: Optional[np.ndarray]) -> np.ndarray:
        if thermal is None:
            return self._synth(lf, self.channel_combo.currentText()) or self._blank()
        # Edge map from synthesized lidar range channel
        range_view = self._synth(lf, "range")
        out = thermal.copy()
        if range_view is not None:
            gray = cv2.cvtColor(range_view, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            out[edges > 0] = (0, 255, 0)  # green
        return out

    def _show(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.image_lbl.width(), self.image_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.image_lbl.setPixmap(pix)

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        super().closeEvent(ev)
