"""Live point cloud popup window.

Features beyond a basic viewer:
  - DBSCAN clustering with stable per-track IDs and frame-to-frame velocity.
  - Per-cluster bounding boxes drawn in the scene.
  - Click-to-select a cluster (info panel shows id/range/size/speed/points).
  - Color modes: range/height/reflectivity/signal/near_ir/cluster_id/THERMAL
    (thermal mode projects the live thermal frame onto the cloud using a
     6-DoF calibration tweakable from the right-hand panel).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph.opengl as gl
    HAS_PG = True
except Exception:
    gl = None
    HAS_PG = False

from clustering import Cluster, ClusterParams, ClusterWorker, HAS_SKLEARN
from fusion import (
    LENS_PRESETS,
    Extrinsics,
    ThermalIntrinsics,
    camera_frustum_lines,
    colorize_with_thermal,
    intrinsics_from_hfov,
    project_points,
)
from lidar import LidarFrame


COLOR_MODES = ["range", "height", "reflectivity", "signal", "near_ir", "cluster_id", "thermal"]


def _turbo_like(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    out = np.empty((v.size, 4), dtype=np.float32)
    out[:, 0] = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
    out[:, 1] = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
    out[:, 2] = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
    out[:, 3] = 1.0
    return out


def _hsv_palette(n: int) -> np.ndarray:
    """N evenly-spaced RGBA colors via HSV."""
    h = (np.arange(n) * 0.61803398875) % 1.0  # golden-ratio hue spread
    s = np.full(n, 0.85)
    v = np.full(n, 0.95)
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i % 6, [v, q, p, p, t, v])
    g = np.choose(i % 6, [t, v, v, q, p, p])
    b = np.choose(i % 6, [p, p, t, v, v, q])
    out = np.ones((n, 4), dtype=np.float32)
    out[:, 0] = r
    out[:, 1] = g
    out[:, 2] = b
    return out


# ----- pickable GL view -----------------------------------------------------

class PickableGLView(gl.GLViewWidget if HAS_PG else QWidget):
    """GLViewWidget that fires `picked(idx)` when the user clicks on a cluster.

    Click-vs-drag is detected by movement: if mouse moves <5 px between press
    and release, treat as a pick; otherwise let the parent rotate the camera.
    """

    picked = Signal(int)  # index into the picking_centroids list (-1 = none)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.picking_centroids: np.ndarray = np.empty((0, 3))  # (N, 3)
        self._press_pos = None

    def set_picking(self, centroids: np.ndarray) -> None:
        self.picking_centroids = centroids

    def mousePressEvent(self, ev) -> None:
        self._press_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        super().mouseReleaseEvent(ev)
        if self._press_pos is None:
            return
        dx = ev.position().x() - self._press_pos.x()
        dy = ev.position().y() - self._press_pos.y()
        moved = (dx * dx + dy * dy) > 25  # 5 px squared
        self._press_pos = None
        if moved or ev.button() != Qt.LeftButton or len(self.picking_centroids) == 0:
            return
        # Project centroids to screen, find nearest within threshold
        proj = self.projectionMatrix()
        view = self.viewMatrix()
        m = proj * view
        w, h = self.width(), self.height()
        click_x = ev.position().x()
        click_y = ev.position().y()
        best_i = -1
        best_d2 = 60.0 ** 2  # 60 px tolerance
        for i, (x, y, z) in enumerate(self.picking_centroids):
            v = m * QVector3D(float(x), float(y), float(z))
            # QVector3D multiplied by a 4x4 returns a QVector3D with perspective divide already applied
            sx = (v.x() * 0.5 + 0.5) * w
            sy = (1.0 - (v.y() * 0.5 + 0.5)) * h
            d2 = (sx - click_x) ** 2 + (sy - click_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        self.picked.emit(best_i)


# ----- main popup window ----------------------------------------------------

class PointCloudWindow(QWidget):
    def __init__(
        self,
        thermal_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Lidar — 3D Point Cloud")
        self.resize(1200, 800)

        self.thermal_provider = thermal_provider  # callable -> latest BGR thermal frame
        self.cluster_params = ClusterParams()
        self.last_clusters: list[Cluster] = []
        self.last_xyz: Optional[np.ndarray] = None
        self.last_frame: Optional[LidarFrame] = None
        self.last_cluster_ms: float = 0.0
        self.selected_track_id: Optional[int] = None

        # Background DBSCAN worker (started lazily when clustering is enabled)
        self.worker: Optional[ClusterWorker] = None

        self.intr = ThermalIntrinsics()
        self.extr = Extrinsics()

        if not HAS_PG:
            lay = QVBoxLayout()
            lay.addWidget(
                QLabel(
                    "pyqtgraph + PyOpenGL not installed.\n"
                    "Run: pip install pyqtgraph PyOpenGL"
                )
            )
            self.setLayout(lay)
            return

        self._build_view()
        self._build_controls()

        # Compose
        root = QHBoxLayout()
        root.addWidget(self.view, 1)
        right = QWidget()
        right.setFixedWidth(320)
        right.setLayout(self.right_layout)
        root.addWidget(right)
        self.setLayout(root)

    # ------------- 3D view setup -------------

    def _build_view(self) -> None:
        self.view = PickableGLView()
        self.view.setBackgroundColor((10, 10, 12))
        self.view.opts["distance"] = 30
        self.view.opts["fov"] = 60

        grid = gl.GLGridItem()
        grid.setSize(40, 40)
        grid.setSpacing(2, 2)
        self.view.addItem(grid)

        # 3-axis frame at lidar origin (X red, Y green, Z blue)
        for axis, color in zip(np.eye(3), [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]):
            line = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], axis * 1.5]),
                color=color, width=2, antialias=True,
            )
            self.view.addItem(line)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32), size=2.0, pxMode=True
        )
        self.view.addItem(self.scatter)

        # Camera-FOV wireframe frustum (apex at camera origin, drawn out to a
        # configurable depth in lidar coords). Updated whenever the calibration
        # or the depth/visibility controls change.
        self.fov_lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 0.85, 0.2, 0.9),
            width=2,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(self.fov_lines)

        # Cluster bounding boxes — recreated every frame
        self.box_items: list[gl.GLLinePlotItem] = []

        self.view.picked.connect(self._on_pick)

    # ------------- right-side controls -------------

    def _build_controls(self) -> None:
        # Display
        self.color_combo = QComboBox()
        self.color_combo.addItems(COLOR_MODES)
        self.color_combo.setCurrentText("range")
        self.color_combo.currentTextChanged.connect(self._redraw)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 10)
        self.size_spin.setValue(2)
        self.size_spin.valueChanged.connect(self._redraw)

        disp = QFormLayout()
        disp.addRow("Color by:", self.color_combo)
        disp.addRow("Point size:", self.size_spin)
        disp_box = QGroupBox("Display")
        disp_box.setLayout(disp)

        # Clustering
        self.cluster_check = QCheckBox("Enable DBSCAN")
        self.cluster_check.setChecked(False)
        self.cluster_check.toggled.connect(self._on_cluster_toggle)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.05, 5.0)
        self.eps_spin.setSingleStep(0.05)
        self.eps_spin.setValue(self.cluster_params.eps)
        self.eps_spin.valueChanged.connect(self._on_param_change)

        self.minpts_spin = QSpinBox()
        self.minpts_spin.setRange(3, 200)
        self.minpts_spin.setValue(self.cluster_params.min_samples)
        self.minpts_spin.valueChanged.connect(self._on_param_change)

        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(-5, 5)
        self.zmin_spin.setValue(self.cluster_params.z_min)
        self.zmin_spin.valueChanged.connect(self._on_param_change)

        self.zmax_spin = QDoubleSpinBox()
        self.zmax_spin.setRange(-5, 10)
        self.zmax_spin.setValue(self.cluster_params.z_max)
        self.zmax_spin.valueChanged.connect(self._on_param_change)

        cl = QFormLayout()
        cl.addRow(self.cluster_check)
        cl.addRow("eps (m):", self.eps_spin)
        cl.addRow("min_samples:", self.minpts_spin)
        cl.addRow("z_min (m):", self.zmin_spin)
        cl.addRow("z_max (m):", self.zmax_spin)
        cl_box = QGroupBox("Clustering" + ("" if HAS_SKLEARN else "  (sklearn missing)"))
        cl_box.setLayout(cl)

        # Selection info
        self.info_lbl = QLabel("(click a cluster to select)")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet("color:#ccc;padding:4px;")
        info_box = QGroupBox("Selected object")
        info_lay = QVBoxLayout()
        info_lay.addWidget(self.info_lbl)
        info_box.setLayout(info_lay)

        # Thermal calibration
        self.lens_combo = QComboBox()
        self.lens_combo.addItems(LENS_PRESETS.keys())
        self.lens_combo.setCurrentText("Boson 14mm (~50°)")
        self.lens_combo.currentTextChanged.connect(self._on_lens_change)

        def make_dspin(lo, hi, val, step=0.01):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(3)
            s.setValue(val)
            s.valueChanged.connect(self._on_extr_change)
            return s

        self.tx_spin = make_dspin(-2, 2, 0.0, 0.01)
        self.ty_spin = make_dspin(-2, 2, 0.0, 0.01)
        self.tz_spin = make_dspin(-2, 2, 0.0, 0.01)
        self.roll_spin = make_dspin(-180, 180, 0.0, 0.5)
        self.pitch_spin = make_dspin(-180, 180, 0.0, 0.5)
        self.yaw_spin = make_dspin(-180, 180, 0.0, 0.5)

        cal = QFormLayout()
        cal.addRow("Lens:", self.lens_combo)
        cal.addRow("tx (m):", self.tx_spin)
        cal.addRow("ty (m):", self.ty_spin)
        cal.addRow("tz (m):", self.tz_spin)
        cal.addRow("roll (°):", self.roll_spin)
        cal.addRow("pitch (°):", self.pitch_spin)
        cal.addRow("yaw (°):", self.yaw_spin)
        cal_box = QGroupBox("Thermal calibration (extrinsics)")
        cal_box.setLayout(cal)

        # Camera-FOV overlay
        self.fov_check = QCheckBox("Show camera FOV frustum")
        self.fov_check.setChecked(True)
        self.fov_check.toggled.connect(self._update_fov)
        self.fov_depth_spin = QDoubleSpinBox()
        self.fov_depth_spin.setRange(1.0, 100.0)
        self.fov_depth_spin.setSingleStep(1.0)
        self.fov_depth_spin.setDecimals(1)
        self.fov_depth_spin.setValue(10.0)
        self.fov_depth_spin.setSuffix(" m")
        self.fov_depth_spin.valueChanged.connect(self._update_fov)
        self.fov_crop_check = QCheckBox("Crop cloud to camera FOV")
        self.fov_crop_check.setChecked(False)
        self.fov_crop_check.toggled.connect(self._redraw)
        fov_l = QFormLayout()
        fov_l.addRow(self.fov_check)
        fov_l.addRow("Depth:", self.fov_depth_spin)
        fov_l.addRow(self.fov_crop_check)
        fov_box = QGroupBox("Camera FOV")
        fov_box.setLayout(fov_l)

        # Status
        self.status_lbl = QLabel("waiting for scans")
        self.status_lbl.setStyleSheet("color:#888;padding:2px;")

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(disp_box)
        self.right_layout.addWidget(cl_box)
        self.right_layout.addWidget(info_box)
        self.right_layout.addWidget(cal_box)
        self.right_layout.addWidget(fov_box)
        self.right_layout.addStretch(1)
        self.right_layout.addWidget(self.status_lbl)

        self._update_fov()

    # ------------- input handlers -------------

    def _on_param_change(self) -> None:
        self.cluster_params.eps = self.eps_spin.value()
        self.cluster_params.min_samples = self.minpts_spin.value()
        self.cluster_params.z_min = self.zmin_spin.value()
        self.cluster_params.z_max = self.zmax_spin.value()
        if self.worker is not None:
            self.worker.update_params(self.cluster_params)
        self._redraw()

    def _on_cluster_toggle(self, on: bool) -> None:
        if on and HAS_SKLEARN:
            if self.worker is None:
                self.worker = ClusterWorker(self.cluster_params)
                self.worker.result.connect(self._on_clusters_ready)
                self.worker.start()
        else:
            if self.worker is not None:
                self.worker.stop()
                self.worker = None
            self.last_clusters = []
        self._redraw()

    def _on_extr_change(self) -> None:
        self.extr.tx = self.tx_spin.value()
        self.extr.ty = self.ty_spin.value()
        self.extr.tz = self.tz_spin.value()
        self.extr.roll_deg = self.roll_spin.value()
        self.extr.pitch_deg = self.pitch_spin.value()
        self.extr.yaw_deg = self.yaw_spin.value()
        self._update_fov()
        if self.color_combo.currentText() == "thermal":
            self._redraw()

    def _on_lens_change(self, name: str) -> None:
        hfov = LENS_PRESETS[name]
        self.intr = intrinsics_from_hfov(self.intr.width, self.intr.height, hfov)
        self._update_fov()
        if self.color_combo.currentText() == "thermal":
            self._redraw()

    def _update_fov(self) -> None:
        """Refresh the camera-FOV wireframe (apex at camera origin in lidar coords)."""
        if not HAS_PG:
            return
        if not self.fov_check.isChecked():
            self.fov_lines.setData(pos=np.zeros((2, 3), dtype=np.float32))
            return
        verts = camera_frustum_lines(self.intr, self.extr, self.fov_depth_spin.value())
        self.fov_lines.setData(pos=verts)

    def set_calibration(self, extr: Extrinsics, intr: ThermalIntrinsics) -> None:
        """Apply calibration from outside (e.g. CalibrationWindow). Updates
        widget values silently and re-renders if in thermal mode."""
        self.extr = Extrinsics(**extr.__dict__)
        self.intr = ThermalIntrinsics(**intr.__dict__)
        for w, v in [
            (self.tx_spin, self.extr.tx), (self.ty_spin, self.extr.ty), (self.tz_spin, self.extr.tz),
            (self.roll_spin, self.extr.roll_deg),
            (self.pitch_spin, self.extr.pitch_deg),
            (self.yaw_spin, self.extr.yaw_deg),
        ]:
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._update_fov()
        if self.color_combo.currentText() == "thermal":
            self._redraw()

    def _on_pick(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.last_clusters):
            self.selected_track_id = None
            self.info_lbl.setText("(click a cluster to select)")
        else:
            c = self.last_clusters[idx]
            self.selected_track_id = c.track_id
            sx, sy, sz = c.size
            self.info_lbl.setText(
                f"<b>Track #{c.track_id}</b><br>"
                f"range:&nbsp;{c.range_m:.2f} m<br>"
                f"size:&nbsp;{sx:.2f} × {sy:.2f} × {sz:.2f} m<br>"
                f"speed:&nbsp;{c.speed_mps:.2f} m/s<br>"
                f"points:&nbsp;{c.n_points}<br>"
                f"age:&nbsp;{c.age} frames"
            )
        self._redraw()

    # ------------- main update path -------------

    def update_cloud(self, frame: LidarFrame) -> None:
        if not HAS_PG or frame.xyz is None:
            return
        self.last_frame = frame

        xyz = frame.xyz.reshape(-1, 3).astype(np.float32)
        rng = np.linalg.norm(xyz, axis=1)
        mask = rng > 0.1
        xyz = xyz[mask]
        if xyz.size == 0:
            return
        self.last_xyz = xyz
        self.last_xyz_mask = mask
        self.last_ranges = rng[mask]

        # Hand the raw frame XYZ to the worker (latest-only queue); rendering
        # uses whatever clusters arrived from the worker most recently.
        if self.cluster_check.isChecked() and HAS_SKLEARN and self.worker is not None:
            self.worker.submit(frame.xyz)

        self._redraw()

    def _on_clusters_ready(self, clusters: list, elapsed_ms: float) -> None:
        self.last_clusters = clusters
        self.last_cluster_ms = elapsed_ms
        self._redraw()

    def _redraw(self) -> None:
        if not HAS_PG or self.last_xyz is None:
            return
        colors = self._compute_colors()
        pos = self.last_xyz
        if self.fov_crop_check.isChecked():
            _, in_fov = project_points(self.last_xyz, self.intr, self.extr)
            pos = pos[in_fov]
            colors = colors[in_fov]
        self.scatter.setData(
            pos=pos, color=colors, size=float(self.size_spin.value())
        )
        self._draw_boxes()
        # Update pickable centroids
        if self.last_clusters:
            cents = np.array([c.centroid for c in self.last_clusters])
        else:
            cents = np.empty((0, 3))
        self.view.set_picking(cents)
        # Status
        n_clusters = len(self.last_clusters)
        parts = [f"{len(self.last_xyz):,} pts", f"{n_clusters} clusters"]
        if self.cluster_check.isChecked():
            if not HAS_SKLEARN:
                parts.append("sklearn missing")
            elif self.last_cluster_ms > 0:
                parts.append(f"DBSCAN {self.last_cluster_ms:.0f}ms")
        self.status_lbl.setText("  |  ".join(parts))

    def _compute_colors(self) -> np.ndarray:
        mode = self.color_combo.currentText()

        if mode == "thermal":
            therm = self.thermal_provider() if self.thermal_provider else None
            if therm is None:
                # Fall back to range coloring with a status note
                self.status_lbl.setText("thermal frame unavailable — connect thermal panel")
                mode = "range"
            else:
                colors, _ = colorize_with_thermal(self.last_xyz, therm, self.intr, self.extr)
                return colors

        if mode == "cluster_id":
            return self._color_by_cluster()

        scalar = self._scalar(mode)
        lo, hi = np.percentile(scalar, (2, 98))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        return _turbo_like((scalar - lo) / (hi - lo))

    def _scalar(self, mode: str) -> np.ndarray:
        if mode == "range":
            return self.last_ranges
        if mode == "height":
            return self.last_xyz[:, 2]
        img = {
            "reflectivity": self.last_frame.reflectivity_img if self.last_frame else None,
            "signal": self.last_frame.signal_img if self.last_frame else None,
            "near_ir": self.last_frame.nearir_img if self.last_frame else None,
        }.get(mode)
        if img is None:
            return self.last_ranges
        return img.reshape(-1)[self.last_xyz_mask].astype(np.float32)

    def _color_by_cluster(self) -> np.ndarray:
        # Default = dim gray; cluster points get their track color.
        out = np.tile(np.array([0.18, 0.18, 0.20, 1.0], dtype=np.float32), (len(self.last_xyz), 1))
        if not self.last_clusters:
            return out
        # Build a KD-like lookup — but simpler: for each cluster, find its points
        # in the displayed cloud by spatial match. Since clustering subsamples,
        # we mark points within bbox of each cluster.
        palette = _hsv_palette(max(64, max(c.track_id for c in self.last_clusters) + 1))
        for c in self.last_clusters:
            in_box = (
                (self.last_xyz[:, 0] >= c.bbox_min[0]) & (self.last_xyz[:, 0] <= c.bbox_max[0])
                & (self.last_xyz[:, 1] >= c.bbox_min[1]) & (self.last_xyz[:, 1] <= c.bbox_max[1])
                & (self.last_xyz[:, 2] >= c.bbox_min[2]) & (self.last_xyz[:, 2] <= c.bbox_max[2])
            )
            color = palette[c.track_id % len(palette)]
            out[in_box] = color
        return out

    def _draw_boxes(self) -> None:
        # Remove old boxes
        for item in self.box_items:
            self.view.removeItem(item)
        self.box_items.clear()
        if not self.last_clusters:
            return
        palette = _hsv_palette(max(64, max(c.track_id for c in self.last_clusters) + 1))
        for c in self.last_clusters:
            color = palette[c.track_id % len(palette)]
            if self.selected_track_id == c.track_id:
                color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                width = 3
            else:
                width = 1
            corners = self._bbox_lines(c.bbox_min, c.bbox_max)
            line = gl.GLLinePlotItem(
                pos=corners, color=tuple(color), width=width, antialias=True, mode="lines"
            )
            self.view.addItem(line)
            self.box_items.append(line)

    def closeEvent(self, ev) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        super().closeEvent(ev)

    @staticmethod
    def _bbox_lines(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
        x0, y0, z0 = mn
        x1, y1, z1 = mx
        c = np.array([
            [x0, y0, z0], [x1, y0, z0],
            [x1, y0, z0], [x1, y1, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y1, z0], [x0, y0, z0],
            [x0, y0, z1], [x1, y0, z1],
            [x1, y0, z1], [x1, y1, z1],
            [x1, y1, z1], [x0, y1, z1],
            [x0, y1, z1], [x0, y0, z1],
            [x0, y0, z0], [x0, y0, z1],
            [x1, y0, z0], [x1, y0, z1],
            [x1, y1, z0], [x1, y1, z1],
            [x0, y1, z0], [x0, y1, z1],
        ], dtype=np.float32)
        return c
