"""Live point cloud popup window.

Features beyond a basic viewer:
  - DBSCAN clustering with stable per-track IDs and frame-to-frame velocity.
  - Jetson PointPillars detections over ZMQ with DBSCAN fallback.
  - Per-cluster bounding boxes drawn in the scene.
  - Click-to-select an object (info panel shows class/range/size/speed/score).
  - Color modes: range/height/reflectivity/signal/near_ir/object_id/THERMAL
    (thermal mode projects the live thermal frame onto the cloud using a
     6-DoF calibration tweakable from the right-hand panel).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
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
    QLineEdit,
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
from detection import DEFAULT_ENDPOINT, Detection, DetectorWorker, HAS_MSGPACK, HAS_ZMQ
from fusion import (
    LENS_PRESETS,
    Extrinsics,
    ThermalIntrinsics,
    camera_frustum_lines,
    colorize_with_thermal,
    intrinsics_from_hfov,
    project_points,
)
from indoor_human import CandidateDebug, IndoorHumanWorker, ROI_CLASS_NAMES, ROI_FEATURE_NAMES
from lidar import LidarFrame


COLOR_MODES = ["range", "height", "reflectivity", "signal", "near_ir", "object_id", "thermal"]
ANALYSIS_MODES = ["off", "indoor_human", "dbscan", "pointpillars", "auto"]

CLASS_COLORS = {
    "car": np.array([1.0, 0.62, 0.12, 1.0], dtype=np.float32),
    "vehicle": np.array([1.0, 0.62, 0.12, 1.0], dtype=np.float32),
    "pedestrian": np.array([1.0, 0.16, 0.28, 1.0], dtype=np.float32),
    "person": np.array([1.0, 0.16, 0.28, 1.0], dtype=np.float32),
    "cyclist": np.array([0.0, 0.85, 1.0, 1.0], dtype=np.float32),
    "standing_person": np.array([1.0, 0.16, 0.28, 1.0], dtype=np.float32),
    "seated_person": np.array([1.0, 0.38, 0.12, 1.0], dtype=np.float32),
    "occupied_chair": np.array([1.0, 0.72, 0.0, 1.0], dtype=np.float32),
}

DEBUG_REASON_COLORS = {
    "accepted": np.array([0.2, 1.0, 0.35, 0.65], dtype=np.float32),
    "empty_chair": np.array([0.1, 0.65, 1.0, 0.45], dtype=np.float32),
    "non_human_clutter": np.array([0.55, 0.55, 0.55, 0.38], dtype=np.float32),
    "uncertain": np.array([1.0, 1.0, 1.0, 0.35], dtype=np.float32),
    "too_short": np.array([0.25, 0.55, 1.0, 0.45], dtype=np.float32),
    "too_tall": np.array([0.2, 0.85, 1.0, 0.45], dtype=np.float32),
    "too_wide": np.array([1.0, 0.55, 0.0, 0.45], dtype=np.float32),
    "too_planar": np.array([0.7, 0.35, 1.0, 0.45], dtype=np.float32),
    "too_linear": np.array([0.8, 0.45, 1.0, 0.45], dtype=np.float32),
    "too_few_points": np.array([0.55, 0.55, 0.55, 0.35], dtype=np.float32),
    "below_confidence": np.array([0.9, 0.9, 0.25, 0.45], dtype=np.float32),
    "below_threshold": np.array([0.9, 0.9, 0.25, 0.45], dtype=np.float32),
    "failed_classifier": np.array([0.9, 0.9, 0.25, 0.45], dtype=np.float32),
    "bad_floor_relation": np.array([0.0, 0.85, 0.85, 0.45], dtype=np.float32),
    "bad_human_geometry": np.array([1.0, 0.2, 0.2, 0.45], dtype=np.float32),
}

SCENE_LABELS = [
    "unknown",
    "no_human",
    "seated_human_present",
    "standing_human_present",
    "empty_chairs",
    "chair_with_bag_or_coat",
    "clutter",
]


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


def _detection_color(det: Detection) -> np.ndarray:
    key = det.label.lower()
    return CLASS_COLORS.get(key, _hsv_palette(16)[det.class_id % 16])


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
        viewport = self.getViewport()
        proj = self.projectionMatrix(viewport, viewport)
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
        web_bridge=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Lidar — 3D Point Cloud")
        self.resize(1200, 800)

        self.thermal_provider = thermal_provider  # callable -> latest BGR thermal frame
        self.web_bridge = web_bridge
        self.cluster_params = ClusterParams()
        self.last_clusters: list[Cluster] = []
        self.last_detections: list[Detection] = []
        self.last_humans: list[Detection] = []
        self.last_human_debug: list[CandidateDebug] = []
        self.last_xyz: Optional[np.ndarray] = None
        self.last_frame: Optional[LidarFrame] = None
        self.last_cluster_ms: float = 0.0
        self.last_detector_ms: float = 0.0
        self.last_human_ms: float = 0.0
        self.human_status: str = "indoor human idle"
        self.detector_status: str = "detector idle"
        self.detector_ok: bool = True
        self.detector_last_ok_t: float = 0.0
        self.analysis_mode = "indoor_human"
        self.selected_track_id: Optional[int] = None
        self.selected_source: Optional[str] = None
        self.max_display_points = 120_000
        self._auto_capture_last_t: float = 0.0
        self._auto_capture_miss_count: int = 0
        self._last_box_log_t: float = 0.0

        # Background DBSCAN worker (started lazily when clustering is enabled)
        self.worker: Optional[ClusterWorker] = None
        self.detector_worker: Optional[DetectorWorker] = None
        self.human_worker: Optional[IndoorHumanWorker] = None

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
        self.debug_box_items: list[gl.GLLinePlotItem] = []
        self.label_items: list[object] = []

        self.view.picked.connect(self._on_pick)

    # ------------- right-side controls -------------

    def _build_controls(self) -> None:
        # Display
        self.color_combo = QComboBox()
        self.color_combo.addItems(COLOR_MODES)
        self.color_combo.setCurrentText("thermal")
        self.color_combo.currentTextChanged.connect(self._redraw)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 10)
        self.size_spin.setValue(2)
        self.size_spin.valueChanged.connect(self._redraw)
        self.debug_check = QCheckBox("Show candidate debug")
        self.debug_check.toggled.connect(self._redraw)
        self.box_log_check = QCheckBox("Log box diagnostics")
        self.save_debug_btn = QPushButton("Save debug frame")
        self.save_debug_btn.clicked.connect(lambda: self._save_debug_frame("manual_debug"))

        disp = QFormLayout()
        disp.addRow("Color by:", self.color_combo)
        disp.addRow("Point size:", self.size_spin)
        disp.addRow(self.debug_check)
        disp.addRow(self.box_log_check)
        disp.addRow(self.save_debug_btn)
        disp_box = QGroupBox("Display")
        disp_box.setLayout(disp)

        # Object detection / clustering
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(ANALYSIS_MODES)
        self.mode_combo.setCurrentText(self.analysis_mode)
        self.mode_combo.currentTextChanged.connect(self._on_analysis_mode_change)

        self.endpoint_edit = QLineEdit(DEFAULT_ENDPOINT)
        self.endpoint_edit.setToolTip("Jetson detector ZMQ endpoint")
        self.endpoint_edit.editingFinished.connect(self._on_detector_endpoint_change)

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
        cl.addRow("Mode:", self.mode_combo)
        cl.addRow("Jetson:", self.endpoint_edit)
        cl.addRow("eps (m):", self.eps_spin)
        cl.addRow("min_samples:", self.minpts_spin)
        cl.addRow("z_min (m):", self.zmin_spin)
        cl.addRow("z_max (m):", self.zmax_spin)
        missing = []
        if not HAS_SKLEARN:
            missing.append("sklearn")
        if not HAS_ZMQ:
            missing.append("pyzmq")
        if not HAS_MSGPACK:
            missing.append("msgpack")
        suffix = "" if not missing else f"  ({', '.join(missing)} missing)"
        cl_box = QGroupBox("Object analysis" + suffix)
        cl_box.setLayout(cl)

        # Capture labels for hard positives/negatives. These are operator
        # annotations for dataset collection, not detector thresholds.
        self.scene_combo = QComboBox()
        self.scene_combo.addItems(SCENE_LABELS)
        self.auto_capture_check = QCheckBox("Auto-save failures")
        self.save_fp_btn = QPushButton("Save false positive")
        self.save_fp_btn.clicked.connect(lambda: self._save_debug_frame("false_positive"))
        self.save_missed_btn = QPushButton("Save missed seated")
        self.save_missed_btn.clicked.connect(lambda: self._save_debug_frame("missed_seated"))
        cap = QFormLayout()
        cap.addRow("Scene:", self.scene_combo)
        cap.addRow(self.auto_capture_check)
        cap.addRow(self.save_fp_btn)
        cap.addRow(self.save_missed_btn)
        cap_box = QGroupBox("Dataset capture")
        cap_box.setLayout(cap)

        # Selection info
        self.info_lbl = QLabel("(click an object to select)")
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
        self.right_layout.addWidget(cap_box)
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

    def _on_detector_endpoint_change(self) -> None:
        endpoint = self.endpoint_edit.text().strip() or DEFAULT_ENDPOINT
        self.endpoint_edit.setText(endpoint)
        if self.detector_worker is not None:
            self.detector_worker.update_endpoint(endpoint)
        self.detector_status = f"detector endpoint {endpoint}"

    def _on_analysis_mode_change(self, mode: str) -> None:
        self.analysis_mode = mode
        self.selected_track_id = None
        self.selected_source = None
        if self.web_bridge is not None:
            self.web_bridge.update_detection_status(f"mode {mode}", mode=mode)
        if mode in ("pointpillars", "auto"):
            self.detector_ok = True
        self._ensure_workers()
        self._redraw()

    def _ensure_workers(self) -> None:
        mode = self.analysis_mode
        wants_human = mode in ("indoor_human", "auto")
        wants_detector = mode in ("pointpillars", "auto")
        wants_dbscan = (
            mode == "dbscan"
            or (mode == "auto" and (not self.detector_ok or not HAS_ZMQ or not HAS_MSGPACK))
        )

        if wants_human and HAS_SKLEARN:
            if self.human_worker is None:
                self.human_worker = IndoorHumanWorker()
                self.human_worker.result.connect(self._on_humans_ready)
                self.human_worker.error.connect(self._on_human_error)
                self.human_worker.start()
                self.human_status = "indoor human starting"
        elif self.human_worker is not None:
            self.human_worker.stop()
            self.human_worker = None
            self.last_humans = []

        if wants_detector and HAS_ZMQ and HAS_MSGPACK:
            endpoint = self.endpoint_edit.text().strip() or DEFAULT_ENDPOINT
            if self.detector_worker is None:
                self.detector_worker = DetectorWorker(endpoint)
                self.detector_worker.result.connect(self._on_detections_ready)
                self.detector_worker.error.connect(self._on_detector_error)
                self.detector_worker.start()
                self.detector_status = f"connecting {endpoint}"
            else:
                self.detector_worker.update_endpoint(endpoint)
        elif self.detector_worker is not None:
            self.detector_worker.stop()
            self.detector_worker = None
            self.detector_ok = False
            self.last_detections = []

        if wants_dbscan and HAS_SKLEARN:
            if self.worker is None:
                self.worker = ClusterWorker(self.cluster_params)
                self.worker.result.connect(self._on_clusters_ready)
                self.worker.start()
        elif self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.last_clusters = []

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
        objects, source = self._active_objects()
        if idx < 0 or idx >= len(objects):
            self.selected_track_id = None
            self.selected_source = None
            self.info_lbl.setText("(click an object to select)")
        else:
            obj = objects[idx]
            self.selected_track_id = obj.track_id
            self.selected_source = source
            if isinstance(obj, Detection):
                sx, sy, sz = obj.size
                cx, cy, cz = obj.center
                metrics = self._box_debug_metrics(obj)
                self.info_lbl.setText(
                    f"<b>{obj.label} #{obj.track_id}</b><br>"
                    f"frame:&nbsp;lidar X fwd, Y left, Z up<br>"
                    f"center:&nbsp;{cx:.2f}, {cy:.2f}, {cz:.2f} m<br>"
                    f"posterior:&nbsp;{obj.score:.2f}<br>"
                    f"model:&nbsp;{obj.model_score:.2f}<br>"
                    f"range:&nbsp;{obj.range_m:.2f} m<br>"
                    f"size dx/dy/dz:&nbsp;{sx:.2f} x {sy:.2f} x {sz:.2f} m<br>"
                    f"yaw:&nbsp;{obj.yaw:.2f} rad / {np.degrees(obj.yaw):.0f} deg<br>"
                    f"support:&nbsp;{obj.support_points} pts / {obj.support_z_span:.2f} m<br>"
                    f"box points:&nbsp;{metrics['points_inside']}<br>"
                    f"sanity:&nbsp;{metrics['sanity']}<br>"
                    f"nearest candidate:&nbsp;{metrics['nearest_candidate_m']:.2f} m<br>"
                    f"nearest cluster:&nbsp;{metrics['nearest_cluster_m']:.2f} m"
                    f" / overlap {metrics['cluster_overlap']}<br>"
                    f"speed:&nbsp;{obj.speed_mps:.2f} m/s<br>"
                    f"age:&nbsp;{obj.age} frames<br>"
                    f"misses:&nbsp;{obj.misses}"
                )
            else:
                sx, sy, sz = obj.size
                self.info_lbl.setText(
                    f"<b>Cluster #{obj.track_id}</b><br>"
                    f"range:&nbsp;{obj.range_m:.2f} m<br>"
                    f"size:&nbsp;{sx:.2f} x {sy:.2f} x {sz:.2f} m<br>"
                    f"speed:&nbsp;{obj.speed_mps:.2f} m/s<br>"
                    f"points:&nbsp;{obj.n_points}<br>"
                    f"age:&nbsp;{obj.age} frames"
                )
        self._redraw()

    def _active_objects(self) -> tuple[list, str]:
        if self.analysis_mode in ("indoor_human", "auto") and self.last_humans:
            return self.last_humans, "humans"
        if self.analysis_mode in ("pointpillars", "auto") and self.last_detections:
            return self.last_detections, "detections"
        if self.analysis_mode in ("dbscan", "auto") and self.last_clusters:
            return self.last_clusters, "clusters"
        return [], ""

    @staticmethod
    def _object_center(obj) -> np.ndarray:
        return obj.center if isinstance(obj, Detection) else obj.centroid

    def _box_debug_metrics(self, det: Detection) -> dict[str, float | int | str]:
        points_inside = 0
        if self.last_xyz is not None:
            points_inside = int(np.count_nonzero(self._points_in_detection(det, self.last_xyz)))

        nearest_candidate = float("nan")
        if self.last_human_debug:
            centers = np.array([d.center for d in self.last_human_debug], dtype=np.float32)
            nearest_candidate = float(np.min(np.linalg.norm(centers - det.center, axis=1)))

        nearest_cluster = float("nan")
        cluster_overlap = 0
        if self.last_clusters:
            best = min(
                self.last_clusters,
                key=lambda c: float(np.linalg.norm(c.centroid - det.center)),
            )
            nearest_cluster = float(np.linalg.norm(best.centroid - det.center))
            cluster_overlap = int(np.count_nonzero(self._points_in_detection(det, best.points)))

        support = det.support_points if det.support_points > 0 else points_inside
        if support <= 2:
            sanity = "LOW_BOX_SUPPORT"
        elif np.isfinite(nearest_candidate) and nearest_candidate > 1.25:
            sanity = "OFFSET_FROM_ROI"
        elif np.isfinite(nearest_cluster) and nearest_cluster > 1.50 and cluster_overlap <= 2:
            sanity = "OFFSET_FROM_CLUSTER"
        else:
            sanity = "ok"

        return {
            "points_inside": points_inside,
            "nearest_candidate_m": nearest_candidate,
            "nearest_cluster_m": nearest_cluster,
            "cluster_overlap": cluster_overlap,
            "sanity": sanity,
        }

    # ------------- main update path -------------

    def update_cloud(self, frame: LidarFrame) -> None:
        if not HAS_PG or frame.xyz is None:
            return
        self.last_frame = frame

        xyz = frame.xyz.reshape(-1, 3).astype(np.float32)
        rng = np.linalg.norm(xyz, axis=1)
        valid_idx = np.flatnonzero(np.isfinite(rng) & (rng > 0.1))
        if valid_idx.size == 0:
            return

        if valid_idx.size > self.max_display_points:
            pick = np.linspace(
                0, valid_idx.size - 1, self.max_display_points, dtype=np.int64
            )
            valid_idx = valid_idx[pick]

        mask = np.zeros(len(rng), dtype=bool)
        mask[valid_idx] = True
        self.last_xyz = xyz[valid_idx]
        self.last_xyz_mask = mask
        self.last_ranges = rng[valid_idx]

        self._ensure_workers()

        # Hand the raw frame to whichever workers are active. Both workers use
        # latest-only queues so the UI never waits for stale analysis.
        if self.worker is not None:
            self.worker.submit(frame.xyz)
        if self.detector_worker is not None:
            self.detector_worker.submit(frame)
        if self.human_worker is not None:
            self.human_worker.submit(frame.xyz)

        self._redraw()

    def _on_clusters_ready(self, clusters: list, elapsed_ms: float) -> None:
        self.last_clusters = clusters
        self.last_cluster_ms = elapsed_ms
        self._redraw()

    def _on_detections_ready(self, detections: list, elapsed_ms: float, status: str) -> None:
        self.last_detections = detections
        self.last_detector_ms = elapsed_ms
        self.detector_status = status
        if self.web_bridge is not None:
            self.web_bridge.update_detections(
                detections,
                mode=self.analysis_mode,
                source="pointpillars",
                status=status,
                elapsed_ms=elapsed_ms,
            )
        self.detector_ok = True
        self.detector_last_ok_t = time.monotonic()
        self._log_box_diagnostics(detections, "pointpillars")
        if self.analysis_mode == "auto" and self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.last_clusters = []
        self._redraw()

    def _on_humans_ready(self, humans: list, elapsed_ms: float, status: str, debug: list) -> None:
        self.last_humans = humans
        self.last_human_ms = elapsed_ms
        self.human_status = status
        self.last_human_debug = debug
        if self.web_bridge is not None:
            self.web_bridge.update_detections(
                humans,
                mode=self.analysis_mode,
                source="indoor_human",
                status=status,
                elapsed_ms=elapsed_ms,
                debug_count=len(debug),
            )
        self._log_box_diagnostics(humans, "indoor_human")
        self._maybe_auto_capture_failures()
        self._redraw()

    def _on_human_error(self, msg: str) -> None:
        self.human_status = f"indoor human: {msg}"
        self.last_humans = []
        self.last_human_debug = []
        if self.web_bridge is not None:
            self.web_bridge.update_detection_status(self.human_status, mode=self.analysis_mode)
        self._redraw()

    def _on_detector_error(self, msg: str) -> None:
        self.detector_ok = False
        self.detector_status = f"detector: {msg}"
        self.last_detections = []
        if self.web_bridge is not None:
            self.web_bridge.update_detection_status(self.detector_status, mode=self.analysis_mode)
        if self.analysis_mode == "auto" and HAS_SKLEARN and self.worker is None:
            self.worker = ClusterWorker(self.cluster_params)
            self.worker.result.connect(self._on_clusters_ready)
            self.worker.start()
        self._redraw()

    def _log_box_diagnostics(self, detections: list[Detection], source: str) -> None:
        if not detections or not getattr(self, "box_log_check", None) or not self.box_log_check.isChecked():
            return
        now = time.monotonic()
        if now - self._last_box_log_t < 1.0:
            return
        self._last_box_log_t = now
        print(
            "box diagnostics:"
            " frame=lidar_sensor_frame_x_forward_y_left_z_up"
            " size_order=dx_dy_dz yaw=radians_about_+Z source="
            f"{source}"
        )
        for det in detections:
            m = self._box_debug_metrics(det)
            print(
                f"  id={det.track_id} class={det.label} score={det.score:.3f}"
                f" model={det.model_score:.3f}"
                f" center=({det.center[0]:.2f},{det.center[1]:.2f},{det.center[2]:.2f})"
                f" size=({det.size[0]:.2f},{det.size[1]:.2f},{det.size[2]:.2f})"
                f" yaw={det.yaw:.3f}"
                f" box_points={m['points_inside']}"
                f" support={det.support_points}"
                f" nearest_candidate_m={m['nearest_candidate_m']:.2f}"
                f" nearest_cluster_m={m['nearest_cluster_m']:.2f}"
                f" cluster_overlap={m['cluster_overlap']}"
                f" sanity={m['sanity']}"
            )

    def _maybe_auto_capture_failures(self) -> None:
        if not getattr(self, "auto_capture_check", None) or not self.auto_capture_check.isChecked():
            self._auto_capture_miss_count = 0
            return
        if self.last_frame is None or self.last_frame.xyz is None:
            return
        scene = self.scene_combo.currentText()
        now = time.monotonic()
        if now - self._auto_capture_last_t < 8.0:
            return

        negative_scenes = {"no_human", "empty_chairs", "chair_with_bag_or_coat", "clutter"}
        if scene in negative_scenes and self.last_humans:
            self._auto_capture_last_t = now
            self._save_debug_frame("auto_false_positive")
            return

        if scene == "seated_human_present":
            if self.last_humans:
                self._auto_capture_miss_count = 0
            else:
                self._auto_capture_miss_count += 1
                if self._auto_capture_miss_count >= 5:
                    self._auto_capture_last_t = now
                    self._auto_capture_miss_count = 0
                    self._save_debug_frame("auto_missed_seated")
        else:
            self._auto_capture_miss_count = 0

    def _save_debug_frame(self, capture_reason: str = "manual_debug") -> None:
        if self.last_frame is None or self.last_frame.xyz is None:
            self.status_lbl.setText("no lidar frame to save")
            return
        out_dir = Path(__file__).parent / "captures" / "indoor_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = out_dir / f"{capture_reason}_{ts}.npz"

        def det_array(items: list[Detection]) -> np.ndarray:
            rows = []
            for det in items:
                metrics = self._box_debug_metrics(det)
                rows.append([
                    *det.center.tolist(),
                    *det.size.tolist(),
                    float(det.yaw),
                    float(det.score),
                    float(det.model_score),
                    float(det.class_id),
                    float(det.track_id),
                    float(det.support_points),
                    float(det.support_z_span),
                    float(metrics["points_inside"]),
                    float(metrics["nearest_candidate_m"]) if isinstance(metrics["nearest_candidate_m"], float) else np.nan,
                    float(metrics["nearest_cluster_m"]) if isinstance(metrics["nearest_cluster_m"], float) else np.nan,
                    float(metrics["cluster_overlap"]),
                ])
            return np.asarray(rows, dtype=np.float32)

        def frame_array(arr) -> np.ndarray:
            if arr is None:
                return np.empty((0,), dtype=np.float32)
            return np.asarray(arr)

        debug_rows = []
        debug_kind = []
        debug_reason = []
        debug_source = []
        debug_features = []
        debug_scores = []
        for item in self.last_human_debug:
            debug_rows.append([
                *item.center.tolist(),
                *item.size.tolist(),
                float(item.yaw),
                float(item.score),
                float(item.points),
                float(item.z_span),
                float(item.accepted),
                float(item.cluster_id),
            ])
            debug_kind.append(item.kind)
            debug_reason.append(item.reason)
            debug_source.append(item.source)
            if item.features.size == len(ROI_FEATURE_NAMES):
                debug_features.append(item.features.astype(np.float32, copy=False))
            else:
                debug_features.append(np.full(len(ROI_FEATURE_NAMES), np.nan, dtype=np.float32))
            if item.class_scores.size == len(ROI_CLASS_NAMES):
                debug_scores.append(item.class_scores.astype(np.float32, copy=False))
            else:
                debug_scores.append(np.full(len(ROI_CLASS_NAMES), np.nan, dtype=np.float32))

        crop_points, crop_counts = self._candidate_crop_tensors(self.last_human_debug)
        debug_arr = (
            np.asarray(debug_rows, dtype=np.float32)
            if debug_rows
            else np.empty((0, 12), dtype=np.float32)
        )
        feature_arr = (
            np.asarray(debug_features, dtype=np.float32)
            if debug_features
            else np.empty((0, len(ROI_FEATURE_NAMES)), dtype=np.float32)
        )
        score_arr = (
            np.asarray(debug_scores, dtype=np.float32)
            if debug_scores
            else np.empty((0, len(ROI_CLASS_NAMES)), dtype=np.float32)
        )

        np.savez_compressed(
            path,
            xyz=self.last_frame.xyz.astype(np.float32, copy=False),
            range_img=frame_array(self.last_frame.range_img),
            signal_img=frame_array(self.last_frame.signal_img),
            reflectivity_img=frame_array(self.last_frame.reflectivity_img),
            nearir_img=frame_array(self.last_frame.nearir_img),
            analysis_mode=np.array(self.analysis_mode),
            scene_label=np.array(self.scene_combo.currentText()),
            capture_reason=np.array(capture_reason),
            box_frame=np.array("lidar_sensor_frame_x_forward_y_left_z_up"),
            box_size_order=np.array("dx_dy_dz"),
            yaw_convention=np.array("radians_about_positive_z"),
            humans=det_array(self.last_humans),
            pointpillars=det_array(self.last_detections),
            candidate_debug=debug_arr,
            candidate_kind=np.asarray(debug_kind),
            candidate_reason=np.asarray(debug_reason),
            candidate_source=np.asarray(debug_source),
            candidate_features=feature_arr,
            candidate_class_scores=score_arr,
            roi_feature_names=np.asarray(ROI_FEATURE_NAMES),
            roi_class_names=np.asarray(ROI_CLASS_NAMES),
            candidate_crop_points=crop_points,
            candidate_crop_counts=crop_counts,
        )
        self.status_lbl.setText(f"saved {path.name}")

    def _candidate_crop_tensors(
        self,
        items: list[CandidateDebug],
        *,
        max_items: int = 48,
        max_points: int = 2048,
    ) -> tuple[np.ndarray, np.ndarray]:
        crops = np.zeros((min(len(items), max_items), max_points, 3), dtype=np.float32)
        counts = np.zeros(min(len(items), max_items), dtype=np.int32)
        if self.last_frame is None or self.last_frame.xyz is None or not items:
            return crops, counts
        pts = self.last_frame.xyz.reshape(-1, 3).astype(np.float32, copy=False)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        for i, item in enumerate(items[:max_items]):
            mask = self._points_in_oriented_box(item.center, item.size, item.yaw, pts, margin=(0.20, 0.20, 0.15))
            crop = pts[mask]
            counts[i] = min(len(crop), max_points)
            if len(crop) == 0:
                continue
            if len(crop) > max_points:
                idx = np.linspace(0, len(crop) - 1, max_points, dtype=np.int64)
                crop = crop[idx]
            crops[i, : len(crop)] = crop
        return crops, counts

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
        objects, source = self._active_objects()
        if objects:
            cents = np.array([self._object_center(o) for o in objects])
        else:
            cents = np.empty((0, 3))
        self.view.set_picking(cents)
        # Status
        parts = [f"{len(self.last_xyz):,} pts", f"mode {self.analysis_mode}"]
        if source == "humans":
            parts.append(f"{len(objects)} humans")
            if self.last_human_ms > 0:
                parts.append(self.human_status)
        elif source == "detections":
            parts.append(f"{len(objects)} detections")
            if self.last_detector_ms > 0:
                parts.append(self.detector_status)
        elif source == "clusters":
            parts.append(f"{len(objects)} clusters")
            if self.analysis_mode == "auto":
                parts.append("fallback DBSCAN")
            if not HAS_SKLEARN:
                parts.append("sklearn missing")
            elif self.last_cluster_ms > 0:
                parts.append(f"DBSCAN {self.last_cluster_ms:.0f}ms")
        elif self.analysis_mode == "indoor_human":
            parts.append(self.human_status)
        elif self.analysis_mode in ("pointpillars", "auto"):
            if self.analysis_mode == "auto":
                parts.append(self.human_status)
            parts.append(self.detector_status)
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

        if mode == "object_id":
            return self._color_by_object()

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

    def _color_by_object(self) -> np.ndarray:
        # Default = dim gray; object points get their track/class color.
        out = np.tile(np.array([0.18, 0.18, 0.20, 1.0], dtype=np.float32), (len(self.last_xyz), 1))
        objects, _ = self._active_objects()
        if not objects:
            return out
        max_id = max(obj.track_id for obj in objects)
        palette = _hsv_palette(max(64, max_id + 1))
        for obj in objects:
            in_box = self._points_in_object(obj)
            color = _detection_color(obj) if isinstance(obj, Detection) else palette[obj.track_id % len(palette)]
            out[in_box] = color
        return out

    def _draw_boxes(self) -> None:
        # Remove old boxes
        for item in self.box_items:
            self.view.removeItem(item)
        self.box_items.clear()
        for item in self.debug_box_items:
            self.view.removeItem(item)
        self.debug_box_items.clear()
        for item in self.label_items:
            self.view.removeItem(item)
        self.label_items.clear()

        if self.debug_check.isChecked():
            self._draw_candidate_debug()

        objects, source = self._active_objects()
        if not objects:
            return

        max_id = max(obj.track_id for obj in objects)
        palette = _hsv_palette(max(64, max_id + 1))
        for obj in objects:
            color = _detection_color(obj) if isinstance(obj, Detection) else palette[obj.track_id % len(palette)]
            if self.selected_track_id == obj.track_id and self.selected_source == source:
                color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                width = 3
            else:
                width = 2 if isinstance(obj, Detection) else 1

            if isinstance(obj, Detection):
                corners = self._oriented_box_lines(obj.center, obj.size, obj.yaw)
            else:
                corners = self._bbox_lines(obj.bbox_min, obj.bbox_max)
            line = gl.GLLinePlotItem(
                pos=corners, color=tuple(color), width=width, antialias=True, mode="lines"
            )
            self.view.addItem(line)
            self.box_items.append(line)
            if isinstance(obj, Detection):
                self._add_detection_label(obj, color)

    def _draw_candidate_debug(self) -> None:
        if not self.last_human_debug:
            return
        for item in self.last_human_debug[:80]:
            color = DEBUG_REASON_COLORS.get(
                item.reason,
                np.array([0.8, 0.8, 0.8, 0.35], dtype=np.float32),
            )
            width = 2 if item.accepted else 1
            line = gl.GLLinePlotItem(
                pos=self._oriented_box_lines(item.center, item.size, item.yaw),
                color=tuple(color),
                width=width,
                antialias=True,
                mode="lines",
            )
            self.view.addItem(line)
            self.debug_box_items.append(line)
            if item.accepted or len(self.debug_box_items) <= 50:
                self._add_debug_label(item, color)

    def _points_in_object(self, obj) -> np.ndarray:
        if isinstance(obj, Detection):
            return self._points_in_detection(obj, self.last_xyz)
        return (
            (self.last_xyz[:, 0] >= obj.bbox_min[0]) & (self.last_xyz[:, 0] <= obj.bbox_max[0])
            & (self.last_xyz[:, 1] >= obj.bbox_min[1]) & (self.last_xyz[:, 1] <= obj.bbox_max[1])
            & (self.last_xyz[:, 2] >= obj.bbox_min[2]) & (self.last_xyz[:, 2] <= obj.bbox_max[2])
        )

    @staticmethod
    def _points_in_detection(det: Detection, points: np.ndarray) -> np.ndarray:
        return PointCloudWindow._points_in_oriented_box(
            det.center,
            det.size,
            det.yaw,
            points,
            margin=(0.25, 0.25, 0.15),
        )

    @staticmethod
    def _points_in_oriented_box(
        center: np.ndarray,
        size: np.ndarray,
        yaw: float,
        points: np.ndarray,
        *,
        margin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        delta = points - center
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        local_x = c * delta[:, 0] + s * delta[:, 1]
        local_y = -s * delta[:, 0] + c * delta[:, 1]
        local_z = delta[:, 2]
        half = np.maximum(size * 0.5, 0.05) + np.array(margin, dtype=np.float32)
        return (
            (np.abs(local_x) <= half[0])
            & (np.abs(local_y) <= half[1])
            & (np.abs(local_z) <= half[2])
        )

    def _add_detection_label(self, det: Detection, color: np.ndarray) -> None:
        if not hasattr(gl, "GLTextItem"):
            return
        pos = det.center + np.array([0.0, 0.0, det.size[2] * 0.5 + 0.25], dtype=np.float32)
        text = f"{det.label.lower()} {det.score:.2f}"
        try:
            item = gl.GLTextItem(pos=pos, text=text, color=tuple(color))
            self.view.addItem(item)
            self.label_items.append(item)
        except Exception:
            pass

    def _add_debug_label(self, item: CandidateDebug, color: np.ndarray) -> None:
        if not hasattr(gl, "GLTextItem"):
            return
        pos = item.center + np.array([0.0, 0.0, item.size[2] * 0.5 + 0.15], dtype=np.float32)
        if item.accepted:
            text = f"{item.kind} {item.score:.2f} c{item.cluster_id}"
        else:
            text = f"{item.reason} {item.kind} {item.score:.2f}"
        try:
            label = gl.GLTextItem(pos=pos, text=text, color=tuple(color))
            self.view.addItem(label)
            self.label_items.append(label)
        except Exception:
            pass

    def closeEvent(self, ev) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        if self.detector_worker is not None:
            self.detector_worker.stop()
            self.detector_worker = None
        if self.human_worker is not None:
            self.human_worker.stop()
            self.human_worker = None
        super().closeEvent(ev)

    @staticmethod
    def _oriented_box_lines(center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
        hx, hy, hz = np.maximum(size * 0.5, 0.05)
        corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ], dtype=np.float32)
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        corners = corners @ rot.T + center
        edges = np.array([
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        ], dtype=np.int64)
        return corners[edges]

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
