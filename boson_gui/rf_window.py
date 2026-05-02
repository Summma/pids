"""Live RF spectrum + waterfall popup for the RTL-SDR."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False

from rf import BAND_PRESETS, SDRThread, Spectrum, Emitter, HAS_RTLSDR


WATERFALL_ROWS = 240


def _fmt_hz(hz: float) -> str:
    if hz >= 1e9:
        return f"{hz / 1e9:.4f} GHz"
    if hz >= 1e6:
        return f"{hz / 1e6:.3f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


class RFWindow(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("RF — RTL-SDR Spectrum")
        self.resize(1200, 800)

        self.sdr: Optional[SDRThread] = None
        self.waterfall: Optional[np.ndarray] = None
        self.last_spec: Optional[Spectrum] = None
        self.last_emitters: list[Emitter] = []

        if not HAS_RTLSDR:
            lay = QVBoxLayout()
            lay.addWidget(QLabel(
                "pyrtlsdr not installed.\nbrew install librtlsdr && pip install pyrtlsdr==0.3.0"
            ))
            self.setLayout(lay)
            return
        if not HAS_PG:
            lay = QVBoxLayout()
            lay.addWidget(QLabel("pyqtgraph not installed."))
            self.setLayout(lay)
            return

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        # Spectrum plot (top)
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setBackground("#111")
        self.spec_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spec_plot.setLabel("left", "Power", units="dB")
        self.spec_plot.showGrid(x=True, y=True, alpha=0.25)
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen("#2bd", width=1))
        self.peak_scatter = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(255, 80, 80, 220), pen=None
        )
        self.spec_plot.addItem(self.peak_scatter)

        # Waterfall (bottom)
        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setBackground("#111")
        self.wf_plot.setLabel("bottom", "Frequency bin")
        self.wf_plot.setLabel("left", "Time (rows, newest at top)")
        self.wf_image = pg.ImageItem()
        self.wf_plot.addItem(self.wf_image)
        # Use a perceptually-uniform colormap
        cmap = pg.colormap.get("turbo", source="matplotlib")
        if cmap is not None:
            self.wf_image.setLookupTable(cmap.getLookupTable(0, 1, 256))

        # Controls — connection
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.toggled.connect(self._toggle_connect)

        # Controls — tuning
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("(custom)")
        self.preset_combo.addItems(BAND_PRESETS.keys())
        self.preset_combo.setCurrentText("ISM 433 MHz (LoRa/drone telem)")
        self.preset_combo.currentTextChanged.connect(self._on_preset)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(24.0, 1750.0)
        self.freq_spin.setDecimals(3)
        self.freq_spin.setSuffix(" MHz")
        self.freq_spin.setValue(433.92)
        self.freq_spin.valueChanged.connect(self._on_freq)

        self.rate_combo = QComboBox()
        for r in (0.25, 1.0, 1.024, 1.4, 1.6, 1.8, 2.048, 2.4, 2.56, 2.8, 3.2):
            self.rate_combo.addItem(f"{r:.3f} MHz", r * 1e6)
        self.rate_combo.setCurrentText("2.400 MHz")
        self.rate_combo.currentTextChanged.connect(self._on_rate)

        self.gain_combo = QComboBox()
        self.gain_combo.addItem("auto", "auto")
        for g in [0, 9, 14, 28, 33, 37, 41, 44, 49]:
            self.gain_combo.addItem(f"{g:.1f} dB", float(g))
        self.gain_combo.currentIndexChanged.connect(self._on_gain)

        tune_box = QGroupBox("Tuning")
        tf = QFormLayout()
        tf.addRow("Preset:", self.preset_combo)
        tf.addRow("Center:", self.freq_spin)
        tf.addRow("Sample rate:", self.rate_combo)
        tf.addRow("Gain:", self.gain_combo)
        tf.addRow(self.connect_btn)
        tune_box.setLayout(tf)

        # Detected emitters table
        self.emit_table = QTableWidget(0, 3)
        self.emit_table.setHorizontalHeaderLabels(["Frequency", "Power (dB)", "SNR (dB)"])
        self.emit_table.horizontalHeader().setStretchLastSection(True)
        self.emit_table.verticalHeader().setVisible(False)
        self.emit_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.emit_table.setSelectionBehavior(QTableWidget.SelectRows)
        emit_box = QGroupBox("Detected emitters")
        eb = QVBoxLayout()
        eb.addWidget(self.emit_table)
        emit_box.setLayout(eb)

        self.status_lbl = QLabel("disconnected")
        self.status_lbl.setStyleSheet("color:#888;padding:4px;")

        right = QVBoxLayout()
        right.addWidget(tune_box)
        right.addWidget(emit_box, 1)
        right.addWidget(self.status_lbl)
        right_w = QWidget()
        right_w.setFixedWidth(340)
        right_w.setLayout(right)

        plots = QVBoxLayout()
        plots.addWidget(self.spec_plot, 1)
        plots.addWidget(self.wf_plot, 1)
        plots_w = QWidget()
        plots_w.setLayout(plots)

        root = QHBoxLayout()
        root.addWidget(plots_w, 1)
        root.addWidget(right_w)
        self.setLayout(root)

    # --------------------------------------------------------- callbacks

    def _toggle_connect(self, on: bool) -> None:
        if on:
            self.sdr = SDRThread(
                center_hz=self.freq_spin.value() * 1e6,
                sample_rate_hz=self.rate_combo.currentData(),
                gain=self.gain_combo.currentData(),
            )
            self.sdr.spectrum_ready.connect(self._on_spectrum)
            self.sdr.emitters_ready.connect(self._on_emitters)
            self.sdr.error.connect(self._on_error)
            self.sdr.info.connect(self._on_info)
            self.sdr.start()
            self.connect_btn.setText("Disconnect")
            self.status_lbl.setText("connecting…")
        else:
            self._stop()
            self.connect_btn.setText("Connect")
            self.status_lbl.setText("disconnected")

    def _stop(self) -> None:
        if self.sdr is not None:
            self.sdr.stop()
            self.sdr = None
        self.waterfall = None

    def _on_preset(self, name: str) -> None:
        if name not in BAND_PRESETS:
            return
        center, rate = BAND_PRESETS[name]
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(center / 1e6)
        self.freq_spin.blockSignals(False)
        # Match rate combo
        for i in range(self.rate_combo.count()):
            if abs(self.rate_combo.itemData(i) - rate) < 1e3:
                self.rate_combo.setCurrentIndex(i)
                break
        if self.sdr is not None:
            self.sdr.request_freq(center)
            self.sdr.request_rate(rate)
        self.waterfall = None  # reset on retune

    def _on_freq(self, mhz: float) -> None:
        if self.sdr is not None:
            self.sdr.request_freq(mhz * 1e6)
        self.waterfall = None
        self.preset_combo.setCurrentText("(custom)")

    def _on_rate(self, _text: str) -> None:
        if self.sdr is not None:
            self.sdr.request_rate(self.rate_combo.currentData())
        self.waterfall = None

    def _on_gain(self, _idx: int) -> None:
        if self.sdr is not None:
            self.sdr.request_gain(self.gain_combo.currentData())

    def _on_info(self, msg: str) -> None:
        self.status_lbl.setText(msg)

    def _on_error(self, msg: str) -> None:
        self.status_lbl.setText(f"error: {msg}")
        self.connect_btn.setChecked(False)

    # ----------------------------------------------------------- update

    def _on_spectrum(self, spec: Spectrum) -> None:
        self.last_spec = spec
        self.spec_curve.setData(spec.freqs_hz, spec.power_db)
        self.spec_plot.setLabel(
            "bottom",
            f"Frequency  ({_fmt_hz(spec.center_hz)} center, "
            f"{spec.sample_rate_hz/1e6:.2f} MHz BW)",
        )

        # Roll waterfall
        N = len(spec.power_db)
        if self.waterfall is None or self.waterfall.shape[1] != N:
            self.waterfall = np.full((WATERFALL_ROWS, N), -120.0, dtype=np.float32)
        self.waterfall = np.roll(self.waterfall, 1, axis=0)
        self.waterfall[0] = spec.power_db
        # Map to 0..1 with adaptive levels (5..99 percentile)
        lo = float(np.percentile(self.waterfall, 5))
        hi = float(np.percentile(self.waterfall, 99))
        if hi - lo < 1.0:
            hi = lo + 1.0
        norm = np.clip((self.waterfall - lo) / (hi - lo), 0.0, 1.0)
        # ImageItem expects (W, H) for default axis order; transpose
        self.wf_image.setImage(norm.T, autoLevels=False, levels=(0.0, 1.0))

    def _on_emitters(self, emitters: list[Emitter]) -> None:
        self.last_emitters = emitters
        # Mark on spectrum
        if self.last_spec is not None:
            xs = [e.freq_hz for e in emitters]
            ys = [e.power_db for e in emitters]
            self.peak_scatter.setData(xs, ys)
        # Fill table
        self.emit_table.setRowCount(len(emitters))
        for r, e in enumerate(emitters):
            self.emit_table.setItem(r, 0, QTableWidgetItem(_fmt_hz(e.freq_hz)))
            self.emit_table.setItem(r, 1, QTableWidgetItem(f"{e.power_db:.1f}"))
            self.emit_table.setItem(r, 2, QTableWidgetItem(f"{e.snr_db:.1f}"))

    def closeEvent(self, ev) -> None:
        self._stop()
        super().closeEvent(ev)
