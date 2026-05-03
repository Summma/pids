"""RTL-SDR I/O.

Background QThread that streams IQ samples from the dongle, computes a
power spectrum (averaged FFT) and emits it to the GUI. Also emits detected
emitter peaks above an adaptive noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    from rtlsdr import RtlSdr
    HAS_RTLSDR = True
except Exception:
    RtlSdr = None  # type: ignore
    HAS_RTLSDR = False


# Useful presets within the v5's coverage (~25 MHz – 1.75 GHz)
BAND_PRESETS: dict[str, tuple[float, float]] = {
    # name: (center_hz, sample_rate_hz)
    "FM broadcast (100 MHz)": (100.0e6, 2.4e6),
    "Aircraft VHF (130 MHz)": (130.0e6, 2.4e6),
    "FRS/GMRS (462.5 MHz)": (462.5e6, 2.4e6),
    "ISM 433 MHz (LoRa/drone telem)": (433.92e6, 2.4e6),
    "ISM 868 MHz (EU)": (868.0e6, 2.4e6),
    "ISM 915 MHz (US)": (915.0e6, 2.4e6),
    "ADS-B (1090 MHz)": (1090.0e6, 2.4e6),
    "Cell 850 uplink": (835.0e6, 2.4e6),
    "Cell 1900 uplink": (1880.0e6, 2.4e6),
    "GPS L1 (1575 MHz)": (1575.42e6, 2.4e6),
}


@dataclass
class Spectrum:
    freqs_hz: np.ndarray  # (N,) absolute frequencies
    power_db: np.ndarray  # (N,) power per bin in dB (relative)
    center_hz: float
    sample_rate_hz: float
    fft_size: int


@dataclass
class Emitter:
    freq_hz: float
    power_db: float
    snr_db: float


class SDRThread(QThread):
    spectrum_ready = Signal(object)         # Spectrum
    emitters_ready = Signal(object)         # list[Emitter]
    error = Signal(str)
    info = Signal(str)

    def __init__(
        self,
        center_hz: float = 433.92e6,
        sample_rate_hz: float = 2.4e6,
        gain="auto",
        fft_size: int = 4096,
        averages: int = 8,
        peak_min_snr_db: float = 10.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.center_hz = center_hz
        self.sample_rate_hz = sample_rate_hz
        self.gain = gain
        self.fft_size = fft_size
        self.averages = averages
        self.peak_min_snr_db = peak_min_snr_db
        self._running = False
        self._sdr: Optional[RtlSdr] = None
        # Pending tune requests applied between blocks (thread-safe via attr writes)
        self._pending_freq: Optional[float] = None
        self._pending_rate: Optional[float] = None
        self._pending_gain = None  # str | float | None

    # --- public control (called from GUI thread) ---

    def request_freq(self, hz: float) -> None:
        self._pending_freq = float(hz)

    def request_rate(self, hz: float) -> None:
        self._pending_rate = float(hz)

    def request_gain(self, gain) -> None:
        self._pending_gain = gain

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    # --- thread body ---

    def run(self) -> None:
        if not HAS_RTLSDR:
            self.error.emit("pyrtlsdr not installed (pip install pyrtlsdr==0.3.0)")
            return
        try:
            self._sdr = RtlSdr()
            self._sdr.sample_rate = self.sample_rate_hz
            self._sdr.center_freq = self.center_hz
            try:
                self._sdr.set_gain(self.gain)
            except Exception:
                self._sdr.set_gain("auto")
            self.info.emit(
                f"tuner={self._sdr.get_tuner_type()} fs={self.sample_rate_hz/1e6:.2f} MHz"
            )
        except Exception as e:
            self.error.emit(f"open failed: {e}")
            return

        self._running = True
        win = np.hanning(self.fft_size).astype(np.float32)
        win_norm = float(np.sum(win**2))

        try:
            while self._running:
                self._apply_pending()
                # Read enough samples for `averages` non-overlapping FFTs
                n = self.fft_size * self.averages
                try:
                    iq = self._sdr.read_samples(n)
                except Exception as e:
                    self.error.emit(f"read failed: {e}")
                    break
                iq = np.asarray(iq, dtype=np.complex64)
                if iq.size < n:
                    continue

                # Average power spectrum over `averages` blocks
                spec_sum = np.zeros(self.fft_size, dtype=np.float32)
                for k in range(self.averages):
                    block = iq[k * self.fft_size : (k + 1) * self.fft_size] * win
                    f = np.fft.fftshift(np.fft.fft(block))
                    spec_sum += (f.real * f.real + f.imag * f.imag)
                power = spec_sum / (self.averages * win_norm)
                power_db = 10.0 * np.log10(power + 1e-12)

                freqs = (
                    self.center_hz
                    + np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1.0 / self.sample_rate_hz))
                )
                spec = Spectrum(
                    freqs_hz=freqs,
                    power_db=power_db,
                    center_hz=self.center_hz,
                    sample_rate_hz=self.sample_rate_hz,
                    fft_size=self.fft_size,
                )
                self.spectrum_ready.emit(spec)

                # Peak detection (cheap CFAR-ish via running median)
                peaks = self._find_peaks(freqs, power_db)
                if peaks:
                    self.emitters_ready.emit(peaks)
        finally:
            try:
                if self._sdr is not None:
                    self._sdr.close()
            except Exception:
                pass
            self._sdr = None

    def _apply_pending(self) -> None:
        if self._sdr is None:
            return
        if self._pending_freq is not None:
            try:
                self._sdr.center_freq = self._pending_freq
                self.center_hz = self._pending_freq
            except Exception:
                pass
            self._pending_freq = None
        if self._pending_rate is not None:
            try:
                self._sdr.sample_rate = self._pending_rate
                self.sample_rate_hz = self._pending_rate
            except Exception:
                pass
            self._pending_rate = None
        if self._pending_gain is not None:
            try:
                self._sdr.set_gain(self._pending_gain)
                self.gain = self._pending_gain
            except Exception:
                pass
            self._pending_gain = None

    def _find_peaks(self, freqs: np.ndarray, power_db: np.ndarray) -> list[Emitter]:
        # Edges are noisy due to DC spike + filter rolloff — ignore outer 5%
        edge = max(1, int(0.05 * len(power_db)))
        view = power_db[edge:-edge]
        view_freqs = freqs[edge:-edge]
        # Noise estimate: median of the bottom 60% of bins
        srt = np.sort(view)
        noise = float(np.mean(srt[: int(0.6 * len(srt))]))
        snr = view - noise
        # Local maxima above threshold
        thr = self.peak_min_snr_db
        is_peak = (
            (snr > thr)
            & (view > np.roll(view, 1))
            & (view > np.roll(view, -1))
        )
        idxs = np.where(is_peak)[0]
        # Suppress tightly-clustered peaks: keep strongest within ±20 bins
        keep: list[int] = []
        for i in sorted(idxs, key=lambda j: -view[j]):
            if all(abs(i - k) > 20 for k in keep):
                keep.append(i)
            if len(keep) >= 12:
                break
        keep.sort()
        return [
            Emitter(
                freq_hz=float(view_freqs[i]),
                power_db=float(view[i]),
                snr_db=float(snr[i]),
            )
            for i in keep
        ]
