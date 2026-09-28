"""Optional vocal / accompaniment separation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass
class Analysis:
    channels: int
    correlation: float
    side_ratio_db: float
    sub_bass_ratio: float
    is_mix: bool
    reason: str

    def describe(self) -> str:
        kind = "mix with accompaniment" if self.is_mix else "voice only"
        return (
            f"{kind}; {self.channels} channels, L/R correlation {self.correlation:.3f}, "
            f"side/mid {self.side_ratio_db:+.1f} dB, energy below 100 Hz "
            f"{self.sub_bass_ratio*100:.1f}%\n  reason: {self.reason}"
        )


def analyze(y: np.ndarray, fs: int) -> Analysis:
    y = np.atleast_2d(np.asarray(y, dtype=float))
    channels = y.shape[0]

    if channels < 2:
        return Analysis(1, 1.0, float("-inf"), _sub_bass(y[0], fs), False,
                        "mono, nothing to separate")

    L, R = y[0], y[1]
    denom = np.sqrt(np.sum(L ** 2) * np.sum(R ** 2))
    corr = float(np.sum(L * R) / denom) if denom > 0 else 1.0

    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    mid_e, side_e = float(np.sum(mid ** 2)), float(np.sum(side ** 2))
    side_db = 10.0 * np.log10(max(side_e, 1e-30) / max(mid_e, 1e-30))
    sub = _sub_bass(mid, fs)

    if corr >= 0.985 and side_db < -35.0:
        return Analysis(channels, corr, side_db, sub, False,
                        "left and right are near identical, mono source")
    if corr >= 0.97 and sub < 0.005 and side_db < -25.0:
        return Analysis(channels, corr, side_db, sub, False,
                        "strongly centred with almost no low end, looks like voice only")

    bits = []
    if corr < 0.97:
        bits.append(f"L/R correlation only {corr:.2f}, stereo instruments present")
    if side_db > -25.0:
        bits.append(f"strong side content ({side_db:+.1f} dB)")
    if sub >= 0.005:
        bits.append(f"{sub*100:.1f}% of energy below 100 Hz, looks like drums or bass")
    return Analysis(channels, corr, side_db, sub, True, "; ".join(bits) or "likely a mix")


def _sub_bass(x: np.ndarray, fs: int, cutoff: float = 100.0) -> float:
    sos = butter(4, cutoff, btype="low", fs=fs, output="sos")
    low = sosfilt(sos, x)
    total = float(np.sum(x ** 2))
    return float(np.sum(low ** 2) / total) if total > 0 else 0.0


def split_mid_side(y: np.ndarray, fs: int, center_keep: float = 0.5,
                   bass_lp_hz: float | None = None):
    """Return (vocal_mono, accompaniment).

    center_keep: 0 removes the centre entirely, 1 keeps the original mix.
    bass_lp_hz: if set, only centre content below this frequency is kept.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] < 2:
        raise ValueError("mid/side separation needs stereo input")

    L, R = y[0], y[1]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    if bass_lp_hz is None:
        body = mid * float(center_keep)
    else:
        low = sosfilt(butter(2, bass_lp_hz, btype="low", fs=fs, output="sos"), mid)
        body = low * float(center_keep)
    accompaniment = np.stack([side + body, -side + body])
    return mid, accompaniment


def demucs_available() -> bool:
    return shutil.which("demucs") is not None


def split_demucs(input_path, target_rate: int, model: str = "htdemucs"):
    """Separate vocals with Demucs. Requires ``pip install demucs``."""
    from . import audio

    if not demucs_available():
        raise RuntimeError(
            "demucs not found. Install it for cleaner separation:\n"
            "    pip install demucs\n"
            "(pulls in PyTorch, roughly 2-3 GB; CPU works but is slow)"
        )

    with tempfile.TemporaryDirectory(prefix="asmr_demucs_") as tmp:
        cmd = ["demucs", "--two-stems=vocals", "-n", model,
               "-o", tmp, str(input_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"demucs failed:\n{result.stderr.strip()[-1500:]}")
        stem_dir = Path(tmp) / model / Path(input_path).stem
        vocal_path = stem_dir / "vocals.wav"
        no_vocal_path = stem_dir / "no_vocals.wav"
        if not vocal_path.exists() or not no_vocal_path.exists():
            raise RuntimeError(f"demucs produced no expected files, got: {list(stem_dir.glob('*'))}")
        vocal, _ = audio.load_mono(vocal_path, target_rate)
        accomp, _ = audio.load_stereo(no_vocal_path, target_rate)
        return vocal, accomp
