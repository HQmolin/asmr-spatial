"""Audio file I/O. MP3, FLAC and OGG are handled by libsndfile directly."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: str | Path, target_rate: int | None = None):
    """Read a file and mix it down to mono. Returns (samples, sample_rate)."""
    data, rate = sf.read(str(path), always_2d=True, dtype="float64")
    mono = data.mean(axis=1)
    if target_rate is not None and rate != target_rate:
        mono = resample(mono, rate, target_rate)
        rate = int(target_rate)
    return mono, rate


def load_stereo(path: str | Path, target_rate: int | None = None):
    """Read a file as two channels. Returns ((2, N) planar array, sample_rate)."""
    data, rate = sf.read(str(path), always_2d=True, dtype="float64")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    planar = np.ascontiguousarray(data.T)
    if target_rate is not None and rate != target_rate:
        planar = np.stack([resample(ch, rate, target_rate) for ch in planar])
        rate = int(target_rate)
    return planar, rate


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return np.asarray(x, dtype=float)
    from scipy.signal import resample_poly

    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(np.asarray(x, dtype=float), dst_rate // g, src_rate // g)


def save(path: str | Path, data: np.ndarray, rate: int, normalize_db: float | None = -1.0):
    """Write audio. ``data`` is (2, N) or (N,). Pass normalize_db=None to skip."""
    y = np.asarray(data, dtype=np.float64)
    interleaved = y.T if (y.ndim == 2 and y.shape[0] <= 2) else y
    if normalize_db is not None and interleaved.size:
        peak = float(np.max(np.abs(interleaved)))
        if peak > 0:
            interleaved = interleaved * (10.0 ** (normalize_db / 20.0) / peak)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {".mp3": "MP3", ".flac": "FLAC", ".ogg": "OGG"}.get(path.suffix.lower(), "WAV")
    sf.write(str(path), interleaved, int(rate), format=fmt)
    return path
