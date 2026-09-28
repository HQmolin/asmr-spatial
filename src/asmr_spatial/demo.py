"""Render a set of comparison samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audio
from .hrtf import HeadModel
from .render import (
    RenderConfig, keyframes, orbit, render, static_position,
)


def _fallback_signal(fs: int, seconds: float = 6.0) -> np.ndarray:
    """Synthesised voice-like signal used when no material is available."""
    rng = np.random.default_rng(7)
    n = int(fs * seconds)
    t = np.arange(n) / fs

    from scipy.signal import butter, sosfilt

    sos = butter(2, [700.0, 6500.0], btype="bandpass", fs=fs, output="sos")
    breath = sosfilt(sos, rng.standard_normal(n))
    breath /= max(np.max(np.abs(breath)), 1e-9)

    env = np.zeros(n)
    pos = 0
    while pos < n:
        dur = int(rng.uniform(0.18, 0.42) * fs)
        gap = int(rng.uniform(0.06, 0.22) * fs)
        seg = np.hanning(min(dur, n - pos))
        env[pos: pos + seg.size] = seg * rng.uniform(0.4, 1.0)
        pos += dur + gap

    voice = breath * env
    voice += 0.15 * np.sin(2 * np.pi * 140 * t) * env
    return voice / max(np.max(np.abs(voice)), 1e-9) * 0.6


def _find_input() -> Path | None:
    root = Path(__file__).resolve().parents[2] / "examples" / "input"
    if not root.is_dir():
        return None
    found = sorted(p for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg")
                   for p in root.glob(ext))
    return found[0] if found else None


DEMO_JOBS = [
    ("01_front_50cm", dict(kind="static", az=0, el=0, dist=0.5)),
    ("02_left_ear_15cm", dict(kind="static", az=-78, el=-8, dist=0.155)),
    ("03_right_ear_15cm", dict(kind="static", az=78, el=-8, dist=0.155)),
    ("04_above", dict(kind="static", az=0, el=75, dist=0.22)),
    ("05_below", dict(kind="static", az=0, el=-75, dist=0.22)),
    ("06_left_ear_2m", dict(kind="static", az=-78, el=-8, dist=2.0)),
    ("07_orbit_18cm", dict(kind="orbit", radius=0.18, duration=8.0)),
    ("08_left_to_right", dict(kind="keys", pts=[
        (0.0, 0, 0, 0.45), (2.5, -78, -8, 0.155),
        (5.0, -40, -5, 0.20), (7.5, 78, -8, 0.155), (10.0, 0, 0, 0.45),
    ])),
    ("09_back_of_head", dict(kind="static", az=180, el=-45, dist=0.15)),
    ("10_approach_left_ear", dict(kind="keys", pts=[
        (0.0, -78, -8, 1.2), (3.0, -78, -8, 0.40), (6.0, -78, -8, 0.16),
    ])),
]


def build_demo(outdir: str | Path, head: HeadModel | None = None,
               config: RenderConfig | None = None, input_path=None):
    head = head or HeadModel()
    cfg = config or RenderConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    src = Path(input_path) if input_path else _find_input()
    if src and src.exists():
        mono, _ = audio.load_mono(src, head.sample_rate)
        origin = f"material: {src.name}"
    else:
        mono = _fallback_signal(head.sample_rate)
        origin = "material: built-in synthesised signal"

    print(f"{head.sample_rate} Hz | head radius {head.radius*100:.1f} cm | "
          f"ear spacing {head.ear_spacing*100:.1f} cm | "
          f"closest {head.min_distance*100:.1f} cm")
    print(origin)
    print(f"{'sample':<24}{'azimuth':>8}{'elev':>6}{'dist':>7}   ITD(ms)  ILD@4k(dB)")

    written = []
    for name, job in DEMO_JOBS:
        if job["kind"] == "static":
            pos = static_position(job["az"], job["el"], job["dist"])
            itd, ild = head.ild_itd(job["az"], job["el"], job["dist"])
            print(f"{name:<24}{job['az']:>8.0f}{job['el']:>6.0f}"
                  f"{job['dist']*100:>6.0f}c   {itd*1000:>6.2f}   {ild:>8.1f}")
        elif job["kind"] == "orbit":
            dur = max(mono.size / head.sample_rate, 1.0)
            pos = orbit(job["radius"], duration=dur)
            print(f"{name:<24}{'orbit':>8}{'':>6}{job['radius']*100:>6.0f}c   "
                  f"{'-':>6}   {'-':>8}")
        else:
            pos = keyframes(job["pts"])
            print(f"{name:<24}{'path':>8}{'':>6}{'-':>7}   {'-':>6}   {'-':>8}")

        y = render(mono, pos, head, cfg)
        path = audio.save(outdir / f"{name}.wav", y, head.sample_rate, cfg.normalize_db)
        written.append(path)

    print(f"\nwrote {len(written)} files to {outdir}")
    return outdir
