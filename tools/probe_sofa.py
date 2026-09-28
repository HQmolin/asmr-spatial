"""Inspect a SOFA cache: ITD against azimuth, quality of the delay alignment,
and the resulting frequency response.

Usage:
    python tools/probe_sofa.py [path/to/cache.npz]
"""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from build_sofa_cache import estimate_lag
from asmr_spatial.sofa import default_path

path = sys.argv[1] if len(sys.argv) > 1 else str(default_path())
d = np.load(path, allow_pickle=False)
fs = float(d["fs"])
az, el, itd, ir = d["az"], d["el"], d["itd"], d["ir"]

print(f"{ir.shape[0]} directions | {ir.shape[2]} taps | {fs:.0f} Hz | "
      f"measured at {float(d['source_distance']):.2f} m")
print(f"source {d['database']} / {d['listener']}")

print("\n-- ITD against azimuth, horizontal plane --")
sel0 = np.abs(el) < np.deg2rad(1.0)
print(f"{'azimuth':>10}{'ITD (ms)':>10}{'residual after alignment':>26}")
for target in (0, 30, 60, 90, 120, 150, 180, -30, -60, -90):
    want = np.deg2rad(target)
    idx = np.where(sel0)[0]
    j = idx[np.argmin(np.abs(np.angle(np.exp(1j * (az[idx] - want)))))]
    lag = estimate_lag(ir[j, 0], ir[j, 1])
    print(f"{target:>10}{itd[j]*1e3:>10.3f}{lag:>26.3f}")

print("\n-- Front vs back --")
rows = {}
for target in (0, 180):
    want = np.deg2rad(target)
    idx = np.where(sel0)[0]
    j = idx[np.argmin(np.abs(np.angle(np.exp(1j * (az[idx] - want)))))]
    spec = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(ir[j, 0], 1024)), 1e-9))
    f = np.fft.rfftfreq(1024, 1 / fs)
    band = (f > 4000) & (f < 10000)
    rows[target] = spec
    print(f"  azimuth {target:>4}: 4-10 kHz left ear mean {spec[band].mean():.1f} dB")

diff = rows[0] - rows[180]
broad = (f > 4000) & (f < 16000)
print(f"  spectral difference over 4-16 kHz: mean {diff[broad].mean():+.1f} dB, "
      f"std {diff[broad].std():.1f} dB, max {np.abs(diff[broad]).max():.1f} dB")
print("  The std and max are what matters: front/back is cued by notch position,")
print("  not by average level.")
for target in (0, 180):
    dip = f[broad][np.argmin(rows[target][broad])]
    print(f"  azimuth {target:>4}: deepest notch in 4-16 kHz at {dip/1000:.2f} kHz")

print("\n-- Left vs right side --")
print(f"{'azimuth':>8}{'ILD @ 1k':>10}{'ILD @ 6k':>10}")
for target in (0, 45, 90, -45, -90):
    want = np.deg2rad(target)
    idx = np.where(sel0)[0]
    j = idx[np.argmin(np.abs(np.angle(np.exp(1j * (az[idx] - want)))))]
    out = []
    for f0 in (1000.0, 6000.0):
        k = int(round(f0 / (fs / 1024)))
        L = abs(np.fft.rfft(ir[j, 0], 1024)[k])
        R = abs(np.fft.rfft(ir[j, 1], 1024)[k])
        out.append(20 * np.log10(L / max(R, 1e-12)))
    print(f"{target:>8}{out[0]:>10.1f}{out[1]:>10.1f}")
