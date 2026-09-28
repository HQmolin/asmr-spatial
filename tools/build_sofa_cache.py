"""Convert a measured SOFA HRIR set into the compact cache this engine uses.

Runs once; h5py is not needed at render time.

    python tools/build_sofa_cache.py <input.sofa> <output.npz> [--decimate 1]

SOFA azimuth is counter-clockwise, so +90 degrees points at the left ear; this
engine uses the opposite convention and negates it. The measured linear-phase
HRIRs carry their own ITD, which is extracted and removed here because delay is
applied geometrically at render time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import correlate


def fractional_shift(x: np.ndarray, samples: float, fs: float) -> np.ndarray:
    n = x.size
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * samples / fs), n=n)


def estimate_lag(left: np.ndarray, right: np.ndarray) -> float:
    """Delay of right relative to left, in samples, with sub-sample refinement."""
    n = left.size
    corr = correlate(right, left, mode="full")
    k = int(np.argmax(np.abs(corr)))
    if 0 < k < corr.size - 1:
        y0, y1, y2 = np.abs(corr[k - 1]), np.abs(corr[k]), np.abs(corr[k + 1])
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-20 else 0.0
    else:
        delta = 0.0
    return (k - (n - 1)) + float(np.clip(delta, -1, 1))


def build(sofa_path: Path, out_path: Path, decimate: int = 1, verbose: bool = True):
    t0 = time.time()
    with h5py.File(sofa_path, "r") as f:
        ir = np.asarray(f["Data.IR"], dtype=np.float64)
        pos = np.asarray(f["SourcePosition"], dtype=np.float64)
        fs = float(np.asarray(f["Data.SamplingRate"]).ravel()[0])
        attrs = {k: v for k, v in f.attrs.items()}
        r_ref = float(pos[:, 2].mean())

    m, n_ears, taps = ir.shape
    if n_ears != 2:
        raise ValueError(f"only binaural HRIR sets are supported, got {n_ears} receivers")

    if decimate > 1:
        keep = np.arange(0, m, decimate)
        ir, pos = ir[keep], pos[keep]
        m = keep.size

    az = -np.deg2rad(pos[:, 0])
    el = np.deg2rad(pos[:, 1])
    unit = np.stack([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ], axis=1)

    aligned = np.empty((m, 2, taps), dtype=np.float32)
    itd = np.empty(m, dtype=np.float32)
    for i in range(m):
        hl, hr = ir[i, 0], ir[i, 1]
        lag = estimate_lag(hl, hr)
        itd[i] = -lag / fs
        aligned[i, 0] = fractional_shift(hl, +lag / 2, fs)
        aligned[i, 1] = fractional_shift(hr, -lag / 2, fs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ir=aligned, az=az.astype(np.float32), el=el.astype(np.float32),
        unit=unit.astype(np.float32), itd=itd,
        fs=np.float32(fs), source_distance=np.float32(r_ref),
        title=np.array(str(attrs.get("Title", "?"))),
        listener=np.array(str(attrs.get("ListenerShortName", "?"))),
        database=np.array(str(attrs.get("DatabaseName", "?"))),
        license=np.array(str(attrs.get("License", "?"))),
        references=np.array(str(attrs.get("References", ""))),
        comment=np.array(str(attrs.get("Comment", ""))),
    )
    if verbose:
        size = out_path.stat().st_size / 1024 ** 2
        print(f"wrote {out_path}  ({size:.1f} MB)")
        print(f"  {m} directions | {taps} taps | {fs:.0f} Hz | measured at {r_ref:.2f} m")
        print(f"  azimuth {np.rad2deg(np.abs(az)).max():.0f} deg range, "
              f"elevation {np.rad2deg(el).min():.0f}..{np.rad2deg(el).max():.0f} deg")
        print(f"  ITD {itd.min()*1e3:+.3f} .. {itd.max()*1e3:+.3f} ms")
        print(f"  source: {attrs.get('DatabaseName','?')} / {attrs.get('ListenerShortName','?')}")
        print(f"  {time.time()-t0:.1f} s")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sofa")
    ap.add_argument("output")
    ap.add_argument("--decimate", type=int, default=1,
                    help="keep every Nth direction (1 keeps all)")
    a = ap.parse_args()
    build(Path(a.sofa), Path(a.output), a.decimate)


if __name__ == "__main__":
    main()
