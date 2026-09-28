"""Probe the rigid-sphere series: truncation order convergence, and where the
elevation difference actually comes from once the pinna layer is switched off."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from asmr_spatial.geometry import sph_to_cart
from asmr_spatial.hrtf import HeadModel, rigid_sphere_magnitude

A, C = 0.09, 343.0
R = 1.0
FREQS = np.array([1000.0, 4000.0, 8000.0])

for lmax in (20, 33, 40, 60, 90, 130, 180):
    vals = {}
    for name, th in (("front", 0.0), ("side", np.pi / 2), ("back", np.pi)):
        m = rigid_sphere_magnitude(np.array([th]), FREQS, A, np.array([R]), C, lmax)[0]
        vals[name] = 20 * np.log10(np.maximum(m, 1e-12))
    print(f"order={lmax:>4}  front={vals['front'].round(1)}  "
          f"side={vals['side'].round(1)}  back={vals['back'].round(1)}")

# Where does the up/down difference come from when the pinna is disabled?
off = HeadModel(sample_rate=48000, pinna=False, torso=False)
r_head = off.min_distance
for el_deg in (60.0, -60.0):
    src = np.asarray(sph_to_cart(0.0, np.deg2rad(el_deg), r_head)).reshape(3)
    v = off.ear_positions[0]
    ct = float(np.dot(src, v) / (np.linalg.norm(src) * np.linalg.norm(v)))
    th = np.arccos(np.clip(ct, -1, 1))
    m8 = rigid_sphere_magnitude(np.array([th]), np.array([8000.0]), off.radius,
                                np.array([r_head]), 343.0, 40)[0, 0]
    m8p = rigid_sphere_magnitude(np.array([th]), np.array([8000.0]), off.radius,
                                 np.array([r_head]), 343.0, 40, shadow_db=0.0)[0, 0]
    print(f"  el={el_deg:+.0f}  theta={np.rad2deg(th):5.1f} deg  "
          f"8k with-absorption {20*np.log10(m8):+.2f} dB | rigid {20*np.log10(m8p):+.2f} dB")
