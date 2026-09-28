"""Geometry and coordinate conversion.

Conventions: right-handed, +x right, +y front, +z up. Head-local coordinates
match world coordinates when the head is unrotated, so head-local +x points at
the right ear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Spherical:
    """Spherical coordinates: azimuth and elevation in radians, distance in metres.

    Fields may be scalars or arrays.
    """

    azimuth: np.ndarray
    elevation: np.ndarray
    distance: np.ndarray

    @property
    def azimuth_deg(self):
        return np.rad2deg(self.azimuth)

    @property
    def elevation_deg(self):
        return np.rad2deg(self.elevation)


def asarray3(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"expected a 3-vector, got {x!r}")
    return a


def sph_to_cart(azimuth, elevation, distance=1.0) -> np.ndarray:
    """Spherical to Cartesian. Broadcasts, returns shape (..., 3)."""
    az = np.asarray(azimuth, dtype=float)
    el = np.asarray(elevation, dtype=float)
    r = np.asarray(distance, dtype=float)
    shape = np.broadcast_shapes(az.shape, el.shape, r.shape)
    x = r * np.cos(el) * np.sin(az)
    y = r * np.cos(el) * np.cos(az)
    z = r * np.sin(el)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(*shape, 3)


def cart_to_sph(vec) -> Spherical:
    """Cartesian to spherical. ``vec`` has shape (..., 3)."""
    v = np.asarray(vec, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError("last axis must have length 3")
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    horiz = np.hypot(x, y)
    r = np.hypot(horiz, z)
    az = np.arctan2(x, y)
    el = np.arctan2(z, horiz)
    return Spherical(az, el, r)


def rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0) -> np.ndarray:
    """Head pose to a 3x3 rotation matrix (head-local vectors left-multiply it).

    Order: yaw about +z, then pitch about +x, then roll about +y. Radians.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return rz @ rx @ ry


def angle_between(a, b) -> np.ndarray:
    """Angle between two vectors in radians. Inputs have shape (..., 3)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    cos = np.einsum("...i,...i->...", a, b) / np.maximum(na * nb, 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))
