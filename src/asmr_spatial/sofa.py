"""Measured HRIR (SOFA) head model. Interchangeable with ``hrtf.HeadModel``.

Supplies what the analytic model cannot: front/back discrimination and real
pinna, ear canal and torso cues, since the data comes from a measured
human-scale head (here a Neumann KU100 dummy head).

The data was measured at a single distance of 1.2 m, so near-field range
dependence and the distance scaling of the ITD are still applied analytically.

Directions are interpolated by angular inverse distance weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import sph_to_cart
from .hrtf import HeadModel, rigid_sphere_magnitude


@dataclass
class SofaHead:
    """Measured SOFA head model. Interface compatible with HeadModel."""

    path: str | Path
    sample_rate: int = 48000
    radius: float = 0.09
    ear_spacing: float = 0.18
    ear_offset: tuple[float, float] = (0.0, 0.0)
    # Closest distance as a multiple of the radius. 1.02 puts the head centre at
    # 9.18 cm, about 2 mm from the ear.
    min_distance_factor: float = 1.02
    # Safe lower bound for the sphere series. Below this, exact geometry is used.
    near_field_safe_factor: float = 1.2
    # Contact coupling: near the ear canal the ear becomes pressure driven and
    # the occlusion effect appears, a large rise below 500 Hz. 0 disables it.
    occlusion_max_db: float = 15.0
    contact_ref_m: float = 0.03
    contact_power: float = 1.5
    contact_corner_hz: float = 500.0
    speed_of_sound: float = 343.0
    n_neighbors: int = 4
    near_field: bool = True
    max_distance_scale: float = 4.0
    max_order: int = 40
    _cache: dict = field(default_factory=dict, repr=False, compare=False)
    _data: dict = field(default_factory=dict, repr=False, compare=False)
    _tree: object = field(default=None, repr=False, compare=False)
    _geom: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        d = np.load(str(self.path), allow_pickle=False)
        self._data = {
            "ir": np.asarray(d["ir"], dtype=np.float32),
            "unit": np.asarray(d["unit"], dtype=np.float64),
            "itd": np.asarray(d["itd"], dtype=np.float64),
            "fs": float(d["fs"]),
            "source_distance": float(d["source_distance"]),
            "database": str(d["database"]) if "database" in d else "?",
            "listener": str(d["listener"]) if "listener" in d else "?",
        }
        self._tree = cKDTree(self._data["unit"])
        self._geom = HeadModel(
            sample_rate=self.sample_rate, radius=self.radius,
            ear_spacing=self.ear_spacing, ear_offset=self.ear_offset,
            speed_of_sound=self.speed_of_sound,
            min_distance_factor=self.min_distance_factor,
        )
        self.hrir_taps = self._data["ir"].shape[2]

    # -------------------------------------------------------------- metadata
    @property
    def source_distance(self) -> float:
        return self._data["source_distance"]

    @property
    def min_distance(self) -> float:
        return self.radius * self.min_distance_factor

    @property
    def ear_positions(self) -> np.ndarray:
        return self._geom.ear_positions

    def describe(self) -> str:
        d = self._data
        return (f"measured SOFA head: {d['database']} / {d['listener']}, "
                f"{d['ir'].shape[0]} directions, {self.hrir_taps} taps, "
                f"{d['fs']:.0f} Hz, measured at {d['source_distance']:.2f} m")

    # -------------------------------------------------------------- geometry
    def path_length(self, source_head_local: np.ndarray) -> np.ndarray:
        return self._geom.path_length(source_head_local)

    # ------------------------------------------------------ interpolation core
    def _weights(self, direction: np.ndarray):
        """Angular inverse distance weighting. Returns (indices, weights)."""
        k = min(self.n_neighbors, self._data["unit"].shape[0])
        dist, idx = self._tree.query(direction, k=k)
        dist = np.atleast_1d(dist)
        idx = np.atleast_1d(idx)
        w = 1.0 / (dist ** 2 + 1e-6)
        return idx, w / w.sum()

    def _interp_ir(self, direction: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (aligned 2 x T HRIR, measured ITD in seconds)."""
        idx, w = self._weights(direction)
        ir = np.tensordot(w, self._data["ir"][idx], axes=(0, 0))    # (2, T)
        itd = float(np.dot(w, self._data["itd"][idx]))
        return ir, itd

    # -------------------------------------------------------- near-field fix
    def _near_field_gain(self, azimuth, elevation, r, r_ref, ears_unit):
        """Near-field magnitude correction per ear, (2, F). ``r`` is model distance."""
        nfft = 2 * self.hrir_taps
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sample_rate)
        freqs[0] = freqs[1]
        # The correction is smooth, so the sphere series is only solved every
        # sixth bin and interpolated back. About six times faster, inaudible.
        stride = 6
        coarse = freqs[::stride]
        if coarse[-1] != freqs[-1]:
            coarse = np.append(coarse, freqs[-1])
        src = np.asarray(sph_to_cart(azimuth, elevation, r)).reshape(3)
        rn = float(np.linalg.norm(src))
        # The series diverges below r/a = 1.15, so it is only evaluated at
        # r_safe. Anything closer uses exact geometry instead: the 1/r law holds
        # exactly there and needs no series.
        r_safe = max(r, self.radius * self.near_field_safe_factor)
        d_now = self._geom.path_length(src.reshape(1, 3))[0]
        d_safe = self._geom.path_length(
            (src * (r_safe / max(rn, 1e-9))).reshape(1, 3))[0]
        out = np.ones((2, freqs.size))
        for e in range(2):
            v = ears_unit[e]
            cos_t = float(np.clip(np.dot(src, v) / (rn * np.linalg.norm(v)), -1, 1))
            theta = np.arccos(cos_t)
            cur = rigid_sphere_magnitude(
                np.array([theta]), coarse, self.radius, np.array([r_safe]),
                self.speed_of_sound, self.max_order)[0]
            ref = rigid_sphere_magnitude(
                np.array([theta]), coarse, self.radius,
                np.array([max(r_ref, self.min_distance)]),
                self.speed_of_sound, self.max_order)[0]
            ratio = cur / np.maximum(ref, 1e-6)
            if r < r_safe:
                g_now = rn / max(d_now[e], 1e-6)
                g_safe = r_safe / max(d_safe[e], 1e-6)
                ratio = ratio * (g_now / g_safe)
            # Only the ear being approached is driven as a pressure source.
            if self.occlusion_max_db > 0:
                contact = float(np.clip(1.0 - d_now[e] / self.contact_ref_m, 0.0, 1.0))
                if contact > 0:
                    gain = 10.0 ** (self.occlusion_max_db * contact ** self.contact_power / 20.0)
                    shelf = 1.0 / (1.0 + (coarse / self.contact_corner_hz) ** 2)
                    ratio = ratio * (1.0 + (gain - 1.0) * shelf)
            out[e] = np.interp(freqs, coarse, ratio)
        return out

    # ----------------------------------------------------------------- HRIR
    def _key(self, az: float, el: float, r: float):
        # Coarser quantisation raises the cache hit rate while moving.
        return (round(float(az) * 180.0 / np.pi / 2.0),
                round(float(el) * 180.0 / np.pi / 2.0),
                round(min(float(r), 3.0) * 250.0))

    def hrir_pair(self, azimuth: float, elevation: float, distance: float):
        """Return (left HRIR, right HRIR). Measured direction cues, no absolute delay."""
        key = self._key(azimuth, elevation, distance)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        r = max(float(distance), self.min_distance)
        direction = np.asarray(sph_to_cart(azimuth, elevation, 1.0)).reshape(3)
        ir, _ = self._interp_ir(direction)

        if self.near_field:
            gain = self._near_field_gain(azimuth, elevation, r,
                                         self.source_distance, self.ear_positions)
            nfft = 2 * self.hrir_taps
            for e in range(2):
                ir[e] = np.fft.irfft(np.fft.rfft(ir[e], nfft) * gain[e], n=nfft)[:self.hrir_taps]

        result = (ir[0].copy(), ir[1].copy())
        if len(self._cache) < 60000:
            self._cache[key] = result
        return result

    # ----------------------------------------------------------------- delay
    def ear_delays(self, source_head_local: np.ndarray) -> np.ndarray:
        """Arrival delay at each ear in seconds, (D, 2).

        Geometry supplies the common delay and the near-field ITD increment; the
        measured ITD supplies the direction-dependent fine structure.
        """
        s = np.atleast_2d(np.asarray(source_head_local, dtype=float))
        c = self.speed_of_sound
        path = self._geom.path_length(s) / c                      # (D,2)
        common = path.mean(axis=1, keepdims=True)
        itd_model = path[:, 0] - path[:, 1]

        ref = self.source_distance
        r_now = np.linalg.norm(s, axis=1)
        r_scale = np.where(r_now > 0, ref / np.maximum(r_now, 1e-6), 1.0)
        path_ref = self._geom.path_length(s * r_scale[:, None]) / c
        itd_model_ref = path_ref[:, 0] - path_ref[:, 1]

        itd_meas = np.empty(s.shape[0])
        for i in range(s.shape[0]):
            direction = s[i] / max(np.linalg.norm(s[i]), 1e-9)
            _, itd_meas[i] = self._interp_ir(direction)

        delta = itd_model - itd_model_ref
        delta = np.clip(delta, -self.max_distance_scale * 1e-3,
                        self.max_distance_scale * 1e-3)
        itd = np.clip(itd_meas + delta, -1.5e-3, 1.5e-3)
        return np.stack([common[:, 0] + itd / 2.0, common[:, 0] - itd / 2.0], axis=1)

    def ild_itd(self, azimuth_deg: float = 90.0, elevation_deg: float = 0.0,
                distance: float = 0.2, freq_hz: float = 4000.0):
        """Return (ITD in seconds, ILD in dB), ILD being left ear over right ear."""
        az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
        r = max(float(distance), self.min_distance)
        src = np.asarray(sph_to_cart(az, el, r)).reshape(1, 3)
        d = self.ear_delays(src)[0]
        hl, hr = self.hrir_pair(az, el, r)
        nfft = 4096
        k = max(1, int(round(freq_hz / (self.sample_rate / nfft))))
        L = abs(np.fft.rfft(hl, nfft)[k])
        R = abs(np.fft.rfft(hr, nfft)[k])
        return float(d[0] - d[1]), float(20.0 * np.log10(max(L, 1e-12) / max(R, 1e-12)))


def default_path() -> Path:
    """Default location of the KU100 cache inside the repository."""
    return Path(__file__).resolve().parents[2] / "data" / "sofa" / "KU100_SADIE2_48k_256tap.npz"


def load(path=None, **kwargs) -> SofaHead:
    p = Path(path) if path else default_path()
    if not p.exists():
        raise FileNotFoundError(
            f"SOFA cache not found at {p}.\n"
            "Convert a .sofa file with tools/build_sofa_cache.py first, "
            "or download the KU100 data from SADIE II."
        )
    return SofaHead(p, **kwargs)
