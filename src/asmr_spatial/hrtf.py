"""Minimal analytic dummy-head model. No HRTF database required.

Handles left/right, distance and elevation. It cannot discriminate front from
back, because a rigid sphere gives identical output for mirrored directions;
that needs a measured HRIR, see sofa.py.

Exact rigid-sphere scattering after Duda & Martens (1998), with a creeping-wave
ITD and a parametric pinna / ear canal / torso layer for elevation.

Coordinates are head-local: azimuth, elevation and distance from the head
centre. Azimuth 0 is ahead, +90 degrees is the right ear, elevation is positive
upwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

from .geometry import angle_between, sph_to_cart

SPEED_OF_SOUND = 343.0     # m/s at 20 C


# ------------------------------------------------------------- rigid sphere

def rigid_sphere_magnitude(
    theta: np.ndarray,
    freqs: np.ndarray,
    radius: float,
    distance: np.ndarray,
    speed_of_sound: float = SPEED_OF_SOUND,
    max_order: int = 40,
    shadow_db: float = 12.0,
) -> np.ndarray:
    """Rigid-sphere magnitude response, shape (D, F).

    theta     : (D,) angle between the source and the ear, radians
    freqs     : (F,) frequencies in Hz
    distance  : (D,) source distance from the head centre, metres
    shadow_db : empirical extra attenuation past 90 degrees, since a real head
                absorbs and a perfect sphere does not. 0 gives a pure sphere.

    Normalised to the free field at the same distance: |H| = 1 means no head.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    dist = np.atleast_1d(np.asarray(distance, dtype=float))
    if dist.size == 1 and theta.size > 1:
        dist = np.full(theta.shape, float(dist[0]))

    cos_theta = np.cos(theta)

    # Floor k: high-order spherical Hankel functions overflow as ka goes small.
    # The floor must not break kr/ka = r/a, which sets the near-field low end.
    k = 2.0 * np.pi * freqs / speed_of_sound
    k = np.maximum(k, 0.05 / radius)

    ka = k * radius                       # (F,)
    kr = np.outer(dist, k)                # (D, F)

    # Order must cover ka, and let (a/r)^m decay, which needs more terms up close.
    x = float(np.clip(radius / max(float(dist.min()), radius * 1.0001), 0.0, 0.9999))
    need = int(np.ceil(-6.0 / np.log10(x))) if 0.0 < x < 1.0 else max_order
    order = int(np.clip(max(np.ceil(ka.max()) + 6, need), 8, max_order))

    total = np.zeros((theta.size, freqs.size), dtype=complex)
    for m in range(order + 1):
        dhm = spherical_jn(m, ka, derivative=True) + 1j * spherical_yn(
            m, ka, derivative=True
        )
        a_m = 1j / (ka ** 2 * dhm)                                    # (F,)
        hm = spherical_jn(m, kr) + 1j * spherical_yn(m, kr)           # (D, F)
        pm = eval_legendre(m, cos_theta)                              # (D,)
        total += (2 * m + 1) * pm[:, None] * a_m[None, :] * hm

    total *= kr                       # normalise to free field (pressure ~ 1/(k r))
    total = np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.abs(total)

    if shadow_db > 0.0:
        # Weight angles past 90 degrees, with a frequency weight since low
        # frequencies diffract around the head.
        excess = np.clip((theta - np.pi / 2.0) / (np.pi / 2.0), 0.0, 1.0)
        f_weight = np.clip((freqs - 500.0) / 3500.0, 0.0, 1.0)
        atten_db = shadow_db * excess[:, None] * f_weight[None, :]
        mag = mag * (10.0 ** (-atten_db / 20.0))
    return mag


# ------------------------------------------------------- minimum phase IR

def minimum_phase_from_magnitude(magnitude: np.ndarray, n_fft: int) -> np.ndarray:
    """Build a minimum-phase impulse response from a one-sided magnitude spectrum.

    Uses the homomorphic method. Returns a causal real sequence of length n_fft.
    """
    mag = np.maximum(np.asarray(magnitude, dtype=float), 1e-8)
    log_mag = np.log(mag)
    full = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cepstrum = np.fft.ifft(full).real
    fold = np.zeros(n_fft)
    fold[0] = 1.0
    fold[1: n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    return np.fft.ifft(np.exp(np.fft.fft(cepstrum * fold))).real


# ------------------------------------------------- pinna / canal / torso

def _peaking(f0: float, q: float, gain_db: float, fs: float):
    """RBJ peaking filter coefficients. A negative gain_db gives a notch."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cw, 1 - alpha * a])
    d = np.array([1 + alpha / a, -2 * cw, 1 - alpha / a])
    return b / d[0], d / d[0]


def _response(b, a, freqs, fs):
    z = np.exp(-2j * np.pi * freqs / fs)
    return (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)


# --------------------------------------------------------------- head model

@dataclass
class HeadModel:
    """Parametric virtual head.

    radius       effective head radius in metres, 0.085 to 0.095 for an adult
    ear_spacing  distance between the ears in metres, 0.15 to 0.18 for an adult
    ear_offset   ear canal offset from "head centre plus lateral axis", as
                 (front, up) in metres
    min_distance_factor
                 closest allowed distance = radius * this factor
    pinna/torso  enable the perceptual detail layers, which carry elevation cues
    hrir_taps    HRIR length in taps; 256 at 48 kHz is about 5.3 ms
    """

    sample_rate: int = 48000
    radius: float = 0.09
    ear_spacing: float = 0.18
    ear_offset: tuple[float, float] = (-0.005, -0.008)
    speed_of_sound: float = SPEED_OF_SOUND
    # Closest source distance from the head centre, as a multiple of the radius.
    # 1.2 puts the head centre at 10.8 cm, about 1.8 cm from the pinna, which is
    # the scale real ASMR uses. Below 1.15 the sphere series is under-truncated
    # and goes numerically unstable.
    min_distance_factor: float = 1.2
    pinna: bool = True
    torso: bool = True
    hrir_taps: int = 256
    hrir_fft: int = 512
    max_order: int = 40
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------- geometry
    @property
    def min_distance(self) -> float:
        return self.radius * self.min_distance_factor

    @property
    def ear_positions(self) -> np.ndarray:
        """Ear positions in head coordinates, (2,3). Row 0 left, row 1 right."""
        half = self.ear_spacing / 2.0
        fy = self.ear_offset[0] if len(self.ear_offset) > 0 else 0.0
        fz = self.ear_offset[1] if len(self.ear_offset) > 1 else 0.0
        return np.array([[-half, fy, fz], [+half, fy, fz]])

    def path_length(self, source_head_local: np.ndarray) -> np.ndarray:
        """Diffraction path length from the source to each ear, (D, 2) metres."""
        s = np.atleast_2d(np.asarray(source_head_local, dtype=float))
        r = np.maximum(np.linalg.norm(s, axis=1), self.radius * 1.001)
        alpha = np.arccos(np.clip(self.radius / r, -1.0, 1.0))
        sqrt_term = np.sqrt(np.maximum(r ** 2 - self.radius ** 2, 0.0))
        out = np.empty((s.shape[0], 2))
        for e in range(2):
            ear = self.ear_positions[e]
            psi = angle_between(s, ear)
            direct = np.linalg.norm(s - ear, axis=1)
            creep = sqrt_term + self.radius * np.maximum(psi - alpha, 0.0)
            out[:, e] = np.where(psi <= alpha, direct, creep)
        return out

    def ear_delays(self, source_head_local: np.ndarray) -> np.ndarray:
        """Arrival delay at each ear in seconds, (D, 2)."""
        return self.path_length(source_head_local) / self.speed_of_sound

    # ----------------------------------------------------------- detail layer
    def _detail_magnitude(self, elevation: float, ipsi: float, freqs, fs) -> np.ndarray:
        """Pinna notches that move with elevation plus ear canal resonances."""
        if not self.pinna or ipsi <= 1e-3:
            return np.ones_like(freqs, dtype=float)
        resp = np.ones_like(freqs, dtype=complex)
        el = float(np.rad2deg(np.clip(elevation, -60.0, 70.0)))
        # Pinna notch frequency rises with elevation.
        f1 = 5500.0 + 3500.0 * np.sin(np.deg2rad(el))
        f2 = 10000.0 + 3500.0 * np.sin(np.deg2rad(el))
        for f_n, depth, q in ((f1, -10.0, 5.0), (f2, -8.0, 8.0)):
            if 0.0 < f_n < 0.45 * fs:
                b, a = _peaking(f_n, q, depth * ipsi, fs)
                resp *= _response(b, a, freqs, fs)
        b, a = _peaking(min(0.42 * fs, 11500.0), 1.2, 2.5 * ipsi, fs)
        resp *= _response(b, a, freqs, fs)
        b, a = _peaking(3000.0, 1.8, 5.5 * ipsi, fs)
        resp *= _response(b, a, freqs, fs)
        b, a = _peaking(9500.0, 2.4, 7.0 * ipsi, fs)
        resp *= _response(b, a, freqs, fs)
        return np.abs(resp)

    def _torso_taps(self, elevation: float, ipsi: float):
        """Torso reflection: (delay in taps, gain, low-pass corner in Hz)."""
        if not self.torso:
            return 0, 0.0, 0.0
        w = float(np.clip(-np.sin(elevation), 0.0, 1.0))
        strength = ipsi * (0.25 + 0.75 * w)
        if strength <= 1e-3:
            return 0, 0.0, 0.0
        d_smp = int(round(2.0 * (0.14 + 0.06 * w) / self.speed_of_sound * self.sample_rate))
        return d_smp, 0.5 * strength, 3500.0

    # ----------------------------------------------------------------- HRIR
    def _key(self, az: float, el: float, r: float):
        return (
            round(float(az) * 180.0 / np.pi / 2.0),
            round(float(el) * 180.0 / np.pi / 2.0),
            round(min(float(r), 3.0) * 500.0),
        )

    def hrir_pair(self, azimuth: float, elevation: float, distance: float):
        """Return (left HRIR, right HRIR). Minimum phase, no absolute delay."""
        key = self._key(azimuth, elevation, distance)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        fs = float(self.sample_rate)
        nfft = self.hrir_fft
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        freqs[0] = freqs[1]

        r = max(float(distance), self.min_distance)
        source = np.asarray(sph_to_cart(azimuth, elevation, r)).reshape(3)
        r_norm = float(np.linalg.norm(source))
        ears = self.ear_positions

        out = []
        for e in range(2):
            v = ears[e]
            cos_t = float(np.clip(np.dot(source, v) / (r_norm * np.linalg.norm(v)), -1, 1))
            # Use the angle directly: the ipsilateral ear sees a small angle and
            # the contralateral ear a large one, which is what separates the two
            # channels. Mirrored front/back directions already give the same
            # angle for a given ear, so nothing else is needed for the cone of
            # confusion.
            theta = np.arccos(cos_t)
            mag = rigid_sphere_magnitude(
                np.array([theta]), freqs, self.radius, np.array([r]),
                self.speed_of_sound, self.max_order,
            )[0]
            ipsi = float(np.clip(0.5 + 0.5 * cos_t, 0.0, 1.0))
            mag = mag * self._detail_magnitude(elevation, ipsi, freqs, fs)
            mag[0] = mag[1]

            ir = minimum_phase_from_magnitude(mag, nfft)
            taps = np.zeros(self.hrir_taps)
            n = min(self.hrir_taps, ir.size)
            taps[:n] = ir[:n]
            fade = max(16, self.hrir_taps // 8)
            taps[-fade:] *= np.linspace(1.0, 0.0, fade) ** 2

            d_smp, g, fc = self._torso_taps(elevation, ipsi)
            if d_smp and (d_smp + 8) < self.hrir_taps:
                t = np.arange(32) / fs
                lp = np.exp(-2.0 * np.pi * fc * t)
                lp = lp - np.concatenate([[0.0], lp[:-1]])
                seg = min(lp.size, self.hrir_taps - d_smp)
                taps[d_smp: d_smp + seg] += g * lp[:seg]
            out.append(taps)

        result = (out[0], out[1])
        if len(self._cache) < 50000:
            self._cache[key] = result
        return result

    # ----------------------------------------------------------- diagnostics
    def ild_itd(self, azimuth_deg: float = 90.0, elevation_deg: float = 0.0,
                distance: float = 0.2, freq_hz: float = 4000.0):
        """Return (ITD in seconds, ILD in dB). ILD is left ear over right ear."""
        az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
        r = max(float(distance), self.min_distance)
        src = np.asarray(sph_to_cart(az, el, r)).reshape(1, 3)
        d = self.path_length(src)[0] / self.speed_of_sound
        ears = self.ear_positions
        mags = []
        for e in range(2):
            v = ears[e]
            ct = float(np.clip(np.dot(src[0], v) / (r * np.linalg.norm(v)), -1, 1))
            mags.append(rigid_sphere_magnitude(
                np.array([np.arccos(ct)]), np.array([freq_hz]),
                self.radius, np.array([r]), self.speed_of_sound, self.max_order,
            )[0, 0])
        ild = 20.0 * np.log10(max(mags[0], 1e-12) / max(mags[1], 1e-12))
        return float(d[0] - d[1]), float(ild)
