"""Binaural rendering engine: re-record a mono signal through a virtual head.

Delay is applied to the signal through a variable fractional delay line rather
than baked into the impulse response, so it varies continuously and Doppler
falls out naturally. The HRIR carries only direction-dependent spectral
colouring, updated per block and crossfaded with a Hann window at 50 percent
overlap. Distance gain follows 1/r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import cart_to_sph, sph_to_cart
from .hrtf import HeadModel


# ---------------------------------------------------------- trajectories

def static_position(azimuth_deg: float, elevation_deg: float, distance: float):
    """Fixed position. Returns f(t) -> (az, el, dist)."""
    az, el, r = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg), float(distance)

    def fn(t):
        t = np.asarray(t, dtype=float)
        return np.full_like(t, az), np.full_like(t, el), np.full_like(t, r)

    return fn


def keyframes(points, smooth: bool = True):
    """Keyframe trajectory: [(t seconds, az degrees, el degrees, distance), ...].

    smooth=True uses monotone cubic interpolation, so velocity is continuous.
    Azimuth is unwrapped, so 170 -> -170 travels around the back.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError("keyframes must be [(t, az_deg, el_deg, dist_m), ...]")
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    ts = pts[:, 0]
    az = np.unwrap(np.deg2rad(pts[:, 1]))
    el = np.deg2rad(pts[:, 2])
    r = np.maximum(pts[:, 3], 1e-3)

    if ts.size == 1:
        return static_position(np.rad2deg(az[0]), np.rad2deg(el[0]), r[0])

    if smooth and ts.size >= 3:
        from scipy.interpolate import PchipInterpolator

        faz, fel, fr = (PchipInterpolator(ts, v) for v in (az, el, r))
    else:
        faz = lambda x: np.interp(x, ts, az)          # noqa: E731
        fel = lambda x: np.interp(x, ts, el)          # noqa: E731
        fr = lambda x: np.interp(x, ts, r)            # noqa: E731

    def fn(t):
        t = np.clip(np.asarray(t, dtype=float), ts[0], ts[-1])
        return faz(t), fel(t), fr(t)

    return fn


def orbit(radius_m: float = 0.18, elevation_deg: float = 0.0,
          turns: float = 1.0, duration: float = 8.0, direction: float = 1.0):
    """Constant-speed orbit. direction=+1 goes front -> right -> back -> left."""
    r, el = float(radius_m), np.deg2rad(elevation_deg)

    def fn(t):
        t = np.asarray(t, dtype=float)
        az = direction * 2.0 * np.pi * turns * np.clip(t, 0.0, duration) / duration
        return az, np.full_like(t, el), np.full_like(t, r)

    return fn


# -------------------------------------------------------- render settings

@dataclass
class RenderConfig:
    """Rendering constants. These are factory settings, not user controls."""

    hop: int = 256               # block hop in samples; 5.3 ms at 48 kHz
    window: int = 512            # block length, must be twice the hop
    reference_distance: float = 0.5   # reference distance for the 1/r gain
    micro_motion_deg: float = 1.5     # automatic handheld drift in degrees
    motion: str = "hover"        # none | hover | lick | scratch
    motion_amount: float = 1.0   # motion scale
    dist_mode: str = "center"    # center = from head centre, ear = from ear canal
    source_radius: float = 0.012  # source radius in metres; 0 = ideal point source
    source_subs: int = 6          # how many sub-sources the source is split into
    mouth_radius: float = 0.02    # piston directivity radius in metres; 0 = none
    normalize_db: float | None = -1.0


# ------------------------------------------------- extended source / directivity

def _frac_shift(x: np.ndarray, samples: float, fs: float) -> np.ndarray:
    """Fractional delay by a frequency-domain phase ramp. Length preserved."""
    n = x.size
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * samples / fs), n=n)


def _piston_directivity(angle: float, freq: np.ndarray, radius: float) -> np.ndarray:
    """Circular piston in a rigid baffle: |2 J1(x)/x| with x = k a sin(theta)."""
    from scipy.special import j1

    if radius <= 0:
        return np.ones_like(freq)
    s = abs(np.sin(angle))
    x = 2.0 * np.pi * np.asarray(freq, dtype=float) * radius * s / 343.0
    out = np.ones_like(x)
    nz = x > 1e-6
    out[nz] = np.abs(2.0 * j1(x[nz]) / x[nz])
    return out


class ExtendedSource:
    """A source with size and directivity rather than a point.

    A real mouth is a few centimetres across and radiates forward. At 2 cm from
    the ear its size and the distance to it are the same order of magnitude, so
    the point source assumption breaks down and each ear receives a different
    part of the source.

    The source is sampled as a set of sub-sources, each with its own direction,
    arrival delay and piston directivity weight.
    """

    def __init__(self, head, radius_m: float = 0.012, n_sub: int = 6,
                 mouth_radius_m: float = 0.02):
        self.head = head
        self.radius = float(radius_m)
        self.n_sub = max(1, int(n_sub))
        self.mouth_radius = float(mouth_radius_m)
        # Once the source is small relative to the distance its angular extent is
        # negligible, so it falls back to a point source. Most of a trajectory
        # sits beyond 30 cm, which skips the extra HRTF work.
        self.min_relative_size = 0.04
        self.sample_rate = head.sample_rate
        self.hrir_taps = head.hrir_taps
        self.min_distance = head.min_distance
        self.speed_of_sound = head.speed_of_sound
        self.ear_positions = head.ear_positions
        self._cache: dict = {}

    # Geometry interface, compatible with HeadModel and SofaHead
    def path_length(self, src):
        return self.head.path_length(src)

    def ear_delays(self, src):
        return self.head.ear_delays(src)

    def _sub_offsets(self, k: int):
        """Direction offset of sub-source k from the source centre, unit vector."""
        if self.n_sub == 1:
            return (0.0, 0.0, 0.0)
        # Golden-angle spiral, spread evenly over the sphere
        i = k + 0.5
        z = 1.0 - 2.0 * i / self.n_sub
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = np.pi * (1.0 + 5.0 ** 0.5) * i
        return (r * np.cos(phi), r * np.sin(phi), z)

    def hrir_pair(self, azimuth, elevation, distance):
        key = (round(azimuth * 180 / np.pi), round(elevation * 180 / np.pi),
               round(min(distance, 3.0) * 500), round(self.radius * 1000),
               self.n_sub, round(self.mouth_radius * 1000))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        fs = float(self.sample_rate)
        r = max(float(distance), self.min_distance)
        if self.mouth_radius <= 0 and (
                self.radius <= 0 or self.radius / r < self.min_relative_size):
            result = self.head.hrir_pair(azimuth, elevation, r)
            self._cache[key] = result
            return result

        center = np.asarray(sph_to_cart(azimuth, elevation, r)).reshape(3)
        center_dir = center / max(np.linalg.norm(center), 1e-9)
        # Two orthogonal directions perpendicular to the line of sight.
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tmp, center_dir)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(center_dir, tmp); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(center_dir, u)

        # Position, direction and arrival delay for every sub-source
        positions, sphs, delays = [], [], []
        for k in range(self.n_sub):
            ox, oy, oz = self._sub_offsets(k)
            pos = center + self.radius * (ox * u + oy * v + oz * center_dir)
            positions.append(pos)
            sph = cart_to_sph(pos)
            sphs.append((float(sph.azimuth), float(sph.elevation),
                         max(float(np.linalg.norm(pos)), self.min_distance)))
            delays.append(self.head.path_length(pos.reshape(1, 3))[0] / self.speed_of_sound)
        delays = np.asarray(delays)
        ref = delays.mean(axis=0)          # mean sub-source arrival per ear

        nfft = 2 * self.hrir_taps
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        freqs[0] = freqs[1]

        out = []
        for e in range(2):
            acc = np.zeros(self.hrir_taps)
            for k in range(self.n_sub):
                hl, hr = self.head.hrir_pair(*sphs[k])
                h = np.array((hl, hr)[e], dtype=float)
                # Arrival delay differences between sub-sources
                dt = (delays[k, e] - ref[e]) * fs
                if abs(dt) > 1e-6:
                    h = _frac_shift(h, dt, fs)
                # Mouth directivity: strong to the front, weak to the sides and rear
                if self.mouth_radius > 0:
                    ear = self.ear_positions[e]
                    to_ear = positions[k] - ear
                    axis = -center_dir
                    cosang = float(np.dot(to_ear / max(np.linalg.norm(to_ear), 1e-9), axis))
                    ang = float(np.arccos(np.clip(cosang, -1.0, 1.0)))
                    dirmag = _piston_directivity(ang, freqs, self.mouth_radius)
                    h = np.fft.irfft(np.fft.rfft(h, nfft) * dirmag, n=nfft)[:self.hrir_taps]
                acc += h
            out.append(acc / self.n_sub)

        result = (out[0], out[1])
        if len(self._cache) < 60000:
            self._cache[key] = result
        return result


# ------------------------------------------------------------ basic operators

def variable_delay(x: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """Variable fractional delay: y[n] = x[n - d[n]], linearly interpolated."""
    n = x.size
    idx = np.arange(n, dtype=float) - np.asarray(delay_samples, dtype=float)
    return np.interp(idx, np.arange(n, dtype=float), x, left=0.0, right=0.0)


#: Motion presets. The tingle people chase in ASMR comes largely from fast, small
#: movement; a static source never produces it however close it sits.
#: (azimuth amplitude deg, elevation amplitude deg, relative distance modulation,
#:  (frequencies Hz), random seed)
MOTION_PRESETS = {
    "none":    None,
    "hover":   (1.0, 0.6, 0.04, (0.13, 0.31, 0.07), 1),      # natural handheld drift
    "lick":    (5.0, 4.0, 0.10, (3.7, 5.3, 0.9), 3),          # centimetres, a few Hz
    "scratch": (9.0, 7.0, 0.16, (7.3, 11.1, 1.7), 5),         # faster and grainier
}


def _osc(t: np.ndarray, rates, seed: int) -> np.ndarray:
    """Sum of incommensurate sines, roughly in [-1, 1]."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=len(rates))
    w = np.array([0.6, 0.3, 0.1])[: len(rates)]
    out = sum(wi * np.sin(2 * np.pi * fi * t + pi)
              for wi, fi, pi in zip(w, rates, phase))
    return out / max(w.sum(), 1e-9)


def _ear_distance_to_center(az, el, d_ear, head, coarse: int = 400):
    """Convert distance-to-near-ear into distance-to-head-centre by bisection.

    Long signals are solved on a coarse subsample and interpolated.
    """
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    d_ear = np.asarray(d_ear, dtype=float)
    flat = az.reshape(-1)
    n = flat.size
    if n == 0:
        return d_ear

    idx = (np.arange(min(n, coarse)) * (n / min(n, coarse))).astype(int)
    idx = np.clip(idx, 0, n - 1)
    a_c, e_c, d_c = np.atleast_1d(az).reshape(-1)[idx], np.atleast_1d(el).reshape(-1)[idx], \
        np.atleast_1d(d_ear).reshape(-1)[idx]

    lo = np.full(idx.shape, head.min_distance)
    hi = np.full(idx.shape, max(4.0, float(np.max(d_c)) * 4.0))
    right_side = np.sin(a_c) >= 0.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        src = np.asarray(sph_to_cart(a_c, e_c, mid)).reshape(-1, 3)
        paths = head.path_length(src)
        near = np.where(right_side, paths[:, 1], paths[:, 0])
        lo = np.where(near < d_c, mid, lo)
        hi = np.where(near < d_c, hi, mid)
    solved = 0.5 * (lo + hi)
    if idx.size == n and np.array_equal(idx, np.arange(n)):
        return solved.reshape(d_ear.shape)
    return np.interp(np.arange(n), idx, solved).reshape(d_ear.shape)


def apply_motion(az, el, dist, t, cfg: RenderConfig, head):
    """Add the preset motion on top of the trajectory. Returns (az, el, dist)."""
    preset = MOTION_PRESETS.get(cfg.motion)
    if preset is None or cfg.motion_amount <= 0:
        return az, el, dist
    amp_az, amp_el, dist_rel, rates, seed = preset
    if cfg.motion == "hover":
        amp_az = cfg.micro_motion_deg
        amp_el = 0.6 * cfg.micro_motion_deg
    amount = cfg.motion_amount
    az = az + np.deg2rad(amp_az * amount) * _osc(t, rates, seed)
    el = el + np.deg2rad(amp_el * amount) * _osc(t, rates, seed + 101)
    dist = dist * (1.0 + dist_rel * amount * _osc(t, rates, seed + 202))
    return az, el, dist


# ------------------------------------------------------------- main renderer

def render(
    signal: np.ndarray,
    position,
    head: HeadModel | None = None,
    config: RenderConfig | None = None,
):
    """Render a mono signal to binaural stereo along a position trajectory.

    signal: (N,) mono. position: f(t) -> (az, el, dist) in radians and metres.
    Returns (2, N + tail).
    """
    head = head or HeadModel()
    cfg = config or RenderConfig()
    if cfg.window != 2 * cfg.hop:
        raise ValueError("window must equal 2 * hop so the Hann windows sum to 1")

    fs = float(head.sample_rate)
    x = np.asarray(signal, dtype=float).reshape(-1)
    n = x.size
    t = np.arange(n, dtype=float) / fs

    az, el, dist = position(t)
    az, el, dist = (np.broadcast_arrays(
        np.asarray(az, dtype=float), np.asarray(el, dtype=float),
        np.asarray(dist, dtype=float)))
    az, el, dist = az.copy(), el.copy(), dist.copy()

    if cfg.dist_mode == "ear":
        dist = _ear_distance_to_center(az, el, dist, head)
    az, el, dist = apply_motion(az, el, dist, t, cfg, head)
    dist = np.maximum(dist, head.min_distance)

    source = head
    if cfg.source_radius > 0 or cfg.mouth_radius > 0:
        source = ExtendedSource(head, cfg.source_radius, cfg.source_subs,
                                 cfg.mouth_radius)

    # Global minimum delay removed so the output starts at zero; ITD unchanged.
    coords = np.asarray(sph_to_cart(az, el, dist)).reshape(n, 3)
    delays = source.path_length(coords) / head.speed_of_sound         # (n, 2)
    delays = delays - delays.min()

    gain = cfg.reference_distance / dist
    per_ear = [
        variable_delay(x, delays[:, e] * fs) * gain for e in range(2)
    ]

    # Block overlap-add, with the HRIR updated per block
    from scipy.signal import fftconvolve

    win, hop, taps = cfg.window, cfg.hop, head.hrir_taps
    seg_len = win + taps - 1
    tail = int(np.ceil(float(delays.max()) * fs)) + taps + win
    out = np.zeros((2, n + tail))
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win) / win))   # periodic Hann

    for start in range(0, n, hop):
        idx = min(start + win // 2, n - 1)
        hl, hr = source.hrir_pair(az[idx], el[idx], dist[idx])
        for e, ir in enumerate((hl, hr)):
            seg = per_ear[e][start: start + win]
            if seg.size < win:
                seg = np.pad(seg, (0, win - seg.size))
            out[e, start: start + seg_len] += fftconvolve(seg * window, ir)

    return out


def render_to_file(input_path, output_path, position, head: HeadModel | None = None,
                   config: RenderConfig | None = None, target_rate: int | None = None):
    """Read a file, render it, write the result. Returns the output path."""
    from . import audio

    head = head or HeadModel()
    if target_rate:
        head.sample_rate = int(target_rate)
    mono, rate = audio.load_mono(input_path, head.sample_rate)
    y = render(mono, position, head, config)
    return audio.save(output_path, y, head.sample_rate,
                      (config or RenderConfig()).normalize_db)


def render_song(
    input_path,
    output_path,
    position,
    head: HeadModel | None = None,
    config: RenderConfig | None = None,
    separate: str = "off",
    vocal_db: float = 0.0,
    accomp_db: float = -6.0,
    center_keep: float = 0.4,
    accomp_bass_lp: float | None = None,
    info: bool = False,
    verbose: bool = True,
):
    """Full render with optional vocal separation.

    separate is always caller-specified; nothing is decided automatically.
      "off"      no separation, whole file rendered binaurally
      "mid-side" mid/side split: vocal close, accompaniment wide
      "demucs"   Demucs separation, install separately

    center_keep: how much centre content the accompaniment keeps, 0 to 1.
    info: print input statistics; nothing is decided from them.
    """
    from . import audio, separate as sep

    head = head or HeadModel()
    cfg = config or RenderConfig()
    fs = head.sample_rate

    stereo, rate = audio.load_stereo(input_path, fs)
    if info:
        print("  " + sep.analyze(stereo, fs).describe())

    # Separation is entirely caller-specified. Nothing is inferred here.
    mode = {"mid_side": "mid-side", "none": "off", "auto": "off"}.get(separate, separate)
    if mode not in ("off", "mid-side", "demucs"):
        raise ValueError(f"unknown separation mode: {separate!r}")
    if mode == "mid-side" and stereo.shape[0] < 2:
        if verbose:
            print("  input is mono, mid/side separation is not possible, using off")
        mode = "off"
    if verbose:
        print(f"  separation: {mode}")

    vocal_signal = stereo.mean(axis=0)
    accompaniment = None
    if mode == "mid-side":
        vocal_signal, accompaniment = sep.split_mid_side(
            stereo, fs, center_keep, accomp_bass_lp)
    elif mode == "demucs":
        vocal_signal, accompaniment = sep.split_demucs(input_path, fs)

    binaural = render(vocal_signal, position, head, cfg)

    if accompaniment is None:
        mixed = binaural * (10.0 ** (vocal_db / 20.0))
    else:
        n = binaural.shape[1]
        acc = np.zeros((2, n))
        m = min(accompaniment.shape[1], n)
        acc[:, :m] = accompaniment[:, :m]
        # The stems are not comparable in level, so they are matched by RMS,
        # which makes accomp_db mean "how far below the vocal this sits".
        v_gain = 10.0 ** (vocal_db / 20.0)
        v_rms = float(np.sqrt(np.mean((binaural * v_gain) ** 2))) or 1e-9
        a_rms = float(np.sqrt(np.mean(acc ** 2))) or 1e-9
        a_gain = (10.0 ** ((accomp_db + vocal_db) / 20.0)) * v_rms / a_rms
        if verbose:
            print(f"  accompaniment RMS matched: was {20*np.log10(a_rms/v_rms):+.1f} dB "
                  f"below the vocal, now compensated")
        mixed = binaural * v_gain + acc * a_gain

    return audio.save(output_path, mixed, fs, cfg.normalize_db)
