"""Physical self-checks for the analytic head model.

    python tests/sanity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asmr_spatial.hrtf import HeadModel, rigid_sphere_magnitude  # noqa: E402
from asmr_spatial.render import static_position, render  # noqa: E402

FS = 48000


def db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-12))


def check_sphere_limits():
    print("-- rigid sphere: limiting behaviour --")
    c, a, r = 343.0, 0.09, 1.0
    freqs = np.array([20.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0])
    m0 = rigid_sphere_magnitude(np.array([0.0]), freqs, a, np.array([r]), c)[0]
    print("  front, dB:", db(m0).round(1))
    # Normalisation check: push the source far away and the low-frequency front
    # response should approach 0 dB, the head being transparent.
    far = rigid_sphere_magnitude(np.array([0.0]), np.array([20.0]), a,
                                 np.array([20.0]), c)[0, 0]
    print(f"  normalisation: 20 Hz at 20 m, front = {db(far):.2f} dB (should tend to 0)")
    assert abs(db(far)) < 0.25, db(far)
    assert np.all(np.diff(db(m0[:5])) > -0.3), "front response should rise with frequency"
    assert 3.0 < db(m0[4]) < 8.0, f"8 kHz bright spot should be about +6 dB, got {db(m0[4]):.2f}"
    side = rigid_sphere_magnitude(np.array([np.pi / 2]), freqs, a, np.array([r]), c)[0]
    back = rigid_sphere_magnitude(np.array([np.pi]), freqs, a, np.array([r]), c)[0]
    print("  side, dB    :", db(side).round(1))
    print("  rear, dB    :", db(back).round(1))
    assert db(back[4]) < db(m0[4]) - 15.0, "shadowed side needs deep attenuation at 8 kHz"
    pure = rigid_sphere_magnitude(np.array([np.pi]), freqs, a, np.array([r]), c,
                                  shadow_db=0.0)[0]
    print(f"  (a pure sphere without absorption gives {db(pure[4]):.1f} dB at 8 kHz; "
          f"the Poisson spot keeps it from going dark)")
    assert np.isfinite(m0).all() and np.isfinite(back).all()


def check_left_right():
    print("-- left/right: interaural difference must grow as the source approaches --")
    head = HeadModel(sample_rate=FS)
    print(f"{'distance':>10} {'ITD(ms)':>9} {'ILD@1k':>8} {'ILD@4k':>8}")
    prev = None
    for dist in (2.0, 1.0, 0.5, 0.3, 0.2, 0.155):
        itd, ild1 = head.ild_itd(90, 0, dist, 1000.0)
        _, ild4 = head.ild_itd(90, 0, dist, 4000.0)
        print(f"{dist*100:>8.1f}cm {abs(itd)*1000:>9.3f} {abs(ild1):>8.1f} {abs(ild4):>8.1f}")
        if prev is not None:
            assert abs(ild4) > prev, "ILD must grow as distance shrinks"
        prev = abs(ild4)
    itd_far = abs(head.ild_itd(90, 0, 20.0)[0]) * 1000
    print(f"  far-field 90 deg ITD = {itd_far:.3f} ms (human range 0.6 to 0.7 ms)")
    assert 0.55 < itd_far < 0.75, itd_far
    itd_back, ild_back = head.ild_itd(180, 0, 0.3)
    print(f"  rear ITD = {itd_back*1000:.3f} ms, ILD = {ild_back:.2f} dB (should be near 0)")
    assert abs(itd_back) < 1e-4 and abs(ild_back) < 1.5


def check_up_down():
    print("-- elevation: the pinna layer must produce an up/down difference --")
    head = HeadModel(sample_rate=FS)
    nfft = 2048
    freqs = np.fft.rfftfreq(nfft, 1.0 / FS)

    def level(head_, el_deg):
        hl, _ = head_.hrir_pair(0.0, np.deg2rad(el_deg), head_.min_distance)
        return 20 * np.log10(np.maximum(np.abs(np.fft.rfft(hl, nfft)), 1e-9))

    mags = {e: level(head, e) for e in (-60.0, 0.0, 60.0)}
    band = (freqs > 5000) & (freqs < 12000)
    up = mags[60.0][band].mean()
    mid = mags[0.0][band].mean()
    down = mags[-60.0][band].mean()
    print(f"  5-12 kHz mean level: up {up:.2f} dB | level {mid:.2f} dB | down {down:.2f} dB")
    assert abs(up - down) > 2.0, "there must be an audible up/down difference"

    off = HeadModel(sample_rate=FS, pinna=False, torso=False)
    raw = level(off, 60.0)[band].mean() - level(off, -60.0)[band].mean()
    print(f"  up/down contrast: with pinna {up-down:+.2f} dB | without {raw:+.2f} dB "
          f"(the residue comes from the ear canal sitting below head centre)")
    assert abs(up - down) > abs(raw) + 1.0, "the pinna layer should widen the elevation difference"

    print("  pinna notch frequency against elevation:")
    dips = []
    for el_deg in (-40.0, 0.0, 40.0):
        el = np.deg2rad(el_deg)
        detail = head._detail_magnitude(el, 1.0, freqs, float(FS))
        sel = (freqs > 2000) & (freqs < 13000)
        f_dip = freqs[sel][np.argmin(detail[sel])]
        dips.append(f_dip)
        print(f"    elevation {el_deg:+5.0f} deg -> notch {f_dip/1000:.2f} kHz")
    assert dips[0] < dips[1] < dips[2], f"notch frequency should rise with elevation: {dips}"


def check_hardware():
    print("-- head geometry and output sanity --")
    head = HeadModel(sample_rate=FS)
    print(f"  ear positions (head coordinates):\n{head.ear_positions}")
    print(f"  closest distance = {head.min_distance*100:.2f} cm")
    for name, kwargs in (("default", {}), ("small", dict(radius=0.08, ear_spacing=0.15)),
                         ("large", dict(radius=0.10, ear_spacing=0.20))):
        h = HeadModel(sample_rate=FS, **kwargs)
        itd = abs(h.ild_itd(90, 0, 20.0)[0]) * 1000
        print(f"  {name}: far-field ITD = {itd:.3f} ms")
        assert 0.4 < itd < 0.9

    x = np.random.default_rng(0).standard_normal(FS) * 0.2
    y = render(x, static_position(-78, -8, 0.155), head)
    assert y.shape[0] == 2 and np.isfinite(y).all()
    e_l, e_r = np.sum(y[0] ** 2), np.sum(y[1] ** 2)
    print(f"  left ear close-up render: left/right energy ratio "
          f"{10*np.log10(e_l/max(e_r,1e-30)):.1f} dB")
    assert e_l > e_r * 4


def main():
    check_sphere_limits()
    check_left_right()
    check_up_down()
    check_hardware()
    print("\nall self-checks passed")


if __name__ == "__main__":
    main()
