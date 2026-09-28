"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .hrtf import HeadModel
from .render import (
    RenderConfig, keyframes, orbit, render_song, static_position,
)


def _build_head(args):
    name = (args.hrtf or "analytic").strip()
    if name == "analytic":
        return HeadModel(
            sample_rate=args.rate,
            radius=args.head_radius,
            ear_spacing=args.ear_spacing,
            pinna=not args.no_pinna,
        )

    from . import sofa as sofa_mod

    path = None if name in ("ku100", "sofa", "default") else name
    head = sofa_mod.load(path, sample_rate=args.rate,
                         occlusion_max_db=args.occlusion_db)
    print(f"  {head.describe()}")
    return head


def _parse_keyframes(text: str):
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace(":", ",").split(",") if p.strip()]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                f"a keyframe needs 4 numbers (t, az, el, dist), got: {chunk!r}")
        pts.append(tuple(float(p) for p in parts))
    if not pts:
        raise argparse.ArgumentTypeError("no keyframes given")
    return pts


def _build_parser():
    p = argparse.ArgumentParser(
        prog="asmr_spatial",
        description="Place audio around a virtual head and re-record it binaurally.",
    )
    p.add_argument("input", nargs="?", help="input audio (wav/mp3/flac/ogg)")
    p.add_argument("output", nargs="?", help="output audio")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--keyframes", type=str,
                   help='path as "t,az,el,dist; ..." (degrees, metres)')
    g.add_argument("--orbit", type=float, metavar="R",
                   help="orbit the head at radius R, metres")
    g.add_argument("--demo", type=str, metavar="OUTDIR",
                   help="render a set of comparison samples into OUTDIR")

    p.add_argument("--az", type=float, default=0.0,
                   help="azimuth in degrees, 0 = front, +90 = right ear")
    p.add_argument("--el", type=float, default=0.0, help="elevation in degrees, + up")
    p.add_argument("--dist", type=float, default=0.3, help="distance in metres")
    p.add_argument("--duration", type=float, default=8.0, help="orbit duration, seconds")
    p.add_argument("--turns", type=float, default=1.0, help="number of orbit turns")
    p.add_argument("--rate", type=int, default=48000, help="internal sample rate")
    p.add_argument("--micro", type=float, default=1.5,
                   help="hover amplitude in degrees, 0 = off")
    p.add_argument("--motion", default="hover",
                   choices=["none", "hover", "lick", "scratch"],
                   help="motion preset: none, hover (handheld drift), lick, scratch")
    p.add_argument("--motion-amount", type=float, default=1.0, help="motion scale")
    p.add_argument("--dist-mode", default="center", choices=["center", "ear"],
                   help="distance reference: head centre (default) or near ear canal")
    p.add_argument("--source-radius", type=float, default=0.012,
                   help="source radius in metres, 0 = ideal point source")
    p.add_argument("--source-subs", type=int, default=4,
                   help="number of sub-sources the source is split into")
    p.add_argument("--mouth-radius", type=float, default=0.0,
                   help="piston directivity radius in metres, 0 = off")
    p.add_argument("--occlusion-db", type=float, default=15.0,
                   help="contact coupling: max low-frequency boost in dB inside 3 cm, "
                        "0 = off")
    p.add_argument("--no-normalize", action="store_true", help="skip peak normalisation")
    p.add_argument("--head-radius", type=float, default=0.09,
                   help="analytic head radius in metres")
    p.add_argument("--ear-spacing", type=float, default=0.18,
                   help="analytic ear spacing in metres")
    p.add_argument("--no-pinna", action="store_true",
                   help="disable the pinna layer, which removes elevation cues")
    p.add_argument("--hrtf", default="analytic",
                   help='head model: "analytic" (no data), "ku100" (measured), '
                        "or a path to an .npz cache")

    p.add_argument("--separate", default="off",
                   choices=["off", "mid-side", "demucs"],
                   help="vocal separation: off, mid-side (built in), demucs (external)")
    p.add_argument("--vocal-db", type=float, default=0.0, help="vocal gain in dB")
    p.add_argument("--accomp-db", type=float, default=-6.0,
                   help="accompaniment level relative to the vocal, in dB")
    p.add_argument("--center-keep", type=float, default=0.4,
                   help="how much centre content the accompaniment keeps, 0 to 1")
    p.add_argument("--accomp-bass-lp", type=float, default=0.0,
                   help="if > 0, keep only centre content below this frequency in Hz")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    p.add_argument("--info", action="store_true",
                   help="print input statistics, informational only")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    head = _build_head(args)
    cfg = RenderConfig(
        micro_motion_deg=args.micro,
        motion=args.motion,
        motion_amount=args.motion_amount,
        dist_mode=args.dist_mode,
        source_radius=args.source_radius,
        source_subs=args.source_subs,
        mouth_radius=args.mouth_radius,
        normalize_db=None if args.no_normalize else -1.0,
    )

    if args.demo:
        from .demo import build_demo

        build_demo(args.demo, head, cfg, input_path=args.input)
        return 0

    if not args.input or not args.output:
        _build_parser().print_help()
        print("\nerror: input and output are required, or use --demo OUTDIR")
        return 2

    if args.keyframes:
        pos = keyframes(_parse_keyframes(args.keyframes))
    elif args.orbit is not None:
        pos = orbit(args.orbit, elevation_deg=args.el, turns=args.turns,
                    duration=args.duration)
    else:
        pos = static_position(args.az, args.el, args.dist)

    out = render_song(
        args.input, args.output, pos, head, cfg,
        separate=args.separate,
        vocal_db=args.vocal_db,
        accomp_db=args.accomp_db,
        center_keep=args.center_keep,
        accomp_bass_lp=(args.accomp_bass_lp or None),
        info=args.info,
        verbose=not args.quiet,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
