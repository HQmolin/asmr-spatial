"""Download and prepare the head HRTF data (SADIE II, Neumann KU100).

    python tools/fetch_data.py [--keep-raw]

Subject D1 of the SADIE II database is a Neumann KU100 dummy head. The data is
Apache-2.0 and may be redistributed with attribution.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_URL = "https://zenodo.org/records/12092466/files/D1_HRIR_SOFA.zip?download=1"
SOFA_NAME = "D1_48K_24bit_256tap_FIR_SOFA.sofa"
CACHE_NAME = "KU100_SADIE2_48k_256tap.npz"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    print(f"  -> {dest}")
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "asmr-spatial"})
    ) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1024**2:6.1f} / {total/1024**2:.1f} MB", end="")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the original .sofa files (otherwise deleted, saves 35 MB)")
    args = ap.parse_args()

    try:
        import h5py  # noqa: F401
    except ImportError:
        sys.exit("h5py is required to parse SOFA files:\n    pip install h5py")

    sys.path.insert(0, str(ROOT / "tools"))
    from build_sofa_cache import build

    cache = ROOT / "data" / "sofa" / CACHE_NAME
    if cache.exists():
        print(f"cache already present, skipping: {cache}")
        return

    raw_dir = ROOT / "data" / "sofa_raw"
    zpath = ROOT / "data" / "D1_HRIR_SOFA.zip"
    _download(ZIP_URL, zpath)

    print("extracting...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw_dir)

    sofa = next(raw_dir.rglob(SOFA_NAME), None)
    if sofa is None:
        sys.exit(f"{SOFA_NAME} not found in the archive")

    build(sofa, cache)
    print("\ndone. The measured head model is now available as --hrtf ku100")
    print("license: SADIE II / Apache-2.0, keep the attribution when redistributing")

    if not args.keep_raw:
        zpath.unlink(missing_ok=True)
        sofa.unlink(missing_ok=True)
        print("removed the raw files (use --keep-raw to keep them)")


if __name__ == "__main__":
    main()
