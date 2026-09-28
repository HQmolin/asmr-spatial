# asmr-spatial

Place any audio at an arbitrary position around a virtual head and re-record it
through a dummy-head HRTF. Output is binaural stereo for headphones.

## What it does

- Puts a sound source anywhere around a virtual head: azimuth, elevation, and
  distance from 2 mm out to 3 m.
- Re-records it through a measured Neumann KU100 dummy-head HRTF, or through an
  analytic head model that needs no data.
- Moves the source along a path, or holds it still.
- Optionally keeps a vocal close while the accompaniment stays wide.
- No room, no reverb. Position is the only effect parameter.

## What it's for

Headphone content that needs a source right at the ear — singing into one ear,
whispering at the back of the head, circling the listener — without booking a
dummy-head recording session.

## Install

```bash
git clone https://github.com/HQmolin/asmr-spatial.git
cd asmr-spatial
python -m venv .venv && .venv/Scripts/python -m pip install -e .
```

On macOS or Linux use `source .venv/bin/activate && pip install -e .`.

## Usage

```bash
# Analytic head model, no data needed
asmr-spatial in.mp3 out.wav --az 90 --el -6 --dist 0.15

# Measured KU100 head model (downloads 36 MB once)
python tools/fetch_data.py
asmr-spatial in.mp3 out.wav --hrtf ku100 --az 90 --el 0 --dist 0.002 --dist-mode ear

# Vocal close to the right ear, accompaniment kept wide
asmr-spatial song.mp3 out.wav --hrtf ku100 --az 90 --el -6 --dist 0.002 \
    --dist-mode ear --separate mid-side --accomp-db -6

# Orbit the head
asmr-spatial in.mp3 out.wav --hrtf ku100 --orbit 0.18

# Path through keyframes: t, azimuth, elevation, distance
asmr-spatial in.mp3 out.wav --hrtf ku100 \
    --keyframes "0,0,0,0.5; 2,90,-10,0.16; 4,270,-10,0.16; 6,0,0,0.5"
```

`--demo OUTDIR` renders a set of comparison samples instead.

### Position

| Option | Default | |
|---|---|---|
| `--az` | 0 | azimuth, degrees. 0 = front, +90 = right ear |
| `--el` | 0 | elevation, degrees, positive up |
| `--dist` | 0.3 | distance, metres |
| `--dist-mode` | `center` | `center` = to the head centre, `ear` = to the near ear canal |
| `--keyframes` | | `"t,az,el,dist; ..."` path, monotone cubic interpolation |
| `--orbit R` | | orbit the head at radius R, metres |
| `--turns` / `--duration` | 1.0 / 8.0 | orbit length |

### Motion

| Option | Default | |
|---|---|---|
| `--motion` | `hover` | `none`, `hover`, `lick`, `scratch` |
| `--motion-amount` | 1.0 | scale the motion |
| `--micro` | 1.5 | `hover` amplitude in degrees, 0 = off |

### Head model

| Option | Default | |
|---|---|---|
| `--hrtf` | `analytic` | `analytic` (no data), `ku100` (measured), or a `.npz` path |
| `--head-radius` | 0.09 | analytic head radius, metres |
| `--ear-spacing` | 0.18 | analytic ear spacing, metres |
| `--no-pinna` | | disable the pinna layer; elevation cues are lost |

### Source

| Option | Default | |
|---|---|---|
| `--source-radius` | 0.012 | source radius, metres. 0 = ideal point source |
| `--source-subs` | 4 | sub-sources used to sample the source |
| `--mouth-radius` | 0 | piston directivity radius, metres. 0 = off |
| `--occlusion-db` | 15 | contact coupling: max low-frequency boost in dB, inside 3 cm. 0 = off |

### Vocal separation

Manual only. The tool never decides this for you.

| Option | Default | |
|---|---|---|
| `--separate` | `off` | `off`, `mid-side`, `demucs` |
| `--vocal-db` | 0 | vocal track gain, dB |
| `--accomp-db` | −6 | accompaniment level relative to the vocal, dB |
| `--center-keep` | 0.4 | how much centre content the accompaniment keeps, 0–1 |
| `--accomp-bass-lp` | 0 | if > 0, keep only centre content below this frequency, Hz |

`demucs` is not bundled. Install it separately for clean separation:

```bash
pip install demucs
```

### Output

| Option | Default | |
|---|---|---|
| `--rate` | 48000 | internal sample rate |
| `--no-normalize` | | skip peak normalisation |
| `--quiet` | | suppress progress output |
| `--info` | | print input statistics; informational only |

### Coordinates

Right-handed. Azimuth 0° is front, +90° is the right ear. Elevation is positive
up. Distances are metres.

## Data

Measured HRTFs come from the **SADIE II Database**, subject D1 — a Neumann KU100
dummy head.

- Database — https://www.york.ac.uk/sadie-project/database.html
- Download record — https://zenodo.org/records/12092466
- License — Apache-2.0

Fetch them with `python tools/fetch_data.py`. The analytic head model follows
[Duda & Martens (1998)](https://doi.org/10.1121/1.423886).

## License

[Apache-2.0](LICENSE)
