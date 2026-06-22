# htdemucs Separation + Analysis — Local Test Harness

Validates the first two pipeline steps locally on Apple Silicon (MPS):

1. **Stem separation** with `htdemucs` (4-stem).
2. **Musical analysis** with `librosa` → `analysis.json` (maps to
   `backend-api-design.md` §2.2).

This is a **validation harness, not production**. Production moves separation to
serverless GPU later.

## Setup

Requires Python **3.11** (3.12+ has lagging audio deps) and `ffmpeg`.

```bash
brew install ffmpeg            # required for mp3/m4a decoding

# This repo's setup uses pyenv's 3.11:
pyenv install -s 3.11.12
~/.pyenv/versions/3.11.12/bin/python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm MPS is visible:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # -> True
```

## Run

```bash
python run.py path/to/song.mp3
python run.py path/to/song.mp3 --device cpu          # force CPU
python run.py path/to/song.mp3 --model htdemucs_6s   # 6-stem (opt-in)
```

Outputs land in `output/<song>/`:

- `stems/{vocals,drums,bass,other}.wav` — separated stems.
- `analysis.json` — tempo, key, beat grid, bars, sections, per-stem loudness,
  a 16-step/bar drum pattern, and **repetition analysis** (see below).

## Repetition analysis

`detect_repetitions()` (in `analyze.py`) clusters every bar into reusable
"parts" from a bar-synchronous fingerprint (harmony + timbre + per-stem
arrangement). It works from the stems alone — no source mix needed. Each bar
gets `loop_label`, `chord`, `active_stems`, and `drum_density`; a top-level
`loops[]` summarizes each part (count, instances/runs, active stems, dominant
chord, drum density, representative bar). This is the seed of the `template.json`
idea: capture each unique part once + an arrangement of references.

### Per-layer patterns

`detect_layer_patterns()` finds repeated patterns **within each stem
independently** — so you can ask "where does *this drum groove* recur?"
regardless of what the other stems do. Features per layer:

- **drums** — the 16-step onset grid per bar, grouped by Hamming distance.
- **bass / vocals / other** — sub-bar chroma + MFCC (z-scored), grouped by
  Euclidean distance (bar-averaged chroma alone is too key-dominated).

Output is a top-level `layers[]` (`{stem: [{label, occurrences[], bar_count,
occurrence_count, representative_bar, repeated}]}`) plus per-bar `layer_ids`
(`{stem: pattern_label}`). Consecutive matching bars are merged into
occurrence runs; one-off (single, non-repeating) bars are dropped.

## App (server + visualizer)

`server.py` is a tiny backend (stdlib only) that serves the visualizer **and**
runs the pipeline from the browser. Start it with the venv python:

```bash
./.venv/bin/python server.py      # http://localhost:8753/visualizer.html
```

From the UI you can:
- **⬆ Upload song** (or drag an audio file in) → separates + analyzes on the
  server, then loads the result. Progress shows in the status line.
- **Songs dropdown** → switch between processed songs in `output/`.
- **Sensitivity toggle** (Low / Med / High) → re-detects per-layer patterns at a
  different cutoff. Higher = more, stricter patterns. This is a fast
  layers-only re-detect (~2–3 s); loops/sections are unchanged.

Endpoints: `GET /api/songs`, `POST /api/upload?name=&sensitivity=` (raw audio
body) → `{job}`, `GET /api/status?job=`, `POST /api/enrich {song, sensitivity}`.

Sensitivity presets live in `analyze.py` (`SENSITIVITY`); the underlying knobs
are `_DRUM_PATTERN_DIST` (Hamming on the step grid) and `_MELODIC_DIST`
(Euclidean on z-scored sub-bar chroma+MFCC).

### Plain static serving (no backend)

If you only want to view existing `analysis.json` files without upload/toggle:

```bash
python -m http.server 8753       # from test-harness/
open http://localhost:8753/visualizer.html
```

It shows: a loop-colored section ribbon, a bar/beat ruler, a per-bar Chords
lane, per-stem waveform lanes with mute/solo/volume (bars are dimmed where a
stem is silent, so you *see* the arrangement), a drum step-grid lane, a synced
playhead (space to play, click to seek), and a clickable Loops panel. It needs
to be served (not `file://`) so the stems can load; it also accepts a
drag-and-dropped `analysis.json` (viz-only without served stems).

**Per-layer patterns:** each stem lane has a colored band at the top, one block
per pattern occurrence (from `layers[]`). Click a block to solo that stem,
highlight every occurrence, and play through them in sequence (status shows
`touring N/total`). Esc, Stop, or a manual seek clears the tour.

## Modules

| File | What it does |
|---|---|
| `separate.py` | Wraps the demucs CLI; MPS→CPU fallback on error or garbled (NaN/empty) output; normalizes layout into `output/<song>/stems/`. |
| `analyze.py` | Full-mix global properties + drum-stem step pattern → analysis dict. |
| `run.py` | Ties it together and writes `analysis.json` + prints a one-line summary. |

Each module is runnable standalone (e.g. `python separate.py song.mp3`,
`python analyze.py song.mp3 --drums output/song/stems/drums.wav`).

## Notes / known pitfalls

- **MPS garbled output** is intermittent on some torch builds; the harness
  retries on CPU automatically, or pass `--device cpu`.
- **mp3/m4a fail to load** → missing `ffmpeg` (the #1 cause).
- **librosa half/double tempo**: beat trackers sometimes lock to 0.5× / 2× the
  true tempo. Sanity-check `tempo_bpm` against a known value.
- **Sample rate**: demucs outputs 44.1 kHz WAV; analysis runs at native rate
  (no silent resampling).
- `madmom` (better beat tracking) is skipped for v1 — finicky on Apple Silicon.
- Drum kick/snare/hat split (band-pass) is a stretch; v1 ships onsets + grid.
