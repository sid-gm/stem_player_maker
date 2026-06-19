# Test Plan: htdemucs Separation + Analysis (Local, Mac M4 Pro)

**Handoff doc for Claude Code.** Goal: stand up a local test harness that validates the first two steps of the pipeline — (1) stem separation with htdemucs and (2) musical analysis — and emits an `analysis.json` matching the schema in `backend-api-design.md` §2.2. This is a **validation harness, not production**; production moves separation to serverless GPU later.

Target machine: Mac mini M4 Pro, 48 GB RAM, Apple Silicon (MPS available).

---

## Acceptance criteria (what "done" looks like)

1. Run `python run.py path/to/song.mp3` and get back, in `output/<song>/`:
   - `stems/{vocals,drums,bass,other}.wav` — four separated stems.
   - `analysis.json` — populated per the schema below.
2. Stems are audibly correct (vocals isolated, drums isolated, etc.) — manual listen check.
3. `analysis.json` has sane values: tempo within ~±2 BPM of the real track, a beat grid spanning the full duration, and at least intro/verse/chorus-level sections.
4. Whole run completes in a few minutes per track on this machine.

---

## Step 1 — Environment

Use Python **3.11** (not 3.12+; some audio deps lag). Use a venv or `uv`.

```bash
# ffmpeg is required for mp3/m4a decoding
brew install ffmpeg

python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# core
pip install demucs soundfile numpy librosa
# torch with MPS ships in the default macOS wheel; demucs pulls it in.
```

Notes:
- **madmom** (better beat tracking) is finicky to install on Apple Silicon (needs Cython + older numpy). Skip it for v1 — use librosa's beat tracker. Add madmom later only if beat accuracy is insufficient.
- Confirm MPS is visible: `python -c "import torch; print(torch.backends.mps.is_available())"` should print `True`.

---

## Step 2 — Separation module (`separate.py`)

Wrap the demucs CLI (simplest reliable path):

```bash
demucs -n htdemucs -d mps -o output/_raw "path/to/song.mp3"
# outputs: output/_raw/htdemucs/<song>/{vocals,drums,bass,other}.wav
```

Requirements:
- Device selection: try `mps`, fall back to `cpu` on error or if MPS produces NaN/garbled output (known intermittent MPS issue). Expose `--device` override.
- Default model `htdemucs` (4-stem). Allow `htdemucs_6s` as an opt-in flag for later.
- Move/rename the demucs output into the clean `output/<song>/stems/` layout.
- Return the four stem paths + the source duration.

---

## Step 3 — Analysis module (`analyze.py`)

Run on the **full mix** for global properties, and on the **drum stem** for the step pattern. Produce the JSON below. Map directly to `backend-api-design.md` §2.2 — same field names so the harness output drops into the real schema later.

Populate these (use librosa unless noted):

| Field | How | Notes |
|---|---|---|
| `duration_sec` | `librosa.get_duration` | |
| `tempo_bpm` + `tempo_confidence` | `librosa.beat.beat_track` | confidence: derive from onset autocorrelation strength, or hardcode 1.0 for v1 |
| `time_signature` | assume `"4/4"` for v1 | real meter detection is a stretch goal |
| `key` + `key_confidence` | chroma (`librosa.feature.chroma_cqt`) → Krumhansl-Schmuckler key profile | standard approach; many gists exist |
| `beat_grid[]` | beat frames from `beat_track` → times; mark every Nth as `is_downbeat` (assume 4/4) | this is the timing source of truth |
| `bars[]` | group beats into bars of 4 | |
| `sections[]` | `librosa.segment` (self-similarity / agglomerative) → label heuristically | v1: unlabeled or simple labels OK; refine later |
| `stems[]` | one entry per stem: type, path, `loudness_lufs` (use `pyloudnorm` or RMS proxy) | |
| `patterns.drums` | onset detection on the drum stem, quantized to a 16-step/bar grid | **stretch** — kick/snare/hat split needs band-pass filtering; ship onsets first |

Output skeleton:

```jsonc
{
  "song_id": "local_test",
  "engine_version": "htdemucs-4.0+librosa",
  "duration_sec": 0,
  "tempo_bpm": 0, "tempo_confidence": 0,
  "time_signature": "4/4",
  "key": "", "key_confidence": 0,
  "beat_grid": [], "bars": [], "sections": [],
  "stems": [], "patterns": {}
}
```

---

## Step 4 — Runner (`run.py`)

Ties it together: `run.py <audio_file> [--device mps|cpu] [--model htdemucs|htdemucs_6s]`
1. Separate → stems.
2. Analyze full mix + drum stem → dict.
3. Write `output/<song>/analysis.json`.
4. Print a one-line summary: tempo, key, #beats, #sections, stem paths.

---

## Step 5 — Verify

- Listen to each stem (bleed is expected and fine — reference clips are guides).
- Check `analysis.json` tempo against a known value (e.g. cross-check one track on a BPM site).
- Confirm `beat_grid` last timestamp ≈ `duration_sec` (grid covers the whole track).
- Try 2–3 genres (a dense pop mix, a sparse acoustic track, a heavy hip-hop track) — separation and beat tracking degrade differently across these.

---

## Suggested structure

```
capcut_for_music/
  test-harness/
    separate.py
    analyze.py
    run.py
    requirements.txt
    README.md
    output/        # gitignored
```

## Known pitfalls

- **MPS garbled output:** intermittent on some torch builds. If a stem sounds like noise, rerun with `-d cpu`.
- **mp3/m4a fail to load:** missing ffmpeg — the #1 cause.
- **librosa tempo half/double errors:** beat trackers sometimes lock to 0.5× or 2× the true tempo. Sanity-check and optionally fold into a 70–160 BPM range.
- **Sample rate:** demucs outputs 44.1 kHz WAV; keep analysis at the native rate, don't silently resample.

## Stretch (once core works)

- Build `template.json` from `analysis.json` per `backend-api-design.md` §2.3 (slice stems into per-bar slots with reference clips) — this validates step 3 of the product flow end-to-end.
- Add a tiny FastAPI wrapper exposing `POST /analysis` as an async job, to mirror the real API contract locally.
