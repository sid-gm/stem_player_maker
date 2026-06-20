# Backend Skeleton — Stem Player / Medley Maker

A map of the backend so a UI/UX redesign knows exactly which endpoints and connectors
to wire back to. Covers the four core flows — **(1) upload + load, (2) select a part,
(3) replace that part, (4) re-render the medley** — plus the modules and external
dependencies each one leans on.

> Backend is a single stdlib HTTP server: `test-harness/server.py`, served at
> `http://localhost:8753`. It does double duty as a **static file server** (the
> visualizer HTML, the stems, the generated JSON/audio) **and** a small JSON API.
> All generated artifacts live under `output/<song>/` and are handed back to the UI
> as `ROOT`-relative paths the same server then serves.

---

## 0. Conventions every endpoint shares

- **`song`** — the song id, which is just the folder name under `output/` (the
  uploaded file's stem). A song "exists" iff `output/<song>/analysis.json` exists.
- **`label`** — a per-layer pattern id like `D14`, `B3`, `O1`, `V2`. First letter =
  stem: **D**rums / **B**ass / **V**ocals / **O**ther.
- **Paths returned to the UI** are rewritten to be relative to the server root so the
  same static server can serve them directly in an `<audio>`/waveform element.
- **Errors**: JSON `{"error": "..."}` with HTTP 400 (bad input) / 404 (no such
  song/label) / 500 (unexpected). Bad-input failures inside the pipeline are raised as
  `SystemExit` and mapped to 400/404.

---

## 1. Upload a song + load

**Goal:** take a raw audio file → separate into stems → analyze structure → make it
loadable in the UI.

### Endpoints

| Method | Endpoint | Body / Query | Returns |
|---|---|---|---|
| `POST` | `/api/upload?name=<filename>&sensitivity=<low\|medium\|high>` | **raw audio bytes** in the body | `{"job": "<id>"}` |
| `GET`  | `/api/status?job=<id>` | — | `{state, step, song, json, error}` |
| `GET`  | `/api/songs` | — | `["<song>", ...]` (every processed song) |

### Flow / connectors

1. `POST /api/upload` writes the bytes to `uploads/<name>`, spins up a **background
   thread** (`_process`), and returns a `job` id immediately (non-blocking).
2. The UI **polls `GET /api/status?job=`** until `state` is `done` or `error`.
   `step` is a human string for the progress line: `"separating stems"` →
   `"analyzing + detecting patterns"` → `"done"`.
3. On `done`, `status` returns `song` and `json` (path to `analysis.json`). The UI
   loads that JSON + the per-stem WAVs to render the timeline.
4. `GET /api/songs` populates the song-switcher dropdown (any folder with an
   `analysis.json`).

### Backend modules behind it

- **`separate.separate(audio_path, OUT, device="mps")`** — wraps the **demucs** CLI
  (4-stem `htdemucs`). Auto-fallback **MPS → CPU** on error/garbled output. Writes
  `output/<song>/stems/{vocals,drums,bass,other}.wav`.
- **`analyze.analyze(...)`** — full-mix + per-stem analysis via **librosa**: tempo,
  key, beat grid, bars, sections, per-stem loudness (LUFS), a 16-step/bar drum
  pattern, repetition `loops[]`, and **per-layer patterns** (`layers[]`). Writes
  `output/<song>/analysis.json`.
- **`analyze.SENSITIVITY`** — preset cutoffs that control how aggressively patterns
  are clustered:
  | preset | drum_dist (Hamming) | melodic_dist (Euclidean) |
  |---|---|---|
  | low | 0.30 | 13.0 |
  | medium | 0.20 | 10.0 |
  | high | 0.10 | 7.0 |

### Re-detect patterns without re-separating (fast toggle)

| Method | Endpoint | Body | Returns |
|---|---|---|---|
| `POST` | `/api/enrich` | `{"song", "sensitivity"}` | `{"ok": true, "song", "sensitivity"}` |

Runs **`enrich_analysis.enrich(song_dir, sensitivity, layers_only=True)`** — a
~2–3 s re-cluster of `layers[]` on the **existing** stems (loops/sections untouched).
This is what the Low/Med/High sensitivity toggle calls; no re-upload, no demucs.

---

## 2. Selecting a part

**Goal:** the user clicks a pattern band; the UI needs the "this is the sound you're
about to replace" preview, and optionally an A/B of how a snap would sound.

### Endpoints

| Method | Endpoint | Query | Returns |
|---|---|---|---|
| `GET` | `/api/reference?song=<song>&label=<label>` | — | `{path, stem, label, representative_bar, start_sec, end_sec, duration_sec, occurrence_count, sr}` |
| `GET` | `/api/snap-preview?song=<song>&label=<label>&source_bar=&target_bar=` | `source_bar`/`target_bar` optional | `{paths: {sample, original, snapped}, ...}` |

### Flow / connectors

- The list of selectable parts comes from `analysis.json` → `layers[<stem>][]`
  (each `{label, occurrences[], representative_bar, occurrence_count, ...}`). The UI
  already has this from step 1 — **no call needed to enumerate parts.**
- `GET /api/reference` cuts (and caches) the **representative bar** of that pattern
  from its stem → a short WAV the UI loops as the preview. Backed by
  **`slice_reference.slice_reference(song_dir, label)`**.
- `GET /api/snap-preview` is optional/demo: renders A/B clips (source sound, original
  target bar, snapped result) so you can hear pitch correction. Backed by
  **`preview_snap.snap_preview(...)`**, which uses the **snap engine** (below).

### Backend modules behind it

- **`snap.build_snap_targets(analysis, label)`** → `list[SnapTarget]`. Pure data
  derived from `analysis.json`: for every bar of every occurrence of the label, where
  it sits in time (`start_sec`, `duration_sec`), the target chord/key/root pitch, the
  target loudness (LUFS), and (drums only) the onset grid. **This is the contract a
  replacement sample is placed against** — it's what connects "a selected part" to
  "where the new sound goes."

---

## 3. Replacing that part (single-part bounce)

**Goal:** swap one part's sound with an uploaded sample at **every** occurrence, then
bounce a full mix.

### Endpoints

| Method | Endpoint | Body / Query | Returns |
|---|---|---|---|
| `POST` | `/api/render?song=<song>&label=<label>&name=<filename>` | **raw audio bytes** (the new sample) in the body | `{paths: {wav, mp3}, stems, label, stem, source, slots_replaced, duration_sec, peak_before_norm, sr, placements[]}` |

### Flow / connectors

1. The UI uploads the chosen sound bite as the **raw body** of `POST /api/render`,
   with `song` + `label` in the query.
2. Server decodes the body to a WAV (`_decode_sample` → **librosa**/**soundfile**, so
   any container mp3/m4a/wav works) under `uploads/`.
3. **`render.render(song_dir, label, sample=...)`** places the sample raw at every
   occurrence and bounces. (If the body is empty, the part's own representative bar is
   used as the stand-in sound — useful for previewing without an upload.)
4. Response `paths.wav`/`paths.mp3` are the new full bounce; `stems` are the edited
   per-stem WAVs so the UI can reload them into the timeline (mute/solo still work).

### How a sample is "placed" (the connector chain)

`render` → `render_medley` (single-placement) → for each `SnapTarget`:
- **`snap.snap(sample, sr, target)`** — **placed RAW**: no time-stretch, no
  pitch-shift, no grid-quantize. It only (a) downmixes to mono, (b) level-matches to
  the slot's stem LUFS, (c) edge-fades ~5 ms so it doesn't click. The sound bite is
  preserved exactly and may ring past its bar.
- **`render._place(...)`** — sums the placed sample into a copy of the stem with an
  ~8 ms crossfade at the leading edge so the swap is click-free.
- Untouched stems pass through; everything is summed and peak-normalized; encoded to
  WAV + MP3 via **soundfile**.

---

## 4. Re-rendering the medley (many parts → many sounds)

**Goal:** stack several part-replacements (possibly across different stems) into one
bounce. Two-step: stash each sample, then render referencing them by id.

### Endpoints

| Method | Endpoint | Body / Query | Returns |
|---|---|---|---|
| `POST` | `/api/sample?name=<filename>` | **raw audio bytes** in body | `{"id": "<sample_id>", "name": "<filename>"}` |
| `POST` | `/api/render-medley` | `{"song", "placements": [{"label", "sample_id"?}], "name"?}` | same shape as `/api/render` (incl. `placements[]`, `slots_replaced`, `paths`, `stems`) |

### Flow / connectors

1. For each new sound the user assigns, `POST /api/sample` once → returns a
   `sample_id` (the stored WAV filename under `uploads/`). The UI holds these ids.
2. `POST /api/render-medley` sends the full arrangement: a list of `{label,
   sample_id}` placements. **A placement with no `sample_id` falls back to that
   part's own representative bar** as the stand-in sound.
3. Server resolves each `sample_id` back to `uploads/<id>` (name-only, no path
   traversal), then calls **`render.render_medley(song_dir, placements,
   write_stems=True)`**.
4. Each placement drops its raw sample at **every occurrence** of its label, across
   whatever stems are involved; touched stems are edited copies, untouched stems pass
   through; summed + peak-normalized → WAV + MP3 (+ per-stem WAVs for timeline
   playback).

### Why two steps

Samples are uploaded once and referenced by id, so re-rendering the medley after
tweaking the arrangement (add/remove a placement, swap one sound) **does not
re-upload audio** — the UI just re-POSTs `/api/render-medley` with an updated
`placements` list.

---

## 5. Dependencies & connectors at a glance

### External / Python packages (`test-harness/requirements.txt`)

| Dependency | Used for | Touched by |
|---|---|---|
| **demucs** (+ torch, torchcodec) | 4-stem source separation (`htdemucs`) | step 1 |
| **librosa** | tempo/key/beat/bar/section/pattern analysis; sample decode & resample | steps 1, 3, 4 |
| **soundfile** | all WAV/MP3 read & write | steps 1–4 |
| **numpy / scipy** | array DSP, feature math | all |
| **scikit-learn** | bar/pattern clustering | step 1 (analyze/enrich) |
| **pyloudnorm** | LUFS loudness measurement (level-matching, analysis) | steps 1, 3, 4 |
| **ffmpeg** (system, via `brew`) | mp3/m4a decode + mp3 encode | upload, sample decode, bounce |

> Runtime: **Python 3.11** (3.12+ has lagging audio deps). torch runs on **MPS**
> (Apple Silicon) with automatic CPU fallback.

### The "effects pack" — `dsp.py` (the operator toolbox / snap engine's hands)

All snap/placement audio correction routes through **`dsp.py`**, the four
librosa-backed operators on mono float arrays. This is the effects layer to keep in
mind when redesigning — the UI never calls it directly, but every render does:

| Operator | Signature | Notes |
|---|---|---|
| `time_stretch(y, sr, ratio)` | length scale | phase-vocoder; **gated off drums** (smears transients). *Not used by raw placement — reserved.* |
| `pitch_shift(y, sr, semitones)` | ± semitones | length-preserving. *Reserved for the pitch-snap path, not raw placement.* |
| `gain_to_lufs(y, sr, target)` | level-match | **actively used** — sample is scaled to the slot's stem LUFS so it sits in the mix. |
| `edge_fade(y, sr, ms)` | anti-click fade | **actively used** on every placement. |

> Current replace/medley flow deliberately uses **only `gain_to_lufs` + `edge_fade`**
> (raw placement). `time_stretch`/`pitch_shift` exist in the pack for a future
> "snap-to-grid / snap-to-key" mode (`snap.SnapTarget` already carries the targets:
> `target_root_pc`, `onset_grid`, `duration_sec`).

### Internal module map

| Module | Role | Entry points |
|---|---|---|
| `server.py` | HTTP API + static server; job queue | all endpoints |
| `separate.py` | demucs wrapper, MPS→CPU fallback | `separate()` |
| `analyze.py` | full analysis + pattern detection; `SENSITIVITY` presets | `analyze()`, `detect_layer_patterns()` |
| `enrich_analysis.py` | fast layers-only re-detect | `enrich()` |
| `slice_reference.py` | representative-bar preview clip | `slice_reference()` |
| `snap.py` | SnapTargets (placement contract) + `snap()` placement | `build_snap_targets()`, `snap()` |
| `preview_snap.py` | A/B snap preview clips | `snap_preview()` |
| `dsp.py` | the effects pack (4 operators) | `time_stretch/pitch_shift/gain_to_lufs/edge_fade` |
| `render.py` | full bounce, single + medley | `render()`, `render_medley()` |

### On-disk layout (the static-served artifacts)

```
output/<song>/
  analysis.json                         # the structure the UI loads
  stems/{drums,bass,vocals,other}.wav   # separated stems (timeline lanes)
  render/<label>/reference_bar<N>.wav   # part previews (step 2)
  render/<label>/snap_*.wav             # A/B snap previews (step 2, optional)
  render/<label>/bounce_<label>.{wav,mp3}   # single-part bounce (step 3)
  render/<name>/bounce_<name>.{wav,mp3}     # medley bounce (step 4)
  render/<name>/stems/<stem>.wav            # edited stems for timeline playback
uploads/                                # raw uploads + decoded samples (sample_id space)
```

---

## Endpoint quick-reference (for wiring the new UI)

```
GET  /api/songs                                          -> [song, ...]
POST /api/upload?name=&sensitivity=        (raw audio)   -> {job}
GET  /api/status?job=                                    -> {state, step, song, json, error}
POST /api/enrich           {song, sensitivity}           -> {ok, song, sensitivity}

GET  /api/reference?song=&label=                         -> {path, stem, representative_bar, ...}
GET  /api/snap-preview?song=&label=&source_bar=&target_bar=  -> {paths:{sample,original,snapped}}

POST /api/render?song=&label=&name=        (raw audio)   -> {paths:{wav,mp3}, stems, slots_replaced, ...}
POST /api/sample?name=                     (raw audio)   -> {id, name}
POST /api/render-medley    {song, placements:[{label, sample_id?}], name?}  -> {paths, stems, ...}
```
