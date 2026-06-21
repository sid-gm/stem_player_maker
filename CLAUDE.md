# jawnsplitter

**jawnsplitter** is the redesign / product name for this stem-player / music-maker
platform. Core loop: **upload a song → split it into 4 stems → swap a part's sound
(with effects) → preview & export the remix.** "any song. your sound."

This repo started as a test harness (`test-harness/`) and is being grown into the
jawnsplitter product.

## Layout

```
capcut_music_maker/
  CLAUDE.md                 # this file
  backend-skeleton.md       # the API contract — READ THIS before wiring UI to the backend
  test-harness/
    server.py               # the backend: stdlib HTTP server, static files + JSON API (:8753)
    analyze.py separate.py render.py snap.py dsp.py ...  # the audio pipeline
    visualizer.html         # the original desktop debug UI (vanilla JS). Source of truth for
                            #   the Web Audio plumbing + the client-side effects (FX) implementation.
    jawnsplitter.html       # NEW: the jawnsplitter MOBILE view (small browsers / phones)
    output/<song>/          # processed songs: analysis.json + stems/*.wav + renders
    uploads/                # raw uploads + decoded swap samples
```

## The mobile view — `test-harness/jawnsplitter.html`

The redesigned **mobile** experience (small browsers / phones), implemented from the
Claude Design wireframe `ui_kits/flow-wireframe` of the "Jawnsplitter Design System"
project (`b5ac4506-c9d1-4d7f-bbca-eebdc29c988e` on claude.ai/design).

- **Single self-contained static file**: React 18 + Babel-standalone via CDN, design
  tokens inlined, fonts from Google Fonts. No build step. Served by `server.py` today
  and trivially deployable to Vercel later (it's just static HTML).
- **Flow**: `Upload → Splitting → Studio (timeline) → Swap (bottom sheet) → Preview`,
  wired to the real backend (see `backend-skeleton.md`):
  - upload → `POST /api/upload` + poll `GET /api/status`
  - "try a sample" → `GET /api/songs`, loads an already-processed song straight to studio
  - studio → loads `output/<song>/analysis.json`, plays the 4 stem WAVs in sync (Web
    Audio, one playhead); lanes are built from `analysis.layers[stem][]`; tap a part →
    `GET /api/reference` loops its representative bar + opens the swap sheet
  - swap → choose a sound (file/mic) + toggle effects → on "use it" the sound is baked
    client-side and held as the swap sample
  - preview → `POST /api/sample` per swap, then `POST /api/render-medley`; A/B toggles
    original ↔ remix by swapping decoded stem buffers; "export jawn" downloads the bounce
- **Effects (FX)** are **client-side Web Audio**, ported from `visualizer.html`
  (the `FX` table + `renderFx`). Each effect is baked into the swap sample
  (`OfflineAudioContext` → mono WAV) before it is uploaded, so the server receives real
  processed audio. The backend render itself is raw placement only (gain-match +
  edge-fade); it does not implement named effects.
- **Configurable API base** for hosting: `API_BASE` resolves from `?api=<url>` or
  `localStorage.jx_api_base`, defaulting to same-origin. Lets the static frontend live
  on Vercel and point at a separately-hosted backend.

## Running locally

```bash
cd test-harness
./.venv/bin/python server.py          # http://localhost:8753
# mobile view:  http://localhost:8753/jawnsplitter.html
# desktop debug UI: http://localhost:8753/visualizer.html
```

Requires Python 3.11 + the deps in `test-harness/requirements.txt` (demucs/torch,
librosa, soundfile, …) and system `ffmpeg`. Splitting a new song runs demucs and takes
a few minutes; "try a sample" loads an already-processed song instantly.

## Hosting note (Vercel)

The frontend (`jawnsplitter.html`) is static and Vercel-ready. The **backend cannot run
on Vercel serverless** — demucs/torch are too heavy and the split is long-running. Plan:
host the static page on Vercel and point `API_BASE` at the Python backend running
elsewhere (a box / Fly / Render / etc.).
