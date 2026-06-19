# Snap Engine — Scope Decision (v1)

> **SUPERSEDED 2026-06-19 (later same day).** Direction pivoted: **snap-to-grid
> (stretch-to-bar + retune) is removed entirely.** The product value is preserving the
> **raw uploaded sound bite** — no time-stretch, no pitch-shift, no grid-quantize. The
> sample is placed raw (mono + level-match + edge-fade) at every occurrence and may ring
> past its bar (overlap is intended). Consequences:
> - `snap.snap()` is now raw-only (no `fit`); `render.render()` lost its `fit`/drum-refusal.
> - **Drums are no longer deferred** — without stretch, dropping a sound on a drum slot is
>   the same raw placement as any other stem, so all layers are replaceable.
> - **New: medley.** `render.render_medley()` + `/api/sample` + `/api/render-medley` layer
>   *many* parts → *many* sounds in one bounce ("messy medley from everywhere").
>
> Everything below is kept for history but no longer describes the build.

**Status:** active scope cut, 2026-06-19. Supplements `snap-engine-plan.md`.

## Decision

For now, **all remaining snap-engine work focuses on bass + melodic layers** (pitch +
time matching) and drives them through to a **finished rendered output**. Drums — and
any percussive / one-shot-style instruments that behave like drums — are **explicitly
deferred to a separate epic**.

## Why

The Step 5 validator proved two things by ear:

- **Bass / melody works.** Stretch-to-bar-length + retune-to-chord-root + level-match
  produces audio that sits in-key at the target slot. The pitch + time math is sound.
- **Drums need a different model.** Drum snap is not "stretch/retune a bar" — it's
  "trigger a **single hit** (one kick/snare) on the song's onset grid." Stretching
  smears transients, and feeding a whole drum bar into the trigger path stacks the bar
  on top of itself many times → mush. This is a fundamentally different input shape
  (one-shot, not a bar-length figure), so it deserves its own design and epic rather
  than being bolted onto the melodic pipeline.

## In scope now (this epic) — "drive bass/melody to an output"

| Layer | Pitch | Time | Level | Notes |
|---|---|---|---|---|
| **bass** (`B`) | shift to chord root | stretch to measured bar length | LUFS match | primary use case |
| **vocals/other** (`V`/`O`) | shift to key tonic (root-shift only) | stretch / tile | LUFS match | polyphonic = root-shift only in v1 |

Work items:
- Steps 1–5 (`snap.py`, `dsp.py`, `preview_snap.py`) — **done**, validated for bass/melody.
- **Step 6 — `render.py` full bounce**, restricted to bass/melodic layers: replace one
  layer's sound at *every* occurrence with the snapped sample, keep all other stems
  untouched, crossfade edges, encode to mp3. This is the "reach an output" milestone.

**Definition of done for this epic:** a full song renders where a chosen bass/melodic
pattern is replaced by a stand-in (or uploaded) sample, auto-corrected at every slot,
and the bounce sounds musically coherent end-to-end.

## Deferred to a separate epic — "percussive / one-shot snap"

- **Layers:** drums (`D`) and any instrument whose snap unit is a single hit rather than
  a sustained bar (claps, percussion, stabs, one-shots).
- **Approach:** extract / accept a **single hit**, then trigger it on
  `patterns.drums.grid` (16 steps/bar) — no stretch, no pitch. The trigger logic already
  exists in `snap._snap_drums`; the missing piece is the **one-shot input** (slice a hit
  from the bar, or take a one-shot upload), not the placement.
- **Validator fix (also deferred):** make `preview_snap.py` extract a single onset for
  drum patterns instead of reusing the whole bar, so drums can be A/B'd fairly.

## What does NOT change

- The `SnapTarget` contract stays as-is — it already carries `onset_grid` for the drum
  path, so the deferred epic needs no schema change.
- `snap()` keeps its drums branch; it's simply not exercised in this epic's render.
- Beat-drift handling (snap to *measured* bar duration, never nominal tempo) stays.

## One line

Ship bass/melody pitch+time snapping all the way to a rendered song now; give drums and
drum-like one-shots their own epic with the single-hit trigger model.
