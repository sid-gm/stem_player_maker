# Snap Engine — Build Plan

**Handoff doc for Claude Code.** Goal: build the backend that lets a user's sample
get **"snapped in place"** into a labeled pattern of an analyzed song — time-,
pitch-, chord-, and level-corrected so it fits the original track automatically.
Think *snap-to-grid in a UI, but for sound.*

This builds the **write/synthesis half** of the product. The read/analysis half
(separation + pattern detection) already exists. **Ignore the user-upload UI for
now** — the job here is to stand up the calculation/backend so that the moment a
sample arrives, it can be snapped into any slot.

Working code lives in `test-harness/`. Run everything with `./.venv/bin/python`.

---

## The goal in one line

User picks a pattern (e.g. `B3` — a recurring bassline), drops in their own sound,
and we render a new song with the *same structure* but *their sound* in that
pattern's place — automatically corrected to the song's tempo, key/chords, and
levels at every occurrence.

---

## Where this fits

```
DONE (read):   audio → separate → stems → analyze → {grid, bars, chords, labels} → visualize/preview
THIS DOC:      {targets} + sample → [SNAP: time/pitch/chord/level correct] → placed audio
LATER:         placed audio + untouched stems → [BOUNCE] → new_song.mp3
```

---

## What we already have — the "map" (the target side of snap)

The analytical work of *knowing what to snap to* is ~90% done. Per slot we already
know when it is, how long, what chord, what level, and (for drums) where the hits
land. Sources in `test-harness/`:

| Capability | File | Output / field |
|---|---|---|
| Stem separation | `separate.py` | `stems/{drums,bass,vocals,other}.wav` |
| Beat grid + downbeats | `analyze.py:_build_beat_grid` | `beat_grid[]` (`is_downbeat`) |
| Bars (timing source of truth) | `analyze.py:_build_bars` | `bars[]` (`start_sec`, `end_sec`) |
| Per-bar chord | `analyze.py:_estimate_chords` | `bars[i].chord` (e.g. `"D#m"`) |
| Song key | `analyze.py:estimate_key` | `key` (e.g. `"C# major"`) |
| Drum onset grid | `analyze.py:_drum_pattern` | `patterns.drums.grid` (16 steps/bar) |
| Per-stem loudness | `analyze.py:_loudness_lufs` | `stems[i].loudness_lufs` |
| Pattern labels + occurrences | `analyze.py:detect_layer_patterns` | `layers[stem][]` (`label`, `occurrences`, `representative_bar`) |
| Representative-bar preview clip | `slice_reference.py` | `output/<song>/render/<label>/reference_bar<N>.wav` |
| Preview endpoint + UI | `server.py` `/api/reference`, `visualizer.html` | loops the clip on pattern select |

**Tempo is implicitly per-bar:** bars drift (e.g. 1.59–1.85s on a 132–143 BPM
track), so always snap to a bar's *measured* `end_sec - start_sec`, never a nominal
tempo. This makes the time-snap auto-correct beat-tracker drift for free.

---

## What "snap" means — the 4 correction axes

When a sample lands on a slot, it must be corrected on each axis. We have the
target for all four; we lack the **operator** (the DSP) for the first two.

| Axis | What it does | Target data — **HAVE** | Operator — **NEED** |
|---|---|---|---|
| **Time** | stretch to the bar's exact length; align sample downbeat to slot downbeat | `bars[i].start/end/duration`, `beat_grid` downbeats | time-stretch (`librosa.effects.time_stretch` ✓) + downbeat-align |
| **Pitch** | shift to the song's key / chord root | `key`, `bars[i].chord` | chord→semitone resolver (build) + pitch-shift (`librosa.effects.pitch_shift` ✓) |
| **Level** | gain so it sits at the stem's loudness | `stems[i].loudness_lufs`, per-bar energy | LUFS gain-match (trivial) |
| **Edge** | crossfade tiled repeats so loops don't click | occurrence boundaries | short fade/crossfade (trivial) |

**Installed now:** `librosa 0.11`, `soundfile`, `numpy`, `scipy`.
**Missing (optional, higher quality):** `pedalboard` (Rubber Band stretch/pitch),
`pyrubberband`, `crepe` (pitch detection — upload side, later).

---

## Layer-specific policy (snap behaves differently per stem)

The label groups the *figure* (rhythm/timbre), **not the pitch** — e.g. `B3` plays
under 6 different chords (`C#, F#m, Fm, D#m, G#, F#`). So:

| Layer (prefix) | Pitch snap | Time handling | Notes |
|---|---|---|---|
| **drums** (`D`) | none | **don't stretch** — trigger one-shots on `patterns.drums.grid` | stretching smears transients |
| **bass** (`B`) | shift to **chord root** (monophonic) | tile one bar, stretch to bar length | the main use case for chord-correction |
| **vocals/other** (`V`/`O`) | shift to **key tonic / nearest chord tone** | tile or sustain | polyphonic content = root-shift only in v1 |

---

## The SnapTarget contract — build this FIRST

Collapse the scattered `analysis.json` fields into one explicit per-slot object the
snapper consumes. Defining this *before* any sample exists is the whole point —
it's the contract a sample must satisfy.

```jsonc
SnapTarget = {
  "label": "B3",
  "stem": "bass",
  "bar_index": 50,
  "occurrence_id": 9,
  "pos_in_occ": 4,            // which repeat within the run (for tiling/crossfade)
  "occ_length": 6,

  "start_sec": 50.86,         // ← bars[i].start_sec (this IS a downbeat)
  "duration_sec": 1.672,      // ← measured bar length (absorbs tempo drift)

  "target_root_pc": 3,        // ← resolved from bars[i].chord "D#m" → D# = pitch-class 3
  "target_chord": "D#m",
  "song_key": "C# major",

  "target_lufs": -19.06,      // ← stems[stem].loudness_lufs
  "onset_grid": [0,1,...],    // ← patterns.drums.grid[i]  (drums only)

  // reserved for the upload side — filled when a real sample arrives:
  "sample_root_pc": null,     // sample's own pitch → pitch_shift = target_root_pc - sample_root_pc
  "sample_native_sec": null
}
```

`build_snap_targets(analysis, label) -> list[SnapTarget]` is pure data — buildable
today from `analysis.json`, no audio.

---

## Build steps

Each step is independently testable. Suggested new files under `test-harness/`.

### Step 1 — `snap.py: build_snap_targets(analysis, label)`
Expand a label's occurrences into one `SnapTarget` per bar, pulling timing/level
from `bars`/`stems` and chord from `bars[i].chord`.
**Acceptance:** for `B3` on Drake, emits 26 targets across 12 occurrences with the
right `start_sec`/`duration_sec` and chords matching the known sequence.

### Step 2 — `snap.py: chord_to_root_pc(chord)` + per-layer pitch policy
Parse `"D#m"` → root pitch-class (0–11). Policy: bass→chord root, drums→none,
melodic→key tonic / nearest chord tone. Populate `target_root_pc`.
**Acceptance:** `"C#"→1, "D#m"→3, "F#m"→6`; drums targets carry `target_root_pc=None`.

### Step 3 — `dsp.py` — the operator toolbox
Thin wrappers: `time_stretch(y, sr, ratio)`, `pitch_shift(y, sr, semitones)`,
`gain_to_lufs(y, sr, target)`, `edge_fade(y, sr, ms)`. librosa-backed for v1.
**Acceptance:** round-trip a clip through each; stretch a 1.6s clip to 1.8s and
confirm new length; shift +2 semitones and confirm pitch moved (spectral centroid).
**Pitfall:** phase vocoder smears transients — gate stretching to non-drum layers.

### Step 4 — `snap.py: snap(sample_y, sr, target) -> y`
The pipeline: `stretch → duration_sec`, align downbeat to `start_sec`,
`pitch_shift by (target_root_pc - sample_root_pc)`, `gain_to_lufs(target_lufs)`,
`edge_fade`. Drums skip stretch/pitch and instead place one-shots on `onset_grid`.
**Acceptance:** given a stub sample + a `SnapTarget`, returns audio of exactly
`duration_sec` at the target level.

### Step 5 — `preview_snap.py` — one-slot snapped preview (validator) ⚠️ do this early
Extend the reference flow: render *"this slot, snapped"* and A/B it against the
original `slice_reference` clip. **This gates everything** — pitch/chord come from
rough estimates, so we must *hear* whether snap is convincing before scaling up.
Use the song's own stem bar as the stand-in "sample" (so we can validate snap math
with no upload). Add a `/api/snap-preview?song=&label=` endpoint mirroring
`/api/reference`.
**Acceptance:** snapping a `B3` bar from one chord onto another occurrence's chord
produces audio that sits in-key by ear.

### Step 6 — `render.py` — full bounce (LATER, after snap is trusted)
`new_mix = (untouched stems) + (replaced layer with snapped audio at every slot)`,
crossfade edges, encode mp3. Leverages the existing stem isolation.

---

## Known pitfalls / risks

- **Transient smearing on stretch** (phase vocoder). Mitigation: never stretch
  drums — trigger one-shots on the onset grid we already have. Stretch only
  sustained/melodic layers. Upgrade to `pedalboard`/Rubber Band when quality bites.
- **Chord-estimate quality** is the gate. `_estimate_chords` is rough template
  matching. Validate by ear (Step 5) before trusting snap-to-chord. Consider
  constraining chords to the detected `key`.
- **Sample root detection** (upload side, later): computing the pitch interval needs
  the sample's own root (`pyin`/`crepe`). Not needed now — the contract reserves
  `sample_root_pc`.
- **Polyphonic chord-correction** (re-voicing a multi-note sample onto a target
  chord) is out of scope for v1 — root-shift only.
- **Beat drift** is handled by snapping to *measured* bar duration, not nominal
  tempo — keep it that way.

---

## In one line

We have the **map** (every slot's time/chord/level target). We need the **hands**
(`dsp.py` stretch/pitch/gain) and the **contract** (`SnapTarget`). Build Steps 1–2
+ 5 first to prove the chord estimates are good enough to snap against; everything
else follows.

---

## Suggested file layout

```
test-harness/
  snap.py          # build_snap_targets, chord_to_root_pc, snap()      (Steps 1,2,4)
  dsp.py           # time_stretch, pitch_shift, gain_to_lufs, edge_fade (Step 3)
  preview_snap.py  # one-slot snapped preview + validator               (Step 5)
  render.py        # full bounce (later)                                (Step 6)
  # existing: separate.py, analyze.py, enrich_analysis.py,
  #           slice_reference.py, server.py, visualizer.html
```

## Data reference (real, for test assertions)

- **Drake high fives** — 143.55 BPM, C# major, 76 bars. `B3`: 26 bars / 12 occ,
  rep bar 50 (`D#m`... actually bar 50 = `F#m`), spans chords `C# F#m Fm D#m G# F#`.
- **E85 DT** — 132.51 BPM. `D7`: 17 bars / 6 occ, rep bar 25; occ lengths 1–7 bars.
- Bar durations drift ~1.59–1.85s → always stretch to measured bar length.
