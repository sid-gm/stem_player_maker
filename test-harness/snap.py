#!/usr/bin/env python
"""Snap engine — turn analysis.json into per-slot SnapTargets (and, later, snap audio).

A *SnapTarget* is the contract a sample must satisfy to be "snapped into place" at
one bar of one pattern occurrence: where it sits in time, what chord/level it must
match, and (for drums) where the hits land. It is pure data, derivable from
analysis.json with no audio.

    python snap.py "output/Drake high fives" B3

Step 1: build_snap_targets(analysis, label) -> list[SnapTarget].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import dsp

# Label prefix -> stem (mirror of analyze._LAYER_PREFIX / slice_reference._PREFIX_STEM).
_PREFIX_STEM = {"D": "drums", "B": "bass", "V": "vocals", "O": "other"}

# Note letter -> pitch class (semitone, C=0). Accidentals applied on top.
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def chord_to_root_pc(chord: str | None) -> int | None:
    """Parse a chord/note name -> root pitch class 0-11. Quality (m/maj/dim...) is
    ignored — the label groups the *figure*, not the chord, so we snap to the root.

        "C#" -> 1, "D#m" -> 3, "F#m" -> 6, "Bb7" -> 10, "N"/"" -> None
    """
    if not chord:
        return None
    s = chord.strip()
    if not s or s[0].upper() not in _NOTE_PC:  # "N", "N.C.", etc.
        return None
    pc = _NOTE_PC[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in ("#", "b", "♯", "♭"):
        pc += 1 if s[i] in ("#", "♯") else -1
        i += 1
    return pc % 12


def key_to_tonic_pc(key: str | None) -> int | None:
    """First token of a key string ("C# major") -> tonic pitch class."""
    if not key:
        return None
    return chord_to_root_pc(key.split()[0])


def pitch_policy(stem: str, target_chord: str | None,
                 song_key: str | None) -> int | None:
    """Per-layer pitch target. bass -> chord root (monophonic); drums -> none
    (pitch-snap smears nothing useful); melodic (vocals/other) -> key tonic in v1
    (polyphonic content is root-shift only, so anchor to the song tonic)."""
    if stem == "drums":
        return None
    if stem == "bass":
        return chord_to_root_pc(target_chord)
    # vocals / other
    tonic = key_to_tonic_pc(song_key)
    return tonic if tonic is not None else chord_to_root_pc(target_chord)


@dataclass
class SnapTarget:
    """One bar of one occurrence — everything a sample must be corrected to."""

    label: str
    stem: str
    bar_index: int
    occurrence_id: int        # index into the pattern's occurrences[]
    pos_in_occ: int           # which repeat within the run (for tiling/crossfade)
    occ_length: int           # number of bars in this occurrence run

    start_sec: float          # bars[i].start_sec — this IS a downbeat
    duration_sec: float       # measured bar length (absorbs tempo drift)

    target_root_pc: int | None  # resolved from target_chord in Step 2
    target_chord: str | None
    song_key: str | None

    target_lufs: float | None   # stems[stem].loudness_lufs
    onset_grid: list[int] | None  # patterns.drums.grid[i] — drums only

    # reserved for the upload side — filled when a real sample arrives:
    sample_root_pc: int | None = None
    sample_native_sec: float | None = None


def _find_pattern(layers: dict, label: str) -> tuple[str, dict]:
    """Locate (stem, pattern) for a label. Try the prefix's stem, then any."""
    stem = _PREFIX_STEM.get(label[:1].upper())
    order = [stem] if stem else []
    order += [s for s in layers if s not in order]
    for s in order:
        for p in layers.get(s) or []:
            if p["label"] == label:
                return s, p
    raise SystemExit(f"label '{label}' not found in any layer "
                     f"(have: { {s: [p['label'] for p in v] for s, v in layers.items()} })")


def _stem_lufs(analysis: dict, stem: str) -> float | None:
    """Loudness of a stem by its type (stems[] entries key on `type`)."""
    for s in analysis.get("stems") or []:
        if s.get("type") == stem:
            return s.get("loudness_lufs")
    return None


def build_snap_targets(analysis: dict, label: str) -> list[SnapTarget]:
    """Expand a label's occurrences into one SnapTarget per bar.

    Timing/level come from bars/stems; chord from bars[i].chord. The drum onset
    grid is attached only for the drums stem. target_root_pc is left None here and
    populated by the Step 2 pitch policy.
    """
    layers = analysis.get("layers") or {}
    if not layers:
        raise SystemExit("analysis.json has no `layers` — run/enrich the song first.")

    stem, pat = _find_pattern(layers, label)
    bars = analysis.get("bars") or []
    key = analysis.get("key")
    lufs = _stem_lufs(analysis, stem)
    drum_grid = (analysis.get("patterns", {}).get("drums", {}) or {}).get("grid") \
        if stem == "drums" else None

    targets: list[SnapTarget] = []
    for occ_id, occ in enumerate(pat.get("occurrences") or []):
        start_bar, end_bar = occ["start_bar"], occ["end_bar"]
        occ_length = end_bar - start_bar + 1
        for pos, bi in enumerate(range(start_bar, end_bar + 1)):
            if not (0 <= bi < len(bars)):
                raise SystemExit(f"bar {bi} out of range (n_bars={len(bars)})")
            bar = bars[bi]
            grid = drum_grid[bi] if drum_grid and bi < len(drum_grid) else None
            targets.append(SnapTarget(
                label=label,
                stem=stem,
                bar_index=bi,
                occurrence_id=occ_id,
                pos_in_occ=pos,
                occ_length=occ_length,
                start_sec=round(float(bar["start_sec"]), 4),
                duration_sec=round(float(bar["end_sec"] - bar["start_sec"]), 4),
                target_root_pc=pitch_policy(stem, bar.get("chord"), key),
                target_chord=bar.get("chord"),
                song_key=key,
                target_lufs=round(float(lufs), 4) if lufs is not None else None,
                onset_grid=grid,
            ))
    return targets


# ── Step 4: snap() — place the raw sample (no stretch/pitch/quantize) ───────────

def nearest_semitone_shift(sample_pc: int, target_pc: int) -> int:
    """Smallest signed semitone move from sample root to target root, in [-6, 6]
    (octave-equivalent — pitch class is mod 12, so prefer the nearest direction)."""
    return ((target_pc - sample_pc + 6) % 12) - 6


def snap(sample_y: np.ndarray, sr: int, target: "SnapTarget") -> np.ndarray:
    """Place a sample at one slot, preserving the raw sound bite, and return mono.

    There is no time-stretch, no pitch-shift, and no grid-quantize: the uploaded
    sound is kept exactly as it is — that's the whole point. We only:
      - downmix to mono,
      - level-match to the slot's stem loudness so it sits in the mix (volume only,
        not character),
      - edge-fade a few ms so the placement doesn't click.

    The sample keeps its native length and may ring past the bar — intended, so a
    medley can overlap messily. The caller decides where each placement lands.
    """
    y = dsp._as_mono_f32(sample_y)
    if y.size == 0:
        return np.zeros(0, dtype=np.float32)
    if target.target_lufs is not None:
        y = dsp.gain_to_lufs(y, sr, target.target_lufs)
    return dsp.edge_fade(y, sr, 5.0)


def load_analysis(song_dir: str | Path) -> dict:
    return json.loads((Path(song_dir) / "analysis.json").read_text())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build SnapTargets for a pattern label.")
    ap.add_argument("song_dir", help="song dir containing analysis.json")
    ap.add_argument("label", help="pattern label, e.g. B3")
    ap.add_argument("--json", action="store_true", help="dump full targets as JSON")
    args = ap.parse_args()

    analysis = load_analysis(args.song_dir)
    targets = build_snap_targets(analysis, args.label)

    if args.json:
        print(json.dumps([asdict(t) for t in targets], indent=2))
    else:
        chords = [t.target_chord for t in targets]
        n_occ = len({t.occurrence_id for t in targets})
        print(f"{args.label}: {len(targets)} targets across {n_occ} occurrences")
        print(f"  stem={targets[0].stem}  key={targets[0].song_key}  "
              f"lufs={targets[0].target_lufs}")
        print(f"  chords: {chords}")
        print(f"  unique chords: {sorted(set(c for c in chords if c))}")
        print(f"  root_pc:  {[t.target_root_pc for t in targets]}")
        for t in targets[:3]:
            print(f"    bar {t.bar_index:>2}  occ {t.occurrence_id} "
                  f"[{t.pos_in_occ}/{t.occ_length}]  "
                  f"{t.start_sec}s +{t.duration_sec}s  "
                  f"{t.target_chord}->pc{t.target_root_pc}")
