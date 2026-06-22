#!/usr/bin/env python
"""preview_snap.py — one-slot snapped preview (the Step 5 validator).

This gates everything downstream: pitch/chord come from rough estimates, so we must
*hear* whether snap is convincing before scaling to a full bounce. With no upload
yet, we use the song's own stem bar as the stand-in "sample" — take one bar of a
pattern (at chord X) and snap it onto another occurrence of the same pattern (at
chord Y). If the chord->pitch math is right, the result sits in-key at Y.

Emits, under output/<song>/render/<label>/:
    snap_sample_bar<src>.wav         raw source bar (chord X, uncorrected)
    snap_original_bar<tgt>.wav       the real target bar (chord Y, ground truth)
    snap_snapped_<src>to<tgt>.wav    source snapped onto the target slot
    snap_ab_<src>to<tgt>.wav         montage: sample | original | snapped (A/B by ear)

    python preview_snap.py "output/Drake high fives" B3

A/B test: snapped should match the *pitch/key* of original (same chord), with the
source's timbre. CLI also drives /api/snap-preview in server.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

import dsp
from snap import (build_snap_targets, chord_to_root_pc, load_analysis,
                  nearest_semitone_shift, snap)


def _read_bar(stem_wav: Path, start_sec: float, end_sec: float) -> tuple[np.ndarray, int]:
    """Mono slice of a stem between two times."""
    sr = sf.info(str(stem_wav)).samplerate
    data, sr = sf.read(str(stem_wav), start=int(start_sec * sr),
                       stop=int(end_sec * sr), always_2d=True)
    return dsp._as_mono_f32(data), sr


def extract_oneshot(stem_wav: Path, bars: list, bar_index: int,
                    onsets_sec: list | None = None,
                    max_len_ms: float = 250.0) -> tuple[np.ndarray, int, float, float]:
    """Slice a single drum hit (one-shot) from a bar — the stand-in for "your one
    drum sound". Takes the first onset in the bar, runs to the next onset (capped),
    keeps the attack sharp (no fade-in) and fades only the tail so it decays clean.
    """
    sr = sf.info(str(stem_wav)).samplerate
    b0, b1 = bars[bar_index]["start_sec"], bars[bar_index]["end_sec"]
    onsets = sorted(t for t in (onsets_sec or []) if b0 <= t < b1)
    if onsets:
        t0 = onsets[0]
        nxt = next((t for t in onsets if t > t0 + 0.02), None)
        end = min(t0 + max_len_ms / 1000.0, b1, nxt if nxt else b1)
        start = max(0.0, t0 - 0.004)  # 4ms pre-roll to capture the transient
    else:  # no detected onset — fall back to the head of the bar
        start, end = b0, min(b0 + max_len_ms / 1000.0, b1)
    y, sr = sf.read(str(stem_wav), start=int(start * sr), stop=int(end * sr), always_2d=True)
    y = dsp._as_mono_f32(y)
    # tiny fade-in (1ms, click guard) + longer fade-out (decay); keep the attack intact
    fi, fo = int(0.001 * sr), int(0.008 * sr)
    if 0 < fi < len(y):
        y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if 0 < fo < len(y):
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return y, sr, round(start, 3), round(end, 3)


def _montage(clips: list[np.ndarray], sr: int, gap_sec: float = 0.35) -> np.ndarray:
    """Concatenate clips with silence between, each edge-faded so it doesn't click."""
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, c in enumerate(clips):
        parts.append(dsp.edge_fade(c, sr, 5.0))
        if i < len(clips) - 1:
            parts.append(gap)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def snap_preview(song_dir: str | Path, label: str,
                 source_bar: int | None = None, target_bar: int | None = None,
                 out_dir: str | Path | None = None) -> dict:
    """Snap one bar of `label` onto another occurrence and write A/B clips.

    Defaults: source = the pattern's representative bar; target = the first slot
    whose chord differs from the source (so the pitch correction is audible).
    """
    song_dir = Path(song_dir)
    analysis = load_analysis(song_dir)
    targets = build_snap_targets(analysis, label)
    if not targets:
        raise SystemExit(f"no targets for label {label}")
    stem = targets[0].stem
    stem_wav = song_dir / "stems" / f"{stem}.wav"
    if not stem_wav.exists():
        raise SystemExit(f"missing stem: {stem_wav}")
    bars = analysis["bars"]
    by_bar = {t.bar_index: t for t in targets}

    # --- pick source ---
    if source_bar is None:
        layers = analysis["layers"][stem]
        source_bar = next(p["representative_bar"] for p in layers if p["label"] == label)
    if source_bar not in by_bar:
        source_bar = targets[0].bar_index
    src_t = by_bar[source_bar]
    sample_pc = chord_to_root_pc(src_t.target_chord)  # the sample's own estimated root

    # --- pick target: first slot whose chord differs from source (audible shift) ---
    if target_bar is None:
        diff = [t for t in targets
                if t.bar_index != source_bar and t.target_chord != src_t.target_chord]
        target_bar = (diff[0] if diff else
                      next(t for t in targets if t.bar_index != source_bar)).bar_index \
            if len(targets) > 1 else source_bar
    tgt_t = by_bar[target_bar]
    tgt_t.sample_root_pc = sample_pc  # tell snap() the sample's pitch -> it computes the shift

    # --- slice the source sound + the original target bar from the stem ---
    # Drums snap a SINGLE hit onto the grid, so the "sample" is one onset, not a bar.
    if stem == "drums":
        onsets = (analysis.get("patterns", {}).get("drums", {}) or {}).get("onsets_sec")
        src_y, sr, _os, _oe = extract_oneshot(stem_wav, bars, source_bar, onsets)
    else:
        src_y, sr = _read_bar(stem_wav, bars[source_bar]["start_sec"], bars[source_bar]["end_sec"])
    orig_y, _ = _read_bar(stem_wav, bars[target_bar]["start_sec"], bars[target_bar]["end_sec"])

    # --- snap source onto target ---
    snapped = snap(src_y, sr, tgt_t)

    semis = (nearest_semitone_shift(sample_pc, tgt_t.target_root_pc)
             if sample_pc is not None and tgt_t.target_root_pc is not None else 0)

    out = Path(out_dir) if out_dir else (song_dir / "render" / label)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "sample": out / f"snap_sample_bar{source_bar}.wav",
        "original": out / f"snap_original_bar{target_bar}.wav",
        "snapped": out / f"snap_snapped_{source_bar}to{target_bar}.wav",
        "ab": out / f"snap_ab_{source_bar}to{target_bar}.wav",
    }
    sf.write(str(paths["sample"]), dsp.edge_fade(src_y, sr, 5.0), sr, subtype="PCM_16")
    sf.write(str(paths["original"]), dsp.edge_fade(orig_y, sr, 5.0), sr, subtype="PCM_16")
    sf.write(str(paths["snapped"]), snapped, sr, subtype="PCM_16")
    sf.write(str(paths["ab"]), _montage([src_y, orig_y, snapped], sr), sr, subtype="PCM_16")

    return {
        "label": label, "stem": stem, "sr": sr,
        "source_bar": source_bar, "source_chord": src_t.target_chord, "source_root_pc": sample_pc,
        "target_bar": target_bar, "target_chord": tgt_t.target_chord,
        "target_root_pc": tgt_t.target_root_pc, "semitone_shift": semis,
        "target_lufs": tgt_t.target_lufs, "duration_sec": tgt_t.duration_sec,
        "snapped_lufs": round(dsp.measure_lufs(snapped, sr) or 0.0, 2),
        "paths": {k: str(v) for k, v in paths.items()},
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render a one-slot snapped A/B preview.")
    ap.add_argument("song_dir", help="song dir containing analysis.json + stems/")
    ap.add_argument("label", help="pattern label, e.g. B3")
    ap.add_argument("--source-bar", type=int, default=None)
    ap.add_argument("--target-bar", type=int, default=None)
    args = ap.parse_args()

    info = snap_preview(args.song_dir, args.label,
                        source_bar=args.source_bar, target_bar=args.target_bar)
    print(f"{info['label']} ({info['stem']}):  "
          f"bar {info['source_bar']} {info['source_chord']}(pc{info['source_root_pc']}) "
          f"-> bar {info['target_bar']} {info['target_chord']}(pc{info['target_root_pc']})  "
          f"shift {info['semitone_shift']:+d} semis")
    print(f"  level: snapped {info['snapped_lufs']} LUFS vs target {info['target_lufs']}")
    for name in ("sample", "original", "snapped", "ab"):
        print(f"  {name:>8}: {info['paths'][name]}")
    print("  A/B: play the 'ab' clip — sample(chordX) | original(chordY) | snapped. "
          "snapped should match original's key.")
