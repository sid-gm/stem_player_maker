#!/usr/bin/env python
"""Cut the representative-bar clip for a layer pattern label.

Given a song dir (analysis.json + stems/) and a pattern label like "D14", find
that pattern's representative bar and slice it out of the matching stem -> a
short WAV the UI loops as the "this is the sound you're replacing" preview.

This is the `reference.audio_slice` from the render-unit spec: the canonical
one-bar unit of the pattern, soloed to a single stem.

    python slice_reference.py "output/Drake high fives" D14
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

# Label prefix -> stem (mirror of analyze._LAYER_PREFIX).
_PREFIX_STEM = {"D": "drums", "B": "bass", "V": "vocals", "O": "other"}


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


def _apply_edge_fade(data: np.ndarray, sr: int, fade_ms: float) -> None:
    """In-place linear fade in/out so the looped clip doesn't click."""
    n = int(fade_ms / 1000.0 * sr)
    if n <= 0 or 2 * n >= len(data):
        return
    ramp = np.linspace(0.0, 1.0, n, dtype=data.dtype)[:, None]
    data[:n] *= ramp
    data[-n:] *= ramp[::-1]


def slice_reference(song_dir: str | Path, label: str,
                    out_dir: str | Path | None = None,
                    fade_ms: float = 5.0) -> dict:
    """Write the representative-bar clip for `label`. Returns clip metadata."""
    song_dir = Path(song_dir)
    d = json.loads((song_dir / "analysis.json").read_text())
    layers = d.get("layers") or {}
    if not layers:
        raise SystemExit("analysis.json has no `layers` — run/enrich the song first.")

    stem, pat = _find_pattern(layers, label)
    bi = pat["representative_bar"]
    bars = d.get("bars") or []
    if not (0 <= bi < len(bars)):
        raise SystemExit(f"representative_bar {bi} out of range (n_bars={len(bars)})")
    start, end = bars[bi]["start_sec"], bars[bi]["end_sec"]

    stem_wav = song_dir / "stems" / f"{stem}.wav"
    if not stem_wav.exists():
        raise SystemExit(f"missing stem: {stem_wav}")
    sr = sf.info(str(stem_wav)).samplerate
    data, sr = sf.read(str(stem_wav), start=int(start * sr), stop=int(end * sr),
                       always_2d=True)
    _apply_edge_fade(data, sr, fade_ms)

    out = Path(out_dir) if out_dir else (song_dir / "render" / label)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"reference_bar{bi}.wav"
    sf.write(str(out_path), data, sr, subtype="PCM_16")

    return {
        "path": str(out_path),
        "stem": stem,
        "label": label,
        "representative_bar": bi,
        "start_sec": round(float(start), 4),
        "end_sec": round(float(end), 4),
        "duration_sec": round(float(end - start), 4),
        "occurrence_count": pat.get("occurrence_count"),
        "sr": sr,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cut a pattern's representative-bar preview clip.")
    ap.add_argument("song_dir", help="song dir containing analysis.json + stems/")
    ap.add_argument("label", help="pattern label, e.g. D14")
    ap.add_argument("--fade-ms", type=float, default=5.0)
    args = ap.parse_args()

    info = slice_reference(args.song_dir, args.label, fade_ms=args.fade_ms)
    print(f"wrote {info['path']}")
    print(f"  {info['stem']} {info['label']}  bar {info['representative_bar']}  "
          f"{info['start_sec']}–{info['end_sec']}s ({info['duration_sec']}s)  "
          f"×{info['occurrence_count']} occurrences")
