#!/usr/bin/env python
"""Add repetition + per-layer pattern data to an existing analysis.json.

Use this on songs that were analyzed before `loops`/`layers` existed, or to
re-run detection at a different sensitivity. Works from the stems beside the
JSON — no source audio needed.

    python enrich_analysis.py "output/E85 DT"
    python enrich_analysis.py "output/E85 DT" --sensitivity high

Sensitivity (higher = more, stricter patterns): low | medium | high.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from analyze import SENSITIVITY, detect_layer_patterns, detect_repetitions

_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg"}


def enrich(target: str | Path, sensitivity: str = "medium",
           layers_only: bool = False) -> dict:
    """Enrich the analysis.json at/under `target`. Returns the updated dict.

    `layers_only=True` skips the (sensitivity-independent) loop/section pass and
    only re-detects per-layer patterns — used by the UI sensitivity toggle so it
    stays fast. Falls back to a full pass if loops aren't present yet.
    """
    p = Path(target)
    if p.suffix.lower() in _AUDIO_EXTS:
        raise SystemExit(
            f"'{p.name}' is an audio file. Use run.py for audio:\n"
            f"    python run.py \"{target}\"\n"
            f"enrich_analysis.py only adds patterns to an already-separated "
            f"song dir (one containing stems/ + analysis.json).")

    json_path = p / "analysis.json" if p.is_dir() else p
    song_dir = json_path.parent
    stems_dir = song_dir / "stems"

    d = json.loads(json_path.read_text())
    bars = d.get("bars") or []
    stem_paths = {}
    for s in d.get("stems", []):
        cand = stems_dir / f"{s['type']}.wav"
        stem_paths[s["type"]] = str(cand if cand.exists() else s["path"])
    if not stem_paths or not bars:
        raise SystemExit("analysis.json is missing stems or bars — run run.py first.")

    th = SENSITIVITY.get(sensitivity, SENSITIVITY["medium"])
    drum_grid = d.get("patterns", {}).get("drums", {}).get("grid")

    if not (layers_only and d.get("loops")):
        per_bar, loops = detect_repetitions(stem_paths, bars, drum_grid)
        for i, bar in enumerate(bars):
            bar.update(per_bar.get(i, {}))
        d["loops"] = loops

    layers, bar_ids = detect_layer_patterns(
        stem_paths, bars, drum_grid,
        drum_dist=th["drum_dist"], melodic_dist=th["melodic_dist"])
    for i, bar in enumerate(bars):
        bar["layer_ids"] = bar_ids.get(i, {})

    d["layers"] = layers
    d["sensitivity"] = sensitivity
    json_path.write_text(json.dumps(d, indent=2))
    return d


def _summary(d: dict, json_path: str) -> None:
    print(f"enriched {json_path}  (sensitivity={d.get('sensitivity')})")
    print(f"  loops:  {[l['label'] for l in d.get('loops', [])]}")
    for stem, pats in d.get("layers", {}).items():
        reps = ", ".join(f"{p['label']}×{p['occurrence_count']}" for p in pats)
        print(f"  {stem:<7}: {len(pats)} patterns  [{reps}]")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Add loops + layer patterns to analysis.json.")
    ap.add_argument("target", help="song dir or analysis.json path")
    ap.add_argument("--sensitivity", default="medium", choices=list(SENSITIVITY))
    args = ap.parse_args()

    d = enrich(args.target, args.sensitivity)
    p = Path(args.target)
    _summary(d, str(p / "analysis.json" if p.is_dir() else p))
