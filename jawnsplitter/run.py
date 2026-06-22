#!/usr/bin/env python
"""End-to-end runner: separate -> analyze -> write output/<song>/analysis.json.

Usage:
    python run.py path/to/song.mp3 [--device mps|cpu] [--model htdemucs|htdemucs_6s]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze import analyze
from separate import separate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="path to an audio file (mp3/m4a/wav/...)")
    ap.add_argument("-o", "--out", default="output", help="output root (default: output)")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--model", default="htdemucs", choices=["htdemucs", "htdemucs_6s"])
    ap.add_argument("--song-id", default=None, help="song_id for analysis.json")
    args = ap.parse_args()

    out_root = Path(args.out)

    print(f"[run] separating {args.audio} (device={args.device}, model={args.model})")
    sep = separate(args.audio, out_root, device=args.device, model=args.model)

    song_id = args.song_id or sep["song"]
    print(f"[run] analyzing full mix + drum stem for '{song_id}'")
    analysis = analyze(args.audio, sep["stems"], song_id=song_id)

    song_dir = Path(sep["song_dir"])
    out_path = song_dir / "analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))

    # One-line summary.
    n_beats = len(analysis["beat_grid"])
    n_sections = len(analysis["sections"])
    print(
        f"[run] done: tempo={analysis['tempo_bpm']} BPM "
        f"(conf {analysis['tempo_confidence']}), key={analysis['key'] or '?'} "
        f"(conf {analysis['key_confidence']}), beats={n_beats}, "
        f"sections={n_sections}, dur={analysis['duration_sec']}s"
    )
    print(f"[run] stems: {', '.join(sep['stems'].values())}")
    print(f"[run] analysis: {out_path}")


if __name__ == "__main__":
    main()
