#!/usr/bin/env python
"""render.py — full song bounce (Step 6).

new_mix = (untouched stems) + (one or more parts whose sound is replaced by your
raw sample at EVERY occurrence). Edges crossfade into the original stem so swaps
don't click; the result is summed and encoded to mp3 + wav.

The sample is placed RAW — no time-stretch, no pitch-shift, no grid-quantize — so
the uploaded sound bite is preserved exactly (see snap.snap). A sample may ring
past its bar; for a messy medley that overlap is the point.

Two entry points:
  - render(song_dir, label, sample=...)            one part -> one sound
  - render_medley(song_dir, placements, ...)       many parts -> many sounds

    python render.py "output/Drake high fives" B3
    python render.py "output/Drake high fives" B3 --sample my_bass.wav
    python render.py "output/Drake high fives" --medley B3=horn.wav O1=vox.wav
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

import dsp
from snap import build_snap_targets, load_analysis, snap

STEMS = ("drums", "bass", "vocals", "other")


def _read_stereo(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data.astype(np.float32), sr


def _mono_to_stereo(y: np.ndarray) -> np.ndarray:
    return np.column_stack([y, y]).astype(np.float32)


def _place(dst: np.ndarray, start: int, repl: np.ndarray, xf: int) -> None:
    """Sum `repl` into `dst` at `start`, crossfading `xf` samples at the leading edge
    from the original stem into the sample so the swap is click-free. The sample is
    allowed to ring past its slot (we don't trim it back into the original) — that
    overlap is what makes a medley messy."""
    end = min(start + len(repl), len(dst))
    n = end - start
    if n <= 0:
        return
    repl = repl[:n].copy()
    seg = dst[start:end]                     # original stem under the slot
    xf = min(xf, n)
    if xf > 0:
        r = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
        repl[:xf] = seg[:xf] * (1 - r) + repl[:xf] * r          # original -> sample
    dst[start:end] = repl


def _occurrences(targets: list["SnapTarget"]) -> list[dict]:
    """Collapse the per-bar SnapTargets of one label into one span per occurrence:
    `{start_sec, end_sec}`. Bars inside an occurrence are contiguous, so the span runs
    from the first bar's start to the last bar's end. Used by confined placement to
    drop one sound per occurrence (instead of one per bar)."""
    by_occ: dict[int, list] = {}
    for t in targets:
        by_occ.setdefault(t.occurrence_id, []).append(t)
    spans = []
    for _id, ts in sorted(by_occ.items()):
        ts.sort(key=lambda t: t.start_sec)
        spans.append({"start_sec": ts[0].start_sec,
                      "end_sec": ts[-1].start_sec + ts[-1].duration_sec})
    return spans


def _load_sample(song_dir: Path, analysis: dict, stem: str, label: str,
                 sample: str | Path | None, sr: int) -> np.ndarray:
    """Mono sample for a placement, resampled to the stem's `sr` so its pitch and
    speed are preserved exactly. With no external sample, the stand-in is the label's
    own representative bar cut from the stem."""
    if sample is not None:
        data, ssr = sf.read(str(Path(sample)), always_2d=True)
        y = dsp._as_mono_f32(data)
        if ssr != sr and y.size:
            import librosa
            y = librosa.resample(y, orig_sr=ssr, target_sr=sr).astype(np.float32)
        return y
    bars = analysis["bars"]
    target_wav = song_dir / "stems" / f"{stem}.wav"
    rep_bar = next(p["representative_bar"] for p in analysis["layers"][stem]
                   if p["label"] == label)
    s0, s1 = int(bars[rep_bar]["start_sec"] * sr), int(bars[rep_bar]["end_sec"] * sr)
    d, _ = sf.read(str(target_wav), start=s0, stop=s1, always_2d=True)
    return dsp._as_mono_f32(d)


def render_medley(song_dir: str | Path, placements: list[dict],
                  out_dir: str | Path | None = None, crossfade_ms: float = 8.0,
                  mp3: bool = True, out_name: str = "medley",
                  write_stems: bool = False, confine: bool = False) -> dict:
    """Bounce one song with many parts replaced at once.

    placements: [{"label": "B3", "sample": "horn.wav" | None}, ...]. Each placement
    drops its raw sample at every occurrence of that part. Parts may live in
    different stems (or several in the same stem); stems with no placement are kept
    untouched. The mix is summed and peak-normalized.

    write_stems: also write each *edited* stem to <out>/stems/<stem>.wav and return
    their paths under "stems" — lets the visualizer load the medley back into its
    per-stem timeline playback (so the main Play button plays it and mute/solo work),
    instead of only playing the flat bounce in a separate audio element.

    confine: place the sample ONCE per occurrence, trimmed to that occurrence's length,
    so the sound fills exactly the part it replaces — no per-bar retrigger and no
    ringing past into the next part. This is the mobile (jawnsplitter) contract: the
    mix at any instant is just the layers active there. Default off keeps the desktop
    "messy medley" behaviour (raw sample dropped at every bar, free to ring past).
    """
    song_dir = Path(song_dir)
    analysis = load_analysis(song_dir)
    stem_dir = song_dir / "stems"
    if not placements:
        raise SystemExit("no placements given")

    # Resolve each placement -> (stem, targets, sample) up front so a bad label fails fast.
    plan = []
    for pl in placements:
        label = pl["label"]
        targets = build_snap_targets(analysis, label)
        if not targets:
            raise SystemExit(f"no targets for label {label}")
        stem = targets[0].stem
        target_wav = stem_dir / f"{stem}.wav"
        if not target_wav.exists():
            raise SystemExit(f"missing stem: {target_wav}")
        plan.append({"label": label, "stem": stem, "targets": targets,
                     "sample": pl.get("sample")})

    sr = sf.info(str(stem_dir / f"{plan[0]['stem']}.wav")).samplerate

    # Apply placements into per-stem editable copies (one copy per stem touched).
    edited: dict[str, np.ndarray] = {}
    xf = int(crossfade_ms / 1000.0 * sr)
    used = []
    for entry in plan:
        stem = entry["stem"]
        if stem not in edited:
            edited[stem], _ = _read_stereo(stem_dir / f"{stem}.wav")
        new_stem = edited[stem]
        sample_y = _load_sample(song_dir, analysis, stem, entry["label"],
                                entry["sample"], sr)
        if confine:
            # One placement per occurrence, trimmed to the occurrence's length: the
            # sound fills exactly the part it replaces (no mid-occurrence restart, no
            # ring-past). level-match once (lufs is per-stem, same for every slot).
            snapped = snap(sample_y, sr, entry["targets"][0])
            spans = _occurrences(entry["targets"])
            for occ in spans:
                n = int(round((occ["end_sec"] - occ["start_sec"]) * sr))
                piece = dsp.edge_fade(snapped[:n], sr, 5.0) if n > 0 else snapped[:0]
                start = int(round(occ["start_sec"] * sr))
                _place(new_stem, start, _mono_to_stereo(piece), xf)
            slots = len(spans)
        else:
            for t in entry["targets"]:
                snapped = snap(sample_y, sr, t)
                start = int(round(t.start_sec * sr))
                _place(new_stem, start, _mono_to_stereo(snapped), xf)
            slots = len(entry["targets"])
        used.append({
            "label": entry["label"], "stem": stem, "slots": slots,
            "source": str(entry["sample"]) if entry["sample"] else "rep bar",
        })

    # Sum: edited stems where touched, original everywhere else.
    layers = []
    for s in STEMS:
        wav = stem_dir / f"{s}.wav"
        if not wav.exists():
            continue
        layers.append(edited[s] if s in edited else _read_stereo(wav)[0])
    n = min(len(x) for x in layers)
    mix = np.zeros((n, 2), dtype=np.float32)
    for x in layers:
        mix += x[:n]

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.999:
        mix *= 0.999 / peak

    out = Path(out_dir) if out_dir else (song_dir / "render" / out_name)
    out.mkdir(parents=True, exist_ok=True)
    wav_path = out / f"bounce_{out_name}.wav"
    sf.write(str(wav_path), mix, sr, subtype="PCM_16")
    paths = {"wav": str(wav_path)}

    # Per-stem edited copies (only the touched stems) for timeline playback.
    stems_out: dict[str, str] = {}
    if write_stems:
        stem_out_dir = out / "stems"
        stem_out_dir.mkdir(parents=True, exist_ok=True)
        for s, arr in edited.items():
            sp = stem_out_dir / f"{s}.wav"
            sf.write(str(sp), arr[:n], sr, subtype="PCM_16")
            stems_out[s] = str(sp)
    if mp3:
        mp3_path = out / f"bounce_{out_name}.mp3"
        try:
            sf.write(str(mp3_path), mix, sr, format="MP3")
            paths["mp3"] = str(mp3_path)
        except Exception as exc:  # noqa: BLE001 - mp3 optional; wav already written
            paths["mp3_error"] = str(exc)

    return {
        "name": out_name, "sr": sr,
        "placements": used,
        "slots_replaced": sum(u["slots"] for u in used),
        "duration_sec": round(n / sr, 2),
        "peak_before_norm": round(peak, 3),
        "paths": paths,
        "stems": stems_out,
    }


def render(song_dir: str | Path, label: str, sample: str | Path | None = None,
           out_dir: str | Path | None = None, crossfade_ms: float = 8.0,
           mp3: bool = True) -> dict:
    """Single-part bounce: replace `label`'s sound with the raw sample at every slot.

    Thin wrapper over render_medley with one placement. Kept for the CLI, the server's
    /api/render, and the snap-preview path.
    """
    info = render_medley(song_dir, [{"label": label, "sample": sample}],
                         out_dir=out_dir, crossfade_ms=crossfade_ms, mp3=mp3,
                         out_name=label)
    # Back-compat surface: callers expect single-part fields.
    p = info["placements"][0]
    info.update(label=p["label"], stem=p["stem"], source=p["source"])
    return info


def _parse_medley(args: list[str]) -> list[dict]:
    """Parse `LABEL[=sample.wav]` tokens into placements."""
    out = []
    for tok in args:
        if "=" in tok:
            label, sample = tok.split("=", 1)
            out.append({"label": label, "sample": sample or None})
        else:
            out.append({"label": tok, "sample": None})
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Bounce a song with a part's sound replaced (raw).")
    ap.add_argument("song_dir", help="song dir with analysis.json + stems/")
    ap.add_argument("label", nargs="?", help="single part label, e.g. B3")
    ap.add_argument("--sample", default=None, help="external sample wav (default: the label's rep bar)")
    ap.add_argument("--medley", nargs="+", metavar="LABEL[=sample.wav]",
                    help="replace several parts at once, e.g. --medley B3=horn.wav O1=vox.wav")
    ap.add_argument("--crossfade-ms", type=float, default=8.0)
    ap.add_argument("--no-mp3", action="store_true")
    args = ap.parse_args()

    if args.medley:
        info = render_medley(args.song_dir, _parse_medley(args.medley),
                             crossfade_ms=args.crossfade_ms, mp3=not args.no_mp3)
        parts = ", ".join(f"{u['label']}({u['stem']})×{u['slots']}" for u in info["placements"])
        print(f"medley: {parts} — {info['slots_replaced']} slots, {info['duration_sec']}s")
    elif args.label:
        info = render(args.song_dir, args.label, sample=args.sample,
                      crossfade_ms=args.crossfade_ms, mp3=not args.no_mp3)
        print(f"bounced {info['label']} ({info['stem']}): replaced {info['slots_replaced']} slots, "
              f"{info['duration_sec']}s")
        print(f"  source: {info['source']}  peak(pre-norm)={info['peak_before_norm']}")
    else:
        ap.error("give a single LABEL or --medley LABEL[=sample] ...")

    for k, v in info["paths"].items():
        print(f"  {k}: {v}")
