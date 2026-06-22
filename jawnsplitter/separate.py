"""Stem separation via the demucs CLI.

Validation harness only — production moves this to serverless GPU later.
Wraps `demucs` (4-stem htdemucs by default), with an MPS->CPU fallback for the
known intermittent Apple-Silicon garbled-output issue, and normalizes the
output into a clean `output/<song>/stems/` layout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "drums", "bass", "other")
STEMS_6S = ("vocals", "drums", "bass", "other", "guitar", "piano")


def _stem_names(model: str) -> tuple[str, ...]:
    return STEMS_6S if model == "htdemucs_6s" else STEMS


def _has_nan(wav_path: Path) -> bool:
    """Cheap garbled-output check: NaN/inf or all-silence in a stem."""
    try:
        data, _ = sf.read(str(wav_path), dtype="float32")
    except Exception:
        return True
    if data.size == 0:
        return True
    if not np.all(np.isfinite(data)):
        return True
    return False


def _run_demucs(audio_path: Path, raw_out: Path, device: str, model: str) -> None:
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-d", device,
        "-o", str(raw_out),
        str(audio_path),
    ]
    print(f"[separate] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def separate(
    audio_path: str | Path,
    out_root: str | Path,
    device: str = "mps",
    model: str = "htdemucs",
) -> dict:
    """Separate ``audio_path`` into stems under ``out_root/<song>/stems/``.

    Returns ``{"song": str, "stems": {name: path}, "duration_sec": float}``.
    Falls back to CPU if MPS errors or produces non-finite output.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    song = audio_path.stem
    out_root = Path(out_root)
    song_dir = out_root / song
    stems_dir = song_dir / "stems"
    raw_out = out_root / "_raw"
    stems_dir.mkdir(parents=True, exist_ok=True)

    names = _stem_names(model)

    def attempt(dev: str) -> dict[str, Path]:
        # demucs writes to raw_out/<model>/<song>/<stem>.wav
        src_dir = raw_out / model / song
        if src_dir.exists():
            shutil.rmtree(src_dir)
        _run_demucs(audio_path, raw_out, dev, model)
        produced = {}
        for name in names:
            src = src_dir / f"{name}.wav"
            if not src.exists():
                raise RuntimeError(f"demucs did not produce {src}")
            dst = stems_dir / f"{name}.wav"
            shutil.copy2(src, dst)
            produced[name] = dst
        return produced

    try:
        stems = attempt(device)
        if device == "mps" and any(_has_nan(p) for p in stems.values()):
            print("[separate] MPS output looked garbled (NaN/empty); retrying on CPU.")
            stems = attempt("cpu")
    except subprocess.CalledProcessError as exc:
        if device != "cpu":
            print(f"[separate] device '{device}' failed ({exc}); retrying on CPU.")
            stems = attempt("cpu")
        else:
            raise

    info = sf.info(str(audio_path))
    duration_sec = info.frames / info.samplerate

    return {
        "song": song,
        "song_dir": str(song_dir),
        "stems": {name: str(p) for name, p in stems.items()},
        "duration_sec": float(duration_sec),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Separate an audio file into stems.")
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="output")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--model", default="htdemucs", choices=["htdemucs", "htdemucs_6s"])
    args = ap.parse_args()

    result = separate(args.audio, args.out, device=args.device, model=args.model)
    print(result)
