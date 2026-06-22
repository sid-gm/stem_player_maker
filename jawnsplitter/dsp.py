#!/usr/bin/env python
"""dsp.py — the operator toolbox (the "hands" of the snap engine).

Thin, librosa-backed wrappers for the four correction operators. Everything works
on a 1-D mono float array `y` at sample rate `sr` (the librosa.effects convention).
The snap pipeline (snap.py Step 4) chains these; exact sample-accurate length is the
caller's job (pad/trim) — these just do the DSP.

    time_stretch(y, sr, ratio)      # ratio = out_len / in_len  (1.125 -> 1.6s->1.8s)
    pitch_shift(y, sr, semitones)   # +/- semitones
    gain_to_lufs(y, sr, target)     # scale so integrated loudness == target LUFS
    edge_fade(y, sr, ms)            # linear fade in/out so loops don't click

Pitfall: phase-vocoder time-stretch smears transients — gate stretching to
non-drum layers (drums trigger one-shots on the onset grid instead).
"""

from __future__ import annotations

import numpy as np
import librosa

try:
    import pyloudnorm as pyln
    _HAVE_PYLN = True
except ImportError:
    _HAVE_PYLN = False


def _as_mono_f32(y: np.ndarray) -> np.ndarray:
    """1-D float32 view. Collapses (samples, ch) or (ch, samples) to mono."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        ax = 0 if y.shape[0] <= y.shape[1] else 1  # channels = the short axis
        y = y.mean(axis=ax)
    return np.ascontiguousarray(y, dtype=np.float32)


def time_stretch(y: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Stretch `y` to `ratio` x its length. ratio>1 = longer/slower, <1 = shorter.

    `sr` is unused by the phase vocoder but kept in the signature for uniformity.
    """
    y = _as_mono_f32(y)
    if ratio <= 0:
        raise ValueError(f"ratio must be > 0, got {ratio}")
    if abs(ratio - 1.0) < 1e-6 or len(y) == 0:
        return y
    # librosa rate>1 SHORTENS (out_len ~= in_len/rate); we want out_len = ratio*in_len.
    return librosa.effects.time_stretch(y, rate=1.0 / ratio).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by `semitones` (length-preserving)."""
    y = _as_mono_f32(y)
    if abs(semitones) < 1e-6 or len(y) == 0:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones)).astype(np.float32)


def measure_lufs(y: np.ndarray, sr: int) -> float | None:
    """Integrated loudness (LUFS) — same method as analyze._loudness_lufs.

    pyloudnorm when available (needs >=~400ms of audio), else an RMS-dBFS proxy.
    """
    y = _as_mono_f32(y)
    if y.size == 0:
        return None
    if _HAVE_PYLN:
        try:
            lufs = float(pyln.Meter(sr).integrated_loudness(y))
            if np.isfinite(lufs):
                return lufs
        except Exception:
            pass  # too short / silent -> fall back to RMS proxy
    rms = float(np.sqrt(np.mean(y**2)))
    return float(20.0 * np.log10(rms)) if rms > 0 else None


def gain_to_lufs(y: np.ndarray, sr: int, target: float) -> np.ndarray:
    """Apply a linear gain so `y`'s integrated loudness == `target` LUFS.

    Silent/unmeasurable input is returned unchanged. Gain is a pure scalar (loudness
    is linear in dB), so no re-measure loop is needed.
    """
    y = _as_mono_f32(y)
    cur = measure_lufs(y, sr)
    if cur is None:
        return y
    gain = float(10.0 ** ((target - cur) / 20.0))
    return (y * gain).astype(np.float32)


def edge_fade(y: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    """Return a copy of `y` with a linear fade in/out of `ms` each end."""
    y = _as_mono_f32(y).copy()
    n = int(ms / 1000.0 * sr)
    if n <= 0 or 2 * n >= len(y):
        return y
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def _selftest() -> None:
    """Step 3 acceptance: round-trip + measurable length/pitch/loudness changes."""
    sr = 22050
    t = np.arange(int(1.6 * sr)) / sr
    y = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)  # 1.6s A3 tone
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    # round-trip: every op returns finite mono audio of sane length
    for nm, out in [("time_stretch", time_stretch(y, sr, 1.0)),
                    ("pitch_shift", pitch_shift(y, sr, 0.0)),
                    ("gain_to_lufs", gain_to_lufs(y, sr, -20.0)),
                    ("edge_fade", edge_fade(y, sr, 5.0))]:
        check(f"{nm} round-trip", out.ndim == 1 and np.all(np.isfinite(out)),
              f"shape={out.shape}")

    # stretch 1.6s -> 1.8s
    st = time_stretch(y, sr, 1.8 / 1.6)
    check("stretch 1.6->1.8s", abs(len(st) / sr - 1.8) < 0.02, f"{len(st)/sr:.3f}s")

    # +2 semitones raises spectral centroid
    c0 = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    sh = pitch_shift(y, sr, 2.0)
    c1 = float(librosa.feature.spectral_centroid(y=sh, sr=sr).mean())
    check("+2 semis raises centroid", c1 > c0 * 1.05, f"{c0:.0f}Hz -> {c1:.0f}Hz")

    # gain hits the LUFS target
    g = gain_to_lufs(y, sr, -23.0)
    meas = measure_lufs(g, sr)
    check("gain_to_lufs(-23)", meas is not None and abs(meas + 23.0) < 0.5,
          f"measured {meas:.2f} LUFS")

    # edge_fade zeroes the endpoints
    f = edge_fade(y, sr, 5.0)
    check("edge_fade endpoints~0", abs(f[0]) < 1e-3 and abs(f[-1]) < 1e-3,
          f"first={f[0]:.4f} last={f[-1]:.4f}")

    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    _selftest()
