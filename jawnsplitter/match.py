#!/usr/bin/env python
"""match.py — the matchmaker engine: rank a sound against a song's parts.

Drop a sound, and for a processed song we rank every part (drums/bass/vocals/
other patterns from analysis.json) by how well the sound fits that slot — best
to worst. For a sound longer than a slot we also find the *best sub-region* of
the sound to use (its `best_offset`).

How it works
------------
Every sound — the candidate and each part's representative bar — is reduced to a
small feature vector built from a per-frame timeline:

    [ perc/pitched ratio, spectral centroid, rolloff, flatness, onset envelope,
      rms, chroma x12, mfcc x13, onset density ]                       (D = 32)

Both queries are the same machine:
  * whole sound  -> aggregate all frames -> one vector
  * sub-region   -> slide a slot-width, onset-snapped window across the sound,
                    aggregate each window, keep the best-scoring offset.

Scoring is a per-dimension-normalized (z-scored against the song's own parts),
weighted distance tuned for ROLE / SLOT fit rather than which stem a sound came
from — so cross-stem creative swaps (a vocal chop into a drum lane) can surface.

Part vectors are derived from each pattern's representative bar (timing already
in analysis.json) and cached in a sidecar `output/<song>/match.json`, so repeat
calls only extract the candidate sound and do arithmetic.

    python match.py "output/E85 DT" uploads/some_sound.wav

Reuses librosa + dsp.py; no new dependencies, CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

import dsp

# ── feature space ──────────────────────────────────────────────────────────────
MATCH_SR = 22050          # everything resampled to this mono rate before features
HOP = 512                 # ~23 ms frames at 22.05 kHz
N_FFT = 2048
ENGINE = 1                # bump to invalidate cached match.json on layout change

N_SCALAR = 6              # perc_ratio, log-centroid, log-rolloff, flatness, onset_env, rms
N_CHROMA = 12
N_MFCC = 13
D = N_SCALAR + N_CHROMA + N_MFCC + 1   # + onset_density  => 32

# Role/slot-fit weighting: heavy on what makes a sound *function* in a slot
# (percussive-vs-pitched, brightness, busyness), light on raw timbre detail, and
# loudness near-zero (every placement is gain-matched at render anyway).
WEIGHTS = np.array(
    [2.6, 1.5, 1.0, 1.6, 1.1, 0.25]        # scalars
    + [0.12] * N_CHROMA                     # chroma (pitch-class profile)
    + [0.10] * N_MFCC                       # mfcc (timbre catch-all)
    + [1.6],                                # onset density
    dtype=np.float64,
)


# ── feature extraction ─────────────────────────────────────────────────────────
def _frame_features(y: np.ndarray, sr: int = MATCH_SR):
    """Per-frame feature matrix for a mono signal.

    Returns (mat[n, D-1], n_frames, onset_times[sec]). The onset-density dim is
    appended per-window at aggregation time (it's a rate, not a per-frame value).
    """
    y = dsp._as_mono_f32(y)
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))
    eps = 1e-9

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr)                       # (12, n)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)  # (13, n)

    # percussive-vs-pitched: harmonic/percussive energy split, per frame
    harm, perc = librosa.effects.hpss(y)
    rms_p = librosa.feature.rms(y=perc, frame_length=N_FFT, hop_length=HOP)[0]
    rms_h = librosa.feature.rms(y=harm, frame_length=N_FFT, hop_length=HOP)[0]
    perc_ratio = rms_p / (rms_p + rms_h + eps)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    n = min(len(rms), len(cent), len(roll), len(flat),
            chroma.shape[1], mfcc.shape[1], len(perc_ratio), len(onset_env))

    def t(a):
        return np.asarray(a)[..., :n]

    scal = np.vstack([
        t(perc_ratio),
        np.log1p(t(cent)),
        np.log1p(t(roll)),
        t(flat),
        t(onset_env),
        t(rms),
    ])
    mat = np.vstack([scal, t(chroma), t(mfcc)]).T                               # (n, D-1)

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env[:n], sr=sr, hop_length=HOP)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)
    return mat, n, np.asarray(onset_times, dtype=np.float64)


def _aggregate(mat: np.ndarray, onset_times: np.ndarray, a_sec: float, b_sec: float) -> np.ndarray:
    """Pool frames in [a_sec, b_sec) into one vector; append the onset rate."""
    fa = int(librosa.time_to_frames(a_sec, sr=MATCH_SR, hop_length=HOP))
    fb = int(librosa.time_to_frames(b_sec, sr=MATCH_SR, hop_length=HOP))
    fa = max(0, min(fa, mat.shape[0] - 1))
    fb = max(fa + 1, min(fb, mat.shape[0]))
    v = mat[fa:fb].mean(axis=0)
    dur = max(1e-3, b_sec - a_sec)
    od = float(np.sum((onset_times >= a_sec) & (onset_times < b_sec))) / dur
    return np.concatenate([v, [od]])


def _z(v, mean, std):
    return (v - mean) / std


def _dist(a_z, b_z) -> float:
    return float(np.sqrt(np.sum(WEIGHTS * (a_z - b_z) ** 2)))


def _nickname(stem: str, raw: np.ndarray) -> str:
    """A light human label for a part (analysis.json has none). Flavor only."""
    bright = raw[1]            # log1p(centroid)
    onset_density = raw[-1]
    if stem == "drums":
        return "groove" if onset_density > 6 else "beat"
    if stem == "bass":
        return "sub line" if bright < np.log1p(900) else "bass line"
    if stem == "vocals":
        return "the hook" if onset_density > 4 else "vocal"
    return "synth stab" if bright > np.log1p(2500) else "melody"


# ── part-vector cache (sidecar match.json) ─────────────────────────────────────
def _build_part_cache(song_dir: Path, analysis: dict) -> dict:
    bars = analysis.get("bars") or []
    layers = analysis.get("layers") or {}
    parts_out, vecs = [], []

    for stem, plist in layers.items():
        wav = song_dir / "stems" / f"{stem}.wav"
        if not wav.exists():
            continue
        sr0 = sf.info(str(wav)).samplerate
        for p in plist or []:
            bi = p.get("representative_bar")
            if bi is None or not (0 <= bi < len(bars)):
                continue
            st, en = float(bars[bi]["start_sec"]), float(bars[bi]["end_sec"])
            if en - st < 0.05:
                continue
            y, _ = sf.read(str(wav), start=int(st * sr0), stop=int(en * sr0), always_2d=False)
            y = dsp._as_mono_f32(y)
            if sr0 != MATCH_SR:
                y = librosa.resample(y, orig_sr=sr0, target_sr=MATCH_SR)
            mat, _n, onsets = _frame_features(y)
            vec = _aggregate(mat, onsets, 0.0, len(y) / MATCH_SR)
            vecs.append(vec)
            parts_out.append({
                "label": p["label"],
                "stem": stem,
                "count": int(p.get("occurrence_count") or 1),
                "slot_sec": round(en - st, 4),
                "vec": vec.tolist(),
            })

    if not vecs:
        raise SystemExit("no part vectors could be built (missing stems or layers?)")

    V = np.array(vecs)
    mean = V.mean(axis=0)
    std = V.std(axis=0)
    std[std < 1e-6] = 1.0
    for po in parts_out:
        po["name"] = _nickname(po["stem"], np.array(po["vec"]))

    return {
        "engine": ENGINE,
        "analysis_mtime": (song_dir / "analysis.json").stat().st_mtime,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "parts": parts_out,
    }


def _load_or_build_cache(song_dir: Path, analysis: dict) -> dict:
    cache_path = song_dir / "match.json"
    if cache_path.exists():
        try:
            c = json.loads(cache_path.read_text())
            fresh = (c.get("engine") == ENGINE
                     and abs(c.get("analysis_mtime", -1)
                             - (song_dir / "analysis.json").stat().st_mtime) < 1e-6)
            if fresh:
                return c
        except Exception:  # noqa: BLE001 - rebuild on any cache problem
            pass
    cache = _build_part_cache(song_dir, analysis)
    cache_path.write_text(json.dumps(cache))
    return cache


# ── windowed sub-region search ─────────────────────────────────────────────────
def _best_window(cmat, conset, cand_len, slot, part_z, mean, std):
    """Best slot-width window of the candidate against one part. Returns (a, b, dist)."""
    if cand_len <= slot + 1e-3:
        v = _z(_aggregate(cmat, conset, 0.0, cand_len), mean, std)
        return 0.0, cand_len, _dist(v, part_z)

    starts = {0.0, cand_len - slot}
    for ot in conset:                                   # onset-snapped cuts
        if 0.0 <= ot <= cand_len - slot:
            starts.add(round(float(ot), 3))
    step = max(0.1, slot / 2)                           # coarse grid fallback
    x = 0.0
    while x <= cand_len - slot:
        starts.add(round(x, 3))
        x += step

    best = None
    for a in sorted(starts)[:64]:                       # cap candidate windows
        b = a + slot
        d = _dist(_z(_aggregate(cmat, conset, a, b), mean, std), part_z)
        if best is None or d < best[2]:
            best = (a, b, d)
    return best


def recommend(song_dir, sound_path, n_best: int = 5, n_worst: int = 3) -> dict:
    """Rank a sound against a song's parts: best fits + a worst-fit 'chaos' tail.

    Returns {sound_len_sec, recommendations:[{label, stem, name, count, slot_sec,
    tier, score, best_offset:{start_sec, end_sec}}]}, best tier first.
    """
    song_dir = Path(song_dir)
    analysis = json.loads((song_dir / "analysis.json").read_text())
    cache = _load_or_build_cache(song_dir, analysis)
    mean = np.array(cache["mean"])
    std = np.array(cache["std"])

    y, _ = librosa.load(str(sound_path), sr=MATCH_SR, mono=True)
    cand_len = max(0.05, len(y) / MATCH_SR)
    cmat, _cn, conset = _frame_features(y)

    # Rank the parts the studio timeline actually shows (repeated patterns); fall back
    # to every part only when there aren't enough repeated ones to fill best+worst.
    pool = [p for p in cache["parts"] if int(p.get("count", 1)) >= 2]
    if len(pool) < n_best + n_worst:
        pool = cache["parts"]

    scored = []
    for po in pool:
        part_z = _z(np.array(po["vec"]), mean, std)
        slot = float(po["slot_sec"])
        a, b, dist = _best_window(cmat, conset, cand_len, slot, part_z, mean, std)
        if cand_len < slot:                             # mild penalty: sound can't fill the slot
            dist *= 1.0 + 0.5 * (1.0 - cand_len / slot)
        scored.append({**po, "dist": dist,
                       "off": [round(a, 3), round(min(b, cand_len), 3)]})

    scored.sort(key=lambda x: x["dist"])
    dmin = scored[0]["dist"]
    dmax = scored[-1]["dist"] or 1.0

    best = scored[:n_best]
    chosen = {id(x) for x in best}
    worst = [x for x in scored if id(x) not in chosen][-n_worst:]

    out = []
    for tier, lst in (("best", best), ("worst", worst)):
        for x in lst:
            # display-only fit 0..1 (UI ranks/tiers; it does not show a number)
            score = round(1.0 - (x["dist"] - dmin) / (dmax - dmin + 1e-9), 3)
            out.append({
                "label": x["label"], "stem": x["stem"], "name": x["name"],
                "count": x["count"], "slot_sec": x["slot_sec"],
                "tier": tier, "score": score,
                "best_offset": {"start_sec": x["off"][0], "end_sec": x["off"][1]},
            })
    return {"sound_len_sec": round(cand_len, 3), "recommendations": out}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rank a sound against a song's parts.")
    ap.add_argument("song_dir", help="song dir containing analysis.json + stems/")
    ap.add_argument("sound", help="path to a candidate sound (wav/mp3/...)")
    ap.add_argument("--best", type=int, default=5)
    ap.add_argument("--worst", type=int, default=3)
    args = ap.parse_args()

    res = recommend(args.song_dir, args.sound, n_best=args.best, n_worst=args.worst)
    print(f"sound: {args.sound}  ({res['sound_len_sec']}s)")
    for r in res["recommendations"]:
        off = r["best_offset"]
        flag = "  💀" if r["tier"] == "worst" else ""
        print(f"  [{r['tier']:>5}] {r['stem']:>6} {r['label']:>4} "
              f"{r['name']:<10} ×{r['count']:<2} slot {r['slot_sec']:.2f}s  "
              f"fit {r['score']:.2f}  uses {off['start_sec']:.2f}-{off['end_sec']:.2f}s{flag}")
