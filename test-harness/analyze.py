"""Musical analysis with librosa.

Runs global properties on the full mix and the step pattern on the drum stem,
emitting a dict that maps directly to the `analysis.json` schema in
backend-api-design.md §2.2 (same field names).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

try:
    import pyloudnorm as pyln
    _HAVE_PYLN = True
except Exception:  # pragma: no cover - optional dep
    _HAVE_PYLN = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - optional dep
    _HAVE_SKLEARN = False

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

BEATS_PER_BAR = 4  # assume 4/4 for v1
STEPS_PER_BAR = 16


def _as_scalar(x) -> float:
    arr = np.asarray(x).ravel()
    return float(arr[0]) if arr.size else 0.0


def estimate_key(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Krumhansl-Schmuckler key estimate. Returns ('C major', confidence)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    if profile.sum() <= 0:
        return "", 0.0
    profile = profile / profile.sum()

    scores = []
    for tonic in range(12):
        maj = np.roll(_KS_MAJOR, tonic)
        minr = np.roll(_KS_MINOR, tonic)
        scores.append((np.corrcoef(profile, maj)[0, 1], f"{NOTE_NAMES[tonic]} major"))
        scores.append((np.corrcoef(profile, minr)[0, 1], f"{NOTE_NAMES[tonic]} minor"))

    scores.sort(key=lambda s: s[0], reverse=True)
    best_corr, best_key = scores[0]
    second_corr = scores[1][0]
    # Confidence: separation between the top two candidates, mapped to 0..1.
    confidence = float(np.clip((best_corr - second_corr) * 2.0, 0.0, 1.0))
    return best_key, confidence


def _tempo_confidence(y: np.ndarray, sr: int, tempo_bpm: float, hop_length: int) -> float:
    """Strength of the onset-envelope autocorrelation at the detected period."""
    if tempo_bpm <= 0:
        return 0.0
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if oenv.size < 2:
        return 0.0
    ac = librosa.autocorrelate(oenv)
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    period_frames = (60.0 / tempo_bpm) * sr / hop_length
    lag = int(round(period_frames))
    if lag <= 0 or lag >= ac.size:
        return 0.0
    return float(np.clip(ac[lag], 0.0, 1.0))


def _loudness_lufs(path: str | Path) -> float | None:
    """Integrated loudness (LUFS) via pyloudnorm, or an RMS-dBFS proxy."""
    data, sr = librosa.load(str(path), sr=None, mono=True)
    if data.size == 0:
        return None
    if _HAVE_PYLN:
        meter = pyln.Meter(sr)
        try:
            lufs = float(meter.integrated_loudness(data))
            # pyloudnorm returns -inf for (near-)silent stems; keep JSON valid.
            return lufs if np.isfinite(lufs) else None
        except Exception:
            pass
    rms = float(np.sqrt(np.mean(data**2)))
    if rms <= 0:
        return None
    return float(20.0 * np.log10(rms))  # dBFS proxy, not true LUFS


def _build_beat_grid(beat_times: np.ndarray) -> list[dict]:
    grid = []
    for i, t in enumerate(beat_times):
        grid.append(
            {
                "beat_index": i,
                "time_sec": round(float(t), 4),
                "bar_number": i // BEATS_PER_BAR,
                "beat_in_bar": i % BEATS_PER_BAR,
                "is_downbeat": (i % BEATS_PER_BAR) == 0,
            }
        )
    return grid


def _build_bars(beat_times: np.ndarray, duration_sec: float) -> list[dict]:
    bars = []
    for bar_idx, start in enumerate(range(0, len(beat_times), BEATS_PER_BAR)):
        chunk = beat_times[start : start + BEATS_PER_BAR]
        if len(chunk) == 0:
            continue
        start_sec = float(chunk[0])
        # End at the next bar's first beat if it exists, else last beat / duration.
        next_start = start + BEATS_PER_BAR
        if next_start < len(beat_times):
            end_sec = float(beat_times[next_start])
        else:
            end_sec = float(min(duration_sec, chunk[-1] + (chunk[-1] - chunk[0]) / max(len(chunk) - 1, 1)))
        bars.append(
            {
                "bar_number": bar_idx,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "beat_indices": list(range(start, min(next_start, len(beat_times)))),
            }
        )
    return bars


def _detect_sections(y: np.ndarray, sr: int, duration_sec: float) -> list[dict]:
    """Agglomerative segmentation on stacked chroma+MFCC, with cluster labels."""
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    feats = np.vstack([librosa.util.normalize(chroma, axis=0),
                       librosa.util.normalize(mfcc, axis=0)])

    n_frames = feats.shape[1]
    if n_frames < 4:
        return [{"index": 0, "label": "A", "start_sec": 0.0,
                 "end_sec": round(duration_sec, 4)}]

    # Aim for one section ~ every 12s, clamped to a sane range.
    k = int(np.clip(round(duration_sec / 12.0), 2, 12))
    k = min(k, n_frames)
    bounds = librosa.segment.agglomerative(feats, k)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)
    edges = np.concatenate([bound_times, [duration_sec]])

    # Label each segment by clustering its mean feature vector.
    seg_means = []
    for i in range(len(bounds)):
        f0, f1 = bounds[i], (bounds[i + 1] if i + 1 < len(bounds) else n_frames)
        seg_means.append(feats[:, f0:f1].mean(axis=1) if f1 > f0 else feats[:, f0])
    labels = _cluster_labels(np.array(seg_means))

    sections = []
    for i in range(len(bounds)):
        sections.append(
            {
                "index": i,
                "label": labels[i],
                "start_sec": round(float(edges[i]), 4),
                "end_sec": round(float(edges[i + 1]), 4),
            }
        )
    return sections


def _cluster_labels(seg_means: np.ndarray) -> list[str]:
    """Cluster segment feature means into letter labels (A, B, C, ...)."""
    n = len(seg_means)
    n_labels = int(np.clip(n // 2, 1, 4)) if n > 1 else 1
    if n_labels <= 1 or n <= 1:
        return ["A"] * n
    try:
        from scipy.cluster.vq import kmeans2

        whitened = seg_means / (seg_means.std(axis=0) + 1e-9)
        _, idx = kmeans2(whitened, n_labels, minit="++", seed=0)
    except Exception:
        idx = np.arange(n) % n_labels
    return [chr(ord("A") + int(i)) for i in idx]


def analyze(
    audio_path: str | Path,
    stem_paths: dict[str, str] | None = None,
    song_id: str = "local_test",
    engine_version: str = "htdemucs-4.0+librosa",
    drum_dist: float | None = None,
    melodic_dist: float | None = None,
) -> dict:
    """Analyze the full mix (and drum stem) into the analysis.json schema."""
    audio_path = Path(audio_path)
    stem_paths = stem_paths or {}

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 512
    tempo_raw, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo_bpm = round(_as_scalar(tempo_raw), 2)
    tempo_conf = round(_tempo_confidence(y, sr, tempo_bpm, hop_length), 4)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    key, key_conf = estimate_key(y, sr)

    beat_grid = _build_beat_grid(beat_times)
    bars = _build_bars(beat_times, duration_sec)
    sections = _detect_sections(y, sr, duration_sec)

    stems = []
    for name, path in stem_paths.items():
        # Store a path relative to the analysis.json location (stems/<name>.wav)
        # so the static server can serve it and the file stays portable; the
        # absolute `path` is still used below for loudness/analysis.
        rel_path = f"{Path(path).parent.name}/{Path(path).name}"
        stems.append(
            {
                "type": name,
                "path": rel_path,
                "loudness_lufs": _loudness_lufs(path),
            }
        )

    patterns = {}
    drum_path = stem_paths.get("drums")
    if drum_path and bars:
        patterns["drums"] = _drum_pattern(drum_path, bars)

    loops, layers = [], {}
    if stem_paths and bars:
        drum_grid = patterns.get("drums", {}).get("grid")
        per_bar, loops = detect_repetitions(stem_paths, bars, drum_grid, sr=sr)
        for i, bar in enumerate(bars):
            bar.update(per_bar.get(i, {}))
        layers, bar_ids = detect_layer_patterns(
            stem_paths, bars, drum_grid, sr=sr,
            drum_dist=drum_dist, melodic_dist=melodic_dist)
        for i, bar in enumerate(bars):
            bar["layer_ids"] = bar_ids.get(i, {})

    return {
        "song_id": song_id,
        "engine_version": engine_version,
        "duration_sec": round(duration_sec, 4),
        "tempo_bpm": tempo_bpm,
        "tempo_confidence": tempo_conf,
        "time_signature": "4/4",
        "key": key,
        "key_confidence": round(key_conf, 4),
        "beat_grid": beat_grid,
        "bars": bars,
        "sections": sections,
        "loops": loops,
        "layers": layers,
        "stems": stems,
        "patterns": patterns,
    }


def _drum_pattern(drum_path: str, bars: list[dict]) -> dict:
    """Onset detection on the drum stem, quantized to a 16-step/bar grid.

    Stretch field: ships onsets + per-bar 16-step grid. Kick/snare/hat split
    (band-pass filtering) is left for later.
    """
    y, sr = librosa.load(str(drum_path), sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    grid = []
    for bar in bars:
        start, end = bar["start_sec"], bar["end_sec"]
        span = max(end - start, 1e-6)
        steps = [0] * STEPS_PER_BAR
        for t in onset_times:
            if start <= t < end:
                step = int((t - start) / span * STEPS_PER_BAR)
                steps[min(step, STEPS_PER_BAR - 1)] = 1
        grid.append(steps)

    return {
        "steps_per_bar": STEPS_PER_BAR,
        "onsets_sec": [round(t, 4) for t in onset_times],
        "grid": grid,
    }


# --------------------------------------------------------------------------- #
# Repetition detection                                                         #
# --------------------------------------------------------------------------- #

# Active-stem gating: a stem counts as "playing" in a bar if its per-bar
# loudness is within this many dB of that stem's loudest bar (and above a floor).
_ACTIVE_REL_DB = 22.0
_ACTIVE_FLOOR_DB = -50.0


def _bar_sync(feat: np.ndarray, bars: list[dict], sr: int, hop: int) -> np.ndarray:
    """Average a frame-wise feature within each bar -> (n_bars, n_dims)."""
    out = []
    for b in bars:
        f0 = int(b["start_sec"] * sr / hop)
        f1 = max(int(b["end_sec"] * sr / hop), f0 + 1)
        seg = feat[:, f0:f1]
        out.append(seg.mean(axis=1) if seg.size else np.zeros(feat.shape[0]))
    return np.asarray(out)


def _estimate_chords(chroma_bars: np.ndarray) -> list[str]:
    """Per-bar major/minor triad estimate via template matching."""
    templates, names = [], []
    for r in range(12):
        maj = np.zeros(12); maj[[r, (r + 4) % 12, (r + 7) % 12]] = 1.0
        minr = np.zeros(12); minr[[r, (r + 3) % 12, (r + 7) % 12]] = 1.0
        templates.append(maj); names.append(NOTE_NAMES[r])
        templates.append(minr); names.append(NOTE_NAMES[r] + "m")
    T = np.asarray(templates)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    out = []
    for v in chroma_bars:
        nv = v / (np.linalg.norm(v) + 1e-9)
        out.append(names[int(np.argmax(T @ nv))])
    return out


def _relabel_by_first_seen(labels: np.ndarray) -> list[str]:
    order, seq = {}, []
    for L in labels:
        if L not in order:
            order[L] = chr(ord("A") + len(order))
        seq.append(order[L])
    return seq


def detect_repetitions(
    stem_paths: dict[str, str],
    bars: list[dict],
    drum_grid: list[list[int]] | None = None,
    sr: int = 44100,
) -> tuple[dict, list[dict]]:
    """Cluster bars into repeated "parts" and summarize each.

    Works from stems alone (sums them to a mix for the harmonic/timbral
    fingerprint, uses per-stem energy for the arrangement). Returns
    ``(per_bar, loops)`` where ``per_bar`` maps bar index -> attributes to merge
    onto each bar, and ``loops`` is the grouped summary list.
    """
    if not stem_paths or not bars:
        return {}, []

    # Load stems; sum to a mix for harmony/timbre, keep each for arrangement.
    waves = {name: librosa.load(str(p), sr=sr, mono=True)[0] for name, p in stem_paths.items()}
    n = min((len(w) for w in waves.values()), default=0)
    if n == 0:
        return {}, []
    mix = sum(w[:n] for w in waves.values())

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=mix, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=mix, sr=sr, hop_length=hop, n_mfcc=13)
    chroma_bars = _bar_sync(chroma, bars, sr, hop)
    mfcc_bars = _bar_sync(mfcc, bars, sr, hop)

    # Per-stem loudness (dB) per bar -> active-stem mask.
    stem_names = list(stem_paths.keys())
    energy_db = np.full((len(bars), len(stem_names)), _ACTIVE_FLOOR_DB)
    for j, name in enumerate(stem_names):
        w = waves[name]
        for i, b in enumerate(bars):
            seg = w[int(b["start_sec"] * sr): int(b["end_sec"] * sr)]
            rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            energy_db[i, j] = 20.0 * np.log10(rms + 1e-9)
    thresh = np.maximum(energy_db.max(axis=0) - _ACTIVE_REL_DB, _ACTIVE_FLOOR_DB)
    active_mask = energy_db > thresh  # (n_bars, n_stems)

    chords = _estimate_chords(chroma_bars)
    density = []
    for i in range(len(bars)):
        if drum_grid and i < len(drum_grid) and drum_grid[i]:
            density.append(round(sum(drum_grid[i]) / len(drum_grid[i]), 3))
        else:
            density.append(0.0)

    # Cluster bars into repetition labels. Feature = harmony + timbre + arrangement.
    e_norm = (energy_db - energy_db.min(0)) / (np.ptp(energy_db, axis=0) + 1e-9)
    feat = np.hstack([chroma_bars, mfcc_bars, e_norm])
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)

    labels = _cluster_bars(feat)
    seq = _relabel_by_first_seen(labels)

    per_bar = {}
    for i in range(len(bars)):
        per_bar[i] = {
            "loop_label": seq[i],
            "chord": chords[i],
            "active_stems": [stem_names[j] for j in range(len(stem_names)) if active_mask[i, j]],
            "drum_density": density[i],
        }

    loops = _summarize_loops(seq, bars, chords, active_mask, energy_db, density,
                             stem_names, feat, labels)
    return per_bar, loops


def _cluster_bars(feat: np.ndarray) -> np.ndarray:
    """KMeans over bar features; picks k by silhouette in [3, 8]."""
    n = len(feat)
    if n < 4 or not _HAVE_SKLEARN:
        return np.zeros(n, dtype=int)
    best = None
    for k in range(3, min(9, n)):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feat)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(feat, km.labels_)
        if best is None or score > best[0]:
            best = (score, km.labels_)
    return best[1] if best else np.zeros(n, dtype=int)


def _summarize_loops(seq, bars, chords, active_mask, energy_db, density,
                     stem_names, feat, raw_labels) -> list[dict]:
    from collections import Counter

    loops = []
    for label in sorted(set(seq), key=lambda L: seq.index(L)):
        idxs = [i for i, c in enumerate(seq) if c == label]
        # contiguous runs -> instances
        instances, run_start, prev = [], idxs[0], idxs[0]
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
            else:
                instances.append((run_start, prev)); run_start = prev = i
        instances.append((run_start, prev))

        active = [s for j, s in enumerate(stem_names)
                  if active_mask[idxs, j].mean() > 0.5]
        stem_energy = {s: round(float(energy_db[idxs, j].mean()), 1)
                       for j, s in enumerate(stem_names)}
        chord_mode = Counter(chords[i] for i in idxs).most_common(1)[0][0]
        n_active = len(active)
        role = "sparse" if n_active <= 1 else ("full" if n_active >= len(stem_names) else "partial")

        # representative bar = closest to the cluster centroid
        cluster_id = raw_labels[idxs[0]]
        members = [i for i in range(len(raw_labels)) if raw_labels[i] == cluster_id]
        centroid = feat[members].mean(0)
        rep_bar = min(members, key=lambda i: float(np.linalg.norm(feat[i] - centroid)))

        loops.append({
            "label": label,
            "role": role,
            "count": len(idxs),
            "bar_count": len(idxs),
            "duration_sec": round(sum(bars[i]["end_sec"] - bars[i]["start_sec"] for i in idxs), 2),
            "active_stems": active,
            "stem_energy_db": stem_energy,
            "chord": chord_mode,
            "drum_density": round(float(np.mean([density[i] for i in idxs])), 3),
            "representative_bar": rep_bar,
            "instances": [
                {"start_bar": a, "end_bar": b,
                 "start_sec": round(bars[a]["start_sec"], 3),
                 "end_sec": round(bars[b]["end_sec"], 3)}
                for a, b in instances
            ],
        })
    return loops


# --------------------------------------------------------------------------- #
# Per-layer pattern detection                                                  #
# --------------------------------------------------------------------------- #

# Similarity cutoffs for grouping bars into repeated patterns, per layer.
_DRUM_PATTERN_DIST = 0.20  # hamming: <= ~3 of 16 steps differ -> same groove
_MELODIC_DIST = 10.0       # euclidean on z-scored sub-bar chroma+MFCC (~74 dims)
_LAYER_PREFIX = {"drums": "D", "bass": "B", "vocals": "V", "other": "O"}

# Sensitivity presets -> distance cutoffs. Higher sensitivity = lower cutoffs =
# more, stricter (more distinct) patterns; lower = fewer, broader patterns.
SENSITIVITY = {
    "low":    {"drum_dist": 0.30, "melodic_dist": 13.0},
    "medium": {"drum_dist": 0.20, "melodic_dist": 10.0},
    "high":   {"drum_dist": 0.10, "melodic_dist": 7.0},
}


def _bar_rms_db(wave: np.ndarray, b: dict, sr: int) -> float:
    seg = wave[int(b["start_sec"] * sr): int(b["end_sec"] * sr)]
    return 20.0 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if seg.size else -120.0


def _bar_sync_sub(feat: np.ndarray, bars: list[dict], sr: int, hop: int, sub: int) -> np.ndarray:
    """Split each bar into `sub` sub-segments and concat their means.

    Captures within-bar contour (movement), so repeated phrases match while
    different ones separate -- bar-averaged chroma alone is too key-dominated.
    """
    out = []
    for b in bars:
        f0 = int(b["start_sec"] * sr / hop)
        f1 = max(int(b["end_sec"] * sr / hop), f0 + 1)
        seg = feat[:, f0:f1]
        if seg.shape[1] == 0:
            out.append(np.zeros(feat.shape[0] * sub))
            continue
        idx = np.linspace(0, seg.shape[1], sub + 1).astype(int)
        out.append(np.concatenate(
            [seg[:, idx[k]:max(idx[k + 1], idx[k] + 1)].mean(axis=1) for k in range(sub)]))
    return np.asarray(out)


def _zstd(mat: np.ndarray, active: list[int], w: float = 1.0) -> np.ndarray:
    """Z-score columns using active-bar statistics, then scale by `w`."""
    if not active:
        return mat * w
    mu = mat[active].mean(axis=0)
    sd = mat[active].std(axis=0) + 1e-9
    return ((mat - mu) / sd) * w


def _group_bars(feats: np.ndarray, metric: str, thr: float) -> list[int]:
    """Agglomerative grouping by a distance cutoff -> cluster id per row."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = len(feats)
    if n == 0:
        return []
    if n == 1:
        return [1]
    d = pdist(feats, metric=metric)
    if not np.any(d > 0):  # all identical
        return [1] * n
    Z = linkage(d, method="average")
    return [int(c) for c in fcluster(Z, t=thr, criterion="distance")]


def detect_layer_patterns(
    stem_paths: dict[str, str],
    bars: list[dict],
    drum_grid: list[list[int]] | None = None,
    sr: int = 44100,
    drum_dist: float | None = None,
    melodic_dist: float | None = None,
) -> tuple[dict, dict]:
    """Find repeated patterns *within each stem independently*.

    For every layer, fingerprint each bar with features suited to that layer
    (drums: the 16-step onset grid; melodic stems: bar-synchronous chroma),
    group near-identical bars, and merge consecutive bars into occurrences.

    Returns ``(layers, per_bar)`` where ``layers[stem]`` is the pattern list and
    ``per_bar[bar_index] = {stem: label_or_None}`` for the UI strips.
    """
    if not stem_paths or not bars:
        return {}, {}

    hop = 512
    dd = _DRUM_PATTERN_DIST if drum_dist is None else drum_dist
    md = _MELODIC_DIST if melodic_dist is None else melodic_dist
    layers: dict = {}
    per_bar: dict = {i: {} for i in range(len(bars))}

    for stem, path in stem_paths.items():
        wave = librosa.load(str(path), sr=sr, mono=True)[0]

        if stem == "drums" and drum_grid:
            feats = np.asarray(drum_grid, dtype=float)
            metric, thr = "hamming", dd
            active = [i for i in range(len(bars))
                      if i < len(drum_grid) and sum(drum_grid[i]) > 0]
        else:
            chroma = _bar_sync_sub(
                librosa.feature.chroma_cqt(y=wave, sr=sr, hop_length=hop), bars, sr, hop, sub=4)
            mfcc = _bar_sync_sub(
                librosa.feature.mfcc(y=wave, sr=sr, hop_length=hop, n_mfcc=13), bars, sr, hop, sub=2)
            db = np.array([_bar_rms_db(wave, b, sr) for b in bars])
            floor = max(db.max() - 22.0, -50.0)
            active = [i for i in range(len(bars))
                      if db[i] > floor and np.linalg.norm(chroma[i]) > 1e-6]
            feats = np.hstack([_zstd(chroma, active), _zstd(mfcc, active, 0.7)])
            metric, thr = "euclidean", md

        layers[stem] = _build_layer(stem, feats, active, bars, metric, thr, per_bar)

    return layers, per_bar


def _build_layer(stem, feats, active, bars, metric, thr, per_bar) -> list[dict]:
    prefix = _LAYER_PREFIX.get(stem, stem[:1].upper())
    active = sorted(active)
    if not active:
        return []

    cluster_ids = _group_bars(feats[active], metric, thr)
    order, bar_label = {}, {}
    for k, bi in enumerate(active):
        cid = cluster_ids[k]
        if cid not in order:
            order[cid] = len(order) + 1
        bar_label[bi] = prefix + str(order[cid])

    # merge consecutive same-label bars into occurrence runs
    runs, cur = [], None
    for bi in range(len(bars)):
        lab = bar_label.get(bi)
        if lab is None:
            cur = None
            continue
        if cur and cur["label"] == lab and cur["end_bar"] == bi - 1:
            cur["end_bar"] = bi
        else:
            cur = {"label": lab, "start_bar": bi, "end_bar": bi}
            runs.append(cur)

    by_label: dict = {}
    for r in runs:
        by_label.setdefault(r["label"], []).append(r)

    patterns = []
    for label in sorted(by_label, key=lambda L: min(r["start_bar"] for r in by_label[L])):
        occs = by_label[label]
        bar_idxs = [bi for r in occs for bi in range(r["start_bar"], r["end_bar"] + 1)]
        if len(bar_idxs) < 2:        # keep only patterns that actually repeat
            continue
        centroid = feats[bar_idxs].mean(axis=0)
        rep = min(bar_idxs, key=lambda i: float(np.linalg.norm(feats[i] - centroid)))
        for bi in bar_idxs:
            per_bar[bi][stem] = label
        patterns.append({
            "label": label,
            "bar_count": len(bar_idxs),
            "occurrence_count": len(occs),
            "repeated": len(occs) >= 2,
            "representative_bar": rep,
            "occurrences": [
                {"start_bar": r["start_bar"], "end_bar": r["end_bar"],
                 "start_sec": round(bars[r["start_bar"]]["start_sec"], 3),
                 "end_sec": round(bars[r["end_bar"]]["end_sec"], 3)}
                for r in occs
            ],
        })

    # null out per-bar labels that didn't survive the >=2-bar filter
    kept = {p["label"] for p in patterns}
    for bi in list(per_bar):
        if per_bar[bi].get(stem) and per_bar[bi][stem] not in kept:
            per_bar[bi].pop(stem, None)
    return patterns


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Analyze an audio file.")
    ap.add_argument("audio")
    ap.add_argument("--drums", help="optional drum stem path for the step pattern")
    args = ap.parse_args()

    stems = {"drums": args.drums} if args.drums else {}
    print(json.dumps(analyze(args.audio, stems), indent=2))
