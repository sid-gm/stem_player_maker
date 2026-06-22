#!/usr/bin/env python
"""Tiny backend for the visualizer: static files + upload/process + re-enrich.

Run from the harness root with the venv python:

    ./.venv/bin/python server.py            # serves http://localhost:8753

Endpoints (all JSON unless noted):
    GET  /                       -> static files (visualizer.html, stems, json)
    GET  /api/songs              -> ["Drake high fives", "E85 DT", ...]
    POST /api/upload?name=&sensitivity=   (raw audio body) -> {"job": id}
    GET  /api/status?job=ID      -> {state, step, song, json, error}
    POST /api/enrich  {song, sensitivity} -> {ok} (fast: re-cluster existing stems)
    POST /api/render?song=&label=&name=   (raw audio body) -> {paths, slots_replaced, ...}
    POST /api/sample?name=                (raw audio body) -> {id} (decoded, stored)
    POST /api/render-medley  {song, placements:[{label, sample_id}], name} -> {paths, ...}
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import urllib.parse
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from analyze import SENSITIVITY, analyze
from enrich_analysis import enrich
from preview_snap import snap_preview
from render import render, render_medley
from separate import separate
from slice_reference import slice_reference

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
UPLOADS = ROOT / "uploads"
OUT.mkdir(exist_ok=True)        # may be a fresh Modal Volume mount on first boot
UPLOADS.mkdir(exist_ok=True)
PORT = 8753

# Stem-separation device: "mps" on Apple Silicon (local), "cuda" on Modal GPU.
# separate() auto-falls back to CPU if the chosen device fails.
DEVICE = os.environ.get("SEP_DEVICE", "mps")
# CORS origin for the Vercel frontend → Modal backend. "*" for now; lock to the
# Vercel domain in prod via the ALLOW_ORIGIN env var.
ALLOW_ORIGIN = os.environ.get("ALLOW_ORIGIN", "*")

JOBS: dict[str, dict] = {}


def _rel_to_root(p) -> str:
    """ROOT-relative artifact path, robust to Modal Volume mounts.

    output/ and uploads/ are Modal Volumes: they mount at /app/output (etc.) but
    resolve() follows the symlink out to /__modal/volumes/<id>/..., which is not
    under ROOT. So try a lexical relative_to(ROOT) first (handles /app/output/...
    paths without touching the fs), then fall back to mapping the resolved path
    back through the OUT/UPLOADS mounts."""
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        pass
    rp = p.resolve()
    for base, name in ((OUT, "output"), (UPLOADS, "uploads")):
        try:
            return str(Path(name) / rp.relative_to(base.resolve()))
        except ValueError:
            continue
    return str(rp.relative_to(ROOT.resolve()))  # last resort


def _process(job_id: str, audio_path: str, sensitivity: str) -> None:
    job = JOBS[job_id]
    try:
        th = SENSITIVITY.get(sensitivity, SENSITIVITY["medium"])
        job.update(state="running", step="separating stems")
        sep = separate(audio_path, OUT, device=DEVICE)  # cuda on Modal, mps locally; auto CPU fallback inside

        job["step"] = "analyzing + detecting patterns"
        result = analyze(
            audio_path, sep["stems"], song_id=sep["song"],
            drum_dist=th["drum_dist"], melodic_dist=th["melodic_dist"])
        result["sensitivity"] = sensitivity

        song_dir = Path(sep["song_dir"])
        (song_dir / "analysis.json").write_text(json.dumps(result, indent=2))
        job.update(state="done", step="done", song=sep["song"],
                   json=f"output/{sep['song']}/analysis.json")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        job.update(state="error", step="error", error=str(exc),
                   trace=traceback.format_exc())


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, *a):  # quieter console
        pass

    def end_headers(self):
        # Inject CORS on EVERY response — JSON API, static stems/json, and renders
        # are all fetched cross-origin by the Vercel frontend (incl. decodeAudioData).
        self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.end_headers()

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----- GET -----
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/songs":
            songs = sorted(d.name for d in OUT.glob("*") if (d / "analysis.json").exists())
            return self._send_json(songs)
        if u.path == "/api/status":
            jid = urllib.parse.parse_qs(u.query).get("job", [""])[0]
            job = JOBS.get(jid)
            if not job:
                return self._send_json({"state": "unknown"}, 404)
            return self._send_json({k: v for k, v in job.items() if k != "trace"})
        if u.path == "/api/reference":
            return self._reference(u)
        if u.path == "/api/snap-preview":
            return self._snap_preview(u)
        return super().do_GET()

    def _reference(self, u):
        """Cut (and cache) a pattern's representative-bar preview clip."""
        q = urllib.parse.parse_qs(u.query)
        song = urllib.parse.unquote(q.get("song", [""])[0])
        label = q.get("label", [""])[0]
        song_dir = OUT / song
        if not song or not (song_dir / "analysis.json").exists():
            return self._send_json({"error": "no such song"}, 404)
        try:
            info = slice_reference(song_dir, label)
            # hand back a ROOT-relative path the static server can serve
            info["path"] = _rel_to_root(info["path"])
            self._send_json(info)
        except SystemExit as exc:  # slice_reference raises these for bad input
            self._send_json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _snap_preview(self, u):
        """Render (and cache) a one-slot snapped A/B preview for a label."""
        q = urllib.parse.parse_qs(u.query)
        song = urllib.parse.unquote(q.get("song", [""])[0])
        label = q.get("label", [""])[0]
        src = q.get("source_bar", [None])[0]
        tgt = q.get("target_bar", [None])[0]
        song_dir = OUT / song
        if not song or not (song_dir / "analysis.json").exists():
            return self._send_json({"error": "no such song"}, 404)
        try:
            info = snap_preview(song_dir, label,
                                source_bar=int(src) if src else None,
                                target_bar=int(tgt) if tgt else None)
            # hand back ROOT-relative paths the static server can serve
            info["paths"] = {k: _rel_to_root(p) for k, p in info["paths"].items()}
            self._send_json(info)
        except SystemExit as exc:  # bad label / missing stem
            self._send_json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    # ----- POST -----
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/upload":
            return self._upload(u)
        if u.path == "/api/enrich":
            return self._enrich()
        if u.path == "/api/render":
            return self._render(u)
        if u.path == "/api/sample":
            return self._sample(u)
        if u.path == "/api/render-medley":
            return self._render_medley()
        self._send_json({"error": "not found"}, 404)

    def _upload(self, u):
        q = urllib.parse.parse_qs(u.query)
        name = urllib.parse.unquote(q.get("name", ["upload.bin"])[0])
        sensitivity = q.get("sensitivity", ["medium"])[0]
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return self._send_json({"error": "empty upload"}, 400)
        data = self.rfile.read(length)

        safe = Path(name).name or "upload.bin"
        dest = UPLOADS / safe
        dest.write_bytes(data)

        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"state": "queued", "step": "queued", "name": safe}
        threading.Thread(target=_process, args=(jid, str(dest), sensitivity),
                         daemon=True).start()
        self._send_json({"job": jid})

    def _enrich(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        song = body.get("song")
        sensitivity = body.get("sensitivity", "medium")
        song_dir = OUT / (song or "")
        if not (song_dir / "analysis.json").exists():
            return self._send_json({"error": "no such song"}, 404)
        try:
            enrich(song_dir, sensitivity, layers_only=True)  # toggle = fast layer re-detect
            self._send_json({"ok": True, "song": song, "sensitivity": sensitivity})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _decode_sample(self, name: str) -> Path:
        """Read a raw audio body and decode to a wav under uploads/ so render's
        soundfile reader handles any container (mp3/m4a/wav/...). Native sr is kept
        (render resamples to the stem on placement). Returns the wav path."""
        import librosa
        import soundfile as sf
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            raise ValueError("empty sample upload")
        raw_path = UPLOADS / f"sample_{uuid.uuid4().hex[:6]}_{Path(name).name or 'sample.bin'}"
        raw_path.write_bytes(self.rfile.read(length))
        y, srr = librosa.load(str(raw_path), sr=None, mono=True)
        sample_wav = raw_path.with_suffix(".wav")
        sf.write(str(sample_wav), y, srr)
        return sample_wav

    def _rel_paths(self, info: dict) -> dict:
        """Rewrite wav/mp3 output paths to ROOT-relative so the static server serves them."""
        info["paths"] = {
            k: (_rel_to_root(v) if k in ("wav", "mp3") else v)
            for k, v in info["paths"].items()
        }
        if info.get("stems"):  # edited per-stem wavs for the timeline -> also ROOT-relative
            info["stems"] = {s: _rel_to_root(p) for s, p in info["stems"].items()}
        return info

    def _render(self, u):
        """Full bounce: replace one part's sound with an uploaded raw sample, every slot."""
        q = urllib.parse.parse_qs(u.query)
        song = urllib.parse.unquote(q.get("song", [""])[0])
        label = q.get("label", [""])[0]
        name = urllib.parse.unquote(q.get("name", ["sample.wav"])[0])
        song_dir = OUT / song
        if not song or not (song_dir / "analysis.json").exists():
            return self._send_json({"error": "no such song"}, 404)
        if not label:
            return self._send_json({"error": "missing label"}, 400)

        sample_wav = None
        if int(self.headers.get("Content-Length", 0)) > 0:
            try:
                sample_wav = self._decode_sample(name)
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": f"could not decode sample: {exc}"}, 400)

        try:
            info = render(song_dir, label, sample=str(sample_wav) if sample_wav else None)
            self._send_json(self._rel_paths(info))
        except SystemExit as exc:  # bad label / missing stem
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc), "trace": traceback.format_exc()}, 500)

    def _sample(self, u):
        """Stash one uploaded sound; return an id the medley render references later."""
        q = urllib.parse.parse_qs(u.query)
        name = urllib.parse.unquote(q.get("name", ["sample.wav"])[0])
        try:
            wav = self._decode_sample(name)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"error": f"could not decode sample: {exc}"}, 400)
        # id = the wav filename; resolved back under UPLOADS on render (never a path).
        self._send_json({"id": wav.name, "name": name})

    def _render_medley(self):
        """Layer many parts -> many sounds into one bounce.

        Body: {song, placements:[{label, sample_id?}], name?}. A missing sample_id
        falls back to that part's own representative bar as the stand-in sound."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        song = body.get("song")
        name = body.get("name") or "medley"
        confine = bool(body.get("confine"))   # mobile: fit each sound to its part
        song_dir = OUT / (song or "")
        if not song or not (song_dir / "analysis.json").exists():
            return self._send_json({"error": "no such song"}, 404)
        raw = body.get("placements") or []
        if not raw:
            return self._send_json({"error": "no placements"}, 400)
        placements = []
        for pl in raw:
            label = (pl or {}).get("label")
            if not label:
                return self._send_json({"error": "placement missing label"}, 400)
            sid = pl.get("sample_id")
            sample = None
            if sid:
                cand = UPLOADS / Path(sid).name  # name only — no path traversal
                if not cand.exists():
                    return self._send_json({"error": f"unknown sample_id: {sid}"}, 400)
                sample = str(cand)
            placements.append({"label": label, "sample": sample})
        try:
            info = render_medley(song_dir, placements, out_name=Path(name).name or "medley",
                                 write_stems=True, confine=confine)
            self._send_json(self._rel_paths(info))
        except SystemExit as exc:  # bad label / missing stem
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc), "trace": traceback.format_exc()}, 500)


if __name__ == "__main__":
    print(f"serving {ROOT} at http://localhost:{PORT}  (visualizer.html)")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
