"""Modal deployment for the jawnsplitter backend (GPU stem separation + static serving).

The existing stdlib server (server.py) is launched verbatim behind Modal's
web_server proxy. Key decisions, and why:

  * GPU (L4)           — demucs/htdemucs runs on CUDA. server.py reads SEP_DEVICE=cuda.
  * Persistent Volumes — output/ and uploads/ are mounted as Modal Volumes so processed
                         songs, stems, renders and uploaded samples survive container
                         sleep. Without this, "try a sample" and serving stems back die.
  * max_containers=1   — upload spawns a background thread + stashes job state in an
                         in-memory JOBS dict; the frontend then polls /api/status. One
                         container guarantees the poll hits the process that owns the job.
  * Baked weights      — htdemucs (~80MB) is downloaded at image-build time so cold
                         starts don't re-fetch it.

Commands:
  modal deploy modal_app.py     # deploy → prints the persistent https URL
  modal serve  modal_app.py     # ephemeral dev URL with live reload
  modal app logs jawnsplitter   # tail logs
"""

import os
import subprocess

import modal

APP_DIR = "/app"
PORT = 8753

app = modal.App("jawnsplitter")

# Persistent stores. Two volumes (output/ and uploads/ are siblings under the app
# dir; one volume mounted at the parent would shadow the code).
output_vol = modal.Volume.from_name("jawnsplitter-output", create_if_missing=True)
uploads_vol = modal.Volume.from_name("jawnsplitter-uploads", create_if_missing=True)


def _bake_demucs_weights():
    """Download htdemucs into the image's torch hub cache at build time."""
    from demucs.pretrained import get_model

    get_model("htdemucs")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # mp3/m4a decode + mp3 encode + torchcodec backend
    .pip_install_from_requirements("requirements.txt")
    .env({"SEP_DEVICE": "cuda", "TORCH_HOME": "/root/.cache/torch"})
    .run_function(_bake_demucs_weights)  # cache htdemucs weights into the image
    .add_local_dir(
        ".",
        APP_DIR,
        # keep the image lean: no venv, no processed audio, no source mp3s/docs
        ignore=[
            ".venv", ".venv/**",
            "output", "output/**",
            "uploads", "uploads/**",
            "__pycache__", "**/__pycache__/**",
            ".git", ".git/**",
            "*.mp3", "*.md",
        ],
    )
)


@app.function(
    image=image,
    volumes={                       # CPU container — no GPU. Demucs is offloaded to separate_gpu.
        f"{APP_DIR}/output": output_vol,
        f"{APP_DIR}/uploads": uploads_vol,
    },
    max_containers=1,        # single container so JOBS dict + status polling stay coherent
    scaledown_window=300,    # CPU is cheap — keep it warm a bit for snappy serving
    timeout=60 * 30,
)
@modal.concurrent(max_inputs=100)  # one warm container serves many simultaneous requests
@modal.web_server(PORT, startup_timeout=120)
def serve():
    # Launch the stdlib server; cwd=APP_DIR so sibling imports (analyze, render, ...) resolve
    # and static files are served from the app dir. SEPARATE_BACKEND=modal tells server.py to
    # offload demucs to the GPU separate_gpu function instead of running it on this CPU box.
    subprocess.Popen(["python", "server.py"], cwd=APP_DIR,
                     env={**os.environ, "SEPARATE_BACKEND": "modal"})


@app.function(image=image, gpu="L4", scaledown_window=10, timeout=60 * 20)
def separate_gpu(audio_bytes: bytes, name: str) -> dict:
    """GPU-only demucs: raw audio bytes in, the 4 stem WAVs out as bytes. Deliberately
    mounts NO Volumes — it's pure compute, so there's no cross-container Volume handoff to
    get wrong. The CPU web container writes the returned stems to the output Volume itself.
    Scales to zero ~10s after the split, so the L4 bills seconds per song, not session time."""
    import pathlib
    import sys
    import tempfile

    sys.path.insert(0, APP_DIR)              # so `from separate import separate` resolves
    from separate import separate

    tmp = pathlib.Path(tempfile.mkdtemp())
    src = tmp / (pathlib.Path(name).name or "upload.bin")
    src.write_bytes(audio_bytes)
    sep = separate(str(src), tmp / "out", device="cuda")  # auto CPU fallback inside
    stems = {n: pathlib.Path(p).read_bytes() for n, p in sep["stems"].items()}
    return {"song": sep["song"], "duration_sec": sep["duration_sec"], "stems": stems}


@app.function(
    image=image,
    volumes={f"{APP_DIR}/output": output_vol},
    timeout=60 * 15,
)
def backfill_mp3():
    """One-off: encode missing stems/*.mp3 for already-split songs (CPU only).
    Run with:  modal run modal_app.py::backfill_mp3"""
    import pathlib

    out = pathlib.Path(f"{APP_DIR}/output")
    n = 0
    for wav in sorted(out.glob("*/stems/*.wav")):
        mp3 = wav.with_suffix(".mp3")
        if mp3.exists():
            continue
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
             "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)],
            check=True)
        n += 1
        print(f"  encoded {mp3}")
    output_vol.commit()
    print(f"backfilled {n} stem mp3s")
