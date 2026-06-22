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
    gpu="L4",
    volumes={
        f"{APP_DIR}/output": output_vol,
        f"{APP_DIR}/uploads": uploads_vol,
    },
    max_containers=1,        # single container so JOBS dict + status polling stay coherent
    scaledown_window=300,    # sleep the GPU after 5 min idle (Volume keeps songs alive)
    timeout=60 * 30,
)
@modal.concurrent(max_inputs=100)  # one warm container serves many simultaneous requests
@modal.web_server(PORT, startup_timeout=120)
def serve():
    # Launch the stdlib server; cwd=APP_DIR so sibling imports (analyze, render, ...)
    # resolve and static files are served from the app dir. server.py listens on PORT.
    subprocess.Popen(["python", "server.py"], cwd=APP_DIR)
