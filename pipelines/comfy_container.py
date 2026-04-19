"""Docker container lifecycle for the always-on ComfyUI server.

Centralizes:
  - container start (with optional extra mounts via COMFY_EXTRA_MOUNTS)
  - container stop / kill
  - readiness polling (/system_stats)
  - docker cp into the container's input/ dir
  - docker exec for ffmpeg helpers (last-frame extract, mp4 concat)
  - atexit cleanup so an aborted run doesn't leak the container
"""
import atexit
import os
import subprocess
import time
import urllib.request

from . import config

_atexit_registered = False


def _extra_mounts():
    """Parse COMFY_EXTRA_MOUNTS into docker -v args.

    Format: comma-separated host:container pairs. Useful when ~/comfy-models
    contains symlinks pointing into volumes that the container also needs to
    see (otherwise readlink fails inside the container).
    """
    out = []
    for spec in os.environ.get("COMFY_EXTRA_MOUNTS", "").split(","):
        spec = spec.strip()
        if spec:
            out += ["-v", spec]
    return out


def stop():
    """Kill + rm the ComfyUI container. Safe to call when not running."""
    subprocess.run(["docker", "kill", config.CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", config.CONTAINER], capture_output=True)
    print("ComfyUI stopped (GPU memory freed)")
    time.sleep(2)


def _atexit_stop():
    # Best-effort silent cleanup — if it's already gone, nothing to do.
    subprocess.run(["docker", "kill", config.CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", config.CONTAINER], capture_output=True)


def start(extra_mounts=None, lowvram=True):
    """Boot ComfyUI in the background, register atexit cleanup."""
    global _atexit_registered
    subprocess.run(["docker", "rm", "-f", config.CONTAINER], capture_output=True)

    mounts = list(extra_mounts) if extra_mounts else []
    mounts += _extra_mounts()

    cmd_inner = (
        "cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 "
        "--output-directory /opt/ComfyUI/output"
    )
    if lowvram:
        cmd_inner += " --lowvram"

    cmd = [
        "docker", "run", "-d", "--name", config.CONTAINER,
        *config.DOCKER_GPU, *config.DOCKER_ENV,
        "-p", "8188:8188",
        "-v", config.MODELS_DIR + ":/opt/ComfyUI/models",
        *mounts,
        "-v", f"{config.OUTPUT_DIR}:/opt/ComfyUI/output",
        config.IMAGE,
        "bash", "-c", cmd_inner,
    ]
    subprocess.run(cmd, check=True)
    print(f"ComfyUI starting{' (--lowvram)' if lowvram else ''}...")

    if not _atexit_registered:
        atexit.register(_atexit_stop)
        _atexit_registered = True


def wait_ready(timeout=180):
    """Poll /system_stats until ComfyUI answers, up to `timeout` seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{config.SERVER}/system_stats", timeout=2)
            print("ComfyUI ready")
            return True
        except Exception:
            time.sleep(3)
    print(f"ComfyUI failed to start within {timeout}s")
    return False


def cp_in(src, dest_basename=None):
    """Copy a host file into the container's /opt/ComfyUI/input/ dir."""
    name = dest_basename or os.path.basename(src)
    rc = subprocess.run([
        "docker", "cp", src,
        f"{config.CONTAINER}:/opt/ComfyUI/input/{name}",
    ]).returncode
    if rc != 0:
        print(f"  WARN: docker cp failed for {name}")
    return rc == 0


def exec_ffmpeg(args, capture=True):
    """Run ffmpeg inside the running container. Args are appended after `ffmpeg`."""
    return subprocess.run(
        ["docker", "exec", config.CONTAINER, "ffmpeg", "-y", *args],
        capture_output=capture, text=True,
    )


def extract_last_frame(mp4_path, png_path):
    """Use container ffmpeg to grab the final frame as a PNG.

    `-update 1` overwrites the output each frame, leaving only the last —
    more reliable than `-sseof` for LTX-2.3 mp4s where the negative seek
    sometimes returns 0 frames.
    """
    rel_in = mp4_path.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
    rel_out = png_path.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
    exec_ffmpeg(["-i", rel_in, "-update", "1", "-q:v", "2", rel_out])
    return os.path.exists(png_path)


def join_mp4s(mp4_paths, out_path):
    """Concat mp4s into one continuous mp4 via ffmpeg concat demuxer.

    Inputs must share codec/resolution/fps — guaranteed when they're all
    from the same workflow.
    """
    if not mp4_paths:
        print("  nothing to join")
        return False
    list_file = config.OUTPUT_DIR + "/chain_concat.txt"
    rel_list = "/opt/ComfyUI/output/chain_concat.txt"
    rel_out = out_path.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
    with open(list_file, "w") as f:
        for m in mp4_paths:
            rel = m.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
            f.write(f"file '{rel}'\n")
    rc = exec_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", rel_list, "-c", "copy", rel_out]
    ).returncode
    try:
        os.remove(list_file)
    except OSError:
        pass
    return rc == 0 and os.path.exists(out_path)
