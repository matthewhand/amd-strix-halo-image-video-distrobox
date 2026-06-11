#!/usr/bin/env python3
"""HeartMuLa music worker — thin HTTP wrapper around heartmula_launcher.py.

Mirrors the qwen_tts_serve.py contract so music gen is URL-configurable like
the other backends:

    POST /music  {prompt, duration}  →  {ok, url, duration, elapsed_s}

The WAV is written into a shared output dir (`/workspace/music` when run via
the compose `heartmula` profile) that the slopfinity dashboard serves at
/files/music/<name>; the orchestrator (run_fleet.heartmula_wav) resolves the
returned URL back to the local file. Generation itself is delegated to the
existing CLI launcher in a subprocess — model load per request, exactly the
cost profile of the legacy `docker run --rm` path it replaces, minus docker.

Run (inside the toolbox container or any env with heartlib + the model):
    HEARTMULA_OUT_DIR=/workspace/music python3 heartmula_serve.py   # :8011
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse


def _first_writable(candidates):
    for cand in candidates:
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".writable-probe")
            with open(probe, "w") as _f:
                _f.write("ok")
            os.unlink(probe)
            return cand
        except (OSError, PermissionError):
            continue
    raise PermissionError(
        f"No writable music output directory found in: {candidates!r}. "
        "Set HEARTMULA_OUT_DIR to a writable path."
    )


OUT_DIR = _first_writable([
    os.environ.get("HEARTMULA_OUT_DIR"),
    "/workspace/music",
    os.path.expanduser("~/.slopfinity/music"),
    "/tmp/slopfinity-music",
])
print(f"🎼 HeartMuLa output dir: {OUT_DIR}", flush=True)

LAUNCHER = os.environ.get("HEARTMULA_LAUNCHER", "/opt/heartmula_launcher.py")
if not os.path.exists(LAUNCHER):
    # Dev fallback: resolve next to this file.
    _local = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "heartmula_launcher.py"
    )
    if os.path.exists(_local):
        LAUNCHER = _local

# Model load + generation is minutes-scale on Strix Halo; bound it so a
# wedged ROCm run frees the worker instead of hanging it forever.
GEN_TIMEOUT_S = float(os.environ.get("HEARTMULA_TIMEOUT_S", "1200"))

app = FastAPI(title="HeartMuLa Music Worker")


@app.get("/health")
def health():
    return {
        "ok": True,
        "launcher": LAUNCHER,
        "launcher_exists": os.path.exists(LAUNCHER),
        "out": OUT_DIR,
        "timeout_s": GEN_TIMEOUT_S,
    }


@app.post("/music")
def music(payload: dict = Body(...)):
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "error": "prompt is required"},
                            status_code=400)
    try:
        duration = max(1.0, float(payload.get("duration") or 30.0))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "duration must be a number"},
                            status_code=400)

    name = f"hm_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(OUT_DIR, name)
    cmd = [
        sys.executable, LAUNCHER,
        "--prompt", prompt,
        "--duration", f"{duration:.1f}",
        "--out", out_path,
        "--real",
    ]
    t0 = time.time()
    print(f"🎼 /music duration={duration:.1f}s prompt={prompt[:60]!r}", flush=True)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=GEN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return JSONResponse(
            {"ok": False, "error": f"generation timed out after {GEN_TIMEOUT_S:.0f}s"},
            status_code=504,
        )
    if res.returncode != 0 or not os.path.exists(out_path):
        tail = (res.stderr or res.stdout or "")[-800:]
        return JSONResponse(
            {"ok": False,
             "error": f"launcher exited {res.returncode}, wav missing={not os.path.exists(out_path)}",
             "detail": tail},
            status_code=502,
        )
    return {
        "ok": True,
        "url": f"/files/music/{name}",
        "duration": duration,
        "elapsed_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
                host=os.environ.get("HEARTMULA_HOST", "0.0.0.0"),
                port=int(os.environ.get("HEARTMULA_PORT", "8011")))
