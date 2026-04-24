#!/usr/bin/env python3
"""Thin FastAPI TTS worker exposing POST /tts on :8010.

Persists generated WAVs to /workspace/tts/ and returns `{ok, url}` pointing
at `/files/tts/<name>` (served by the slopfinity dashboard).

Implementation strategy: shell out to qwen_tts_launcher.py so model load
failures don't take the HTTP server down. The launcher enforces the
HSA_OVERRIDE_GFX_VERSION=11.0.0 override and disk-space guard.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

OUT_DIR = os.environ.get("TTS_OUT_DIR", "/workspace/tts")
os.makedirs(OUT_DIR, exist_ok=True)

LAUNCHER = os.environ.get(
    "QWEN_TTS_LAUNCHER",
    "/opt/qwen_tts_launcher.py",
)
if not os.path.exists(LAUNCHER):
    # Dev fallback: resolve next to this file.
    _local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_tts_launcher.py")
    if os.path.exists(_local):
        LAUNCHER = _local

DEFAULT_MODEL = os.environ.get(
    "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)

app = FastAPI(title="Qwen3-TTS Worker")


@app.get("/health")
def health():
    return {"ok": True, "launcher": LAUNCHER, "out": OUT_DIR}


@app.post("/tts")
def tts(data: dict = Body(...)):
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "ryan"
    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)

    fname = f"tts_{voice}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.wav"
    out_path = os.path.join(OUT_DIR, fname)

    cmd = [
        sys.executable,
        LAUNCHER,
        "--text", text,
        "--voice", voice,
        "--out", out_path,
        "--model", DEFAULT_MODEL,
    ]
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "synthesis timeout"}, status_code=504)

    if proc.returncode == 2:
        # Disk guard tripped.
        return JSONResponse(
            {"ok": False, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "disk guard"},
            status_code=507,
        )
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        return JSONResponse(
            {"ok": False, "error": "launcher failed", "stderr": "\n".join(tail)},
            status_code=500,
        )

    url = f"/files/tts/{fname}"
    return {
        "ok": True,
        "status": "ok",
        "url": url,
        "audio_path": url,
        "voice": voice,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("TTS_PORT", "8010")))
