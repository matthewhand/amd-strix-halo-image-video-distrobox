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

def _first_writable(candidates):
    """Pick the first directory we can create + write to."""
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
        f"No writable TTS output directory found in: {candidates!r}. "
        "Set TTS_OUT_DIR to a writable path."
    )


OUT_DIR = _first_writable([
    os.environ.get("TTS_OUT_DIR"),
    "/workspace/tts",
    os.path.expanduser("~/.slopfinity/tts"),
    "/tmp/slopfinity-tts",
])
print(f"🎙️  TTS output dir: {OUT_DIR}", flush=True)

QWEN_LAUNCHER = os.environ.get(
    "QWEN_TTS_LAUNCHER",
    "/opt/qwen_tts_launcher.py",
)
KOKORO_LAUNCHER = os.environ.get(
    "KOKORO_TTS_LAUNCHER",
    "/opt/kokoro_tts_launcher.py",
)
for _name in ("QWEN_LAUNCHER", "KOKORO_LAUNCHER"):
    _path = locals()[_name]
    if not os.path.exists(_path):
        # Dev fallback: resolve next to this file.
        _local = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.basename(_path),
        )
        if os.path.exists(_local):
            globals()[_name] = _local
            locals()[_name] = _local

DEFAULT_MODEL = os.environ.get(
    "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)

# Voices that route to the qwen-tts launcher. Anything else goes to
# kokoro by default. Qwen voices are short proper names; kokoro voices
# follow the `xx_yyy` pattern (af_heart, am_eric, bm_lewis, ...).
QWEN_VOICES = {
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
}

# Default engine when the operator hasn't pinned per-call. `kokoro`
# because Qwen3-TTS hits a HIP kernel mismatch on Strix Halo gfx1151
# and Kokoro-82M ONNX runs cleanly on CPU. Operators can flip back to
# Qwen via TTS_ENGINE=qwen if they have working kernels.
DEFAULT_ENGINE = os.environ.get("TTS_ENGINE", "kokoro").lower()

app = FastAPI(title="Qwen3-TTS Worker")


@app.get("/health")
def health():
    return {
        "ok": True,
        "default_engine": DEFAULT_ENGINE,
        "qwen_launcher": QWEN_LAUNCHER,
        "kokoro_launcher": KOKORO_LAUNCHER,
        "out": OUT_DIR,
    }


def _pick_engine(voice: str, requested: str | None) -> str:
    """Decide which TTS engine to use:
       1. Explicit `engine` field in the request body wins.
       2. Else: known qwen voice names route to qwen.
       3. Else: DEFAULT_ENGINE (kokoro by default)."""
    if requested:
        v = requested.strip().lower()
        if v in ("qwen", "kokoro"):
            return v
    if voice and voice.strip().lower() in QWEN_VOICES:
        return "qwen"
    return DEFAULT_ENGINE


@app.post("/tts")
def tts(data: dict = Body(...)):
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "af_heart"
    engine = _pick_engine(voice, data.get("engine"))
    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)

    fname = f"tts_{engine}_{voice}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.wav"
    out_path = os.path.join(OUT_DIR, fname)

    if engine == "qwen":
        cmd = [
            sys.executable, QWEN_LAUNCHER,
            "--text", text,
            "--voice", voice,
            "--out", out_path,
            "--model", DEFAULT_MODEL,
        ]
        env = os.environ.copy()
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    else:  # kokoro
        cmd = [
            sys.executable, KOKORO_LAUNCHER,
            "--text", text,
            "--voice", voice,
            "--out", out_path,
            "--lang", data.get("lang") or "en-us",
            "--speed", str(float(data.get("speed") or 1.0)),
        ]
        env = os.environ.copy()  # no HSA override needed; kokoro is CPU ONNX

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
