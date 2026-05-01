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


@app.get("/voices")
def voices():
    """Enumerate voices each engine knows about. The kokoro path lazy-
    loads the model files (~325 MB) so the first call may pay a small
    download/disk cost; subsequent calls hit the in-process cache."""
    out = {
        "ok": True,
        "default_engine": DEFAULT_ENGINE,
        "engines": {
            "qwen": {
                "default_voice": "ryan",
                "voices": sorted(QWEN_VOICES),
            },
        },
    }
    try:
        from kokoro_onnx import Kokoro  # type: ignore
        # Touch the kokoro launcher's model resolver so we share the
        # download path. Avoid double-downloading.
        import importlib.util as _u
        spec = _u.spec_from_file_location("_kkl", KOKORO_LAUNCHER)
        if spec and spec.loader:
            mod = _u.module_from_spec(spec); spec.loader.exec_module(mod)
            mp, vp = mod._resolve_model_files(os.environ.get("KOKORO_MODEL_DIR", mod.DEFAULT_DIR))
            k = Kokoro(mp, vp)
            kokoro_voices = sorted(k.get_voices())
            out["engines"]["kokoro"] = {
                "default_voice": "af_heart",
                "voices": kokoro_voices,
            }
    except Exception as e:
        out["engines"]["kokoro"] = {"error": f"{type(e).__name__}: {e}", "voices": []}
    return out


def _prune_old_tts_files(keep: int = 0) -> int:
    """Delete oldest TTS files beyond the keep cap.

    Without this, /tmp/slopfinity-tts (or whichever OUT_DIR resolves to)
    grows unbounded — every /tts hit writes a fresh ~100-2700 KB WAV.
    Default keep=50 (~50 MB at the high end) tunable via env
    TTS_KEEP_FILES. Keep=0 disables the prune entirely (operator
    manages cleanup elsewhere).

    Returns number of files removed; logs to stderr on each delete.
    Cheap O(n log n) sort over the OUT_DIR — at 50 files this is
    sub-millisecond. Running this after each generation amortises the
    cost across writes rather than accumulating debt.
    """
    if keep <= 0:
        return 0
    try:
        files = []
        for n in os.listdir(OUT_DIR):
            p = os.path.join(OUT_DIR, n)
            if os.path.isfile(p) and n.endswith(".wav"):
                files.append((os.path.getmtime(p), p))
        if len(files) <= keep:
            return 0
        files.sort(reverse=True)  # newest first
        to_delete = files[keep:]
        for _, p in to_delete:
            try:
                os.unlink(p)
            except OSError as e:
                print(f"⚠ prune: failed to delete {p}: {e}", file=sys.stderr, flush=True)
        if to_delete:
            print(f"🧹 pruned {len(to_delete)} TTS file(s); kept {keep} most recent", file=sys.stderr, flush=True)
        return len(to_delete)
    except Exception as e:
        print(f"⚠ prune skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 0


KEEP_TTS_FILES = int(os.environ.get("TTS_KEEP_FILES", "50"))


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
    # Storage prune — keep the OUT_DIR bounded. Runs on success path so
    # we don't penalize errors. Worst case: a hot loop of /tts requests
    # paying ~1ms of dirent scan + a few unlinks per call. Scales fine
    # at the single-digit-RPS the dashboard sees.
    _prune_old_tts_files(KEEP_TTS_FILES)
    return {
        "ok": True,
        "status": "ok",
        "url": url,
        "audio_path": url,
        "voice": voice,
        "engine": engine,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("TTS_PORT", "8010")))
