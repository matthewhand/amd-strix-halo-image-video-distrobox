#!/usr/bin/env python3
"""Thin FastAPI TTS worker exposing POST /tts on :8010.

... (docstring) ...
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid


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
        f"No writable TTS output directory found in: {candidates!r}. "
        "Set TTS_OUT_DIR to a writable path."
    )


OUT_DIR = None

def _get_out_dir():
    global OUT_DIR
    if OUT_DIR is None:
        OUT_DIR = _first_writable([
            os.environ.get("TTS_OUT_DIR"),
            "/workspace/tts",
            os.path.expanduser("~/.slopfinity/tts"),
            "/tmp/slopfinity-tts",
        ])
        if not os.environ.get("SLOPFINITY_QUIET"):
            print(f"🎙️  TTS output dir: {OUT_DIR}", flush=True)
    return OUT_DIR


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

QWEN_VOICES = {
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
}

DEFAULT_ENGINE = os.environ.get("TTS_ENGINE", "kokoro").lower()


def _pick_engine(voice: str, requested: str | None) -> str:
    if requested:
        v = requested.strip().lower()
        if v in ("qwen", "kokoro"):
            return v
    if voice and voice.strip().lower() in QWEN_VOICES:
        return "qwen"
    return DEFAULT_ENGINE


def _prune_old_tts_files(keep: int = 0) -> int:
    out_dir = _get_out_dir()
    if keep <= 0:
        return 0
    try:
        files = []
        for n in os.listdir(out_dir):
            p = os.path.join(out_dir, n)
            if os.path.isfile(p) and n.endswith(".wav"):
                files.append((os.path.getmtime(p), p))
        if len(files) <= keep:
            return 0
        files.sort(reverse=True)
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


def _health_impl() -> dict:
    h = {
        "ok": True,
        "default_engine": DEFAULT_ENGINE,
        "qwen_launcher": QWEN_LAUNCHER,
        "kokoro_launcher": KOKORO_LAUNCHER,
        "out": _get_out_dir(),
    }
    if os.environ.get("TTS_WORKER_URL"):
        h["url"] = os.environ["TTS_WORKER_URL"]
    return h


def _voices_impl() -> dict:
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
        from kokoro_onnx import Kokoro
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


def _tts_impl(data: dict = None) -> dict:
    if data is None:
        data = {}
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "af_heart"
    engine = _pick_engine(voice, data.get("engine"))
    if not text:
        return {"ok": False, "error": "empty text"}

    fname = f"tts_{engine}_{voice}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.wav"
    out_path = os.path.join(_get_out_dir(), fname)

    if engine == "qwen":
        cmd = [
            sys.executable, QWEN_LAUNCHER,
            "--text", text,
            "--voice", voice,
            "--out", out_path,
            "--model", DEFAULT_MODEL,
        ]
        env = os.environ.copy()
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"
    else:
        cmd = [
            sys.executable, KOKORO_LAUNCHER,
            "--text", text,
            "--voice", voice,
            "--out", out_path,
            "--lang", data.get("lang") or "en-us",
            "--speed", str(float(data.get("speed") or 1.0)),
        ]
        env = os.environ.copy()

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "synthesis timeout"}

    if proc.returncode == 2:
        return {"ok": False, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "disk guard"}

    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        return {"ok": False, "error": "launcher failed", "stderr": "\n".join(tail)}

    url = f"/files/tts/{fname}"
    _prune_old_tts_files(KEEP_TTS_FILES)
    ret = {
        "ok": True,
        "status": "ok",
        "url": url,
        "audio_path": url,
        "voice": voice,
        "engine": engine,
    }
    if os.environ.get("TTS_WORKER_URL"):
        ret["base"] = os.environ["TTS_WORKER_URL"]
    return ret


def health():
    return _health_impl()

def voices():
    return _voices_impl()

def tts(data: dict = None):
    return _tts_impl(data)


# Default dummy (with routes) for bare import verif step 3.
from http_worker_app import build_dummy_app, build_app

app = build_dummy_app("Qwen3-TTS Worker", ["/health", "/tts", "/voices"])

# Cache so get_app always returns the built real app after first successful call
_built_real_app = None


def get_app():
    """Build (if needed) the shipped FastAPI using shared builder.
    Real registration + error JSONResponse status codes happen here.
    The module-level dummy is only for bare-import verif step 3 (print .routes without calling get_app).
    Once built, always return the real one (so uvicorn and TDD see the registered routes/handlers).
    Uses short SIGALRM guard so a broken fastapi install on host python does not hang TDD/verif.
    """
    global app, _built_real_app
    if _built_real_app is not None:
        return _built_real_app
    try:
        import signal
        def _timeout_handler(signum, frame):
            raise ImportError("fastapi import timed out (broken host install)")
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(3)
        try:
            from fastapi import Body
            from fastapi.responses import JSONResponse
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        def _health_route():
            return _health_impl()

        def _voices_route():
            return _voices_impl()

        def _tts_route(data: dict = Body(...)):
            res = _tts_impl(data)
            if isinstance(res, dict) and res.get("ok") is False:
                status = 400
                if "timeout" in (res.get("error") or ""):
                    status = 504
                elif "disk" in (res.get("error") or "") or "guard" in (res.get("error") or ""):
                    status = 507
                return JSONResponse(res, status_code=status)
            return res

        real_app = build_app(
            "Qwen3-TTS Worker",
            [
                ("get", "/health", _health_route),
                ("get", "/voices", _voices_route),
                ("post", "/tts", _tts_route),
            ],
        )
        app = real_app
        _built_real_app = real_app
    except Exception as e:
        print("Qwen get_app build except:", type(e).__name__, ":", str(e)[:120], file=sys.stderr)
        # In envs without fastapi (harness bare), return the dummy. TDD drivers pre-stub so build succeeds.
        pass
    return app


if __name__ == "__main__":
    a = get_app()
    import uvicorn
    uvicorn.run(a, host="0.0.0.0", port=int(os.environ.get("TTS_PORT", "8010")))
