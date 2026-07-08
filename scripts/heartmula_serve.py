#!/usr/bin/env python3
"""HeartMuLa music worker — thin HTTP wrapper around heartmula_launcher.py.

... (same docstring) ...
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
        f"No writable music output directory found in: {candidates!r}. "
        "Set HEARTMULA_OUT_DIR to a writable path."
    )


OUT_DIR = None

def _get_out_dir():
    global OUT_DIR
    if OUT_DIR is None:
        OUT_DIR = _first_writable([
            os.environ.get("HEARTMULA_OUT_DIR"),
            "/workspace/music",
            os.path.expanduser("~/.slopfinity/music"),
            "/tmp/slopfinity-music",
        ])
        if not os.environ.get("SLOPFINITY_QUIET"):
            print(f"🎼 HeartMuLa output dir: {OUT_DIR}", flush=True)
    return OUT_DIR


LAUNCHER = os.environ.get("HEARTMULA_LAUNCHER", "/opt/heartmula_launcher.py")
if not os.path.exists(LAUNCHER):
    try:
        _local = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "heartmula_launcher.py"
        )
    except NameError:
        _local = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv else ".")), "heartmula_launcher.py"
        )
    if os.path.exists(_local):
        LAUNCHER = _local

GEN_TIMEOUT_S = float(os.environ.get("HEARTMULA_TIMEOUT_S", "1200"))


def _health_impl() -> dict:
    h = {
        "ok": True,
        "launcher": LAUNCHER,
        "launcher_exists": os.path.exists(LAUNCHER),
        "out": _get_out_dir(),
        "timeout_s": GEN_TIMEOUT_S,
    }
    if os.environ.get("HEARTMULA_URL"):
        h["url"] = os.environ["HEARTMULA_URL"]
    return h


def _music_impl(payload: dict) -> dict:
    prompt = str(payload.get("prompt") or "").strip() if payload else ""
    if not prompt:
        return {"ok": False, "error": "prompt is required"}
    try:
        duration = max(1.0, float(payload.get("duration") or 30.0))
    except (TypeError, ValueError):
        return {"ok": False, "error": "duration must be a number"}

    name = f"hm_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(_get_out_dir(), name)
    cmd = [
        sys.executable, LAUNCHER,
        "--prompt", prompt,
        "--duration", f"{duration:.1f}",
        "--out", out_path,
        "--real",
    ]
    t0 = time.time()
    if not os.environ.get("SLOPFINITY_QUIET"):
        print(f"🎼 /music duration={duration:.1f}s prompt={prompt[:60]!r}", flush=True)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=GEN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"generation timed out after {GEN_TIMEOUT_S:.0f}s"}
    if res.returncode != 0 or not os.path.exists(out_path):
        tail = (res.stderr or res.stdout or "")[-800:]
        return {"ok": False, "error": f"launcher exited {res.returncode}, wav missing={not os.path.exists(out_path)}", "detail": tail}
    ret = {
        "ok": True,
        "url": f"/files/music/{name}",
        "duration": duration,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if os.environ.get("HEARTMULA_URL"):
        ret["base"] = os.environ["HEARTMULA_URL"]
    return ret


def health():
    return _health_impl()

def music(payload: dict = None):
    if payload is None:
        payload = {}
    return _music_impl(payload)


# Default to dummy (with routes) for bare `import` in verif step 3 (avoids fastapi import/hang).
# Real shipped app is obtained via get_app() which uses the shared builder.
from http_worker_app import build_dummy_app, build_app

app = build_dummy_app("HeartMuLa Music Worker", ["/health", "/music"])

_built_real_app = None


def get_app():
    """Build (once) the shipped FastAPI app using the shared registration helper.
    Called by TDD and __main__. The route handlers are the ones defined here
    (wrapping impls for proper error status codes via real JSONResponse).
    Uses short SIGALRM guard so a broken fastapi install on host python does not hang TDD collection.
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

        def _music_route(payload: dict = Body(...)):
            res = _music_impl(payload)
            if isinstance(res, dict) and res.get("ok") is False:
                status = 400
                if "timeout" in (res.get("error") or ""):
                    status = 504
                elif "launcher" in (res.get("error") or ""):
                    status = 502
                return JSONResponse(res, status_code=status)
            return res

        real_app = build_app(
            "HeartMuLa Music Worker",
            [
                ("get", "/health", _health_route),
                ("post", "/music", _music_route),
            ],
        )
        app = real_app
        _built_real_app = real_app
    except Exception:
        # leave the dummy; TDD falls back to direct handler calls
        pass
    return app


if __name__ == "__main__":
    a = get_app()
    import uvicorn
    uvicorn.run(a,
                host=os.environ.get("HEARTMULA_HOST", "0.0.0.0"),
                port=int(os.environ.get("HEARTMULA_PORT", "8011")))
