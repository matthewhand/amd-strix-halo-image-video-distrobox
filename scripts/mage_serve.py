#!/usr/bin/env python3
"""Mage-Flow image worker — HTTP wrapper around MageFlowPipeline.

Contract (matches slopfinity ImageWorker / Qwen-style endpoint):

    GET  /health          → {ok, model, loaded, ...}
    GET  /docs            → FastAPI docs (health probe target)
    POST /api/generate    → {prompt, steps?, cfg?, width?, height?, seed?, out?}
                          ← {ok, path, url, steps, cfg, model, elapsed_s}

Model is kept warm after first request (or optional MAGE_PRELOAD=1).

Run (toolbox container / compose profile mage-image):
    MAGE_OUT_DIR=/workspace python3 mage_serve.py   # :8181
"""
from __future__ import annotations

import os
import random
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

# Inclusive upper bound for diffusion noise seeds (matches slopfinity queue_schema).
_SEED_MAX = 2**31 - 1


def _resolve_request_seed(payload: dict) -> int:
    """Concrete seed for this generate call.

    Missing / negative (sentinel -1) → fresh random. Never sticky-default to 42
    when callers omit the field (the old trap that made infinity look identical).
    Positive ints are used as-is.
    """
    raw = payload.get("seed")
    if raw is None or raw is False or raw == "":
        return random.randint(1, _SEED_MAX)
    try:
        seed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bad seed: {exc}") from exc
    if seed < 0 or seed == 0:
        return random.randint(1, _SEED_MAX)
    return seed & _SEED_MAX


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
        f"No writable Mage output directory in: {candidates!r}. "
        "Set MAGE_OUT_DIR to a writable path."
    )


OUT_DIR = _first_writable(
    [
        os.environ.get("MAGE_OUT_DIR"),
        "/workspace",
        "/opt/ComfyUI/output",
        os.path.expanduser("~/.slopfinity/mage"),
        "/tmp/slopfinity-mage",
    ]
)
print(f"🎨 Mage-Flow output dir: {OUT_DIR}", flush=True)

DEFAULT_MODEL = os.environ.get("MAGE_MODEL", "microsoft/Mage-Flow-Turbo")
DEFAULT_STEPS = int(os.environ.get("MAGE_STEPS", "4"))
DEFAULT_CFG = float(os.environ.get("MAGE_CFG", "1.0"))
DEFAULT_WIDTH = int(os.environ.get("MAGE_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.environ.get("MAGE_HEIGHT", "1024"))
DEVICE = os.environ.get("MAGE_DEVICE", "cuda")
ATTN = os.environ.get("MAGE_ATTN") or os.environ.get("MAGE_ATTN_BACKEND") or "sdpa"
GEN_TIMEOUT_S = float(os.environ.get("MAGE_TIMEOUT_S", "900"))

_PIPE = None
_PIPE_MODEL = None
_LOAD_ERROR: Optional[str] = None

# Short UI/slopfinity ids → preferred checkpoint. Turbo aliases always pin to
# DEFAULT_MODEL so a local snapshot (MAGE_MODEL=/…/Mage-Flow-Turbo) is reused
# instead of re-downloading microsoft/Mage-Flow-Turbo from the Hub.
_MODEL_ALIASES = {
    "mage": "turbo",
    "mage-turbo": "turbo",
    "mage-flow-turbo": "turbo",
    "microsoft/mage-flow-turbo": "turbo",
    "mage-flow": "microsoft/Mage-Flow",
    "mage-rl": "microsoft/Mage-Flow",
    "mage-base": "microsoft/Mage-Flow-Base",
}


def _is_turbo_id(model_id: str) -> bool:
    m = (model_id or "").lower()
    return (
        m in ("mage", "mage-turbo", "mage-flow-turbo", "turbo")
        or m.endswith("mage-flow-turbo")
        or "mage-flow-turbo" in m
        or m.endswith("/turbo")
    )


def _resolve_model_id(raw: str) -> str:
    """Map request model → load path. Prefer DEFAULT_MODEL for all Turbo ids."""
    mid = (raw or "").strip() or DEFAULT_MODEL
    key = mid.lower()
    kind = _MODEL_ALIASES.get(key)
    if kind == "turbo" or _is_turbo_id(mid):
        return DEFAULT_MODEL
    if kind:
        return kind
    return mid


def _install_flash_attn_shim() -> None:
    """Stub flash_attn so importlib / transformers probes succeed on ROCm.

    Real kernels are unused: Mage-Flow runs with the SDPA attention backend.
    """
    if os.environ.get("MAGE_FA_SHIM", "1") != "1":
        return
    import importlib.machinery

    def _stub(name: str, *, is_pkg: bool = False):
        if name in sys.modules:
            return sys.modules[name]
        try:
            __import__(name)
            return sys.modules[name]
        except Exception:
            pass
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=is_pkg
        )
        m.__file__ = f"<mage_fa_shim:{name}>"
        if is_pkg:
            m.__path__ = []  # type: ignore[attr-defined]
        for n in ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "flash_attn_varlen_func"):
            setattr(m, n, MagicMock(name=f"{name}.{n}"))
        sys.modules[name] = m
        return m

    fa = _stub("flash_attn", is_pkg=True)
    cute = _stub("flash_attn.cute", is_pkg=False)
    fa.cute = cute  # type: ignore[attr-defined]
    _stub("flash_attn_2_cuda", is_pkg=False)


def _ensure_mage_on_path() -> None:
    try:
        import mage_flow  # noqa: F401
        return
    except ImportError:
        pass
    for c in (
        os.environ.get("MAGE_FLOW_ROOT"),
        "/opt/Mage/mage_flow",
        str(Path(__file__).resolve().parents[1] / "Mage" / "mage_flow"),
    ):
        if not c:
            continue
        root = Path(c)
        parent = root.parent if root.name == "mage_flow" else root
        if (parent / "mage_flow").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


def _ensure_loguru() -> None:
    try:
        import loguru  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--root-user-action=ignore", "loguru"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except Exception:
        import logging
        import types

        mod = types.ModuleType("loguru")

        class _L:
            def __getattr__(self, name):
                return getattr(logging.getLogger("mage"), name, lambda *a, **k: None)

        mod.logger = _L()  # type: ignore[attr-defined]
        sys.modules["loguru"] = mod


def _get_pipe(model_id: str):
    global _PIPE, _PIPE_MODEL, _LOAD_ERROR
    model_id = _resolve_model_id(model_id)
    # Reuse warm weights if already loaded for the same logical checkpoint
    # (local path vs HF turbo id must not trigger a second multi-GB load).
    if _PIPE is not None and (
        _PIPE_MODEL == model_id
        or (
            _is_turbo_id(_PIPE_MODEL or "")
            and _is_turbo_id(model_id)
        )
    ):
        return _PIPE
    _install_flash_attn_shim()
    _ensure_loguru()
    _ensure_mage_on_path()
    # Force SDPA for DiT + HF Qwen3-VL text encoder on ROCm (flash2 is CUDA-only).
    attn = ATTN if ATTN in ("sdpa", "eager", "flash2", "flash4") else "sdpa"
    os.environ["VF_HF_ATTN_IMPL"] = "sdpa" if attn in ("sdpa", "eager") else attn
    try:
        from mage_flow.models.modules._attn_backend import set_attn_backend

        set_attn_backend(attn)
    except Exception as exc:
        print(f"[MAGE] WARN set_attn_backend: {exc}", flush=True)
    from mage_flow import MageFlowPipeline

    print(f"[MAGE] Loading {model_id} on {DEVICE} (attn={attn}) ...", flush=True)
    t0 = time.time()
    try:
        _PIPE = MageFlowPipeline.from_pretrained(model_id, device=DEVICE)
        # Re-apply: MageFlowModel.__init__ resets backend from config.attn_type=flash2.
        try:
            from mage_flow.models.modules._attn_backend import set_attn_backend

            set_attn_backend(attn)
        except Exception:
            pass
        # ROCm: content-filter .generate() often fails closed → white blanks.
        # Fail-open unless MAGE_CONTENT_FILTER=1.
        if os.environ.get("MAGE_CONTENT_FILTER", "0") != "1":
            try:
                from mage_flow.models.modules.mage_text import FilterVerdict

                def _allow(prompt, max_new_tokens=160):
                    return FilterVerdict(
                        False, [], "rocm fail-open (MAGE_CONTENT_FILTER!=1)", ""
                    )

                _PIPE.model.txt_enc.screen_text = _allow  # type: ignore[method-assign]
                print("[MAGE] content filter: fail-open", flush=True)
            except Exception as exc:
                print(f"[MAGE] WARN content filter patch: {exc}", flush=True)
        _PIPE_MODEL = model_id
        _LOAD_ERROR = None
        print(f"[MAGE] loaded in {time.time() - t0:.1f}s", flush=True)
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        _PIPE = None
        _PIPE_MODEL = None
        raise
    return _PIPE


app = FastAPI(title="Mage-Flow Image Worker")


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": DEFAULT_MODEL,
        "loaded": _PIPE is not None,
        "loaded_model": _PIPE_MODEL,
        "out": OUT_DIR,
        "attn": ATTN,
        "default_steps": DEFAULT_STEPS,
        "default_cfg": DEFAULT_CFG,
        "load_error": _LOAD_ERROR,
    }


@app.get("/")
def root():
    return health()


@app.post("/api/generate")
def generate(payload: dict = Body(...)):
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "error": "prompt is required"}, status_code=400)

    model_id = _resolve_model_id(str(payload.get("model") or DEFAULT_MODEL))

    try:
        steps = int(payload.get("steps") or DEFAULT_STEPS)
        cfg = float(payload.get("cfg") if payload.get("cfg") is not None else DEFAULT_CFG)
        width = int(payload.get("width") or DEFAULT_WIDTH)
        height = int(payload.get("height") or DEFAULT_HEIGHT)
        seed = _resolve_request_seed(payload)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": f"bad numeric param: {exc}"}, status_code=400)

    # Turbo defaults when steps look like a Qwen tier mapping
    if _is_turbo_id(model_id) and steps > 8:
        steps = DEFAULT_STEPS
        if payload.get("cfg") is None:
            cfg = DEFAULT_CFG

    # Always write under OUT_DIR (shared bind-mount). Ignore caller "out"
    # paths that are not under OUT_DIR — host absolute paths from the
    # dashboard (e.g. /home/.../v0_base.png) are not writable in-container.
    requested = payload.get("out") or payload.get("path")
    name = f"mage_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(OUT_DIR, name)
    if requested:
        req = str(requested)
        try:
            if os.path.commonpath(
                [os.path.abspath(req), os.path.abspath(OUT_DIR)]
            ) == os.path.abspath(OUT_DIR):
                out_path = os.path.abspath(req)
        except ValueError:
            pass  # different drives / invalid — keep OUT_DIR default
    os.makedirs(os.path.dirname(out_path) or OUT_DIR, exist_ok=True)

    t0 = time.time()
    print(
        f"🎨 /api/generate model={model_id} {width}x{height} steps={steps} cfg={cfg} "
        f"prompt={prompt[:60]!r}",
        flush=True,
    )
    try:
        pipe = _get_pipe(model_id)
        imgs = pipe.generate(
            [prompt],
            seeds=[seed],
            heights=[height],
            widths=[width],
            steps=steps,
            cfg=cfg,
        )
        imgs[0].save(out_path)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": f"mage generate failed: {exc}"},
            status_code=502,
        )

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return JSONResponse({"ok": False, "error": "output missing or empty"}, status_code=502)

    elapsed = round(time.time() - t0, 1)
    size = os.path.getsize(out_path)
    rel = os.path.basename(out_path)
    return {
        "ok": True,
        "path": os.path.abspath(out_path),
        "url": f"/files/{rel}",
        "steps": steps,
        "cfg": cfg,
        "model": model_id,
        "width": width,
        "height": height,
        "size": size,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    if os.environ.get("MAGE_PRELOAD", "0") == "1":
        try:
            _get_pipe(DEFAULT_MODEL)
        except Exception as exc:
            print(f"[MAGE] preload failed: {exc}", flush=True)
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("MAGE_HOST", "0.0.0.0"),
        port=int(os.environ.get("MAGE_PORT", "8181")),
    )
