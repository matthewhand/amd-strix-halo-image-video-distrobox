#!/usr/bin/env python3
"""HOMIE video worker — HTTP wrapper around subject-to-video generation.

Contract (matches slopfinity VideoWorker / WAN-style endpoint):

    GET  /health          → {ok, loaded, model, ...}
    GET  /docs            → FastAPI docs
    POST /generate        → {prompt, image|images|ref_image, out?, size?, frames?, steps?, cfg?}
                          ← {ok, path, url, model, elapsed_s, ...}
    POST /api/generate    → same (alias)

Pipeline stays warm after first request when HOMIE_PRELOAD=1 or first generate.

Run (compose profile homie-video):
    python3 homie_serve.py   # :8192
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any, List, Optional
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
        f"No writable HOMIE output directory in: {candidates!r}. "
        "Set HOMIE_OUT_DIR to a writable path."
    )


OUT_DIR = _first_writable(
    [
        os.environ.get("HOMIE_OUT_DIR"),
        "/opt/ComfyUI/output/homie",
        "/workspace",
        os.path.expanduser("~/.slopfinity/homie"),
        "/tmp/slopfinity-homie",
    ]
)
print(f"🎬 HOMIE output dir: {OUT_DIR}", flush=True)

DEFAULT_SIZE = os.environ.get("HOMIE_SIZE", "832*480")
DEFAULT_FRAMES = int(os.environ.get("HOMIE_FRAMES", "49"))
DEFAULT_STEPS = int(os.environ.get("HOMIE_STEPS", "50"))
DEFAULT_CFG = float(os.environ.get("HOMIE_CFG", "5.0"))
DEFAULT_FPS = int(os.environ.get("HOMIE_FPS", "16"))
CKPT_DIR = os.environ.get(
    "HOMIE_CKPT_DIR",
    os.environ.get("WAN_CKPT_DIR", "/opt/weights/Wan2.1-T2V-14B-Diffusers"),
)
HOMIE_WEIGHTS = os.environ.get("HOMIE_WEIGHTS", "/opt/weights/HOMIE-Wan-Model")
HOMIE_ROOT = os.environ.get("HOMIE_ROOT", "/opt/HOMIE")
LAUNCHER = os.environ.get(
    "HOMIE_LAUNCHER",
    "/opt/homie_launcher.py" if os.path.isfile("/opt/homie_launcher.py") else "",
)
GEN_TIMEOUT_S = float(os.environ.get("HOMIE_TIMEOUT_S", "3600"))

_PIPE = None
_LOAD_ERROR: Optional[str] = None


def _install_flash_attn_shim() -> None:
    if os.environ.get("HOMIE_FA_SHIM", "1") != "1":
        return
    for mod_name in ("flash_attn", "flash_attn_2_cuda", "flash_attn.cute"):
        if mod_name in sys.modules:
            continue
        try:
            __import__(mod_name)
        except Exception:
            m = types.ModuleType(mod_name)
            if mod_name == "flash_attn":
                m.flash_attn_varlen_func = MagicMock()  # type: ignore[attr-defined]
            for name in ("fwd", "bwd", "varlen_fwd", "varlen_bwd"):
                setattr(m, name, MagicMock())
            sys.modules[mod_name] = m


def _ensure_homie_on_path() -> Path:
    for c in (
        os.environ.get("HOMIE_ROOT"),
        HOMIE_ROOT,
        str(Path(__file__).resolve().parents[1] / "HOMIE"),
        "/opt/HOMIE",
    ):
        if not c:
            continue
        root = Path(c)
        if (root / "generate.py").is_file() and (root / "homie_wan").is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise RuntimeError(
        "HOMIE source not found (set HOMIE_ROOT). "
        "Clone https://github.com/YIYANGCAI/HOMIE"
    )


def _normalize_refs(payload: dict) -> List[str]:
    """Accept image / images / ref_image / references fields."""
    refs: List[str] = []
    for key in ("ref_image", "image", "seed_image"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            refs.extend([p.strip() for p in v.split(",") if p.strip()])
    for key in ("images", "references", "ref_images"):
        v = payload.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    refs.append(item.strip())
                elif isinstance(item, list):
                    # HOMIE meta format: list-of-lists per subject
                    for sub in item:
                        if isinstance(sub, str) and sub.strip():
                            refs.append(sub.strip())
    # de-dupe preserve order
    seen = set()
    out: List[str] = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _get_pipe():
    """Optional in-process warm pipeline (heavy; off by default, use launcher)."""
    global _PIPE, _LOAD_ERROR
    if _PIPE is not None:
        return _PIPE
    if os.environ.get("HOMIE_INPROCESS", "0") != "1":
        return None
    _install_flash_attn_shim()
    root = _ensure_homie_on_path()
    os.chdir(root)
    try:
        import homie_wan
        from homie_wan.configs import get_config

        cfg = get_config(os.environ.get("HOMIE_TASK", "s2v-14B"))
        print(f"[HOMIE] Loading in-process pipeline from {HOMIE_WEIGHTS}…", flush=True)
        t0 = time.time()
        _PIPE = homie_wan.HomieWanS2V(
            config=cfg,
            ckpt_dir=CKPT_DIR,
            homie_ckpt=HOMIE_WEIGHTS,
            device_id=0,
            rank=0,
            t5_cpu=os.environ.get("HOMIE_T5_CPU", "1") == "1",
        )
        _LOAD_ERROR = None
        print(f"[HOMIE] loaded in {time.time() - t0:.1f}s", flush=True)
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        _PIPE = None
        raise
    return _PIPE


def _run_launcher(
    prompt: str,
    refs: List[str],
    out_path: str,
    size: str,
    frames: int,
    steps: int,
    cfg: float,
    fps: int,
    seed: int,
    qwen_feature: Optional[str],
) -> None:
    launcher = LAUNCHER
    if not launcher or not os.path.isfile(launcher):
        # fall back to scripts next to this file
        cand = Path(__file__).resolve().parent / "homie_launcher.py"
        if cand.is_file():
            launcher = str(cand)
        else:
            raise RuntimeError("homie_launcher.py not found")

    cmd = [
        sys.executable,
        launcher,
        "--prompt",
        prompt,
        "--ref-image",
        ",".join(refs),
        "--out",
        out_path,
        "--ckpt-dir",
        CKPT_DIR,
        "--homie-ckpt",
        HOMIE_WEIGHTS,
        "--size",
        size,
        "--frame-num",
        str(frames),
        "--sample-steps",
        str(steps),
        "--sample-guide-scale",
        str(cfg),
        "--sample-fps",
        str(fps),
        "--seed",
        str(seed),
    ]
    if qwen_feature:
        cmd += ["--qwen-feature", qwen_feature]

    env = os.environ.copy()
    env.setdefault("HOMIE_ROOT", str(_ensure_homie_on_path()))
    env.setdefault("HOMIE_FA_SHIM", "1")
    print(f"[HOMIE] exec: {' '.join(cmd[:8])}…", flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        timeout=GEN_TIMEOUT_S,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"homie_launcher rc={proc.returncode}: {err}")


app = FastAPI(title="HOMIE Subject-to-Video Worker")


@app.get("/health")
def health():
    weights_ok = os.path.isdir(HOMIE_WEIGHTS) or os.path.isfile(HOMIE_WEIGHTS)
    ckpt_ok = os.path.isdir(CKPT_DIR)
    return {
        "ok": True,
        "model": "homie",
        "task": os.environ.get("HOMIE_TASK", "s2v-14B"),
        "loaded": _PIPE is not None,
        "load_error": _LOAD_ERROR,
        "out": OUT_DIR,
        "ckpt_dir": CKPT_DIR,
        "ckpt_present": ckpt_ok,
        "homie_weights": HOMIE_WEIGHTS,
        "weights_present": weights_ok,
        "default_size": DEFAULT_SIZE,
        "default_frames": DEFAULT_FRAMES,
        "default_steps": DEFAULT_STEPS,
        "source": "https://github.com/YIYANGCAI/HOMIE",
    }


@app.get("/")
def root():
    return health()


@app.post("/generate")
@app.post("/api/generate")
def generate(payload: dict = Body(...)):
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "error": "prompt is required"}, status_code=400)

    refs = _normalize_refs(payload)
    if not refs:
        return JSONResponse(
            {
                "ok": False,
                "error": "at least one reference image required "
                "(image / ref_image / images)",
            },
            status_code=400,
        )
    missing = [r for r in refs if not os.path.isfile(r)]
    if missing:
        return JSONResponse(
            {"ok": False, "error": f"reference image(s) not found: {missing}"},
            status_code=400,
        )

    size = str(payload.get("size") or DEFAULT_SIZE)
    try:
        frames = int(payload.get("frames") or payload.get("frame_num") or DEFAULT_FRAMES)
        steps = int(payload.get("steps") or payload.get("sample_steps") or DEFAULT_STEPS)
        cfg = float(
            payload.get("cfg")
            if payload.get("cfg") is not None
            else payload.get("sample_guide_scale")
            if payload.get("sample_guide_scale") is not None
            else DEFAULT_CFG
        )
        fps = int(payload.get("fps") or payload.get("sample_fps") or DEFAULT_FPS)
        seed = _resolve_request_seed(payload)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": f"bad numeric param: {exc}"}, status_code=400)

    out_path = payload.get("out") or payload.get("path") or payload.get("save_file")
    if out_path:
        out_path = str(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    else:
        name = f"homie_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        out_path = os.path.join(OUT_DIR, name)

    qwen_feature = payload.get("qwen_feature")
    if qwen_feature:
        qwen_feature = str(qwen_feature)

    t0 = time.time()
    print(
        f"🎬 /generate refs={len(refs)} size={size} frames={frames} steps={steps} "
        f"prompt={prompt[:60]!r}",
        flush=True,
    )
    try:
        # Prefer subprocess launcher (isolates CUDA/ROCm crashes from the HTTP process).
        # Set HOMIE_INPROCESS=1 to keep a warm pipeline instead.
        if os.environ.get("HOMIE_INPROCESS", "0") == "1":
            pipe = _get_pipe()
            from PIL import Image
            from homie_wan.configs import SIZE_CONFIGS
            from homie_wan.utils.utils import cache_video

            images = [Image.open(p).convert("RGB") for p in refs]
            labels = [[chr(ord("a") + i)] for i in range(len(images))]
            video = pipe.generate(
                input_prompt=prompt,
                ref_images=images,
                qwen_feature=None,
                reference_id_labels=labels,
                size=SIZE_CONFIGS.get(size, (832, 480)),
                frame_num=frames,
                shift=3.0,
                sampling_steps=steps,
                guide_scale=cfg,
                seed=seed,
                offload_model=os.environ.get("HOMIE_OFFLOAD", "1") == "1",
            )
            if video is None:
                raise RuntimeError("pipeline returned None")
            cfg_fps = getattr(getattr(pipe, "config", None), "sample_fps", fps)
            cache_video(
                video[None],
                save_file=out_path,
                fps=cfg_fps or fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            _run_launcher(
                prompt=prompt,
                refs=refs,
                out_path=out_path,
                size=size,
                frames=frames,
                steps=steps,
                cfg=cfg,
                fps=fps,
                seed=seed,
                qwen_feature=qwen_feature,
            )
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": f"homie generate failed: {exc}"},
            status_code=502,
        )

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return JSONResponse({"ok": False, "error": "output missing or empty"}, status_code=502)

    elapsed = round(time.time() - t0, 1)
    size_b = os.path.getsize(out_path)
    return {
        "ok": True,
        "path": os.path.abspath(out_path),
        "url": f"/files/{os.path.basename(out_path)}",
        "model": "homie",
        "size": size,
        "frames": frames,
        "steps": steps,
        "cfg": cfg,
        "refs": refs,
        "bytes": size_b,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    if os.environ.get("HOMIE_PRELOAD", "0") == "1" and os.environ.get(
        "HOMIE_INPROCESS", "0"
    ) == "1":
        try:
            _get_pipe()
        except Exception as exc:
            print(f"[HOMIE] preload failed: {exc}", flush=True)
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOMIE_HOST", "0.0.0.0"),
        port=int(os.environ.get("HOMIE_PORT", "8192")),
    )
