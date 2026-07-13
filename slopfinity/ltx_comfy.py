"""LTX-2.3 via ComfyUI HTTP — video i2v, single-frame image, and spatial upscale.

Replaces the missing `/opt/ltx_launcher.py` docker path. Generation stays on
ComfyUI (`SLOPFINITY_COMFY_URL`); lifecycle is owned by service_registry
(ensure comfyui before calling).

Workflows reuse the proven node graph from `scripts/generate_ltx23_workflow.py`
and the spatial upscaler graph from `scripts/upscale_t2v.py` / `upscale_smoke.py`.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

# Repo root for comfy-input / comfy-outputs and scripts/
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

MODEL = "ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA = "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DEFAULT_SIGMAS = (
    "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
)


def comfy_base_url() -> str:
    return (os.environ.get("SLOPFINITY_COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")


def comfy_server_hostport() -> str:
    """Host:port form used by scripts/comfyui_api.py."""
    base = comfy_base_url()
    return base.replace("http://", "").replace("https://", "").rstrip("/")


def comfy_input_dir() -> Path:
    return Path(os.environ.get("COMFY_INPUT") or _ROOT / "comfy-input")


def comfy_output_dir() -> Path:
    return Path(os.environ.get("COMFY_OUTPUTS") or _ROOT / "comfy-outputs")


def _http_json(method: str, url: str, payload: Optional[dict] = None, timeout: float = 60) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def submit_prompt(workflow: dict, *, client_id: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id
    body = _http_json("POST", f"{comfy_base_url()}/prompt", payload, timeout=60)
    pid = body.get("prompt_id")
    if not pid:
        raise RuntimeError(f"Comfy submit missing prompt_id: {body}")
    return str(pid)


def wait_history(
    prompt_id: str,
    *,
    timeout_s: float = 1800,
    poll_s: float = 5.0,
) -> dict:
    """Block until /history/{id} has a terminal status. Returns history entry."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            h = _http_json("GET", f"{comfy_base_url()}/history/{prompt_id}", timeout=15)
            if prompt_id in h:
                entry = h[prompt_id]
                status = (entry.get("status") or {}).get("status_str")
                if status == "success":
                    return entry
                if status == "error":
                    msgs = (entry.get("status") or {}).get("messages") or []
                    for m in msgs:
                        if m and m[0] == "execution_error":
                            raise RuntimeError(
                                f"{m[1].get('exception_type')}: {m[1].get('exception_message')}"
                            )
                    raise RuntimeError(f"Comfy job error: {msgs[:3]}")
                # pending / running
        except RuntimeError:
            raise
        except Exception as ex:
            last_err = ex
        time.sleep(poll_s)
    raise TimeoutError(f"Comfy job {prompt_id} timed out after {timeout_s}s ({last_err})")


def stage_input_file(host_path: str, *, prefix: str = "ltx") -> str:
    """Copy host file into Comfy input dir; return basename for LoadImage."""
    src = Path(host_path)
    if not src.is_file():
        raise FileNotFoundError(f"input missing: {host_path}")
    dest_dir = comfy_input_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex[:10]}{src.suffix or '.png'}"
    dest = dest_dir / name
    shutil.copy2(src, dest)
    return name


def _find_output_file(entry: dict, prefer_exts: Tuple[str, ...] = (".mp4", ".png", ".webm")) -> Optional[Path]:
    """Locate newest matching file from history outputs or comfy-outputs."""
    out = entry.get("outputs") or {}
    names: List[str] = []
    for node_out in out.values():
        if not isinstance(node_out, dict):
            continue
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key) or []:
                fn = item.get("filename")
                if fn:
                    names.append(fn)
                sub = item.get("subfolder") or ""
                if fn and sub:
                    names.append(str(Path(sub) / fn))
    out_dir = comfy_output_dir()
    for n in names:
        p = out_dir / n
        if p.is_file():
            return p
        # also search recursively by basename
        matches = list(out_dir.rglob(Path(n).name))
        if matches:
            return max(matches, key=lambda x: x.stat().st_mtime)
    # Fallback: newest file with preferred extension
    candidates = []
    for ext in prefer_exts:
        candidates.extend(out_dir.rglob(f"*{ext}"))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def _copy_result(src: Path, dest: str) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

def workflow_image(
    prompt: str,
    *,
    width: int = 768,
    height: int = 512,
    seed: Optional[int] = None,
    prefix: str = "ltx_img",
) -> dict:
    """Single-frame LTX image (EmptyLTXVLatentVideo length=1 + SaveImage)."""
    seed = seed if seed is not None else random.randint(1, 10**9)
    return {
        "1": {"class_type": "LowVRAMCheckpointLoader", "inputs": {"ckpt_name": MODEL}},
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {"gemma_path": GEMMA, "ltxv_path": MODEL, "max_length": 1024},
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality", "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "cfg": 1.0,
            },
        },
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "7": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": 1, "batch_size": 1},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["5", 0],
                "sampler": ["6", 0],
                "sigmas": ["7", 0],
                "latent_image": ["9", 0],
            },
        },
        "11": {
            "class_type": "LTXVTiledVAEDecode",
            "inputs": {
                "vae": ["1", 2],
                "latents": ["10", 0],
                "horizontal_tiles": 1,
                "vertical_tiles": 1,
                "overlap": 8,
                "last_frame_fix": False,
                "working_dtype": "float32",
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": prefix, "images": ["11", 0]},
        },
    }


def workflow_video_i2v(
    image_filename: str,
    prompt: str,
    *,
    width: int = 768,
    height: int = 512,
    frames: int = 49,
    fps: int = 24,
    seed: Optional[int] = None,
    prefix: str = "ltx_vid",
    include_audio: bool = False,
) -> dict:
    """I2V via fleet-proven CFGGuider graph (video latent only).

    Note: MultimodalGuider (create_workflow with include_audio) expects AV
    packed latents; using it without audio latents raises unpack errors.
    The simpler CFGGuider path matches run_fleet.generate_video_ltx and is
    reliable for smoke + product i2v.
    """
    seed = seed if seed is not None else random.randint(1, 10**9)
    # Fleet graph (run_fleet.generate_video_ltx) — SaveImage frames, not SaveVideo.
    # We still prefer SaveVideo when available via create_workflow audio path only.
    if include_audio:
        from generate_ltx23_workflow import create_workflow  # type: ignore
        return create_workflow(
            prompt=prompt,
            image_filename=image_filename,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            seed=seed,
            output_prefix=prefix,
            include_audio=True,
            model_name=MODEL,
        )
    return {
        "1": {
            "class_type": "LowVRAMCheckpointLoader",
            "inputs": {"ckpt_name": MODEL},
        },
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {
                "gemma_path": GEMMA,
                "ltxv_path": MODEL,
                "max_length": 1024,
            },
        },
        "3": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "cfg": 1.0,
            },
        },
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            },
        },
        "11": {
            "class_type": "LTXVImgToVideoConditionOnly",
            "inputs": {
                "vae": ["1", 2],
                "image": ["3", 0],
                "latent": ["10", 0],
                "strength": 1.0,
            },
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["6", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["11", 0],
            },
        },
        "13": {
            "class_type": "LTXVTiledVAEDecode",
            "inputs": {
                "vae": ["1", 2],
                "latents": ["12", 0],
                "horizontal_tiles": 1,
                "vertical_tiles": 1,
                "overlap": 8,
                "last_frame_fix": False,
                "working_dtype": "float32",
            },
        },
        "14": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["13", 0], "fps": float(fps)},
        },
        "15": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["14", 0],
                "filename_prefix": prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def workflow_upscale_i2v_spatial(
    image_filename: str,
    prompt: str,
    *,
    width: int = 640,
    height: int = 384,
    frames: int = 25,
    fps: int = 24,
    seed: Optional[int] = None,
    prefix: str = "ltx_up",
) -> dict:
    """I2V (CFGGuider) + LTXVLatentUpsampler x2 + SaveVideo (ltx-spatial)."""
    seed = seed if seed is not None else random.randint(1, 10**9)
    # Prefer LowVRAMLatentUpscaleModelLoader when present (registry node name).
    return {
        "1": {"class_type": "LowVRAMCheckpointLoader", "inputs": {"ckpt_name": MODEL}},
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {"gemma_path": GEMMA, "ltxv_path": MODEL, "max_length": 1024},
        },
        "3": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "cfg": 1.0,
            },
        },
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            },
        },
        "11": {
            "class_type": "LTXVImgToVideoConditionOnly",
            "inputs": {
                "vae": ["1", 2],
                "image": ["3", 0],
                "latent": ["10", 0],
                "strength": 1.0,
            },
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["6", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["11", 0],
            },
        },
        "30": {
            "class_type": "LowVRAMLatentUpscaleModelLoader",
            "inputs": {"model_name": UPSCALER},
        },
        "31": {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {
                "samples": ["12", 0],
                "upscale_model": ["30", 0],
                "vae": ["1", 2],
            },
        },
        "40": {
            "class_type": "LTXVTiledVAEDecode",
            "inputs": {
                "vae": ["1", 2],
                "latents": ["31", 0],
                "horizontal_tiles": 4,
                "vertical_tiles": 4,
                "overlap": 2,
                "last_frame_fix": False,
                "working_dtype": "float32",
            },
        },
        "41": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["40", 0], "fps": float(fps)},
        },
        "42": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["41", 0],
                "filename_prefix": prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


# ---------------------------------------------------------------------------
# High-level generate APIs (return 0 on success, like workers.py)
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    out_path: str,
    *,
    width: int = 768,
    height: int = 512,
    timeout_s: float = 900,
) -> int:
    try:
        prefix = f"ltx_img_{uuid.uuid4().hex[:8]}"
        wf = workflow_image(prompt, width=width, height=height, prefix=prefix)
        pid = submit_prompt(wf)
        entry = wait_history(pid, timeout_s=timeout_s)
        found = _find_output_file(entry, prefer_exts=(".png", ".jpg", ".webp"))
        if not found:
            # prefix search
            matches = list(comfy_output_dir().rglob(f"{prefix}*"))
            found = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
        if not found or not found.is_file():
            return 2
        _copy_result(found, out_path)
        return 0 if Path(out_path).is_file() and Path(out_path).stat().st_size > 0 else 2
    except Exception:
        return 1


def generate_video(
    prompt: str,
    out_path: str,
    *,
    image_path: str = "",
    width: int = 768,
    height: int = 512,
    frames: int = 49,
    fps: int = 24,
    timeout_s: float = 1800,
) -> int:
    try:
        img_name = None
        if image_path:
            img_name = stage_input_file(image_path, prefix="ltx_seed")
        prefix = f"ltx_vid_{uuid.uuid4().hex[:8]}"
        if img_name:
            wf = workflow_video_i2v(
                img_name, prompt, width=width, height=height,
                frames=frames, fps=fps, prefix=prefix, include_audio=False,
            )
        else:
            from generate_ltx23_workflow import create_workflow  # type: ignore
            wf = create_workflow(
                prompt=prompt,
                image_filename=None,
                width=width,
                height=height,
                frames=frames,
                fps=fps,
                output_prefix=prefix,
                include_audio=False,
                model_name=MODEL,
            )
        pid = submit_prompt(wf)
        entry = wait_history(pid, timeout_s=timeout_s)
        found = _find_output_file(entry, prefer_exts=(".mp4", ".webm"))
        if not found:
            matches = list(comfy_output_dir().rglob(f"{prefix}*"))
            found = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
        if not found or not found.is_file():
            return 2
        _copy_result(found, out_path)
        return 0 if Path(out_path).is_file() and Path(out_path).stat().st_size > 0 else 2
    except Exception:
        return 1


def _extract_seed_frame(video_path: str, dest_png: Path) -> bool:
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    # Prefer last frame (end of clip) as upscale seed
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-sseof", "-0.1", "-i", video_path,
        "-frames:v", "1", str(dest_png),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and dest_png.is_file() and dest_png.stat().st_size > 0:
        return True
    # Fallback: first frame
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-frames:v", "1", str(dest_png),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dest_png.is_file() and dest_png.stat().st_size > 0


def upscale_video(
    in_path: str,
    out_path: str,
    *,
    prompt: str = "cinematic continuity, high detail, sharp",
    frames: int = 25,
    timeout_s: float = 1800,
) -> int:
    """Upscale via LTX spatial upsampler (seed frame → short i2v + latent x2).

    True pixel-perfect re-encode of long clips is not exposed by the stock LTX
    graph; this uses the proven LTXVLatentUpsampler path seeded from the input
    clip's last frame so post/upscale stages exercise the real model.
    """
    try:
        if not Path(in_path).is_file():
            return 2
        seed_png = comfy_input_dir() / f"ltx_up_seed_{uuid.uuid4().hex[:8]}.png"
        if not _extract_seed_frame(in_path, seed_png):
            return 2
        img_name = seed_png.name
        prefix = f"ltx_up_{uuid.uuid4().hex[:8]}"
        wf = workflow_upscale_i2v_spatial(
            img_name, prompt, frames=frames, prefix=prefix,
        )
        pid = submit_prompt(wf)
        entry = wait_history(pid, timeout_s=timeout_s)
        found = _find_output_file(entry, prefer_exts=(".mp4", ".webm"))
        if not found:
            matches = list(comfy_output_dir().rglob(f"{prefix}*"))
            found = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
        if not found or not found.is_file():
            return 2
        _copy_result(found, out_path)
        return 0 if Path(out_path).is_file() and Path(out_path).stat().st_size > 0 else 2
    except Exception:
        return 1
