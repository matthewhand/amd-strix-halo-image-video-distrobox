"""Thin async wrappers around the existing `docker run --rm` fleet calls.

Each wrapper acquires the scheduler's GPU lock + budget gate via
`acquire_gpu(stage, model)` and then shells out to the same docker
command `run_philosophical_experiments.py` uses today.

Deliberately minimal — no replacement for the fleet runner, just the
library surface a future orchestrator can call.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from typing import List, Optional

from .scheduler import acquire_gpu

IMAGE = "amd-strix-halo-image-video-toolbox:latest"


def _hf_cache() -> str:
    return os.path.expanduser("~/.cache/huggingface")


def _workspace() -> str:
    return os.getcwd()


def _base_docker_cmd(extra_env: Optional[List[str]] = None) -> List[str]:
    cmd = ["docker", "run", "--rm"]
    for e in extra_env or []:
        cmd += ["-e", e]
    cmd += [
        "-v", f"{_workspace()}:/workspace",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-w", "/workspace",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        IMAGE,
    ]
    return cmd


# Hard cap (seconds) for docker GPU worker call. On timeout return 124
# so acquire_gpu releases the lock instead of hanging forever.
WORKER_TIMEOUT_S = 600


async def _run(cmd: List[str], timeout: int = WORKER_TIMEOUT_S) -> int:
    def _do() -> int:
        try:
            return subprocess.run(cmd, check=False, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"⏱  worker worker timeout after {timeout}s", flush=True)
            return 124  # conventional timeout exit code → treated as failure
    return await asyncio.to_thread(_do)


async def run_image_qwen(prompt: str, out: str) -> int:
    async with acquire_gpu("image", "qwen"):
        cmd = _base_docker_cmd(["PYTHONPATH=/opt/qwen-image-studio/src"]) + [
            "python3", "/opt/qwen_launcher.py", "generate",
            "--prompt", prompt, "--out", out,
        ]
        return await _run(cmd)


async def run_image_ernie(prompt: str, out: str, steps: int = 8) -> int:
    async with acquire_gpu("image", "ernie"):
        cmd = _base_docker_cmd() + [
            "python3", "/opt/ernie_launcher.py",
            "--prompt", prompt,
            "--model", "baidu/ERNIE-Image-Turbo",
            "--steps", str(steps),
            "--out", out,
        ]
        return await _run(cmd)


def _ltx_prefer_comfy() -> bool:
    return (os.environ.get("SLOPFINITY_LTX_MODE") or "http").strip().lower() != "docker"


async def run_image_ltx(prompt: str, out: str) -> int:
    async with acquire_gpu("image", "ltx-2.3"):
        if _ltx_prefer_comfy():
            try:
                from . import ltx_comfy
                from . import service_registry as _svc
                await asyncio.to_thread(_svc.ensure_for_stage, "image", "ltx-2.3")
                return await asyncio.to_thread(ltx_comfy.generate_image, prompt, out)
            except Exception:
                pass  # fall through to docker launcher
        # Host path for --out may be absolute; keep as-is (workspace mount).
        cmd = _base_docker_cmd(["PYTHONPATH=/workspace"]) + [
            "python3", "/opt/ltx_launcher.py",
            "--mode", "image",
            "--prompt", prompt, "--out", out,
        ]
        return await _run(cmd)


async def run_video_ltx(prompt: str, in_img: str, out: str) -> int:
    async with acquire_gpu("video", "ltx-2.3"):
        if _ltx_prefer_comfy():
            try:
                from . import ltx_comfy
                from . import service_registry as _svc
                await asyncio.to_thread(_svc.ensure_for_stage, "video", "ltx-2.3")
                return await asyncio.to_thread(
                    ltx_comfy.generate_video, prompt, out, image_path=in_img or "",
                )
            except Exception:
                pass
        cmd = _base_docker_cmd(["PYTHONPATH=/workspace"]) + [
            "python3", "/opt/ltx_launcher.py",
            "--mode", "video",
            "--prompt", prompt,
            "--image", in_img,
            "--out", out,
        ]
        return await _run(cmd)


async def run_video_wan(prompt: str, in_img: str, out: str, model: str = "wan2.2") -> int:
    from .wan_cli import wan_launcher_argv, wan_paths

    async with acquire_gpu("video", model):
        cfg = wan_paths(model)
        ckpt = cfg["ckpt"]
        lora = cfg["lora"]
        # Free UMA peers before huge WAN load
        try:
            from . import service_registry as _svc
            await asyncio.to_thread(_svc.ensure_for_stage, "video", model)
        except Exception:
            pass
        extra_vols = [
            "-v", f"{ckpt}:/models/{os.path.basename(ckpt.rstrip('/'))}:ro",
        ]
        if lora and os.path.isdir(lora):
            extra_vols += [
                "-v", f"{lora}:/models/lightning/{os.path.basename(lora.rstrip('/'))}:ro",
            ]
        base = _base_docker_cmd(["WAN_ATTENTION_BACKEND=sdpa"])
        image = base[-1]
        cmd = base[:-1] + extra_vols + [image] + wan_launcher_argv(prompt, in_img, out, model)
        return await _run(cmd)


async def run_audio_heartmula(prompt: str, out: str) -> int:
    async with acquire_gpu("audio", "heartmula"):
        cmd = _base_docker_cmd() + [
            "python3", "/opt/heartmula_launcher.py",
            "--prompt", prompt, "--out", out,
            "--real",
        ]
        return await _run(cmd)


async def run_tts_qwen(text: str, out: str) -> int:
    async with acquire_gpu("tts", "qwen-tts"):
        cmd = _base_docker_cmd() + [
            "python3", "/opt/qwen_tts_launcher.py",
            "--text", text, "--out", out,
        ]
        return await _run(cmd)


async def run_tts_kokoro(text: str, out: str, voice: str = "ryan") -> int:
    async with acquire_gpu("tts", "kokoro"):
        cmd = _base_docker_cmd() + [
            "python3", "/opt/kokoro_launcher.py",
            "--text", text, "--voice", voice, "--out", out,
        ]
        return await _run(cmd)


async def run_upscale_ltx(in_path: str, out: str) -> int:
    async with acquire_gpu("upscale", "ltx-spatial"):
        if _ltx_prefer_comfy():
            try:
                from . import ltx_comfy
                from . import service_registry as _svc
                await asyncio.to_thread(_svc.ensure_for_stage, "upscale", "ltx-spatial")
                return await asyncio.to_thread(ltx_comfy.upscale_video, in_path, out)
            except Exception:
                pass
        cmd = _base_docker_cmd(["PYTHONPATH=/workspace"]) + [
            "python3", "/opt/ltx_launcher.py",
            "--mode", "upscale",
            "--input", in_path, "--out", out,
        ]
        return await _run(cmd)
