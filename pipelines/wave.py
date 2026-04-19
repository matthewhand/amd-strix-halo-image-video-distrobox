"""Two-phase wave orchestration: Qwen seeds, then LTX-2.3 i2v queue.

This is the loop currently duplicated four times across the wave scripts.
Phase 1 stops the always-on container so Qwen can take all VRAM, then runs
one Qwen container per scene that has a `qwen_prompt`. Phase 2 starts ComfyUI
and submits each scene as an LTX-2.3 i2v job (returns immediately; ComfyUI
serializes the queue).
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Callable

from . import comfy_container, config, qwen_runner
from . import ltx_runner  # noqa: F401  (ensures scripts/ is on sys.path)
from .scenes import Scene

# Re-establish scripts/ on sys.path for create_workflow imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from generate_ltx23_workflow import create_workflow  # noqa: E402
import comfyui_api  # noqa: E402


def _default_prefix(scene: Scene) -> str:
    return f"wave_{scene.label}"


def run_wave(
    scenes: list[Scene],
    *,
    skip_qwen: bool = False,
    watch: bool = False,
    output_prefix_fn: Callable[[Scene], str] | None = None,
    image_path_fn: Callable[[Scene], str] | None = None,
    include_audio: bool = True,
) -> list[Path]:
    """Run a two-phase wave. Returns list of submitted prompt_ids by way of mp4 paths.

    Note: actual mp4 paths are produced asynchronously by ComfyUI. We return
    the *expected* output stems (the prefix becomes the mp4 filename) so the
    caller can grep /tmp/comfy-outputs after the queue drains.
    """
    output_prefix_fn = output_prefix_fn or _default_prefix
    image_path_fn = image_path_fn or (
        lambda s: os.path.join(config.OUTPUT_DIR, f"qwen_input_{s.label}.png")
    )

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Phase 1: Qwen seeds ---
    images: dict[str, str] = {}  # label -> basename for ComfyUI input/
    qwen_scenes = [s for s in scenes if s.qwen_prompt]
    if skip_qwen:
        for s in scenes:
            path = image_path_fn(s)
            if os.path.exists(path):
                images[s.label] = os.path.basename(path)
            else:
                print(f"  MISSING: {path}")
    elif qwen_scenes:
        print("\n--- Phase 1: Qwen image generation ---")
        comfy_container.stop()
        for s in qwen_scenes:
            print(f"\n[{s.label}]")
            path = image_path_fn(s)
            got = qwen_runner.generate_image(s.qwen_prompt, path)
            if got:
                images[s.label] = os.path.basename(got)
        # Pick up any non-Qwen scenes that already have images (rare).
        for s in scenes:
            if s.label not in images:
                path = image_path_fn(s)
                if os.path.exists(path):
                    images[s.label] = os.path.basename(path)

    if not images:
        print("\nNo images available. Exiting.")
        return []

    # --- Phase 2: ComfyUI + LTX queue ---
    print(f"\n--- Phase 2: LTX-2.3 video generation ({len(images)} jobs) ---")
    comfy_container.start()
    if not comfy_container.wait_ready():
        return []

    for label, filename in images.items():
        src = os.path.join(config.OUTPUT_DIR, filename)
        comfy_container.cp_in(src, filename)

    submitted_paths: list[Path] = []
    for s in scenes:
        if s.label not in images:
            continue
        prefix = output_prefix_fn(s)
        wf = create_workflow(
            prompt=s.video_prompt,
            image_filename=images[s.label],
            width=s.width, height=s.height, frames=s.frames, fps=s.fps,
            include_audio=include_audio,
            output_prefix=prefix,
        )
        client_id = str(uuid.uuid4())
        try:
            resp = comfyui_api.submit(wf, config.SERVER, client_id)
            pid = resp.get("prompt_id", "?")
            print(f"  Queued {s.label} {s.width}x{s.height}/{s.frames}f -> {pid}")
            if watch:
                comfyui_api.watch(config.SERVER, client_id, pid)
            submitted_paths.append(Path(config.OUTPUT_DIR) / f"{prefix}.mp4")
        except RuntimeError as e:
            print(f"  FAIL submit {s.label}: {e}")

    return submitted_paths
