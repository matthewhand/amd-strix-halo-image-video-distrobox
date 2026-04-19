"""Shared infra for ComfyUI / LTX-2.3 / Qwen wave scripts.

Public API — import from here for stable callers:

    from pipelines import (
        config, comfy_container, qwen_runner, ltx_runner,
        cost_model, cli,
    )
    from pipelines.scenes import Scene, load_from_json, filter_by
    from pipelines.wave import run_wave
"""
from . import config, comfy_container, qwen_runner, ltx_runner, cost_model, cli
from .scenes import Scene, load_from_json, filter_by
from .wave import run_wave

__all__ = [
    "config",
    "comfy_container",
    "qwen_runner",
    "ltx_runner",
    "cost_model",
    "cli",
    "Scene",
    "load_from_json",
    "filter_by",
    "run_wave",
]
