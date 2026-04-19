"""Wall-time cost model for Strix Halo gfx1151 with --lowvram + distilled-fp8.

Quadratic fit derived from session measurements:
  480x320  *  49f =  7.5 Mvox -> 1.5 min
  768x512  * 121f = 47.6 Mvox -> 4.0 min
  1024x576 * 193f =113.8 Mvox -> 10.3 min
  1280x720 * 145f =133.6 Mvox -> 39.7 min

Plus ~30s/scene fixed for model swap + queue marshalling.
"""
from __future__ import annotations

from .scenes import Scene


QWEN_MIN_PER_IMAGE = 3.0


def estimate_min_per_scene(width: int, height: int, frames: int) -> float:
    """Estimated wall-time for one LTX i2v generation."""
    mvox = (width * height * frames) / 1_000_000
    return 0.5 + 0.07 * mvox + 0.0019 * (mvox ** 2)


def fit_to_budget(
    scenes: list[Scene],
    hours: float,
    *,
    skip_qwen: bool = False,
    per_scene_min: float | None = None,
    qwen_min_per_image: float = QWEN_MIN_PER_IMAGE,
) -> list[Scene]:
    """Trim `scenes` to fit a wall-time budget. Preserves order."""
    if not scenes or hours <= 0:
        return []
    budget_min = hours * 60
    # Use the first scene's size as the per-scene proxy if not supplied.
    if per_scene_min is None:
        s = scenes[0]
        per_scene_min = estimate_min_per_scene(s.width, s.height, s.frames)
    cost_each = per_scene_min + (0 if skip_qwen else qwen_min_per_image)
    if cost_each <= 0:
        return list(scenes)
    k = max(0, int(budget_min // cost_each))
    return list(scenes)[:k]
