"""Scene dataclass + JSON load/filter helpers.

A Scene unifies the (label, qwen_prompt, video_prompt, ...) tuples currently
used by every wave script. Optional fields (tone, audio_kind, ...) let the
matrix runner do classifier-eval filtering without subclassing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass
class Scene:
    label: str
    qwen_prompt: str | None  # None means chain-only (uses prior video's last frame)
    video_prompt: str
    width: int = 768
    height: int = 432
    frames: int = 97
    fps: int = 24
    tone: str | None = None
    audio_kind: str | None = None
    moderation_category: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_from_json(path: str) -> list[Scene]:
    """Load scenes from a JSON file.

    Accepts the older formats (`qwen` instead of `qwen_prompt`, `video`
    instead of `video_prompt`) as well as the canonical Scene field names.
    """
    with open(path) as f:
        raw = json.load(f)
    out: list[Scene] = []
    for item in raw:
        out.append(Scene(
            label=item["label"],
            qwen_prompt=item.get("qwen_prompt", item.get("qwen")),
            video_prompt=item.get("video_prompt", item.get("video")),
            width=item.get("width", 768),
            height=item.get("height", 432),
            frames=item.get("frames", 97),
            fps=item.get("fps", 24),
            tone=item.get("tone"),
            audio_kind=item.get("audio_kind"),
            moderation_category=item.get("moderation_category"),
        ))
    return out


def filter_by(
    scenes: Iterable[Scene],
    *,
    tones: list[str] | None = None,
    audio_kinds: list[str] | None = None,
    moderation_categories: list[str] | None = None,
    scenes_per_tone: int | None = None,
) -> list[Scene]:
    """Apply tone / audio / moderation filters and an optional per-tone cap."""
    out: list[Scene] = []
    seen_per_tone: dict[str | None, int] = {}
    for s in scenes:
        if tones and s.tone not in tones:
            continue
        if audio_kinds and s.audio_kind not in audio_kinds:
            continue
        if moderation_categories and s.moderation_category not in moderation_categories:
            continue
        if scenes_per_tone is not None:
            n = seen_per_tone.get(s.tone, 0)
            if n >= scenes_per_tone:
                continue
            seen_per_tone[s.tone] = n + 1
        out.append(s)
    return out
