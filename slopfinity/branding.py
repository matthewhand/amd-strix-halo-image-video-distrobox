"""Branding profile loader.

Users can customise every user-facing string and colour without touching
code. Profiles live in ``<repo>/branding/<name>.json`` (gitignored); the
committed ``<name>.json.example`` files are fallbacks so fresh clones
still render.

Active profile is selected by ``config.json -> branding.active`` (string
name). If the named profile isn't found, the loader falls back through:

    branding/<name>.json
    branding/<name>.json.example
    branding/slopfinity.json
    branding/slopfinity.json.example
    BUILTIN_DEFAULTS   (hard-coded below, guarantees the UI never breaks)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

BUILTIN_DEFAULTS: dict[str, Any] = {
    "app": {
        "name": "Slopfinity",
        "tagline": "Philosophical Video Fleet",
        "short_name": "slop",
        "logo_emoji": "♾️",
    },
    "infinity_mode": {
        "name": "Smithgetti Mode",
        "emoji": "🍝",
        "description": "Autonomous conceptual loops",
    },
    "theme": {
        "default": "dracula",
        "available": ["dracula", "synthwave", "night", "forest"],
    },
    "colors": {
        "primary": "#ff79c6",
        "secondary": "#bd93f9",
        "accent": "#50fa7b",
        "warning": "#f1fa8c",
        "danger": "#ff5555",
    },
    "labels": {
        "pipeline": "Pipeline Config",
        "queue": "Mission Queue",
        "inject": "Inject Task",
        "ai_magic": "AI Magic",
        "gallery_live": "Live Gallery",
        "gallery_completed": "Completed Gallery",
        "gallery_keyframes": "Conceptual Keyframes",
        "concept": "Concept",
        "base_image": "Base Image",
        "video_chain": "Video Chain",
        "audio_music": "Audio / Music",
        "post_process": "Post Process",
        "final_merge": "Final Merge",
    },
    "footer": {"tagline": "STRIX HALO GFX1151 • UNIFIED ARCHITECTURE"},
}


def _repo_root() -> Path:
    # slopfinity/branding.py -> slopfinity/ -> repo root
    return Path(__file__).resolve().parent.parent


def branding_dir() -> Path:
    override = os.environ.get("SLOPFINITY_BRANDING_DIR")
    if override:
        return Path(override)
    return _repo_root() / "branding"


def list_profiles() -> list[str]:
    """Return the set of profile names available (from .json or .json.example)."""
    d = branding_dir()
    if not d.is_dir():
        return ["slopfinity"]
    names: set[str] = set()
    for p in d.iterdir():
        if p.name.endswith(".json"):
            names.add(p.name[: -len(".json")])
        elif p.name.endswith(".json.example"):
            names.add(p.name[: -len(".json.example")])
    return sorted(names) or ["slopfinity"]


def _load_profile_file(name: str) -> dict[str, Any] | None:
    d = branding_dir()
    for candidate in (d / f"{name}.json", d / f"{name}.json.example"):
        if candidate.is_file():
            try:
                with candidate.open("r") as f:
                    data = json.load(f)
                data.pop("_description", None)
                return data
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(active: str | None = None) -> dict[str, Any]:
    """Return a fully-populated branding dict for the named profile.

    Missing keys are filled from ``slopfinity`` profile, then from
    ``BUILTIN_DEFAULTS``. Never raises — always returns a usable dict.
    """
    resolved = BUILTIN_DEFAULTS
    if active and active != "slopfinity":
        fallback = _load_profile_file("slopfinity")
        if fallback:
            resolved = _deep_merge(resolved, fallback)
    chosen_name = active or "slopfinity"
    chosen = _load_profile_file(chosen_name)
    if chosen:
        resolved = _deep_merge(resolved, chosen)
    return resolved
