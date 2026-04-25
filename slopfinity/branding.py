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
    # Aesthetics: page-level background controls. Every field is optional;
    # leave `image_url` blank to disable. The dashboard wires this to the
    # body via inline CSS variables so themes / branding profiles can
    # ship a signature look (logo wallpaper, repeating SVG pattern, hero
    # photo) without code changes.
    #
    #   image_url:     URL or relative /static path to a raster image,
    #                  SVG file, or a `data:` URI carrying inline SVG /
    #                  base64 PNG. Repeating patterns work — set
    #                  repeat="repeat" + size="<px>".
    #   svg:           Inline SVG markup. When set (non-empty) it
    #                  takes precedence over image_url and is rendered
    #                  via a `data:image/svg+xml` URI.
    #   repeat:        "repeat" | "repeat-x" | "repeat-y" | "no-repeat" |
    #                  "space" | "round" — passed through to background-repeat.
    #   size:          "cover" | "contain" | a CSS length (e.g. "120px") —
    #                  passed through to background-size.
    #   position:      CSS background-position (default "center").
    #   blur_px:       Pixels of blur applied to the background only when
    #                  cards are overlaid. 0 disables. Cards retain crisp
    #                  text because the blur is on the background layer.
    #   dim_pct:       0..100; how much the background is darkened so
    #                  card text reads cleanly (default 30 = 30 % black).
    #   parallax:      true/false — when true, the background uses
    #                  `background-attachment: fixed` so it stays put
    #                  while the page scrolls.
    "aesthetics": {
        "image_url": "",
        "svg": "",
        "repeat": "no-repeat",
        "size": "cover",
        "position": "center",
        "blur_px": 6,
        "dim_pct": 30,
        "parallax": True,
    },
}


# Environment-variable overlay for aesthetics. These let `.env` (or any
# shell env) override the active branding profile's background without
# editing JSON. Useful for per-deploy tweaks (e.g. a deployment-specific
# logo wallpaper) without forking a branding profile.
_ENV_AESTHETICS_KEYS = {
    "SLOPFINITY_BG_IMAGE":    ("image_url", str),
    "SLOPFINITY_BG_SVG":      ("svg", str),
    "SLOPFINITY_BG_REPEAT":   ("repeat", str),
    "SLOPFINITY_BG_SIZE":     ("size", str),
    "SLOPFINITY_BG_POSITION": ("position", str),
    "SLOPFINITY_BG_BLUR_PX":  ("blur_px", int),
    "SLOPFINITY_BG_DIM_PCT":  ("dim_pct", int),
    "SLOPFINITY_BG_PARALLAX": ("parallax", lambda v: v.strip().lower() in ("1", "true", "yes", "y", "on")),
}


def _env_aesthetics_overlay() -> dict[str, Any]:
    """Build a partial aesthetics dict from environment variables.
    Each SLOPFINITY_BG_* env var maps to one field; values are coerced to
    the right type. Empty/unset vars are skipped so they don't blow away
    profile-supplied defaults."""
    out: dict[str, Any] = {}
    for env_key, (cfg_key, coerce) in _ENV_AESTHETICS_KEYS.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            out[cfg_key] = coerce(raw)
        except (ValueError, TypeError):
            # Bad input → leave field unset; the profile/builtin default
            # remains in effect rather than breaking the page.
            continue
    return out


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
    # Layer .env / process-env overrides onto the aesthetics block last so
    # they win over both profile JSON and builtin defaults.
    env_aesthetics = _env_aesthetics_overlay()
    if env_aesthetics:
        resolved = _deep_merge(resolved, {"aesthetics": env_aesthetics})
    return resolved
