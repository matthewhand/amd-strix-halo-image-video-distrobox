"""Known-broken model/config states on this hardware (gfx1151 / Strix Halo).

Single source of truth, consumed by:
  - slopfinity.routers.config.update_config  → surfaces warnings to the dashboard UI
  - run_fleet.py / slopfinity.worker_sh       → guards so a selection can't GPU-hang

Background (empirically verified 2026-06-03): ERNIE-Image's VAE decode
hard-hangs the GPU (``HW Exception ... GPU Hang``) above 512² on gfx1151 —
the CK grouped-conv MIOpen kernel isn't resolved at runtime. WAN 2.x video
is timeout-prone here, and qwen-tts is broken. Keep this list in sync with
COMPATIBILITY_MATRIX.md and the slop-permutation skill notes.
"""
from __future__ import annotations

# ERNIE VAE decode hard-hangs the GPU above this square dimension on gfx1151.
ERNIE_MAX_DIM = 512

# Declarative registry. Each rule matches ONE config field against a value
# (or list of values). `auto_fix=True` means the pipeline guard silently
# corrects the state (so the UI says "capped" rather than "this will break").
KNOWN_ISSUES = [
    {
        "id": "ernie-hires-vae-hang",
        "role": "base_model",
        "value": "ernie",
        "severity": "danger",
        "message": (
            f"ERNIE-Image GPU-hangs during VAE decode above {ERNIE_MAX_DIM}² on "
            f"this GPU (gfx1151) — it runs capped to {ERNIE_MAX_DIM}×{ERNIE_MAX_DIM}."
        ),
        "auto_fix": True,
    },
    {
        "id": "wan-video-flake",
        "role": "video_model",
        "value": ["wan2.2", "wan2.5"],
        "severity": "warning",
        "message": (
            "WAN 2.x video is unreliable on this hardware — expect ComfyUI "
            "timeouts. LTX-2.3 is the stable video model."
        ),
        "auto_fix": False,
    },
    {
        "id": "qwen-tts-broken",
        "role": "tts_model",
        "value": "qwen-tts",
        "severity": "warning",
        "message": (
            "qwen-tts is broken on this hardware (gfx1151 + disk-low guard). "
            "Use kokoro or dramabox."
        ),
        "auto_fix": False,
    },
]


def _matches(rule: dict, value) -> bool:
    want = rule["value"]
    if isinstance(want, (list, tuple, set)):
        return value in want
    return value == want


def check_config(config: dict) -> list[dict]:
    """Return UI-facing warnings ({id, role, severity, message}) for every
    known-broken state present in `config`. Empty list ⇒ nothing to warn about."""
    out = []
    for rule in KNOWN_ISSUES:
        if _matches(rule, config.get(rule["role"])):
            out.append({
                "id": rule["id"],
                "role": rule["role"],
                "severity": rule["severity"],
                "message": rule["message"],
            })
    return out


def clamp_ernie_dims(width: int | None = None, height: int | None = None) -> tuple[int, int]:
    """Clamp ERNIE width/height to the GPU-safe ceiling. None ⇒ the ceiling."""
    w = min(int(width), ERNIE_MAX_DIM) if width else ERNIE_MAX_DIM
    h = min(int(height), ERNIE_MAX_DIM) if height else ERNIE_MAX_DIM
    return w, h
