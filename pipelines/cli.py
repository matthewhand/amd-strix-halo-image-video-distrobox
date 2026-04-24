"""Shared argparse helpers — keeps CLI surface consistent across wave scripts.

Backward-compatible: every flag added here matches the name and behavior of
existing per-script flags, so the user's muscle memory still works.
"""
import argparse

from . import cost_model

TONES = ["ironic", "sardonic", "comedic", "absurd"]
AUDIO_KINDS = ["prescribed_speech", "general_speech", "sound_effects", "music"]
MODERATION_CATEGORIES = [
    "safe_baseline",
    "violence_action",
    "weapons",
    "suggestive_intimate",
    "regulated_substances",
    "medical_distress",
    "frightening_imagery",
]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach the standard wave-runner flag set to a parser."""
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--skip-qwen", action="store_true",
                        help="Skip Qwen phase, use existing seed images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan + estimate, run nothing")
    parser.add_argument("--scenes", nargs="+",
                        help="Subset of scene labels (or 1-indexed integers)")
    parser.add_argument("--scenes-per-tone", type=int, default=None,
                        help="Cap scenes per tone")
    parser.add_argument("--hours", type=float, default=None,
                        help="Time budget in hours; planner trims to fit")
    parser.add_argument("--per-scene-min", type=float, default=None,
                        help="Override estimated wall-time per LTX scene")
    parser.add_argument("--qwen-min-per-image", type=float,
                        default=cost_model.QWEN_MIN_PER_IMAGE,
                        help="Override Qwen still-image cost (default 3.0 min)")
    parser.add_argument("--tones", nargs="+", choices=TONES,
                        help="Restrict to a subset of tones")
    parser.add_argument("--audio-kinds", nargs="+", choices=AUDIO_KINDS,
                        help="Restrict to a subset of audio kinds")
    parser.add_argument("--moderation-categories", nargs="+",
                        choices=MODERATION_CATEGORIES,
                        help="Restrict to a subset of moderation categories")
