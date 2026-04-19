#!/usr/bin/env python3
"""
Organize /tmp/comfy-outputs/ into a sensible folder layout.

Sources of mess (April 2026):
- PNGs and MP4s share the same flat directory
- Multiple test scripts wrote different prefixes for the same purpose
- chain_*.png ambiguity: some are Qwen seeds, some are extracted frames

Layout this script produces:
    /tmp/comfy-outputs/
    ├── images/         qwen_*  + extracted-frame_*  (PNGs grouped by wave)
    ├── videos/         current waves' LTX outputs (MP4s grouped by wave)
    ├── inspection/     spot-check PNGs from frame extraction
    └── archive/        legacy one-off outputs (i2v_, ltx2_, lowvram_, smoke_)

Renames are minimal and reversible — the wave/scene labels stay; only the
top-level prefix changes for clarity.

Usage:
    python scripts/organize_outputs.py            # apply
    python scripts/organize_outputs.py --dry-run  # preview
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

OUTPUT_DIR = Path("/tmp/comfy-outputs")

# (source_glob, dest_subdir) — order matters: more specific first.
RULES = [
    # Inspection PNGs from spot-check tooling
    ("insp_*.png", "inspection"),
    # Chained-wave outputs: distinguish Qwen seed (1st) from extracted frames
    # (2nd onward). The seed has scene index 1; extracted frames have 2..N.
    # We rename for clarity:
    #   chain_1_*.png             -> images/qwen_chain_1_*.png  (seed)
    #   chain_{2,3,4,5}_*.png     -> images/frame_chain_{N}_*.png (extracted)
    ("chain_*.mp4", "videos"),
    ("chain_1_*.png", "images:rename:qwen_"),  # special: rename
    ("chain_*.png", "images:rename:frame_"),    # everything else extracted
    # Ironic-wave outputs
    ("wave_*.mp4", "videos"),
    ("qwen_input_*.png", "images:rename:qwen_"),
    # Legacy / one-off outputs
    ("i2v_*.mp4", "archive"),
    ("ltx2_*.mp4", "archive"),
    ("lowvram_*.mp4", "archive"),
    ("smoke_*.mp4", "archive"),
    ("test_*.png", "archive"),
]


def plan_moves():
    moves = []
    seen = set()
    for pattern, dest in RULES:
        for src in sorted(OUTPUT_DIR.glob(pattern)):
            if src in seen:
                continue
            seen.add(src)
            if ":rename:" in dest:
                subdir, _, prefix = dest.partition(":rename:")
                target_name = prefix + src.name
                target = OUTPUT_DIR / subdir / target_name
            else:
                target = OUTPUT_DIR / dest / src.name
            moves.append((src, target))
    return moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not OUTPUT_DIR.exists():
        sys.exit(f"{OUTPUT_DIR} doesn't exist")

    moves = plan_moves()
    if not moves:
        print("Nothing to organize.")
        return 0

    by_dest = {}
    for src, target in moves:
        by_dest.setdefault(target.parent.name, []).append((src, target))

    for dest_name in sorted(by_dest):
        items = by_dest[dest_name]
        print(f"\n{dest_name}/  ({len(items)} files)")
        for src, target in items:
            arrow = "->" if src.name != target.name else "  "
            print(f"  {src.name:50s} {arrow} {target.name}")

    if args.dry_run:
        print("\n(dry run — no changes made)")
        return 0

    for src, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))

    untouched = sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())
    if untouched:
        print(f"\nUnclassified (left in root): {len(untouched)}")
        for n in untouched:
            print(f"  {n}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
