#!/usr/bin/env python3
"""Hardlink marathon outputs into Slopfinity EXP_DIR so /assets and /files serve them.

Also writes prompt sidecars (``<file>.json``) from marathon_manifest.jsonl so the
Live Gallery detail pane shows the generation prompt via GET /asset/{filename}.

Usage:
  python3 scripts/publish_marathon_to_slopfinity.py
  SLOPFINITY_EXP_DIR=/path/to/experiments python3 scripts/publish_marathon_to_slopfinity.py
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(
    os.environ.get("SLOPFINITY_EXP_DIR")
    or os.environ.get("SLOPFINITY_STATE_DIR")
    or (ROOT / "comfy-outputs" / "experiments")
)
EXP.mkdir(parents=True, exist_ok=True)

MANIFEST_CANDIDATES = [
    Path(os.environ.get("MAGE_MARATHON_SCRATCH", "")) / "marathon_manifest.jsonl"
    if os.environ.get("MAGE_MARATHON_SCRATCH")
    else None,
    ROOT / "marathon-run" / "marathon_manifest.jsonl",
    Path("/tmp/grok-goal-f1c445a80234/implementer/marathon_manifest.jsonl"),
]


def load_manifest() -> list[dict]:
    for p in MANIFEST_CANDIDATES:
        if p and p.is_file():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return []


def publish(src: Path, dest_name: str) -> str:
    dest = EXP / dest_name
    if dest.exists():
        try:
            if dest.stat().st_ino == src.stat().st_ino:
                return "exists"
            if dest.stat().st_mtime >= src.stat().st_mtime and dest.stat().st_size == src.stat().st_size:
                return "exists"
        except OSError:
            pass
        try:
            dest.unlink()
        except OSError:
            pass
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def write_sidecar(dest: Path, meta: dict) -> None:
    sidecar = Path(str(dest) + ".json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    rows = load_manifest()
    by_tick: dict[int, dict] = {}
    by_name: dict[str, dict] = {}
    for r in rows:
        if r.get("tick") is not None:
            by_tick[int(r["tick"])] = r
        path = r.get("path") or ""
        if path:
            by_name[Path(path).name] = r
            by_name[f"marathon_{Path(path).name}"] = r
        slug = r.get("slug") or ""
        if slug:
            by_name[f"{slug}.png"] = r
            by_name[f"marathon_{slug}.png"] = r

    actions: list[str] = []
    sidecars = 0

    def maybe_sidecar(dest: Path) -> None:
        nonlocal sidecars
        r = by_name.get(dest.name)
        if not r and dest.name.startswith("marathon_t"):
            # marathon_t00661_subject.png → tick 661
            rest = dest.stem[len("marathon_t") :]
            num = ""
            for ch in rest:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num:
                r = by_tick.get(int(num))
        if not r or not r.get("prompt"):
            return
        write_sidecar(
            dest,
            {
                "prompt": r.get("prompt"),
                "model": r.get("model"),
                "kind": (
                    "image"
                    if dest.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                    else "video"
                    if dest.suffix.lower() == ".mp4"
                    else "audio"
                ),
                "tick": r.get("tick"),
                "slug": r.get("slug"),
                "modality": r.get("modality"),
                "source": "marathon_manifest",
                "ok": r.get("ok"),
            },
        )
        sidecars += 1

    for p in sorted((ROOT / "comfy-outputs/mage/marathon").glob("*.png")):
        actions.append(publish(p, f"marathon_{p.name}"))
        maybe_sidecar(EXP / f"marathon_{p.name}")
    for p in sorted((ROOT / "comfy-outputs/mage/marathon_video").glob("*.mp4")):
        actions.append(publish(p, f"marathon_{p.name}"))
        maybe_sidecar(EXP / f"marathon_{p.name}")
    for dname in ("music", "music_marathon", "tts_marathon", "tts"):
        d = ROOT / "comfy-outputs/experiments" / dname
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.wav")):
            name = p.name if p.name.startswith("marathon_") else f"marathon_{p.name}"
            actions.append(publish(p, name))
            maybe_sidecar(EXP / name)

    # Also sidecar any already-published marathon_* without re-copy
    for p in EXP.glob("marathon_*"):
        if p.suffix.lower() in {".png", ".mp4", ".wav"}:
            maybe_sidecar(p)

    print(
        json.dumps(
            {
                "exp": str(EXP),
                "n": len(actions),
                "actions": dict(Counter(actions)),
                "sidecars_written": sidecars,
                "manifest_rows": len(rows),
            },
            indent=2,
        )
    )
    print("Browse: http://127.0.0.1:9099/  → Live Gallery (click asset for prompt)")
    print("API:    curl -s http://127.0.0.1:9099/asset/marathon_<file>.png | jq .prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
