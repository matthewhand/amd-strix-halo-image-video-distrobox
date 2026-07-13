#!/usr/bin/env python3
"""Serial memory sweep across pipeline permutations.

For each (base × video × audio × tts) perm, walk stages one at a time under
stage_gate hard-floor rules, record MemAvailable before/after warm/park.

Does NOT run full image/video generation (too long / OOM risk). Measures:
  - service warm cost via ensure_up (comfy / heartmula / qwen-tts)
  - budget estimates for docker --rm one-shots (qwen/ernie)

Writes TSV + summary markdown under --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slopfinity.stage_gate import (  # noqa: E402
    DEFAULT_SAFETY_GB,
    can_start,
    headroom_gb,
    need_gb,
    reclaim_all,
    remaining_after_load,
    stage_gate,
)
from slopfinity import service_registry as svc  # noqa: E402

BASES = ["qwen", "ernie", "ltx-2.3"]
VIDEOS = ["ltx-2.3"]
AUDIOS = ["none", "heartmula"]
TTS = ["none", "kokoro", "qwen-tts", "dramabox"]

# role, model → long-lived service id (None = estimate-only / docker --rm)
SERVICE = {
    ("image", "qwen"): None,
    ("image", "ernie"): None,
    ("image", "ltx-2.3"): "comfyui",
    ("video", "ltx-2.3"): "comfyui",
    ("audio", "heartmula"): "heartmula",
    ("tts", "kokoro"): "qwen-tts",
    ("tts", "qwen-tts"): "qwen-tts",
    ("tts", "qwen"): "qwen-tts",
    ("tts", "dramabox"): "qwen-tts",
}


def _stages(base: str, video: str, audio: str, tts: str):
    yield ("image", base)
    if audio and audio != "none":
        yield ("audio", audio)
    if tts and tts != "none":
        tm = "qwen" if tts == "qwen-tts" else tts
        yield ("tts", tm)
    if video and video != "none":
        yield ("video", video)


def measure_stage(role: str, model: str, safety: float) -> dict:
    free0 = headroom_gb()
    need = need_gb(role, model)
    est_after = remaining_after_load(free0, need)
    sid = SERVICE.get((role, model))
    # normalize tts keys
    if sid is None and role == "tts":
        sid = "qwen-tts"

    row = {
        "role": role,
        "model": model,
        "service_id": sid or "",
        "free_before_gb": round(free0, 2),
        "budget_need_gb": need,
        "est_free_after_load_gb": round(est_after, 2),
        "pre_ok_keep_10gb": can_start(free0, need, safety),
        "free_after_warm_gb": "",
        "delta_warm_gb": "",
        "free_after_park_gb": "",
        "post_ok_keep_10gb": "",
        "status": "estimate_only",
        "error": "",
    }

    if sid is None:
        row["status"] = "estimate_only_no_longlived_service"
        return row

    if not can_start(free0, need, safety):
        try:
            reclaim_all(f"sweep pre {role}/{model}", registry=svc)
        except Exception as e:
            row["error"] = f"reclaim:{e}"
        free0 = headroom_gb()
        row["free_before_gb"] = round(free0, 2)
        row["est_free_after_load_gb"] = round(remaining_after_load(free0, need), 2)
        row["pre_ok_keep_10gb"] = can_start(free0, need, safety)

    if not can_start(free0, need, safety):
        row["status"] = "refused_insufficient"
        row["error"] = f"need {need}+{safety}, free {free0:.1f}"
        return row

    try:
        with stage_gate(
            role,
            model,
            safety_gb=safety,
            registry=svc,
            service_id=sid,
            ensure_up=True,
            keep_after=False,
        ) as info:
            # hold warm briefly so RSS settles
            time.sleep(2.0)
            free_w = headroom_gb()
            row["free_after_warm_gb"] = round(free_w, 2)
            row["delta_warm_gb"] = round(free0 - free_w, 2)
            row["post_ok_keep_10gb"] = free_w >= safety
            row["status"] = "warmed"
        # after park
        time.sleep(1.5)
        free_p = headroom_gb()
        row["free_after_park_gb"] = round(free_p, 2)
        _ = info  # gate metadata used for side effects only
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        try:
            reclaim_all(f"sweep err {role}/{model}", registry=svc)
        except Exception:
            pass
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "comfy-outputs" / "experiments" / "mem_sweep"),
    )
    ap.add_argument("--safety-gb", type=float, default=DEFAULT_SAFETY_GB)
    ap.add_argument(
        "--skip-warm",
        action="store_true",
        help="budget/plan only — no ensure_up",
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safety = float(args.safety_gb)

    started = datetime.now(timezone.utc).isoformat()
    free_idle = headroom_gb()
    print(f"# mem_perm_sweep start {started} idle_free={free_idle:.1f}GB safety={safety}")

    # Global reclaim before sweep
    try:
        r = reclaim_all("sweep start", registry=svc)
        print(f"reclaim: {r}")
    except Exception as e:
        print(f"reclaim warn: {e}")
    time.sleep(2)
    free_idle = headroom_gb()

    stage_rows = []
    perm_rows = []
    # Cache per (role,model) warm measurement once
    cache = {}

    for base in BASES:
        for video in VIDEOS:
            for audio in AUDIOS:
                for tts in TTS:
                    label = f"{base}|{video}|{audio}|{tts}"
                    free_perm0 = headroom_gb()
                    stages = list(_stages(base, video, audio, tts))
                    serial_peak = max((need_gb(r, m) for r, m in stages), default=0)
                    naive = sum(need_gb(r, m) for r, m in stages)
                    stage_detail = []
                    ok = True
                    for role, model in stages:
                        key = (role, model)
                        if key not in cache:
                            if args.skip_warm:
                                free0 = headroom_gb()
                                need = need_gb(role, model)
                                cache[key] = {
                                    "role": role,
                                    "model": model,
                                    "service_id": SERVICE.get(key) or "",
                                    "free_before_gb": round(free0, 2),
                                    "budget_need_gb": need,
                                    "est_free_after_load_gb": round(
                                        remaining_after_load(free0, need), 2
                                    ),
                                    "pre_ok_keep_10gb": can_start(free0, need, safety),
                                    "free_after_warm_gb": "",
                                    "delta_warm_gb": "",
                                    "free_after_park_gb": "",
                                    "post_ok_keep_10gb": "",
                                    "status": "estimate_only",
                                    "error": "",
                                }
                            else:
                                print(f"  warm measure {role}/{model} …", flush=True)
                                cache[key] = measure_stage(role, model, safety)
                                stage_rows.append(cache[key])
                                print(
                                    f"    → {cache[key]['status']} "
                                    f"delta={cache[key].get('delta_warm_gb')} "
                                    f"free_warm={cache[key].get('free_after_warm_gb')}",
                                    flush=True,
                                )
                        d = cache[key]
                        stage_detail.append(
                            f"{role}/{model}:need={d['budget_need_gb']}"
                            f":delta={d.get('delta_warm_gb') or 'est'}"
                            f":status={d['status']}"
                        )
                        if d["status"] in ("refused_insufficient", "error"):
                            ok = False
                    free_perm1 = headroom_gb()
                    perm_rows.append(
                        {
                            "base": base,
                            "video": video,
                            "audio": audio,
                            "tts": tts,
                            "label": label,
                            "free_at_perm_start_gb": round(free_perm0, 2),
                            "free_at_perm_end_gb": round(free_perm1, 2),
                            "serial_peak_gb": serial_peak,
                            "serial_need_plus_safety_gb": serial_peak + safety,
                            "naive_all_warm_gb": naive,
                            "naive_plus_safety_gb": naive + safety,
                            "est_ok_serial_10gb_free": free_idle - serial_peak >= safety
                            or serial_peak + safety <= free_idle,
                            "naive_oom_risk": naive + safety > free_idle,
                            "stages": " ; ".join(stage_detail),
                            "all_stages_ok": ok,
                        }
                    )
                    print(
                        f"PERM {label} serial={serial_peak} naive={naive} "
                        f"free={free_perm1:.1f} ok={ok}",
                        flush=True,
                    )

    # Write TSV
    stage_tsv = out / "stage_warm_readings.tsv"
    perm_tsv = out / "perm_readings.tsv"
    if stage_rows:
        with open(stage_tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
            w.writeheader()
            w.writerows(stage_rows)
    with open(perm_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perm_rows[0].keys()))
        w.writeheader()
        w.writerows(perm_rows)

    # Markdown summary
    md = out / "mem_perm_report.md"
    lines = [
        f"# Memory permutation sweep",
        f"",
        f"- started: `{started}`",
        f"- finished: `{datetime.now(timezone.utc).isoformat()}`",
        f"- idle free (after reclaim): **{free_idle:.1f} GB**",
        f"- safety floor: **{safety} GB free after load**",
        f"- host total: ~125 GB UMA",
        f"",
        f"## Stage warm readings (empirical service up)",
        f"",
        f"| role | model | service | free before | need | est free after | Δ warm | free after warm | free after park | status |",
        f"| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    seen = set()
    for r in stage_rows or list(cache.values()):
        k = (r["role"], r["model"])
        if k in seen:
            continue
        seen.add(k)
        lines.append(
            f"| {r['role']} | {r['model']} | {r.get('service_id') or '—'} | "
            f"{r['free_before_gb']} | {r['budget_need_gb']} | {r['est_free_after_load_gb']} | "
            f"{r.get('delta_warm_gb') or '—'} | {r.get('free_after_warm_gb') or '—'} | "
            f"{r.get('free_after_park_gb') or '—'} | {r['status']} |"
        )
    lines += [
        f"",
        f"## All pipeline permutations (n={len(perm_rows)})",
        f"",
        f"| base | video | audio | tts | serial peak | serial+10 | naive all-warm | naive+10 | naive OOM vs idle? | stages ok |",
        f"| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for p in perm_rows:
        lines.append(
            f"| {p['base']} | {p['video']} | {p['audio']} | {p['tts']} | "
            f"{p['serial_peak_gb']} | {p['serial_need_plus_safety_gb']} | "
            f"{p['naive_all_warm_gb']} | {p['naive_plus_safety_gb']} | "
            f"{'YES' if p['naive_oom_risk'] else 'no'} | {p['all_stages_ok']} |"
        )
    lines += [
        f"",
        f"## Files",
        f"- `{stage_tsv}`",
        f"- `{perm_tsv}`",
        f"",
        f"## Notes",
        f"- `delta_warm` = free_before − free_after_warm (positive ⇒ service used RAM).",
        f"- Qwen/Ernie image are docker `--rm` one-shots → estimate_only (no long-lived warm).",
        f"- Rule: free after load must stay ≥ {safety} GB.",
    ]
    md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {md}\n{perm_tsv}\n{stage_tsv}")
    # final reclaim
    try:
        reclaim_all("sweep end", registry=svc)
    except Exception:
        pass
    print(f"final free={headroom_gb():.1f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
