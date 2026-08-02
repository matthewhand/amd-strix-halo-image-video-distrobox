#!/usr/bin/env python3
"""
Capacity sweep — climb a (resolution × frames) ladder from smallest to largest
while sweeping LoRA strengths at each rung. Goal: find the maximum viable
(resolution, frames, lora_strength) for a given (model, lora) pair without
guessing.

Default ladder (rungs are SMALLEST first so failures abort early):
    480x320 / 49f       (~3 min/run, smoke baseline)
    640x384 / 49f       (~4 min)
    768x432 / 97f       (~7 min)
    1024x576 / 145f     (~15 min)
    1280x720 / 193f     (~30 min)

Default strengths: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0  (6 variants per rung)

For each rung, ALL strength variants are tested before climbing. If every
variant on a rung fails (OOM, NaN audio, etc.), the climb aborts — the
larger rungs would only fail harder. This bounds total runtime.

Designed for unattended overnight runs. Override any dimension via CLI.

Usage:
    # Default: full 2D sweep
    python tests/run_lora_sweep.py

    # Just one rung
    python tests/run_lora_sweep.py --rungs 768x432:97

    # Climb but only 3 strengths
    python tests/run_lora_sweep.py --strengths 0.2 0.5 0.8

    # Skip the climb, run a flat single config
    python tests/run_lora_sweep.py --rungs 480x320:49 --strengths 0.5

Output naming:
    sweep_<model_short>_<lora_short>_<WxH>_<F>f_s050_NNNNN.mp4
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts"))

from generate_ltx23_workflow import create_workflow  # noqa: E402
import comfyui_api  # noqa: E402

SERVER = "127.0.0.1:8188"
COMFYUI_CONTAINER = "comfyui-ltx23"
OUTPUT_DIR = "/tmp/comfy-outputs"

# (width, height, frames) — smallest first; climb stops at first all-fail rung
DEFAULT_RUNGS = [
    (480, 320, 49),
    (640, 384, 49),
    (768, 432, 97),
    (1024, 576, 145),
    (1280, 720, 193),
]

DEFAULT_STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

DEFAULT_MODEL = "ltx-2.3-22b-dev.safetensors"
DEFAULT_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DEFAULT_PROMPT = (
    "a serene cosmic brain hovers in a starfield gently pulsing with golden light, "
    "audio is ambient cosmic drone with chime accents, no dialogue"
)
DEFAULT_IMAGE = "qwen_input_tone_violence_action_rooftop_chase_floating_brains.png"


def short_name(filename: str, max_len: int = 14) -> str:
    base = filename.replace(".safetensors", "").replace(".sft", "")
    return base[:max_len].rstrip("_-.")


def parse_rung(spec: str) -> tuple[int, int, int]:
    """Parse 'WxH:F' format. e.g. '768x432:97' -> (768, 432, 97)"""
    res, frames = spec.split(":")
    w, h = res.split("x")
    return int(w), int(h), int(frames)


def submit_one(width, height, frames, strength, args, prefix) -> tuple[str, str]:
    label = f"{width}x{height}_{frames}f_s{int(strength * 100):03d}"
    wf = create_workflow(
        prompt=args.prompt,
        image_filename=args.image,
        width=width, height=height, frames=frames, fps=args.fps,
        include_audio=True,
        model_name=args.model,
        lora_name=args.lora if strength > 0 else None,
        lora_strength=strength,
        output_prefix=f"{prefix}_{label}",
    )
    resp = comfyui_api.submit(wf, SERVER, str(uuid.uuid4()))
    return label, resp.get("prompt_id", "?")


def wait_for_pid(pid: str, label: str, timeout_s: int) -> str:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(f"http://{SERVER}/history/{pid}", timeout=5) as r:
                h = json.loads(r.read().decode())
                if h:
                    s = list(h.values())[0].get("status", {}).get("status_str")
                    if s in ("success", "error"):
                        elapsed = int(time.time() - start)
                        print(f"  [{label}] {s} in {elapsed}s")
                        return s
        except Exception:
            pass
        time.sleep(15)
    return "timeout"


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--rungs", nargs="+", type=parse_rung,
                   help=f"Resolution+frames rungs as WxH:F (default: {DEFAULT_RUNGS})")
    p.add_argument("--strengths", nargs="+", type=float, default=DEFAULT_STRENGTHS,
                   help=f"LoRA strengths (default: {DEFAULT_STRENGTHS})")
    p.add_argument("--no-abort-on-fail", action="store_true",
                   help="Continue climbing even if all strengths failed at a rung")
    p.add_argument("--per-rung-timeout", type=int, default=2400,
                   help="Per-job timeout in seconds (default 40min)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lora", default=DEFAULT_LORA)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help="Filename in ComfyUI's input/ dir")
    p.add_argument("--copy-image", help="Host path to docker-cp into input/ first")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rungs = args.rungs or DEFAULT_RUNGS
    prefix = f"sweep_{short_name(args.model)}_{short_name(args.lora)}"

    # Estimate wall time using the same model from cost_model
    try:
        from pipelines.cost_model import estimate_min_per_scene
        per_run_estimates = [estimate_min_per_scene(w, h, f) for w, h, f in rungs]
    except Exception:
        per_run_estimates = [4 * (w * h * f) / (480 * 320 * 49) for w, h, f in rungs]
    total_est = sum(per_run_estimates) * len(args.strengths)

    print("=" * 70)
    print(f"Capacity Sweep: model={args.model} + lora={args.lora}")
    print(f"  Rungs ({len(rungs)} resolutions × {len(args.strengths)} strengths "
          f"= {len(rungs) * len(args.strengths)} runs):")
    for (w, h, f), est in zip(rungs, per_run_estimates):
        print(f"    {w}x{h}/{f}f  ~{est:.1f} min/run × {len(args.strengths)} = ~{est * len(args.strengths):.0f} min")
    print(f"  Strengths: {args.strengths}")
    print(f"  Total estimated wall time: {total_est:.0f} min ({total_est/60:.1f}h)")
    print(f"  Abort on full-rung failure: {not args.no_abort_on_fail}")
    print("=" * 70)

    if args.dry_run:
        return 0

    if args.copy_image:
        fname = os.path.basename(args.copy_image)
        subprocess.run([
            "docker", "cp", args.copy_image,
            f"{COMFYUI_CONTAINER}:/opt/ComfyUI/input/{fname}",
        ], check=True)
        args.image = fname

    results = []  # list of (rung_str, strength, status, output_path)

    for w, h, f in rungs:
        rung_str = f"{w}x{h}/{f}f"
        print(f"\n=== RUNG {rung_str} ===")
        rung_successes = 0
        for s in args.strengths:
            label, pid = submit_one(w, h, f, s, args, prefix)
            print(f"  submitted [{label}] strength={s:.2f} -> {pid[:8]}")
            status = wait_for_pid(pid, label, timeout_s=args.per_rung_timeout)
            results.append((rung_str, s, status))
            if status == "success":
                rung_successes += 1

        if rung_successes == 0 and not args.no_abort_on_fail:
            print(f"\nALL {len(args.strengths)} STRENGTHS FAILED at {rung_str} — aborting climb")
            print("(use --no-abort-on-fail to keep going anyway)")
            break

    # Summary
    print("\n" + "=" * 70)
    print(f"Capacity sweep complete: {len(results)} runs total")
    by_status = {}
    for _, _, status in results:
        by_status[status] = by_status.get(status, 0) + 1
    for status, n in sorted(by_status.items()):
        print(f"  {status:10s}  {n}")
    print()
    print(f"{'rung':20s} {'strength':>8s}  status")
    print("-" * 50)
    for rung_str, s, status in results:
        mark = "OK" if status == "success" else "X "
        print(f"{rung_str:20s} {s:8.2f}  {mark} {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
