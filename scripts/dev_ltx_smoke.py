#!/usr/bin/env python3
"""
LTX-2.3 dev (non-distilled bf16) smoke test — for use after ROCm stack upgrade.

Purpose: verify whether ltx-2.3-22b-dev.safetensors still NaNs on gfx1151
after a container rebuild with PYTORCH_ROCM_ARCH=gfx1151 and fresh TheRock
wheels.

Known-broken under earlier stack: NaN audio + black video frames at all
tested LoRA strengths, 8 or 30 sigma steps. If this smoke produces a valid
mp4 with non-black frames, we've unlocked the dev model.

Usage:
    python3 scripts/dev_ltx_smoke.py [--lora distillation.safetensors --lora-strength 0.5]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from pipelines import config, comfy_container  # noqa: E402
import comfyui_api  # noqa: E402
from generate_ltx23_workflow import create_workflow  # noqa: E402

DEV_MODEL = "ltx-2.3-22b-dev.safetensors"
SEED_PROMPT = (
    "a tired office worker in a button-down shirt stands at a generic "
    "corporate break room counter, pouring coffee from a glass carafe "
    "into a white ceramic mug, microwave and sink visible in the "
    "background, fluorescent overhead lighting"
)
VIDEO_PROMPT = (
    "the worker pours coffee in slow careful motion, steam rising from "
    "the mug, head tilted slightly down with mild boredom"
)
# Use an already-produced chain scene 1 PNG as the i2v seed so we don't
# need to re-run Qwen. chain_1_attic_discovery.png from earlier runs.
SEED_IMAGE_NAME = "chain_1_attic_discovery.png"


def wait_for_result(server, prompt_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                f"http://{server}/history/{prompt_id}", timeout=5) as r:
                h = json.loads(r.read().decode())
                if h:
                    st = list(h.values())[0].get("status", {})
                    s = st.get("status_str")
                    if s == "success":
                        return ("success", None, int(time.time() - start))
                    if s == "error":
                        msgs = st.get("messages", [])
                        err = "unknown"
                        for m in msgs:
                            if m[0] == "execution_error":
                                err = (f"{m[1].get('exception_type')}: "
                                       f"{m[1].get('exception_message')}")
                        return ("error", err, int(time.time() - start))
        except Exception:
            pass
        time.sleep(15)
    return ("timeout", None, int(time.time() - start))


def ensure_seed_image():
    """Copy the seed image into the container's input dir. Generates a
    placeholder via ffmpeg if the real PNG isn't on disk."""
    src = os.path.join(config.OUTPUT_DIR, SEED_IMAGE_NAME)
    if not os.path.exists(src):
        print(f"  Seed {SEED_IMAGE_NAME} not found at {src}; aborting")
        sys.exit(2)
    ok = comfy_container.cp_in(src, SEED_IMAGE_NAME)
    if not ok:
        sys.exit("  failed to docker cp seed image")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", default=None,
                   help="LoRA filename under models/loras/ (optional)")
    p.add_argument("--lora-strength", type=float, default=0.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--frames", type=int, default=97)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--prefix", default="dev_smoke")
    p.add_argument("--no-audio", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print("LTX-2.3 dev (bf16) smoke — post-ROCm-upgrade retry")
    print(f"  model : {DEV_MODEL}")
    print(f"  lora  : {args.lora} @ {args.lora_strength}")
    print(f"  shape : {args.width}x{args.height} / {args.frames}f @ {args.fps}fps")
    print(f"  audio : {'no' if args.no_audio else 'yes'}")
    print("=" * 70)

    # Start container with current config.DOCKER_ENV (should include
    # PYTORCH_ROCM_ARCH=gfx1151 after latest edit).
    comfy_container.start()
    if not comfy_container.wait_ready(timeout=240):
        sys.exit("  ComfyUI failed to come up")

    ensure_seed_image()

    wf = create_workflow(
        prompt=VIDEO_PROMPT,
        image_filename=SEED_IMAGE_NAME,
        width=args.width, height=args.height,
        frames=args.frames, fps=args.fps,
        include_audio=not args.no_audio,
        model_name=DEV_MODEL,
        lora_name=args.lora,
        lora_strength=args.lora_strength,
        output_prefix=args.prefix,
    )

    client_id = str(uuid.uuid4())
    resp = comfyui_api.submit(wf, config.SERVER, client_id)
    pid = resp.get("prompt_id")
    print(f"  queued dev smoke: {pid}")

    status, err, secs = wait_for_result(config.SERVER, pid, timeout=2400)
    print(f"  {status} in {secs}s{'  — ' + err if err else ''}")

    if status == "success":
        # Verify we have a non-empty mp4 output
        import glob
        matches = sorted(
            glob.glob(f"{config.OUTPUT_DIR}/{args.prefix}_*.mp4"),
            key=os.path.getmtime,
        )
        if matches:
            size_kb = os.path.getsize(matches[-1]) / 1024
            print(f"  output: {os.path.basename(matches[-1])}  {size_kb:.0f} KB")
            if size_kb < 50:
                print("  WARNING: file is tiny — probably black-frame NaN output "
                      "like the previous stack (known-broken symptom)")
                return 1
            print("  → dev model appears to have PRODUCED output. "
                  "Visually inspect the mp4 to confirm it's not black frames.")
            return 0
        print("  SUCCESS reported but no mp4 found — odd")
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
