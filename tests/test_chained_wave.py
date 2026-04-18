#!/usr/bin/env python3
"""
Chained-narrative LTX-2.3 wave: last frame of scene N seeds scene N+1.

Theme: "The Coffee Break" — 5 scenes of escalating absurdity. Qwen generates
the first frame; every subsequent scene starts from the previous video's
final frame so the action chains visually into one ~30s continuous arc.

Uses LTX-2.3 22B *dev* + distillation LoRA at strength 0.5 (matches the
official Lightricks 2.3 example). dev gives more headroom for prompt
adherence; distillation LoRA keeps the 8-step inference budget.

Pipeline per scene:
    docker cp <input.png> -> ComfyUI input/
    submit i2v workflow, watch via WebSocket
    docker cp <wave_*.mp4> -> host
    ffmpeg extracts last frame -> next scene's input

Usage:
    python tests/test_chained_wave.py
    python tests/test_chained_wave.py --scenes 1 2     # only first two
    python tests/test_chained_wave.py --skip-qwen      # use existing scene-1 png
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from generate_ltx23_workflow import create_workflow  # noqa: E402
import comfyui_api  # noqa: E402

SERVER = "127.0.0.1:8188"
COMFYUI_CONTAINER = "comfyui-ltx23"
IMAGE = "amd-strix-halo-image-video-toolbox:latest"
OUTPUT_DIR = "/tmp/comfy-outputs"
MODEL = "ltx-2.3-22b-dev.safetensors"
LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
LORA_STRENGTH = 0.5
WIDTH, HEIGHT, FRAMES, FPS = 1024, 576, 145, 24

DOCKER_ENV = [
    "-e", "HSA_OVERRIDE_GFX_VERSION=11.5.1",
    "-e", "LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib",
    "-e", "LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib",
]
DOCKER_GPU = [
    "--device", "/dev/dri", "--device", "/dev/kfd",
    "--security-opt", "seccomp=unconfined",
]

# 5 escalating absurdity beats. Scene 1 has a Qwen seed prompt (static still);
# all scenes have an LTX action+sound prompt. Scene N>1 inherits its visual
# from scene N-1's last frame, so the Qwen prompt for them is None.
SCENES = [
    {
        "label": "1_office_coffee",
        "qwen": "photorealistic mid shot of a tired office worker in a button-down shirt standing at a generic corporate break room counter, pouring coffee from a glass carafe into a white ceramic mug, microwave and sink visible in background, fluorescent overhead lighting, sterile beige walls, mundane Tuesday morning energy",
        "video": "the worker pours coffee in slow careful motion, steam rising from the mug, head tilted slightly down with mild boredom, audio is mundane office break room ambience: distant printer humming, refrigerator compressor cycling, faint hold music from a nearby phone, a dripping tap",
    },
    {
        "label": "2_coffee_swirls",
        "qwen": None,
        "video": "the coffee inside the mug suddenly begins spinning impossibly fast in a tight glowing vortex, faint blue light emanating from inside the cup, the worker's eyes widen and they slowly lower the carafe in confusion, audio shifts: room ambience drops to dead silence then a low electrical hum builds, metal stress sounds, faint whispering",
    },
    {
        "label": "3_dragon_eye",
        "qwen": None,
        "video": "a beam of golden light bursts upward from the mug revealing a tiny shimmering portal hovering above the coffee surface, an enormous slitted reptilian eye peers through the portal blinking once, the worker's hair blows back in the conjured wind, audio is reverb-drenched whoosh into a deep bass dragon snort, crackling magical energy, the worker letting out an involuntary squeak",
    },
    {
        "label": "4_dragon_ride",
        "qwen": None,
        "video": "the worker is now mid-air clinging to the back of a fully grown red scaled dragon flying down the office hallway, papers and cubicle walls exploding outward in their wake, terrified coworkers diving for cover, fluorescent lights raining down sparks, audio is full chaos: dragon roar, building alarms blaring, screaming, glass shattering, wind rush, and orchestral cinematic strings swelling",
    },
    {
        "label": "5_snow_globe_cosmos",
        "qwen": None,
        "video": "the camera rapidly pulls back zooming out through office windows then through the building roof revealing the entire office tower is encased inside a glass snow globe being gently shaken by an enormous cosmic toddler sitting cross-legged on a starry void, dozens of other planet-sized snow globes float around them, audio is full ambient cosmic drone with wind chimes, then a giggling baby laugh that distorts and echoes across the void",
    },
]


def stop_comfyui():
    subprocess.run(["docker", "kill", COMFYUI_CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", COMFYUI_CONTAINER], capture_output=True)
    print("ComfyUI stopped (GPU memory freed)")
    time.sleep(2)


def start_comfyui():
    subprocess.run(["docker", "rm", "-f", COMFYUI_CONTAINER], capture_output=True)
    cmd = [
        "docker", "run", "-d", "--name", COMFYUI_CONTAINER,
        *DOCKER_GPU, *DOCKER_ENV,
        "-p", "8188:8188",
        "-v", os.path.expanduser("~/comfy-models") + ":/opt/ComfyUI/models",
        "-v", f"{OUTPUT_DIR}:/opt/ComfyUI/output",
        IMAGE,
        "bash", "-c",
        "cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 "
        "--output-directory /opt/ComfyUI/output --lowvram",
    ]
    subprocess.run(cmd, check=True)
    print("ComfyUI starting (--lowvram)...")
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://{SERVER}/system_stats", timeout=2)
            print("ComfyUI ready")
            return True
        except Exception:
            time.sleep(3)
    print("ComfyUI failed to start within 180s")
    return False


def generate_qwen_image(label, prompt):
    output_path = os.path.join(OUTPUT_DIR, f"chain_{label}.png")
    if os.path.exists(output_path):
        print(f"  Image exists: {output_path}")
        return output_path

    scripts_dir = os.path.join(ROOT, "scripts")
    escaped_prompt = prompt.replace("'", "\\'")

    cmd = [
        "docker", "run", "--rm",
        *DOCKER_GPU, *DOCKER_ENV,
        "-v", os.path.expanduser("~/.cache/huggingface") + ":/root/.cache/huggingface",
        "-v", f"{OUTPUT_DIR}:/output",
        "-v", f"{scripts_dir}/apply_qwen_patches.py:/opt/apply_qwen_patches.py:ro",
        IMAGE,
        "python3", "-c", f"""
import sys, shutil, glob, os
sys.path.insert(0, '/opt/qwen-image-studio/src')
sys.path.insert(0, '/opt')
from apply_qwen_patches import apply_comprehensive_patches
apply_comprehensive_patches()
from qwen_image_mps.cli import generate_image
import random

class Args:
    prompt = '{escaped_prompt}'
    steps = 8
    num_images = 1
    size = '16:9'
    ultra_fast = False
    model = 'Qwen/Qwen-Image'
    no_mmap = True
    lora = None
    edit = False
    input_image = None
    output_dir = '/tmp'
    seed = random.randint(1, 10000)
    guidance_scale = 1.0
    negative_prompt = 'blurry, low quality, distorted, watermark'
    batman = False
    fast = False
    targets = 'all'

generate_image(Args())
files = glob.glob('/root/.qwen-image-studio/*.png')
if files:
    latest = max(files, key=os.path.getmtime)
    shutil.copy2(latest, '/output/chain_{label}.png')
    print(f'Saved: chain_{label}.png')
""",
    ]
    print("  Generating Qwen seed image (docker run --rm)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr[-300:]}")
        return None
    if os.path.exists(output_path):
        print(f"  OK: {output_path}")
        return output_path
    print("  FAIL: image not produced")
    return None


def submit_and_wait(label, image_filename, video_prompt):
    workflow = create_workflow(
        prompt=video_prompt,
        image_filename=image_filename,
        width=WIDTH, height=HEIGHT, frames=FRAMES, fps=FPS,
        include_audio=True,
        model_name=MODEL,
        lora_name=LORA,
        lora_strength=LORA_STRENGTH,
        output_prefix=f"chain_{label}",
    )
    client_id = str(uuid.uuid4())
    print(f"  Submitting LTX-2.3 dev+LoRA i2v ({WIDTH}x{HEIGHT}/{FRAMES}f)...")
    try:
        resp = comfyui_api.submit(workflow, SERVER, client_id)
        pid = resp.get("prompt_id", "?")
        print(f"  Queued {pid} — watching")
    except RuntimeError as e:
        print(f"  FAIL submit: {e}")
        return None

    # Block until prompt completes by polling history
    start = time.time()
    while True:
        try:
            with urllib.request.urlopen(f"http://{SERVER}/history/{pid}", timeout=5) as r:
                import json
                h = json.loads(r.read().decode())
                if h:
                    status = list(h.values())[0].get("status", {})
                    s = status.get("status_str")
                    if s == "success":
                        elapsed = int(time.time() - start)
                        print(f"  DONE in {elapsed}s")
                        return s
                    if s == "error":
                        msgs = status.get("messages", [])
                        for m in msgs:
                            if m[0] == "execution_error":
                                print(f"  ERROR: {m[1].get('exception_type')} - {m[1].get('exception_message')}")
                        return None
        except Exception:
            pass
        time.sleep(15)


def find_latest_mp4(label):
    """SaveVideo node names files <prefix>_NNNNN_.mp4 — find the freshest one."""
    import glob
    matches = sorted(glob.glob(f"{OUTPUT_DIR}/chain_{label}_*.mp4"), key=os.path.getmtime)
    return matches[-1] if matches else None


def extract_last_frame(mp4_path, png_path):
    """Use the running comfyui container's ffmpeg to grab the final frame.

    The `-update 1` flag overwrites the output each frame, leaving only the
    last. More reliable than `-sseof` for LTX-2.3 mp4s — the negative seek
    sometimes returns 0 frames.
    """
    rel_in = mp4_path.replace(OUTPUT_DIR, "/opt/ComfyUI/output")
    rel_out = png_path.replace(OUTPUT_DIR, "/opt/ComfyUI/output")
    subprocess.run([
        "docker", "exec", COMFYUI_CONTAINER, "ffmpeg", "-y",
        "-i", rel_in, "-update", "1", "-q:v", "2", rel_out,
    ], capture_output=True, text=True)
    return os.path.exists(png_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", type=int,
                   help="1-indexed subset (e.g. --scenes 1 2)")
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen seed (chain_1_office_coffee.png must exist)")
    args = p.parse_args()

    scenes = SCENES
    if args.scenes:
        scenes = [SCENES[i - 1] for i in args.scenes if 1 <= i <= len(SCENES)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LTX-2.3 Chained Narrative — 'The Coffee Break'")
    print(f"Model: {MODEL}  +  LoRA: {LORA} @ {LORA_STRENGTH}")
    print(f"Per-scene: {WIDTH}x{HEIGHT}  {FRAMES}f @ {FPS}fps  ({FRAMES/FPS:.1f}s)")
    print("=" * 70)
    for i, s in enumerate(scenes, 1):
        print(f"  {i}. {s['label']}")

    # Phase 1: Qwen seed for scene 1
    seed_img = None
    if not args.skip_qwen:
        first = scenes[0]
        if first["qwen"]:
            print(f"\n--- Phase 1: Qwen seed for {first['label']} ---")
            stop_comfyui()
            seed_img = generate_qwen_image(first["label"], first["qwen"])
            if not seed_img:
                print("Qwen seed failed; aborting.")
                return 1
    else:
        seed_img = os.path.join(OUTPUT_DIR, f"chain_{scenes[0]['label']}.png")
        if not os.path.exists(seed_img):
            print(f"--skip-qwen but {seed_img} doesn't exist; aborting.")
            return 1

    # Phase 2: chained video gen
    print(f"\n--- Phase 2: Chained LTX-2.3 ({len(scenes)} scenes) ---")
    if not start_comfyui():
        return 1

    current_input_path = seed_img
    completed = []

    for idx, scene in enumerate(scenes, 1):
        label = scene["label"]
        print(f"\n[{idx}/{len(scenes)}] {label}")
        print(f"  Input image: {os.path.basename(current_input_path)}")

        # Copy input image into ComfyUI input/
        in_filename = os.path.basename(current_input_path)
        rc = subprocess.run([
            "docker", "cp", current_input_path,
            f"{COMFYUI_CONTAINER}:/opt/ComfyUI/input/{in_filename}",
        ]).returncode
        if rc != 0:
            print(f"  FAIL: docker cp input")
            break

        status = submit_and_wait(label, in_filename, scene["video"])
        if status != "success":
            print(f"  Aborting chain at scene {idx}")
            break

        mp4 = find_latest_mp4(label)
        if not mp4:
            print(f"  FAIL: no mp4 produced")
            break
        size_kb = os.path.getsize(mp4) // 1024
        print(f"  Video: {os.path.basename(mp4)} ({size_kb} KB)")
        completed.append(mp4)

        # Extract last frame for next scene's input (unless this is the last)
        if idx < len(scenes):
            next_input = os.path.join(OUTPUT_DIR, f"chain_{scenes[idx]['label']}.png")
            if extract_last_frame(mp4, next_input):
                print(f"  Extracted last frame -> {os.path.basename(next_input)}")
                current_input_path = next_input
            else:
                print(f"  FAIL: ffmpeg last-frame extract; aborting chain")
                break

    print("\n" + "=" * 70)
    print(f"Completed {len(completed)}/{len(scenes)} scenes:")
    for m in completed:
        print(f"  {m}")
    return 0 if len(completed) == len(scenes) else 1


if __name__ == "__main__":
    sys.exit(main())
