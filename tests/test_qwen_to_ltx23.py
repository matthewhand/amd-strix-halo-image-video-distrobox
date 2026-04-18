#!/usr/bin/env python3
"""
Qwen image -> LTX-2.3 video+audio pipeline.

Builds the LTX-2.3 workflow via scripts/generate_ltx23_workflow.create_workflow,
submits via scripts/comfyui_api, and orchestrates Qwen <-> ComfyUI container
swaps so only one model occupies VRAM at a time (Strix Halo unified memory).

Usage:
    python tests/test_qwen_to_ltx23.py --smoke              # 1 scene, low res
    python tests/test_qwen_to_ltx23.py                      # all scenes
    python tests/test_qwen_to_ltx23.py --scenes octopus_accountant cats_boardroom
    python tests/test_qwen_to_ltx23.py --frames 49

Verified config (smoke 480x320, 49f, audio on): ~48s on Strix Halo gfx1151
with the 22B distilled fp8 model + Gemma-3 12B encoder. Always-on host
containers consume ~6 GB; LTX-2.3 needs ~60 GB peak.
"""
import argparse
import os
import random
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

DOCKER_ENV = [
    "-e", "HSA_OVERRIDE_GFX_VERSION=11.5.1",
    "-e", "LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib",
    "-e", "LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib",
]
DOCKER_GPU = [
    "--device", "/dev/dri", "--device", "/dev/kfd",
    "--security-opt", "seccomp=unconfined",
]

# (label, qwen_prompt, video_prompt, frames)
SCENES = [
    (
        "octopus_accountant",
        "hyperrealistic photograph of an octopus wearing tiny reading glasses sitting at a desk covered in spreadsheets and tax forms, each tentacle holding a different pen or calculator, office cubicle background, fluorescent lighting, mundane corporate setting",
        "the octopus slowly looks up from the spreadsheets directly at the camera with an expression of existential dread, papers rustling, pens moving, calculator buttons clicking, a phone rings in the background",
        97,
    ),
    (
        "medieval_astronaut",
        "oil painting in renaissance style of an astronaut in a full NASA spacesuit sitting for a formal portrait in a medieval castle throne room, golden frame, dramatic chiaroscuro lighting, servants in period costume attending",
        "the astronaut slowly raises the visor revealing a confused medieval knight face underneath, servants gasping, candles flickering, dramatic orchestral music swelling",
        97,
    ),
    (
        "cats_boardroom",
        "corporate photograph of a serious business meeting in a luxury boardroom, but all the executives are cats in tiny suits and ties, one cat standing at a whiteboard with a laser pointer, charts showing fish stock prices, mahogany table",
        "the CEO cat slams its paw on the table and meows loudly, the other cats turn in shock, papers fly everywhere, the stock chart on the screen crashes, dramatic boardroom tension",
        97,
    ),
    (
        "dinosaur_barista",
        "detailed illustration of a friendly T-Rex working as a barista in a modern hipster coffee shop, tiny apron on its massive body, struggling to hold a tiny espresso cup with its small arms, customers waiting patiently in line, chalkboard menu",
        "the T-Rex carefully pours latte art into the tiny cup with trembling small arms, milk splashing everywhere, the cup crumbles in its claws, customers clapping encouragingly, coffee machine steaming",
        145,
    ),
    (
        "underwater_library",
        "photorealistic wide shot of a grand classical library that is completely submerged underwater, fish swimming between the bookshelves, an old librarian octopus organizing books, coral growing on marble columns, shafts of sunlight from above, books floating open with pages drifting",
        "camera slowly glides through the underwater library, pages turning by themselves in the current, small fish dart between shelves, bubbles rising from an open book, whale song echoing through the halls, peaceful and surreal",
        145,
    ),
    (
        "angry_ai_user",
        "photorealistic close up of a frustrated middle aged man in a messy home office, red faced and furious, gripping a keyboard with white knuckles, multiple monitors showing AI chatbot responses, energy drink cans everywhere, dramatic lighting from screen glow",
        "the man slams the keyboard on the desk and yells profanity at the screen, veins bulging on his forehead, a monitor flickers, he grabs his coffee mug and throws it, angry shouting and crashing sounds, keyboard keys flying",
        97,
    ),
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
    print("ComfyUI starting...")
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
    output_path = os.path.join(OUTPUT_DIR, f"qwen_input_{label}.png")
    if os.path.exists(output_path):
        print(f"  Image exists: {output_path}")
        return output_path

    scripts_dir = os.path.join(ROOT, "scripts")
    seed = random.randint(1, 10000)
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
    seed = {seed}
    guidance_scale = 1.0
    negative_prompt = 'blurry, low quality, distorted, watermark'
    batman = False
    fast = False
    targets = 'all'

generate_image(Args())
files = glob.glob('/root/.qwen-image-studio/*.png')
if files:
    latest = max(files, key=os.path.getmtime)
    shutil.copy2(latest, '/output/qwen_input_{label}.png')
    print(f'Saved: qwen_input_{label}.png')
""",
    ]

    print("  Generating Qwen image (docker run --rm)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr[-300:]}")
        return None
    if os.path.exists(output_path):
        print(f"  OK: {output_path}")
        return output_path
    print("  FAIL: image not produced")
    return None


def submit_ltx23(label, image_filename, video_prompt, *, frames, width, height,
                 fps, include_audio, watch):
    workflow = create_workflow(
        prompt=video_prompt,
        image_filename=image_filename,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        include_audio=include_audio,
        output_prefix=f"i2v23_{label}",
    )
    client_id = str(uuid.uuid4())
    try:
        resp = comfyui_api.submit(workflow, SERVER, client_id)
        pid = resp.get("prompt_id", "?")
        print(f"  Queued LTX-2.3 i2v: {pid}")
        if watch:
            comfyui_api.watch(SERVER, client_id, pid)
        return pid
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="1 scene, 49 frames, 480x320 (fast verification)")
    p.add_argument("--scenes", nargs="+", help="Subset of scene labels to run")
    p.add_argument("--frames", type=int, help="Override frame count for all scenes")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--no-audio", action="store_true",
                   help="DOES NOT WORK with LTX-2.3 — the MultimodalGuider+"
                        "SamplerCustomAdvanced stack returns an AV-tuple "
                        "regardless and unpacking fails downstream. Audio is "
                        "always produced; this flag is rejected.")
    p.add_argument("--watch", action="store_true",
                   help="Block on each job until it completes (one-at-a-time)")
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen phase, use existing qwen_input_*.png")
    args = p.parse_args()

    if args.no_audio:
        sys.exit("--no-audio is incompatible with LTX-2.3 (see help)")

    if args.smoke:
        scenes = SCENES[:1]
        args.frames = 49
        args.width, args.height = 480, 320
        args.watch = True
    elif args.scenes:
        wanted = set(args.scenes)
        scenes = [s for s in SCENES if s[0] in wanted]
        if not scenes:
            print(f"No scenes matched {sorted(wanted)}")
            print(f"Available: {[s[0] for s in SCENES]}")
            return 1
    else:
        scenes = SCENES

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Qwen -> LTX-2.3 Image-to-Video Pipeline")
    print("=" * 60)
    print(f"Scenes: {[s[0] for s in scenes]}")
    print(f"Resolution: {args.width}x{args.height}, audio: {not args.no_audio}")

    images = {}
    if args.skip_qwen:
        for label, *_ in scenes:
            path = os.path.join(OUTPUT_DIR, f"qwen_input_{label}.png")
            if os.path.exists(path):
                images[label] = os.path.basename(path)
            else:
                print(f"  MISSING: {path}")
    else:
        print("\n--- Phase 1: Generate images with Qwen ---")
        stop_comfyui()
        for label, qwen_prompt, _, _ in scenes:
            print(f"\n[{label}] Generating source image...")
            path = generate_qwen_image(label, qwen_prompt)
            if path:
                images[label] = os.path.basename(path)

    if not images:
        print("\nNo images available. Exiting.")
        return 1

    print(f"\n--- Phase 2: Animate with LTX-2.3 ({len(images)} videos) ---")
    if not start_comfyui():
        return 1

    for label, filename in images.items():
        src = os.path.join(OUTPUT_DIR, filename)
        rc = subprocess.run(
            ["docker", "cp", src, f"{COMFYUI_CONTAINER}:/opt/ComfyUI/input/{filename}"]
        ).returncode
        if rc != 0:
            print(f"  WARN: docker cp failed for {filename}")

    submitted = []
    for label, _qwen_prompt, video_prompt, default_frames in scenes:
        if label not in images:
            continue
        frames = args.frames if args.frames else default_frames
        print(f"\n[{label}] frames={frames}")
        pid = submit_ltx23(
            label, images[label], video_prompt,
            frames=frames, width=args.width, height=args.height, fps=args.fps,
            include_audio=not args.no_audio, watch=args.watch,
        )
        if pid:
            submitted.append((label, pid))

    print("\n" + "=" * 60)
    print(f"Submitted {len(submitted)} jobs. Outputs land in {OUTPUT_DIR}/")
    print("Monitor: python scripts/comfyui_api.py queue")
    print("         python scripts/comfyui_api.py watch <prompt_id>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
