#!/usr/bin/env python3
"""
10-scene ironic audio+video wave on LTX-2.3 22B distilled.

Each scene pairs a Qwen *static composition* prompt with an LTX *action+sound*
prompt that intentionally clashes — silent visuals with loud audio, dignified
performers with absurd subjects, etc.

Resolution and frame count step up across the wave so we sweep performance
from smoke-size to near-max within the 64 GB HIP cap (using --lowvram).

Phase 1: stop ComfyUI, generate 10 Qwen images (sequential containers, 16:9).
Phase 2: start ComfyUI with --lowvram, queue 10 LTX-2.3 jobs, ComfyUI serializes.

Usage:
    python tests/test_ironic_wave.py                        # full wave
    python tests/test_ironic_wave.py --scenes 1 2 3         # 1-indexed subset
    python tests/test_ironic_wave.py --skip-qwen            # use cached images
    python tests/test_ironic_wave.py --dry-run              # print plan only
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

# (label, qwen_prompt, video_prompt, width, height, frames)
# Qwen prompt = static composition. LTX prompt = motion + sound design.
SCENES = [
    (
        "heavy_metal_librarian",
        "photorealistic interior of a hushed grand library with tall oak shelves, an elderly librarian in a leather biker vest covered in band patches stands behind the wooden circulation desk, long grey hair, reading glasses, calm expression, patrons silently reading at study tables in the background, warm afternoon light through stained glass windows",
        "the librarian explodes into furious silent headbanging behind the desk while the patrons read on undisturbed, hair whipping around, fist pumping, audio is full speed thrash metal with double-kick drums and shredded electric guitar at maximum volume",
        480, 320, 49,
    ),
    (
        "mime_opera_singer",
        "photograph of a classic white-faced street mime in black and white striped shirt, beret, white gloves, performing on a Parisian sidewalk in front of a small chalk-marked semicircle, cobblestones, an iron lamppost, a few onlookers watching politely, soft overcast daylight",
        "the mime stays completely silent with sealed lips and exaggerated facial expressions while a thunderous Italian operatic tenor aria booms out of nowhere, full orchestra and chorus, audio is Pavarotti-scale Nessun Dorma at concert hall volume, the mime simply tilts head and bows",
        480, 320, 97,
    ),
    (
        "astronaut_city_bus",
        "documentary photograph of an astronaut in a full white NASA spacesuit with helmet on, sitting on a bench seat of a crowded city bus during morning commute, holding a metal coffee thermos, other passengers reading phones and newspapers oblivious, fluorescent overhead lighting, urban morning light through bus windows",
        "the astronaut sips coffee through the closed helmet visor in slow motion as the bus rattles down the street, audio is full NASA mission control radio chatter with countdown commands, beeping telemetry, and Houston we are go for liftoff over crackling comms",
        640, 384, 97,
    ),
    (
        "trex_piano_recital",
        "elegant photograph of a full grown Tyrannosaurus Rex in a black tuxedo and white bow tie seated at a polished black grand piano on a concert hall stage, tiny arms barely reaching the keys, large head leaning down in concentration, audience in formal attire visible in the front rows, dramatic spotlight",
        "the T-Rex carefully picks at the keys with its tiny arms in slow concentrated motion, head sways with the music, audio is full Chopin Nocturne in E-flat major performed at concert quality with rich piano resonance, occasional polite applause swells from the audience",
        768, 432, 97,
    ),
    (
        "cat_evening_news",
        "photograph of a professional television news studio, two well-groomed cats wearing tiny suits and ties seated at a sleek anchor desk with the city skyline backdrop behind them, broadcast graphics, teleprompter visible, studio lighting, papers stacked neatly, the channel logo on the desk",
        "the lead anchor cat looks up at camera with serious gravitas while the co-anchor cat nods solemnly, audio is the full evening news theme music swelling then a deep authoritative male newscaster voice introducing tonight's top stories interrupted by occasional dignified meows",
        768, 512, 121,
    ),
    (
        "knight_orders_latte",
        "photograph of a medieval knight in full polished plate armor including helmet with visor up, standing patiently at the counter of a modern hipster coffee shop, chalkboard menu visible, espresso machine and pastry case, barista in apron taking the order, customers at small tables on laptops, warm pendant lighting",
        "the knight slowly counts coins onto the counter from a leather pouch in metal-clad fingers, audio is full coffee shop ambience: hissing espresso machine, milk steaming, ceramic cups clinking, indie acoustic music, then the barista calls out clearly: oat milk latte for Sir Reginald",
        768, 512, 145,
    ),
    (
        "underwater_chef",
        "photograph of a French chef in a tall white toque and pristine chef's coat working at a fully submerged underwater stainless steel kitchen line, fish swimming between hanging copper pots, coral growing on the prep surfaces, vegetables floating, soft sunbeams from above through the water",
        "the chef performs precise knife work in slow motion as bubbles rise from each cut and ingredients drift weightlessly, audio is full restaurant kitchen chaos: sizzling pans, oil popping, ticket printer, expediter shouting orders, dishes clattering, completely incongruous with the silent underwater scene",
        896, 512, 145,
    ),
    (
        "toddler_ceo_emergency_meeting",
        "photograph of a corporate boardroom with a long mahogany table, executives in suits seated around it looking concerned, at the head of the table a serious-faced toddler in a tiny pinstripe business suit and tie sits in an oversized leather chair, financial charts on the wall display, crayons and a juice box in front of the toddler",
        "the toddler CEO slams a small fist on the table while drawing furiously with crayons across a quarterly report, executives lean in attentively taking notes, audio is full tense corporate orchestral strings building dramatically, gavel bangs, urgent telephones ringing in the background",
        1024, 576, 145,
    ),
    (
        "viking_startup_pitch",
        "photograph of a fierce Viking warrior in full historical regalia, fur cloak, horned helmet, large round wooden shield slung on back, long beard, standing in front of a sleek modern Silicon Valley conference room projector screen displaying a startup pitch deck slide titled Series A, audience of casually-dressed tech investors in chairs",
        "the Viking gestures emphatically at the projector screen with a battle axe, slowly turning to point at key bullet points, audio is upbeat startup pitch narration in TED-talk cadence: our blockchain-enabled longship logistics platform disrupts traditional pillage and tribute markets at a 10x runtime, applause swells",
        1024, 576, 193,
    ),
    (
        "statue_ballet_performance",
        "photograph of three classical white marble Greek sculptures of muscular male figures arranged on a grand ballet stage, soft theatrical lighting, red velvet curtain backdrop, polished wooden stage floor, ornate proscenium visible at the edges, an empty orchestra pit in the foreground",
        "the marble statues glide impossibly into precise synchronized ballet poses in slow motion, holding arabesques and pirouettes, audio is full Tchaikovsky Swan Lake orchestral score at concert quality with sweeping strings and timpani, punctuated by faint stone-on-stone chiseling sounds at every movement",
        1280, 720, 145,
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


def submit_ltx23(label, image_filename, video_prompt, width, height, frames):
    workflow = create_workflow(
        prompt=video_prompt,
        image_filename=image_filename,
        width=width, height=height, frames=frames, fps=24,
        include_audio=True,
        output_prefix=f"wave_{label}",
    )
    client_id = str(uuid.uuid4())
    try:
        resp = comfyui_api.submit(workflow, SERVER, client_id)
        pid = resp.get("prompt_id", "?")
        print(f"  Queued {label} {width}x{height}/{frames}f -> {pid}")
        return pid
    except RuntimeError as e:
        print(f"  FAIL submit {label}: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", type=int,
                   help="1-indexed subset of scenes (e.g. --scenes 1 2 3)")
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen phase, use existing qwen_input_*.png")
    p.add_argument("--dry-run", action="store_true",
                   help="Print scene plan only, no actions")
    args = p.parse_args()

    scenes = SCENES
    if args.scenes:
        scenes = [SCENES[i - 1] for i in args.scenes if 1 <= i <= len(SCENES)]

    print("=" * 70)
    print("LTX-2.3 Ironic Audio+Video Wave (--lowvram)")
    print("=" * 70)
    for i, (label, _qp, _vp, w, h, f) in enumerate(scenes, 1):
        print(f"  {i:2d}. {label:30s} {w}x{h}  {f}f")

    if args.dry_run:
        return 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = {}
    if args.skip_qwen:
        for label, *_ in scenes:
            path = os.path.join(OUTPUT_DIR, f"qwen_input_{label}.png")
            if os.path.exists(path):
                images[label] = os.path.basename(path)
            else:
                print(f"  MISSING: {path} (run without --skip-qwen)")
    else:
        print("\n--- Phase 1: Qwen image generation ---")
        stop_comfyui()
        for label, qwen_prompt, _, _, _, _ in scenes:
            print(f"\n[{label}]")
            path = generate_qwen_image(label, qwen_prompt)
            if path:
                images[label] = os.path.basename(path)

    if not images:
        print("\nNo images available. Exiting.")
        return 1

    print(f"\n--- Phase 2: LTX-2.3 video generation ({len(images)} jobs) ---")
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
    for label, _qp, video_prompt, w, h, frames in scenes:
        if label not in images:
            continue
        pid = submit_ltx23(label, images[label], video_prompt, w, h, frames)
        if pid:
            submitted.append((label, pid))

    print("\n" + "=" * 70)
    print(f"Submitted {len(submitted)}/{len(scenes)} jobs to ComfyUI queue.")
    print(f"Outputs land in {OUTPUT_DIR}/wave_*.mp4")
    print("Monitor: python scripts/comfyui_api.py queue")
    print("         watch -n 30 'ls -lt /tmp/comfy-outputs/wave_*.mp4 2>/dev/null | head'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
