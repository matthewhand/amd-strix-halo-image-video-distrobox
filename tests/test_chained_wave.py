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
import json
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
# Dev (43GB) + LoRA (7.6GB) at strength 0.5 produced NaN/Inf in the AAC
# audio encode (avcodec_send_frame returned 22). The LoRA was trained
# alongside the distilled checkpoint and doesn't compose cleanly with the
# dev model's audio pathway. Falling back to distilled-fp8 (no LoRA) which
# wave 1 proved works at 1024x576/145f.
MODEL = "ltx-2.3-22b-distilled-fp8.safetensors"
LORA = None
LORA_STRENGTH = 0.0
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
        # /mnt/downloads is the symlink target for large models (e.g. dev-only
        # downloads kept off the root partition). Mount it so symlinks under
        # ~/comfy-models/checkpoints/ → /mnt/downloads/... resolve inside.
        "-v", "/mnt/downloads:/mnt/downloads",
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


EXTENSION_TEMPLATES = [
    "the previous scene's final moment holds for a breath then erupts as {escalation}, audio swells from the previous scene into {audio_shift}",
]
EXTENSION_ESCALATIONS = [
    "the entire frame flooding with impossible rainbow light as geometry warps",
    "a gigantic cosmic hand reaching in from above to rearrange the scene like chess pieces",
    "all objects gaining cartoon eyes and legs and scurrying in every direction",
    "the ground tearing open to reveal a deeper layer where the same events repeat smaller",
    "gravity reversing and everyone floating up into a purple vortex",
    "the camera pulling so far back that the whole scene is revealed to be inside a snow globe held by something even larger and more absurd",
    "the moment freezing solid as ice then shattering and reforming as the same scene but underwater",
    "time running backward for three seconds then forward again with everything slightly misaligned",
]
EXTENSION_AUDIO = [
    "a deep cosmic drone with whispered counting backward from ten",
    "a polite studio audience laugh track then sudden complete silence",
    "overlapping news anchor voices reporting contradictory versions of the scene",
    "a lonely cello solo rising over distant church bells",
    "a vintage telephone ringing from inside the viewer's head",
    "a children's choir humming a lullaby that slowly becomes a horror score",
]


def extend_scenes(base_scenes, count):
    """If user wants more scenes than defined, fabricate continuations using
    escalation templates. Earlier scenes get more novelty; later ones recycle.
    """
    import random
    scenes = list(base_scenes)
    while len(scenes) < count:
        i = len(scenes)
        esc = EXTENSION_ESCALATIONS[i % len(EXTENSION_ESCALATIONS)]
        aud = EXTENSION_AUDIO[i % len(EXTENSION_AUDIO)]
        tmpl = EXTENSION_TEMPLATES[0]
        scenes.append({
            "label": f"{i+1}_extension_{i}",
            "qwen": None,  # chained from previous last frame
            "video": tmpl.format(escalation=esc, audio_shift=aud),
        })
    return scenes[:count]


def join_mp4s(mp4_paths, out_path):
    """Concat a list of mp4s into one continuous mp4 using ffmpeg concat
    demuxer. Requires all inputs to share codec/resolution/fps — guaranteed
    here because they're all produced by the same ComfyUI workflow.
    """
    if not mp4_paths:
        print("  nothing to join")
        return False
    list_file = OUTPUT_DIR + "/chain_concat.txt"
    rel_list = "/opt/ComfyUI/output/chain_concat.txt"
    rel_out = out_path.replace(OUTPUT_DIR, "/opt/ComfyUI/output")
    with open(list_file, "w") as f:
        for m in mp4_paths:
            rel = m.replace(OUTPUT_DIR, "/opt/ComfyUI/output")
            f.write(f"file '{rel}'\n")
    rc = subprocess.run([
        "docker", "exec", COMFYUI_CONTAINER, "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", rel_list,
        "-c", "copy", rel_out,
    ], capture_output=True, text=True).returncode
    os.remove(list_file)
    return rc == 0 and os.path.exists(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", type=int,
                   help="1-indexed subset (e.g. --scenes 1 2)")
    p.add_argument("--count", type=int, default=None,
                   help="Number of chained scenes. <len(SCENES) truncates; "
                        ">len(SCENES) auto-extends with generic escalation "
                        "prompts (series mode) or repeats (repeat mode).")
    p.add_argument("--mode", choices=["series", "repeat"], default="series",
                   help="series: each scene has its own prompt (narrative arc, "
                        "default). repeat: same prompt across all chained scenes "
                        "(extends one moment by chaining the visual continuity).")
    p.add_argument("--prompt", type=str, default=None,
                   help="In repeat mode: video prompt to use for every scene "
                        "(default: scene 1's video prompt)")
    p.add_argument("--qwen-prompt", type=str, default=None,
                   help="In repeat mode: Qwen seed prompt (default: scene 1's "
                        "qwen prompt)")
    p.add_argument("--prompts-file", type=str, default=None,
                   help="JSON list [{label, qwen (null for mid-chain), video}, ...] — "
                        "overrides SCENES entirely (only meaningful in series mode)")
    p.add_argument("--no-join", action="store_true",
                   help="Don't concat completed scenes into a single mp4")
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen seed (scene 1 png must already exist)")
    args = p.parse_args()

    if args.prompts_file:
        with open(args.prompts_file) as f:
            scenes = json.load(f)
    else:
        scenes = SCENES

    if args.mode == "repeat":
        # Take seed prompts from scene 1 (or CLI overrides), then replicate
        # the video prompt across N scenes. Each chained scene visually
        # continues the previous via last-frame seeding.
        base_qwen = args.qwen_prompt or scenes[0]["qwen"]
        base_video = args.prompt or scenes[0]["video"]
        n = args.count or 5
        base_label = scenes[0]["label"].split("_", 1)[1] if "_" in scenes[0]["label"] else "repeat"
        scenes = [
            {
                "label": f"{i+1}_{base_label}",
                "qwen": base_qwen if i == 0 else None,
                "video": base_video,
            }
            for i in range(n)
        ]
    elif args.count is not None:
        scenes = extend_scenes(scenes, args.count)

    if args.scenes:
        scenes = [scenes[i - 1] for i in args.scenes if 1 <= i <= len(scenes)]

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

    # Auto-join into one continuous mp4 (default on)
    if not args.no_join and len(completed) >= 2:
        joined = os.path.join(OUTPUT_DIR, f"chain_joined_{int(time.time())}.mp4")
        print(f"\nJoining {len(completed)} clips into {os.path.basename(joined)}...")
        if join_mp4s(completed, joined):
            size_mb = os.path.getsize(joined) / 1e6
            print(f"  OK: {joined} ({size_mb:.1f} MB)")
        else:
            print(f"  FAIL: ffmpeg concat")

    return 0 if len(completed) == len(scenes) else 1


if __name__ == "__main__":
    sys.exit(main())
