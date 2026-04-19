#!/usr/bin/env python3
"""
Tone-variety wave — 12 scenes spanning four humor registers.

Categories (3 scenes each):
  - ironic:   image and audio expectations clash
  - sardonic: bitter / mocking / corporate cynicism
  - comedic:  lighthearted slapstick / wholesome silliness
  - absurd:   dream-logic / nonsensical premise played straight

All scenes at 768x432 / 97f / 24fps (~4s clips). distilled-fp8 with
--lowvram (proven path from wave 1). Estimated wall: ~5-6 min/scene =
~70 min for 12 scenes.

Usage:
    python tests/test_tone_wave.py                       # full 12
    python tests/test_tone_wave.py --tones absurd        # one category
    python tests/test_tone_wave.py --scenes 1 5 9        # subset by index
    python tests/test_tone_wave.py --dry-run             # plan only
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

# Defaults — overridable via CLI. Module-level mutables set by main().
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FRAMES, DEFAULT_FPS = 768, 432, 97, 24
WIDTH, HEIGHT, FRAMES, FPS = DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FRAMES, DEFAULT_FPS

# Wall-time model derived from this session's observations on Strix Halo
# gfx1151 with --lowvram, distilled-fp8 22B + Gemma-3 12B encoder.
# Measured (resolution * frames -> minutes):
#   480x320  *  49f =  7.5 Mvox -> 1.5 min  (ironic_wave heavy_metal_librarian)
#   768x512  * 121f = 47.6 Mvox -> 4.0 min  (ironic_wave cat_evening_news)
#   1024x576 * 193f =113.8 Mvox -> 10.3 min (ironic_wave viking_startup_pitch)
#   1280x720 * 145f =133.6 Mvox -> 39.7 min (ironic_wave statue_ballet)
# Cost is super-linear in pixels (VAE decode tiling overhead). Quadratic fit:
#   minutes ~= 0.07 * Mvox + 0.0019 * Mvox^2
# Plus ~30s/scene overhead for model swap + queue marshalling.
def estimate_min_per_scene(width, height, frames):
    mvox = (width * height * frames) / 1_000_000
    return 0.5 + 0.07 * mvox + 0.0019 * (mvox ** 2)


# Qwen still-image generation cost (per image, observed ~3 min from
# the ironic wave Phase 1 timings).
QWEN_MIN_PER_IMAGE = 3.0

DOCKER_ENV = [
    "-e", "HSA_OVERRIDE_GFX_VERSION=11.5.1",
    "-e", "LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib",
    "-e", "LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib",
]
DOCKER_GPU = [
    "--device", "/dev/dri", "--device", "/dev/kfd",
    "--security-opt", "seccomp=unconfined",
]

# (tone, label, qwen_prompt, video_prompt)
SCENES = [
    # === IRONIC: visual/audio expectation reversal ===
    ("ironic", "yoga_road_rage",
     "photograph of a serene blissful yoga instructor in immaculate white linen sitting cross-legged in lotus pose on a folded mat in the middle of a gridlocked downtown intersection, eyes closed in meditative peace, soft morning sunlight, cars and trucks bumper-to-bumper around her, billboards in the background",
     "the yoga instructor remains perfectly still and serene with hands in mudra position, audio is full road rage cacophony: blaring car horns, shouting drivers, screeching tires, a man yelling profanity at length, ambulance siren in distance, then a single tibetan bowl bell rings clearly through it all"),
    ("ironic", "funeral_kazoo",
     "wide shot photograph of a somber graveside funeral on a grey overcast afternoon, mourners in dark clothes standing around a polished casket draped with flowers, the priest in black robes holding a bible, a folded flag on a stand, weeping family in the foreground, leafless trees and headstones in the background",
     "the priest slowly raises a single bright pink kazoo to his lips and the entire group of mourners produces a loud unison kazoo rendition of taps, audio is a comically buzzy kazoo dirge with tinny harmonies and one wrong note, distant crow caws, a single sob blending into a kazoo trill"),
    ("ironic", "asmr_construction",
     "photograph of a cozy ASMR studio with soft pink LED lighting, a content creator in noise-cancelling headphones leaning close to a fluffy windscreened microphone, hands holding delicate items, comfortable beige fabric backdrop, intimate close-up framing",
     "the creator gently mouths into the microphone but the audio is full industrial construction site: jackhammer pounding, circular saw screaming through metal, dump truck reversing alarm, hard hat hammering rebar, foreman shouting orders through a megaphone, no soft sounds at all"),

    # === SARDONIC: bitter / corporate cynicism ===
    ("sardonic", "linkedin_parking_lot",
     "photograph of a polished motivational influencer in a sharp navy suit and bright smile holding up a smartphone for a selfie video in an empty fluorescent-lit suburban office park parking lot at twilight, his black SUV in the background, a laptop bag at his feet, mass-produced brick office building looms",
     "the influencer's confident smile cracks the moment he stops recording, his eyes go vacant and exhausted, he slowly sits down on the curb still holding the phone, audio is his own voiceover playing back: hashtag rise and grind, never give up, your dreams are valid, with a faint cracked sob underneath and the buzz of the parking lot lights"),
    ("sardonic", "open_plan_inspirational",
     "photograph of a vast open-plan tech office floor packed wall-to-wall with identical desks and rows of pale faces glued to monitors under harsh fluorescent ceiling lights, a giant inspirational wall decal reading WORK HARD HAVE FUN MAKE HISTORY in cheery hand-lettering above them, gray industrial carpet, no windows visible",
     "the camera slowly dollies across the rows of motionless workers as a single cubicle plant tips over and dies, audio is upbeat corporate inspirational stock music with chirpy ukulele and clapping rhythm playing relentlessly over dead silence from the workers, then a single keyboard key being pressed once every two seconds"),
    ("sardonic", "wedding_no_show",
     "photograph of an elaborately decorated outdoor wedding reception venue at sunset with chandeliers in the trees and round tables set with crystal and roses, a four-tier white wedding cake centerpiece, all chairs perfectly arranged and completely empty, a string quartet plays in the corner of the empty space, a tearful bride alone at the head table",
     "the bride slowly raises her champagne glass for a toast to the empty chairs, the quartet keeps playing pachelbel's canon dutifully, audio includes the polite string quartet performance, the soft clink of one glass meeting absolutely nothing, distant party laughter from a successful wedding next door, a single helium balloon popping"),

    # === COMEDIC: slapstick / wholesome silliness ===
    ("comedic", "penguin_dance_off",
     "photograph of an emperor penguin standing on a frozen ice sheet wearing tiny mirrored disco sunglasses and a single gold chain around its neck, colorful disco party lights casting purple and pink across the ice, other penguins arranged in a loose circle as audience, snowflakes drifting",
     "the penguin attempts an elaborate hip-hop dance routine but immediately slips backward onto its belly and slides across the ice spinning, then determinedly waddle-stands and tries again falling forward, audio is full upbeat funk dance track with thumping bass plus the penguin's surprised honking and the audience penguins cheering"),
    ("comedic", "toddler_dog_ball",
     "photograph of a sunny suburban backyard with green grass, a happy golden retriever proudly holding a slobbery tennis ball in its mouth standing in front of an enthralled toddler in overalls and one shoe, a wooden fence and flowering hydrangeas behind them, soft late afternoon golden hour light",
     "in slow motion the toddler executes a perfect tackle stealing the ball from the dog's mouth then runs in a triumphant zigzag across the lawn, the dog looks at the camera in shocked betrayal then chases excitedly, audio is jaunty comedic chase music with kazoo and tuba, toddler shrieks of laughter, dog barks of joy, ball squeak"),
    ("comedic", "cat_zoom_meeting",
     "photograph of a fluffy orange tabby cat sitting upright in a small office chair behind a sleek modern desk wearing wire-rimmed reading glasses, a laptop displaying a tile grid of human coworkers' faces in a video call, a tiny mug of coffee labeled WORLDS BEST CAT, plant in the background, soft window light",
     "the cat solemnly raises one paw to its chin like it's deeply considering the conversation then suddenly knocks the coffee mug clean off the desk with absolute deliberation while making direct eye contact with the camera, audio is muffled human zoom call discussing quarterly metrics, the mug shattering, faint human gasps over the call"),

    # === ABSURD: dream-logic / nonsense played straight ===
    ("absurd", "stapler_wedding",
     "wide photograph of a small ornate wedding ceremony in a candlelit chapel where a beige office stapler stands at the altar in a tiny tuxedo bowtie next to a yellow legal-pad bride wearing a lace veil, a fountain pen in officiant robes officiates with reading glasses, a roll of tape and a paperclip act as flower-girl and ring-bearer respectively, friends and family of office supplies fill the pews in formal attire",
     "the fountain pen officiant declares the union and the stapler dramatically staples a paper marriage certificate three times in solemn ceremony, the legal pad bride flutters its top page like a veil, audio is a full traditional pipe organ wedding march, congregation murmurs of approval, a single sniffle from a binder clip in the front row"),
    ("absurd", "fridge_keynote",
     "photograph of a glossy stainless steel refrigerator standing alone on a giant convention center stage under a single dramatic spotlight, a clip-on lavalier microphone attached to its handle, a massive screen behind reads THE FUTURE OF COLD in sleek tech-conference branding, an audience of hundreds of other appliances in folding chairs visible in the dark space",
     "the refrigerator's door slowly swings open then shut once for emphasis as if speaking, internal light pulsing in cadence like speech, audio is a deeply confident TED-talk-style male narrator delivering a passionate keynote: thank you all for being here, today we are going to talk about preserving freshness in an increasingly warm world, applause swells from the appliance audience, a microwave whoops"),
    ("absurd", "goldfish_talk_show",
     "photograph of a small round glass fishbowl on a sleek talk show desk under bright studio lights, inside the bowl a single bright orange goldfish hovers behind a tiny clip-on microphone, an empty matching guest fishbowl sits across from it, the studio backdrop reads AROUND THE BOWL in glowing letters, audience seating visible in the soft blur",
     "the goldfish swims a slow self-important lap then hovers gesturing with its fins as if making an interview point, occasional bubble rises, audio is a polished late-night-talk-show host voice asking a guest about their new memoir, recorded studio audience laughter at the perfect rhythm, an opening jingle plays, the host says: welcome back to around the bowl"),
]


def _extra_mounts():
    """Optional extra bind mounts via env var COMFY_EXTRA_MOUNTS, formatted
    as colon-separated host:container pairs, comma-separated for multiple.
    Useful when models live on a separate volume and ~/comfy-models has
    symlinks pointing into it (the container needs the symlink target
    mounted too, or readlink fails). Example:
        COMFY_EXTRA_MOUNTS=/data/ml:/data/ml,/scratch:/scratch
    """
    out = []
    for spec in os.environ.get("COMFY_EXTRA_MOUNTS", "").split(","):
        spec = spec.strip()
        if spec:
            out += ["-v", spec]
    return out


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
        *_extra_mounts(),
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
    output_path = os.path.join(OUTPUT_DIR, f"qwen_input_tone_{label}.png")
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
    shutil.copy2(latest, '/output/qwen_input_tone_{label}.png')
    print(f'Saved: qwen_input_tone_{label}.png')
""",
    ]
    print("  Generating Qwen image...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr[-300:]}")
        return None
    if os.path.exists(output_path):
        print(f"  OK: {output_path}")
        return output_path
    print("  FAIL: image not produced")
    return None


def submit_ltx23(tone, label, image_filename, video_prompt):
    workflow = create_workflow(
        prompt=video_prompt,
        image_filename=image_filename,
        width=WIDTH, height=HEIGHT, frames=FRAMES, fps=FPS,
        include_audio=True,
        output_prefix=f"tone_{tone}_{label}",
    )
    client_id = str(uuid.uuid4())
    try:
        resp = comfyui_api.submit(workflow, SERVER, client_id)
        pid = resp.get("prompt_id", "?")
        print(f"  Queued [{tone}] {label} -> {pid}")
        return pid
    except RuntimeError as e:
        print(f"  FAIL submit: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tones", nargs="+",
                   choices=["ironic", "sardonic", "comedic", "absurd"],
                   help="Restrict to a subset of tones (default: all four)")
    p.add_argument("--scenes-per-tone", type=int, default=None,
                   help="Cap scenes per tone (default: take all available, max 3)")
    p.add_argument("--scenes", nargs="+", type=int,
                   help="Override: 1-indexed subset by global scene number")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--hours", type=float, default=None,
                   help="Time budget in hours; planner picks largest subset that fits")
    p.add_argument("--per-scene-min", type=float, default=None,
                   help="Override estimated wall-time per LTX scene (default: derived "
                        "from --width/--height/--frames using session benchmarks)")
    p.add_argument("--qwen-min-per-image", type=float, default=QWEN_MIN_PER_IMAGE,
                   help="Override Qwen still-image generation cost (default 3.0 min)")
    p.add_argument("--prompts-file", type=str, default=None,
                   help="Load scenes from a JSON file instead of the built-in list. "
                        "Format: [{\"tone\":\"...\",\"label\":\"...\",\"qwen\":\"...\","
                        "\"video\":\"...\"}, ...]")
    p.add_argument("--theme-prefix", type=str, default="",
                   help="Text prepended to every Qwen + LTX prompt. Strongest "
                        "weight (early tokens dominate attention). Use for "
                        "style/aesthetic: 'noir film grain, 1940s, '")
    p.add_argument("--theme-suffix", type=str, default="",
                   help="Text appended to every Qwen + LTX prompt. Use for "
                        "technical/quality tags: ', 8k, hyperrealistic'")
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen phase, use existing qwen_input_tone_*.png")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan + time estimate, don't run anything")
    args = p.parse_args()

    # Load scenes from file if requested
    if args.prompts_file:
        import json
        with open(args.prompts_file) as f:
            raw = json.load(f)
        scenes_source = [(s["tone"], s["label"], s["qwen"], s["video"]) for s in raw]
    else:
        scenes_source = SCENES

    # Apply theme prefix/suffix to every prompt (both qwen and video)
    if args.theme_prefix or args.theme_suffix:
        wrapped = []
        for tone, label, qwen, video in scenes_source:
            wq = f"{args.theme_prefix}{qwen}{args.theme_suffix}".strip()
            wv = f"{args.theme_prefix}{video}{args.theme_suffix}".strip()
            wrapped.append((tone, label, wq, wv))
        scenes_source = wrapped

    # Apply tone filter
    scenes = scenes_source
    if args.tones:
        scenes = [s for s in scenes if s[0] in args.tones]
    # Apply per-tone cap
    if args.scenes_per_tone is not None:
        capped = []
        seen_per_tone = {}
        for s in scenes:
            n = seen_per_tone.get(s[0], 0)
            if n < args.scenes_per_tone:
                capped.append(s)
                seen_per_tone[s[0]] = n + 1
        scenes = capped
    # Manual override by global index (highest priority, against scenes_source)
    if args.scenes:
        scenes = [scenes_source[i - 1] for i in args.scenes
                  if 1 <= i <= len(scenes_source)]

    # Cost model
    per_scene = (args.per_scene_min if args.per_scene_min is not None
                 else estimate_min_per_scene(args.width, args.height, args.frames))
    qwen_phase = 0 if args.skip_qwen else len(scenes) * args.qwen_min_per_image
    ltx_phase = len(scenes) * per_scene
    total_min = qwen_phase + ltx_phase

    # Hours-budget trim
    if args.hours is not None:
        budget_min = args.hours * 60
        if total_min > budget_min:
            # Solve for max scene count k that fits:
            #   k * (per_scene + qwen_per_image_if_applicable) <= budget_min
            cost_each = per_scene + (0 if args.skip_qwen else args.qwen_min_per_image)
            k = max(0, int(budget_min // cost_each))
            print(f"Budget {args.hours}h ({budget_min:.0f}min) < estimated "
                  f"{total_min:.0f}min — trimming {len(scenes)} -> {k} scenes")
            scenes = scenes[:k]
            qwen_phase = 0 if args.skip_qwen else len(scenes) * args.qwen_min_per_image
            ltx_phase = len(scenes) * per_scene
            total_min = qwen_phase + ltx_phase

    # Apply runtime overrides into module-level constants used by helpers.
    global WIDTH, HEIGHT, FRAMES, FPS
    WIDTH, HEIGHT, FRAMES, FPS = args.width, args.height, args.frames, args.fps

    print("=" * 70)
    print(f"LTX-2.3 Tone-Variety Wave  ({WIDTH}x{HEIGHT} {FRAMES}f @ {FPS}fps)")
    print("=" * 70)
    by_tone = {}
    for i, (tone, label, _, _) in enumerate(scenes, 1):
        by_tone.setdefault(tone, []).append((i, label))
    for t in ["ironic", "sardonic", "comedic", "absurd"]:
        if t in by_tone:
            print(f"  {t}:")
            for idx, label in by_tone[t]:
                print(f"    {idx:2d}. {label}")

    print(f"\nEstimated wall time: {total_min:.0f} min "
          f"({per_scene:.1f}min/LTX-scene"
          f"{'' if args.skip_qwen else f', {args.qwen_min_per_image:.1f}min/Qwen-image'})")
    if args.hours is not None:
        print(f"  Budget: {args.hours}h = {args.hours*60:.0f} min")

    if args.dry_run:
        return 0
    if not scenes:
        print("No scenes selected; nothing to do.")
        return 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = {}
    if args.skip_qwen:
        for _, label, *_ in scenes:
            path = os.path.join(OUTPUT_DIR, f"qwen_input_tone_{label}.png")
            if os.path.exists(path):
                images[label] = os.path.basename(path)
            else:
                print(f"  MISSING: {path}")
    else:
        print("\n--- Phase 1: Qwen image generation ---")
        stop_comfyui()
        for tone, label, qwen_prompt, _ in scenes:
            print(f"\n[{tone}/{label}]")
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
    for tone, label, _qp, video_prompt in scenes:
        if label not in images:
            continue
        pid = submit_ltx23(tone, label, images[label], video_prompt)
        if pid:
            submitted.append((tone, label, pid))

    print("\n" + "=" * 70)
    print(f"Submitted {len(submitted)}/{len(scenes)} jobs.")
    print(f"Outputs: {OUTPUT_DIR}/tone_<tone>_<label>_*.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
