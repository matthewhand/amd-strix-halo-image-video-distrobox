#!/usr/bin/env python3
"""
Chained-narrative LTX-2.3 wave: last frame of scene N seeds scene N+1.

Theme: "The Coffee Break" — 5 scenes of escalating absurdity. Qwen generates
the first frame; every subsequent scene starts from the previous video's
final frame so the action chains visually into one ~30s continuous arc.

Pipeline per scene:
    docker cp <input.png> -> ComfyUI input/
    submit i2v workflow, wait via /history polling
    docker cp <wave_*.mp4> -> host
    ffmpeg extracts last frame -> next scene's input

Usage:
    python tests/run_chained_wave.py
    python tests/run_chained_wave.py --scenes 1 2     # only first two
    python tests/run_chained_wave.py --skip-qwen      # use existing scene-1 png
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipelines import comfy_container, config, ltx_runner, qwen_runner  # noqa: E402

# Pull create_workflow once; pipelines/wave.py already wires sys.path.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from generate_ltx23_workflow import create_workflow  # noqa: E402

# Dev (43GB) + LoRA (7.6GB) at strength 0.5 produced NaN/Inf in the AAC
# audio encode (avcodec_send_frame returned 22). Falling back to distilled-fp8
# (no LoRA) which wave 1 proved works at 1024x576/145f.
MODEL = "ltx-2.3-22b-distilled-fp8.safetensors"
LORA = None
LORA_STRENGTH = 0.0
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FRAMES, DEFAULT_FPS = 1024, 576, 145, 24

SCENES = [
    {
        "label": "1_office_coffee",
        "qwen": "photorealistic mid shot of a tired office worker in a button-down shirt standing at a generic corporate break room counter, pouring coffee from a glass carafe into a white ceramic mug, microwave and sink visible in background, fluorescent overhead lighting, sterile beige walls, mundane Tuesday morning energy",
        "video": "the worker pours coffee in slow careful motion, steam rising from the mug, head tilted slightly down with mild boredom, audio is mundane office break room ambience: distant printer humming, refrigerator compressor cycling, faint hold music from a nearby phone, a dripping tap",
    },
    {
        "label": "2_coffee_swirls", "qwen": None,
        "video": "the coffee inside the mug suddenly begins spinning impossibly fast in a tight glowing vortex, faint blue light emanating from inside the cup, the worker's eyes widen and they slowly lower the carafe in confusion, audio shifts: room ambience drops to dead silence then a low electrical hum builds, metal stress sounds, faint whispering",
    },
    {
        "label": "3_dragon_eye", "qwen": None,
        "video": "a beam of golden light bursts upward from the mug revealing a tiny shimmering portal hovering above the coffee surface, an enormous slitted reptilian eye peers through the portal blinking once, the worker's hair blows back in the conjured wind, audio is reverb-drenched whoosh into a deep bass dragon snort, crackling magical energy, the worker letting out an involuntary squeak",
    },
    {
        "label": "4_dragon_ride", "qwen": None,
        "video": "the worker is now mid-air clinging to the back of a fully grown red scaled dragon flying down the office hallway, papers and cubicle walls exploding outward in their wake, terrified coworkers diving for cover, fluorescent lights raining down sparks, audio is full chaos: dragon roar, building alarms blaring, screaming, glass shattering, wind rush, and orchestral cinematic strings swelling",
    },
    {
        "label": "5_snow_globe_cosmos", "qwen": None,
        "video": "the camera rapidly pulls back zooming out through office windows then through the building roof revealing the entire office tower is encased inside a glass snow globe being gently shaken by an enormous cosmic toddler sitting cross-legged on a starry void, dozens of other planet-sized snow globes float around them, audio is full ambient cosmic drone with wind chimes, then a giggling baby laugh that distorts and echoes across the void",
    },
]


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
    scenes = list(base_scenes)
    while len(scenes) < count:
        i = len(scenes)
        scenes.append({
            "label": f"{i+1}_extension_{i}",
            "qwen": None,
            "video": EXTENSION_TEMPLATES[0].format(
                escalation=EXTENSION_ESCALATIONS[i % len(EXTENSION_ESCALATIONS)],
                audio_shift=EXTENSION_AUDIO[i % len(EXTENSION_AUDIO)],
            ),
        })
    return scenes[:count]


def find_latest_mp4(label):
    matches = sorted(
        glob.glob(f"{config.OUTPUT_DIR}/chain_{label}_*.mp4"),
        key=os.path.getmtime,
    )
    return matches[-1] if matches else None


def submit_and_wait_chain(label, image_filename, video_prompt,
                          width, height, frames, fps):
    wf = create_workflow(
        prompt=video_prompt,
        image_filename=image_filename,
        width=width, height=height, frames=frames, fps=fps,
        include_audio=True,
        model_name=MODEL,
        lora_name=LORA,
        lora_strength=LORA_STRENGTH,
        output_prefix=f"chain_{label}",
    )
    print(f"  Submitting LTX-2.3 i2v ({width}x{height}/{frames}f)...")
    status, err = ltx_runner.submit_and_wait(wf)
    if status != "success":
        print(f"  FAIL: {err}")
        return None
    return "success"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", type=int,
                   help="1-indexed subset (e.g. --scenes 1 2)")
    p.add_argument("--count", type=int, default=None,
                   help="Number of chained scenes. <len(SCENES) truncates; "
                        ">len(SCENES) auto-extends with generic escalation "
                        "prompts (series mode) or repeats (repeat mode).")
    p.add_argument("--mode", choices=["series", "repeat"], default="series",
                   help="series: each scene has its own prompt (default). "
                        "repeat: same prompt across all chained scenes.")
    p.add_argument("--prompt", type=str, default=None,
                   help="In repeat mode: video prompt to use for every scene")
    p.add_argument("--qwen-prompt", type=str, default=None,
                   help="In repeat mode: Qwen seed prompt")
    p.add_argument("--prompts-file", type=str, default=None,
                   help="JSON list [{label, qwen (null for mid-chain), video}, ...]")
    p.add_argument("--no-join", action="store_true",
                   help="Don't concat completed scenes into a single mp4")
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen seed (scene 1 png must already exist)")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = p.parse_args()
    width, height, frames, fps = args.width, args.height, args.frames, args.fps

    if args.prompts_file:
        with open(args.prompts_file) as f:
            scenes = json.load(f)
    else:
        scenes = SCENES

    if args.mode == "repeat":
        base_qwen = args.qwen_prompt or scenes[0]["qwen"]
        base_video = args.prompt or scenes[0]["video"]
        n = args.count or 5
        base_label = (scenes[0]["label"].split("_", 1)[1]
                      if "_" in scenes[0]["label"] else "repeat")
        scenes = [
            {"label": f"{i+1}_{base_label}",
             "qwen": base_qwen if i == 0 else None,
             "video": base_video}
            for i in range(n)
        ]
    elif args.count is not None:
        scenes = extend_scenes(scenes, args.count)

    if args.scenes:
        scenes = [scenes[i - 1] for i in args.scenes if 1 <= i <= len(scenes)]

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LTX-2.3 Chained Narrative — 'The Coffee Break'")
    print(f"Model: {MODEL}  +  LoRA: {LORA} @ {LORA_STRENGTH}")
    print(f"Per-scene: {width}x{height}  {frames}f @ {fps}fps  ({frames/fps:.1f}s)")
    print("=" * 70)
    for i, s in enumerate(scenes, 1):
        print(f"  {i}. {s['label']}")

    # Phase 1: Qwen seed for scene 1
    seed_img = None
    if not args.skip_qwen:
        first = scenes[0]
        if first["qwen"]:
            print(f"\n--- Phase 1: Qwen seed for {first['label']} ---")
            comfy_container.stop()
            seed_img = qwen_runner.generate_image(
                first["qwen"],
                os.path.join(config.OUTPUT_DIR, f"chain_{first['label']}.png"),
            )
            if not seed_img:
                print("Qwen seed failed; aborting.")
                return 1
    else:
        seed_img = os.path.join(config.OUTPUT_DIR, f"chain_{scenes[0]['label']}.png")
        if not os.path.exists(seed_img):
            print(f"--skip-qwen but {seed_img} doesn't exist; aborting.")
            return 1

    # Phase 2: chained video gen
    print(f"\n--- Phase 2: Chained LTX-2.3 ({len(scenes)} scenes) ---")
    comfy_container.start()
    if not comfy_container.wait_ready():
        return 1

    current_input_path = seed_img
    completed = []

    for idx, scene in enumerate(scenes, 1):
        label = scene["label"]
        print(f"\n[{idx}/{len(scenes)}] {label}")
        print(f"  Input image: {os.path.basename(current_input_path)}")

        in_filename = os.path.basename(current_input_path)
        if not comfy_container.cp_in(current_input_path, in_filename):
            print(f"  FAIL: docker cp input")
            break

        if submit_and_wait_chain(label, in_filename, scene["video"],
                                 width, height, frames, fps) != "success":
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
            next_input = os.path.join(
                config.OUTPUT_DIR, f"chain_{scenes[idx]['label']}.png"
            )
            if comfy_container.extract_last_frame(mp4, next_input):
                print(f"  Extracted last frame -> {os.path.basename(next_input)}")
                current_input_path = next_input
            else:
                print(f"  FAIL: ffmpeg last-frame extract; aborting chain")
                break

    print("\n" + "=" * 70)
    print(f"Completed {len(completed)}/{len(scenes)} scenes:")
    for m in completed:
        print(f"  {m}")

    if not args.no_join and len(completed) >= 2:
        # narrative is the prompts_file basename (e.g. lost_letter.json -> lost_letter);
        # fall back to timestamp if not provided (t2v / SCENES[].label chain).
        if args.prompts_file:
            narrative = os.path.splitext(os.path.basename(args.prompts_file))[0]
            joined_name = f"chain_{narrative}_{width}x{height}_{frames}f.mp4"
        else:
            joined_name = f"chain_joined_{int(time.time())}.mp4"
        joined = os.path.join(config.OUTPUT_DIR, joined_name)
        print(f"\nJoining {len(completed)} clips into {os.path.basename(joined)}...")
        if comfy_container.join_mp4s(completed, joined):
            size_mb = os.path.getsize(joined) / 1e6
            print(f"  OK: {joined} ({size_mb:.1f} MB)")
        else:
            print(f"  FAIL: ffmpeg concat")

    return 0 if len(completed) == len(scenes) else 1


if __name__ == "__main__":
    sys.exit(main())
