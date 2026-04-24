#!/usr/bin/env python3
"""
Qwen image -> LTX-2.3 video+audio pipeline.

Builds the LTX-2.3 workflow via scripts/generate_ltx23_workflow.create_workflow,
submits via scripts/comfyui_api, and orchestrates Qwen <-> ComfyUI container
swaps so only one model occupies VRAM at a time (Strix Halo unified memory).

Usage:
    python tests/run_smoke.py --smoke              # 1 scene, low res
    python tests/run_smoke.py                      # all scenes
    python tests/run_smoke.py --scenes octopus_accountant cats_boardroom
    python tests/run_smoke.py --frames 49

Verified config (smoke 480x320, 49f, audio on): ~48s on Strix Halo gfx1151
with the 22B distilled fp8 model + Gemma-3 12B encoder.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipelines import Scene, run_wave  # noqa: E402

# (label, qwen_prompt, video_prompt, default_frames)
_RAW_SCENES = [
    ("octopus_accountant",
     "hyperrealistic photograph of an octopus wearing tiny reading glasses sitting at a desk covered in spreadsheets and tax forms, each tentacle holding a different pen or calculator, office cubicle background, fluorescent lighting, mundane corporate setting",
     "the octopus slowly looks up from the spreadsheets directly at the camera with an expression of existential dread, papers rustling, pens moving, calculator buttons clicking, a phone rings in the background",
     97),
    ("medieval_astronaut",
     "oil painting in renaissance style of an astronaut in a full NASA spacesuit sitting for a formal portrait in a medieval castle throne room, golden frame, dramatic chiaroscuro lighting, servants in period costume attending",
     "the astronaut slowly raises the visor revealing a confused medieval knight face underneath, servants gasping, candles flickering, dramatic orchestral music swelling",
     97),
    ("cats_boardroom",
     "corporate photograph of a serious business meeting in a luxury boardroom, but all the executives are cats in tiny suits and ties, one cat standing at a whiteboard with a laser pointer, charts showing fish stock prices, mahogany table",
     "the CEO cat slams its paw on the table and meows loudly, the other cats turn in shock, papers fly everywhere, the stock chart on the screen crashes, dramatic boardroom tension",
     97),
    ("dinosaur_barista",
     "detailed illustration of a friendly T-Rex working as a barista in a modern hipster coffee shop, tiny apron on its massive body, struggling to hold a tiny espresso cup with its small arms, customers waiting patiently in line, chalkboard menu",
     "the T-Rex carefully pours latte art into the tiny cup with trembling small arms, milk splashing everywhere, the cup crumbles in its claws, customers clapping encouragingly, coffee machine steaming",
     145),
    ("underwater_library",
     "photorealistic wide shot of a grand classical library that is completely submerged underwater, fish swimming between the bookshelves, an old librarian octopus organizing books, coral growing on marble columns, shafts of sunlight from above, books floating open with pages drifting",
     "camera slowly glides through the underwater library, pages turning by themselves in the current, small fish dart between shelves, bubbles rising from an open book, whale song echoing through the halls, peaceful and surreal",
     145),
    ("angry_ai_user",
     "photorealistic close up of a frustrated middle aged man in a messy home office, red faced and furious, gripping a keyboard with white knuckles, multiple monitors showing AI chatbot responses, energy drink cans everywhere, dramatic lighting from screen glow",
     "the man slams the keyboard on the desk and yells profanity at the screen, veins bulging on his forehead, a monitor flickers, he grabs his coffee mug and throws it, angry shouting and crashing sounds, keyboard keys flying",
     97),
]


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
        raw = _RAW_SCENES[:1]
        args.frames = 49
        args.width, args.height = 480, 320
        args.watch = True
    elif args.scenes:
        wanted = set(args.scenes)
        raw = [s for s in _RAW_SCENES if s[0] in wanted]
        if not raw:
            print(f"No scenes matched {sorted(wanted)}")
            print(f"Available: {[s[0] for s in _RAW_SCENES]}")
            return 1
    else:
        raw = _RAW_SCENES

    scenes = [
        Scene(label=label, qwen_prompt=qp, video_prompt=vp,
              width=args.width, height=args.height,
              frames=(args.frames or default_frames), fps=args.fps)
        for label, qp, vp, default_frames in raw
    ]

    print("Qwen -> LTX-2.3 Image-to-Video Pipeline")
    print("=" * 60)
    print(f"Scenes: {[s.label for s in scenes]}")
    print(f"Resolution: {args.width}x{args.height}, audio: True")

    paths = run_wave(
        scenes,
        skip_qwen=args.skip_qwen,
        watch=args.watch,
        output_prefix_fn=lambda s: f"i2v23_{s.label}",
    )

    print("\n" + "=" * 60)
    print(f"Submitted {len(paths)} jobs.")
    print("Monitor: python scripts/comfyui_api.py queue")
    return 0 if paths else 1


if __name__ == "__main__":
    sys.exit(main())
