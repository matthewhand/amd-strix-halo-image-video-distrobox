#!/usr/bin/env python3
"""
Tone-variety wave — 12 scenes spanning four humor registers.

Categories (3 scenes each):
  - ironic:   image and audio expectations clash
  - sardonic: bitter / mocking / corporate cynicism
  - comedic:  lighthearted slapstick / wholesome silliness
  - absurd:   dream-logic / nonsensical premise played straight

All scenes default to 768x432 / 97f / 24fps (~4s clips). distilled-fp8 with
--lowvram (proven path from wave 1).

Usage:
    python tests/run_tone_wave.py                       # full 12
    python tests/run_tone_wave.py --tones absurd        # one category
    python tests/run_tone_wave.py --scenes 1 5 9        # subset by index
    python tests/run_tone_wave.py --dry-run             # plan only
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipelines import Scene, cli, config, cost_model, run_wave  # noqa: E402
from pipelines.scenes import filter_by  # noqa: E402

# (tone, label, qwen, video). Templates only; CLI overrides resolution/frames.
SCENE_TEMPLATES = [
    # === IRONIC ===
    ("ironic", "yoga_road_rage",
     "photograph of a serene blissful yoga instructor in immaculate white linen sitting cross-legged in lotus pose on a folded mat in the middle of a gridlocked downtown intersection, eyes closed in meditative peace, soft morning sunlight, cars and trucks bumper-to-bumper around her, billboards in the background",
     "the yoga instructor remains perfectly still and serene with hands in mudra position, audio is full road rage cacophony: blaring car horns, shouting drivers, screeching tires, a man yelling profanity at length, ambulance siren in distance, then a single tibetan bowl bell rings clearly through it all"),
    ("ironic", "funeral_kazoo",
     "wide shot photograph of a somber graveside funeral on a grey overcast afternoon, mourners in dark clothes standing around a polished casket draped with flowers, the priest in black robes holding a bible, a folded flag on a stand, weeping family in the foreground, leafless trees and headstones in the background",
     "the priest slowly raises a single bright pink kazoo to his lips and the entire group of mourners produces a loud unison kazoo rendition of taps, audio is a comically buzzy kazoo dirge with tinny harmonies and one wrong note, distant crow caws, a single sob blending into a kazoo trill"),
    ("ironic", "asmr_construction",
     "photograph of a cozy ASMR studio with soft pink LED lighting, a content creator in noise-cancelling headphones leaning close to a fluffy windscreened microphone, hands holding delicate items, comfortable beige fabric backdrop, intimate close-up framing",
     "the creator gently mouths into the microphone but the audio is full industrial construction site: jackhammer pounding, circular saw screaming through metal, dump truck reversing alarm, hard hat hammering rebar, foreman shouting orders through a megaphone, no soft sounds at all"),

    # === SARDONIC ===
    ("sardonic", "linkedin_parking_lot",
     "photograph of a polished motivational influencer in a sharp navy suit and bright smile holding up a smartphone for a selfie video in an empty fluorescent-lit suburban office park parking lot at twilight, his black SUV in the background, a laptop bag at his feet, mass-produced brick office building looms",
     "the influencer's confident smile cracks the moment he stops recording, his eyes go vacant and exhausted, he slowly sits down on the curb still holding the phone, audio is his own voiceover playing back: hashtag rise and grind, never give up, your dreams are valid, with a faint cracked sob underneath and the buzz of the parking lot lights"),
    ("sardonic", "open_plan_inspirational",
     "photograph of a vast open-plan tech office floor packed wall-to-wall with identical desks and rows of pale faces glued to monitors under harsh fluorescent ceiling lights, a giant inspirational wall decal reading WORK HARD HAVE FUN MAKE HISTORY in cheery hand-lettering above them, gray industrial carpet, no windows visible",
     "the camera slowly dollies across the rows of motionless workers as a single cubicle plant tips over and dies, audio is upbeat corporate inspirational stock music with chirpy ukulele and clapping rhythm playing relentlessly over dead silence from the workers, then a single keyboard key being pressed once every two seconds"),
    ("sardonic", "wedding_no_show",
     "photograph of an elaborately decorated outdoor wedding reception venue at sunset with chandeliers in the trees and round tables set with crystal and roses, a four-tier white wedding cake centerpiece, all chairs perfectly arranged and completely empty, a string quartet plays in the corner of the empty space, a tearful bride alone at the head table",
     "the bride slowly raises her champagne glass for a toast to the empty chairs, the quartet keeps playing pachelbel's canon dutifully, audio includes the polite string quartet performance, the soft clink of one glass meeting absolutely nothing, distant party laughter from a successful wedding next door, a single helium balloon popping"),

    # === COMEDIC ===
    ("comedic", "penguin_dance_off",
     "photograph of an emperor penguin standing on a frozen ice sheet wearing tiny mirrored disco sunglasses and a single gold chain around its neck, colorful disco party lights casting purple and pink across the ice, other penguins arranged in a loose circle as audience, snowflakes drifting",
     "the penguin attempts an elaborate hip-hop dance routine but immediately slips backward onto its belly and slides across the ice spinning, then determinedly waddle-stands and tries again falling forward, audio is full upbeat funk dance track with thumping bass plus the penguin's surprised honking and the audience penguins cheering"),
    ("comedic", "toddler_dog_ball",
     "photograph of a sunny suburban backyard with green grass, a happy golden retriever proudly holding a slobbery tennis ball in its mouth standing in front of an enthralled toddler in overalls and one shoe, a wooden fence and flowering hydrangeas behind them, soft late afternoon golden hour light",
     "in slow motion the toddler executes a perfect tackle stealing the ball from the dog's mouth then runs in a triumphant zigzag across the lawn, the dog looks at the camera in shocked betrayal then chases excitedly, audio is jaunty comedic chase music with kazoo and tuba, toddler shrieks of laughter, dog barks of joy, ball squeak"),
    ("comedic", "cat_zoom_meeting",
     "photograph of a fluffy orange tabby cat sitting upright in a small office chair behind a sleek modern desk wearing wire-rimmed reading glasses, a laptop displaying a tile grid of human coworkers' faces in a video call, a tiny mug of coffee labeled WORLDS BEST CAT, plant in the background, soft window light",
     "the cat solemnly raises one paw to its chin like it's deeply considering the conversation then suddenly knocks the coffee mug clean off the desk with absolute deliberation while making direct eye contact with the camera, audio is muffled human zoom call discussing quarterly metrics, the mug shattering, faint human gasps over the call"),

    # === ABSURD ===
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


def main():
    p = argparse.ArgumentParser()
    cli.add_common_args(p)
    p.add_argument("--prompts-file", type=str, default=None,
                   help="Load scenes from a JSON file (overrides built-in templates)")
    p.add_argument("--theme-prefix", type=str, default="",
                   help="Text prepended to every prompt (early-token attention dominance)")
    p.add_argument("--theme-suffix", type=str, default="",
                   help="Text appended to every prompt")
    args = p.parse_args()

    # Source rows: (tone, label, qwen, video)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            raw = json.load(f)
        rows = [(s["tone"], s["label"], s.get("qwen"), s.get("video"))
                for s in raw]
    else:
        rows = list(SCENE_TEMPLATES)

    if args.theme_prefix or args.theme_suffix:
        rows = [
            (t, l,
             f"{args.theme_prefix}{q}{args.theme_suffix}".strip() if q else None,
             f"{args.theme_prefix}{v}{args.theme_suffix}".strip())
            for t, l, q, v in rows
        ]

    scenes_source = [
        Scene(label=l, qwen_prompt=q, video_prompt=v,
              width=args.width, height=args.height,
              frames=args.frames, fps=args.fps,
              tone=t)
        for t, l, q, v in rows
    ]

    # Filtering: tones, then per-tone cap, then optional manual --scenes override
    scenes = filter_by(scenes_source, tones=args.tones,
                       scenes_per_tone=args.scenes_per_tone)
    if args.scenes:
        # Allow integer indexing OR label matching for backward compat
        try:
            indices = [int(s) for s in args.scenes]
            scenes = [scenes_source[i - 1] for i in indices
                      if 1 <= i <= len(scenes_source)]
        except ValueError:
            wanted = set(args.scenes)
            scenes = [s for s in scenes_source if s.label in wanted]

    # Cost model + budget trim
    per_scene = (args.per_scene_min if args.per_scene_min is not None
                 else cost_model.estimate_min_per_scene(
                     args.width, args.height, args.frames))
    if args.hours is not None:
        before = len(scenes)
        scenes = cost_model.fit_to_budget(
            scenes, args.hours,
            skip_qwen=args.skip_qwen,
            per_scene_min=per_scene,
            qwen_min_per_image=args.qwen_min_per_image,
        )
        if len(scenes) != before:
            print(f"Budget {args.hours}h trimmed {before} -> {len(scenes)} scenes")

    qwen_phase = 0 if args.skip_qwen else len(scenes) * args.qwen_min_per_image
    total_min = qwen_phase + len(scenes) * per_scene

    print("=" * 70)
    print(f"LTX-2.3 Tone-Variety Wave  ({args.width}x{args.height} "
          f"{args.frames}f @ {args.fps}fps)")
    print("=" * 70)
    by_tone: dict[str, list] = {}
    for i, s in enumerate(scenes, 1):
        by_tone.setdefault(s.tone or "?", []).append((i, s.label))
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

    paths = run_wave(
        scenes,
        skip_qwen=args.skip_qwen,
        output_prefix_fn=lambda s: f"tone_{s.tone}_{s.label}",
        image_path_fn=lambda s: os.path.join(
            config.OUTPUT_DIR, f"qwen_input_tone_{s.label}.png"
        ),
    )

    print("\n" + "=" * 70)
    print(f"Submitted {len(paths)}/{len(scenes)} jobs.")
    return 0 if paths or not scenes else 1


if __name__ == "__main__":
    sys.exit(main())
