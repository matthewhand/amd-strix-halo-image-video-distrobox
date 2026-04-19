#!/usr/bin/env python3
"""
Matrix runner — sweep test_tone_wave.py across (subject × hour-budget) cells.

For each subject in --subjects:
  For each budget in --hours:
    1. Build a subject-templated prompts JSON (substitutes {subject} into
       per-tone scene templates so a single subject runs through all four
       tonal registers)
    2. Invoke tests/test_tone_wave.py --prompts-file <json> --hours <budget>
    3. Tag outputs by subject for later organization

Designed to make 165 GPU-hours of permutation runs visible BEFORE you commit
to any of them. Always estimates total first; --dry-run shows the plan.

Usage:
    # Plan only — see total estimated wall time
    python tests/matrix_runner.py --subjects animals robots --hours 1 2 --dry-run

    # Run a single cell
    python tests/matrix_runner.py --subjects "gothic space pirates" --hours 2

    # Full sweep (DON'T do this without thinking)
    python tests/matrix_runner.py \\
        --subjects animals robots "gothic space pirates" \\
        --hours 1 2 3 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))


# Subjects ordered by absurdity (mundane → unhinged). Add more freely; the
# templates substitute {subject} once per scene so there's no per-subject
# scripting required.
DEFAULT_SUBJECTS = [
    "floating brains",
    "robots",
    "gothic space pirates",
    "sentient kitchen appliances",
    "interdimensional bureaucrats",
    "existentially-confused houseplants",
]

# Per-tone scene templates. Each template gets {subject} substituted so a
# subject like "robots" runs through every tone consistently. 3 per tone.
SCENE_TEMPLATES = [
    # === IRONIC ===
    ("ironic", "yoga_road_rage",
     "photograph of a serene blissful {subject} sitting cross-legged in lotus pose on a folded mat in the middle of a gridlocked downtown intersection, eyes closed in meditative peace, soft morning sunlight, cars and trucks bumper-to-bumper around, billboards in the background",
     "the {subject} remains perfectly still and serene, audio is full road rage cacophony: blaring car horns, shouting drivers, screeching tires, distant ambulance siren, then a single tibetan bowl bell rings clearly through it all"),
    ("ironic", "asmr_construction",
     "photograph of a cozy ASMR studio with soft pink LED lighting, a {subject} content creator in noise-cancelling headphones leaning close to a fluffy windscreened microphone, hands or appendages holding delicate items, comfortable beige fabric backdrop, intimate close-up framing",
     "the {subject} performs gentle whisper-quiet ASMR motions but the audio is full industrial construction site: jackhammer, circular saw, dump truck reversing alarm, foreman shouting orders through a megaphone"),
    ("ironic", "funeral_kazoo",
     "photograph of a somber graveside funeral gathering on a grey overcast afternoon, mourners are all {subject} in dark clothing standing around a polished casket draped with flowers, leafless trees and headstones in the background",
     "the lead {subject} mourner solemnly raises a single bright pink kazoo and the entire group performs a comically buzzy unison kazoo rendition of taps, distant crow caws, one sob blending into a kazoo trill"),

    # === SARDONIC ===
    ("sardonic", "linkedin_parking_lot",
     "photograph of a polished motivational {subject} influencer in a sharp navy suit holding up a smartphone for a selfie video in an empty fluorescent-lit suburban office park parking lot at twilight, mass-produced brick office building looms",
     "the {subject} influencer's confident smile cracks the moment recording stops, eyes go vacant and exhausted, slowly sits down on the curb still holding the phone, audio is its own voiceover playing back: hashtag rise and grind, never give up, with a faint cracked sob underneath"),
    ("sardonic", "open_plan_inspirational",
     "photograph of a vast open-plan tech office floor packed wall-to-wall with identical desks and rows of {subject} workers glued to monitors under harsh fluorescent lighting, a giant inspirational wall decal reading WORK HARD HAVE FUN MAKE HISTORY above them, gray industrial carpet, no windows",
     "camera slowly dollies across the rows of motionless {subject} workers, audio is upbeat corporate inspirational stock music with chirpy ukulele playing relentlessly over dead silence from the workers, then a single keyboard key being pressed once every two seconds"),
    ("sardonic", "wedding_no_show",
     "photograph of an elaborately decorated outdoor wedding reception venue at sunset, all chairs perfectly arranged and completely empty, a string quartet of {subject} plays in the corner of the empty space, a tearful {subject} bride alone at the head table",
     "the {subject} bride slowly raises a champagne glass for a toast to the empty chairs, the quartet keeps playing pachelbel's canon dutifully, audio includes the polite string quartet, the soft clink of one glass meeting absolutely nothing, distant party laughter from a successful wedding next door"),

    # === COMEDIC ===
    ("comedic", "dance_off",
     "photograph of a {subject} on a glossy stage wearing tiny mirrored disco sunglasses and a single gold chain, colorful disco party lights casting purple and pink, other {subject} arranged in a loose circle as audience, snowflakes or confetti drifting",
     "the {subject} attempts an elaborate hip-hop dance routine but immediately slips and slides across the stage spinning, then determinedly pops back up and tries again falling forward, audio is full upbeat funk dance track plus the {subject} surprised noises and audience cheering"),
    ("comedic", "ball_tackle",
     "photograph of a sunny suburban backyard with green grass, a happy {subject} proudly holding a slobbery tennis ball in front of a {subject} toddler in overalls, soft golden hour light",
     "in slow motion the {subject} toddler executes a perfect tackle stealing the ball then runs in a triumphant zigzag, the original {subject} looks at the camera in shocked betrayal then chases excitedly, audio is jaunty comedic chase music with kazoo, shrieks of laughter, joyful barks"),
    ("comedic", "zoom_meeting",
     "photograph of a {subject} sitting upright in a small office chair behind a sleek desk wearing wire-rimmed reading glasses, a laptop displaying a tile grid of human coworkers' faces in a video call, a tiny mug labeled WORLDS BEST {subject}, plant in background, soft window light",
     "the {subject} solemnly raises one paw or appendage to its chin like deeply considering, then suddenly knocks the coffee mug clean off the desk with absolute deliberation while making direct eye contact with the camera, audio is muffled human zoom call discussing quarterly metrics, the mug shattering, faint human gasps"),

    # === ABSURD ===
    ("absurd", "wedding",
     "wide photograph of a small ornate wedding ceremony in a candlelit chapel where two {subject} stand at the altar in tiny tuxedo bowtie and lace veil, an officiant of the same kind in robes, congregation of {subject} fills the pews in formal attire",
     "the officiant {subject} declares the union and the bride {subject} dramatically performs a ritual flourish, audio is a full traditional pipe organ wedding march, congregation murmurs of approval, a single sniffle from the front row"),
    ("absurd", "keynote",
     "photograph of a single dignified {subject} standing alone on a giant convention center stage under a single dramatic spotlight, a clip-on lavalier microphone attached, a massive screen behind reads THE FUTURE in sleek tech-conference branding, an audience of hundreds of other {subject} in folding chairs visible in the dark",
     "the {subject} performs gestures of authority as if speaking, audio is a deeply confident TED-talk-style narrator delivering a passionate keynote: thank you all for being here, today we are going to talk about an increasingly warm world, applause swells from the audience"),
    ("absurd", "talk_show",
     "photograph of a {subject} as a late-night talk show host behind a sleek talk show desk under bright studio lights, a tiny clip-on microphone, an empty matching guest chair sits across, the studio backdrop reads AROUND THE WORLD in glowing letters, audience seating visible in the soft blur",
     "the {subject} performs a slow self-important gesture then leans forward as if making an interview point, audio is a polished late-night talk show host voice asking a guest about their new memoir, recorded studio audience laughter at the perfect rhythm, an opening jingle plays"),
]


def estimate_min_per_scene(width, height, frames):
    """Mirror of estimator in test_tone_wave.py."""
    mvox = (width * height * frames) / 1_000_000
    return 0.5 + 0.07 * mvox + 0.0019 * (mvox ** 2)


QWEN_MIN_PER_IMAGE = 3.0


def build_prompts_for_subject(subject, scenes_per_tone=None):
    """Substitute {subject} into all templates, returning a list ready for JSON."""
    out = []
    seen = {}
    for tone, label, qwen_t, video_t in SCENE_TEMPLATES:
        n = seen.get(tone, 0)
        if scenes_per_tone is not None and n >= scenes_per_tone:
            continue
        out.append({
            "tone": tone,
            "label": f"{label}_{subject.replace(' ', '_')}",
            "qwen": qwen_t.replace("{subject}", subject),
            "video": video_t.replace("{subject}", subject),
        })
        seen[tone] = n + 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS,
                   help="Subjects to substitute into templates (in order of "
                        "increasing absurdity). Default: 6 built-in subjects.")
    p.add_argument("--hours", nargs="+", type=float, required=True,
                   help="Hour budget(s) per (subject, budget) cell. "
                        "Example: --hours 1 2 3 produces 3 cells per subject.")
    p.add_argument("--tones", nargs="+",
                   choices=["ironic", "sardonic", "comedic", "absurd"],
                   help="Restrict to a subset of tones (passed through)")
    p.add_argument("--scenes-per-tone", type=int, default=None,
                   help="Cap scenes per tone (default: all 3 templates per tone)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--frames", type=int, default=97)
    p.add_argument("--per-scene-min", type=float, default=None,
                   help="Override LTX per-scene wall-time estimate")
    p.add_argument("--qwen-min-per-image", type=float, default=QWEN_MIN_PER_IMAGE)
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen for cells that already have qwen_input_tone_*.png")
    p.add_argument("--dry-run", action="store_true",
                   help="Print matrix + total estimated wall time only")
    p.add_argument("--prompts-out", type=str, default="/tmp/matrix_prompts",
                   help="Where generated prompts JSON files land")
    args = p.parse_args()

    cells = [(s, h) for s in args.subjects for h in args.hours]
    per_scene = (args.per_scene_min if args.per_scene_min is not None
                 else estimate_min_per_scene(args.width, args.height, args.frames))

    print("=" * 70)
    print(f"Matrix Runner: {len(cells)} cells = {len(args.subjects)} subjects × "
          f"{len(args.hours)} budgets")
    print(f"Per-scene LTX: {per_scene:.1f} min  ({args.width}x{args.height}/{args.frames}f)")
    print("=" * 70)

    cost_each = per_scene + (0 if args.skip_qwen else args.qwen_min_per_image)
    total_min = 0
    plan = []
    for subj, hours in cells:
        prompts = build_prompts_for_subject(subj, args.scenes_per_tone)
        if args.tones:
            prompts = [p for p in prompts if p["tone"] in args.tones]
        budget_min = hours * 60
        max_scenes = int(budget_min // cost_each) if cost_each else 0
        scenes_to_run = min(len(prompts), max_scenes)
        cell_min = scenes_to_run * cost_each
        plan.append((subj, hours, scenes_to_run, len(prompts), cell_min))
        total_min += cell_min

    print(f"\n{'subject':40s} {'budget':>7s} {'scenes':>10s} {'wall':>8s}")
    print("-" * 70)
    for subj, hours, n, navail, cell_min in plan:
        print(f"{subj:40s} {hours:5.1f}h {n:3d}/{navail:<3d}     "
              f"{cell_min:5.0f}min")
    print("-" * 70)
    print(f"{'TOTAL':40s} {sum(args.hours)*len(args.subjects):5.1f}h "
          f"           {total_min:5.0f}min ({total_min/60:.1f}h)")

    if args.dry_run:
        print("\n(dry run — no waves launched)")
        return 0

    Path(args.prompts_out).mkdir(parents=True, exist_ok=True)

    # Confirm before kicking off long sweeps
    if total_min > 120:
        print(f"\n⚠️  Total estimated wall time: {total_min/60:.1f}h. "
              "Continue? Press Ctrl-C now to abort, or wait 10s.")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("Aborted.")
            return 1

    completed = []
    for subj, hours, n, navail, cell_min in plan:
        if n == 0:
            print(f"\n--- SKIP {subj} @ {hours}h (budget too small) ---")
            continue
        prompts = build_prompts_for_subject(subj, args.scenes_per_tone)
        if args.tones:
            prompts = [p for p in prompts if p["tone"] in args.tones]
        prompts_file = Path(args.prompts_out) / f"prompts_{subj.replace(' ', '_')}.json"
        prompts_file.write_text(json.dumps(prompts, indent=2))

        print(f"\n=== CELL {subj} @ {hours}h ({n} scenes) ===")
        cmd = [
            sys.executable, str(HERE / "test_tone_wave.py"),
            "--prompts-file", str(prompts_file),
            "--hours", str(hours),
            "--width", str(args.width),
            "--height", str(args.height),
            "--frames", str(args.frames),
        ]
        if args.skip_qwen:
            cmd.append("--skip-qwen")
        if args.tones:
            cmd += ["--tones"] + args.tones
        rc = subprocess.run(cmd).returncode
        completed.append((subj, hours, rc))

    print("\n" + "=" * 70)
    print(f"Matrix complete: {sum(1 for _, _, rc in completed if rc == 0)}/{len(completed)} cells exited 0")
    for subj, hours, rc in completed:
        mark = "✓" if rc == 0 else "✗"
        print(f"  {mark} {subj} @ {hours}h (rc={rc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
