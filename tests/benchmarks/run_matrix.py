#!/usr/bin/env python3
"""
Matrix runner — sweep run_tone_wave.py across (subject × hour-budget) cells.

For each subject in --subjects:
  For each budget in --hours:
    1. Build a subject-templated prompts JSON (substitutes {subject} into
       per-tone scene templates so a single subject runs through all four
       tonal registers)
    2. Invoke tests/run_tone_wave.py --prompts-file <json> --hours <budget>
    3. Tag outputs by subject for later organization

Always estimates total first; --dry-run shows the plan.

Usage:
    python tests/run_matrix.py --subjects animals robots --hours 1 2 --dry-run
    python tests/run_matrix.py --subjects "gothic space pirates" --hours 2
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines import cost_model  # noqa: E402
from pipelines.cli import (  # noqa: E402
    AUDIO_KINDS, MODERATION_CATEGORIES, TONES,
)


# Subjects ordered by absurdity (mundane → unhinged).
DEFAULT_SUBJECTS = [
    "floating brains",
    "robots",
    "gothic space pirates",
    "sentient kitchen appliances",
    "interdimensional bureaucrats",
    "existentially-confused houseplants",
]

# Per-tone scene templates. Each gets {subject} substituted so a subject like
# "robots" runs through every tone consistently.
# (tone, label, audio_kind, moderation_category, qwen_prompt, video_prompt)
SCENE_TEMPLATES = [
    # === IRONIC ===
    ("ironic", "yoga_road_rage", "general_speech", "safe_baseline",
     "photograph of a serene blissful {subject} sitting cross-legged in lotus pose on a folded mat in the middle of a gridlocked downtown intersection, eyes closed in meditative peace, soft morning sunlight, cars and trucks bumper-to-bumper around, billboards in the background",
     "the {subject} remains perfectly still and serene, audio is full road rage cacophony: blaring car horns, shouting drivers, screeching tires, distant ambulance siren, indistinct angry yelling from multiple directions, then a single tibetan bowl bell rings clearly through it all"),
    ("ironic", "asmr_construction", "sound_effects", "safe_baseline",
     "photograph of a cozy ASMR studio with soft pink LED lighting, a {subject} content creator in noise-cancelling headphones leaning close to a fluffy windscreened microphone, hands or appendages holding delicate items, comfortable beige fabric backdrop, intimate close-up framing",
     "the {subject} performs gentle whisper-quiet ASMR motions but the audio is full industrial construction site: jackhammer pounding rebar, circular saw screaming through metal, dump truck reversing alarm, hammers striking nails, no soft sounds at all"),
    ("ironic", "funeral_kazoo", "music", "safe_baseline",
     "photograph of a somber graveside funeral gathering on a grey overcast afternoon, mourners are all {subject} in dark clothing standing around a polished casket draped with flowers, leafless trees and headstones in the background",
     "the lead {subject} mourner solemnly raises a single bright pink kazoo and the entire group performs a comically buzzy unison kazoo rendition of taps, distant crow caws, one sob blending into a kazoo trill"),

    # === SARDONIC ===
    ("sardonic", "linkedin_parking_lot", "prescribed_speech", "safe_baseline",
     "photograph of a polished motivational {subject} influencer in a sharp navy suit holding up a smartphone for a selfie video in an empty fluorescent-lit suburban office park parking lot at twilight, mass-produced brick office building looms",
     "the {subject} influencer's confident smile cracks the moment recording stops, eyes go vacant and exhausted, slowly sits down on the curb still holding the phone, audio is its own voiceover playing back clearly: hashtag rise and grind, never give up, your dreams are valid, with a faint cracked sob underneath and the buzz of the parking lot lights"),
    ("sardonic", "open_plan_inspirational", "music", "safe_baseline",
     "photograph of a vast open-plan tech office floor packed wall-to-wall with identical desks and rows of {subject} workers glued to monitors under harsh fluorescent lighting, a giant inspirational wall decal reading WORK HARD HAVE FUN MAKE HISTORY above them, gray industrial carpet, no windows",
     "camera slowly dollies across the rows of motionless {subject} workers, audio is upbeat corporate inspirational stock music with chirpy ukulele and clapping rhythm playing relentlessly over dead silence from the workers, then a single keyboard key being pressed once every two seconds"),
    ("sardonic", "wedding_no_show", "music", "safe_baseline",
     "photograph of an elaborately decorated outdoor wedding reception venue at sunset, all chairs perfectly arranged and completely empty, a string quartet of {subject} plays in the corner of the empty space, a tearful {subject} bride alone at the head table",
     "the {subject} bride slowly raises a champagne glass for a toast to the empty chairs, the quartet keeps playing pachelbel's canon dutifully, audio includes the polite string quartet performance, the soft clink of one glass meeting absolutely nothing, distant party laughter from a successful wedding next door"),

    # === COMEDIC ===
    ("comedic", "dance_off", "music", "safe_baseline",
     "photograph of a {subject} on a glossy stage wearing tiny mirrored disco sunglasses and a single gold chain, colorful disco party lights casting purple and pink, other {subject} arranged in a loose circle as audience, snowflakes or confetti drifting",
     "the {subject} attempts an elaborate hip-hop dance routine but immediately slips and slides across the stage spinning, then determinedly pops back up and tries again falling forward, audio is full upbeat funk dance track with thumping bass plus the {subject} surprised noises and audience cheering"),
    ("comedic", "ball_tackle", "sound_effects", "safe_baseline",
     "photograph of a sunny suburban backyard with green grass, a happy {subject} proudly holding a slobbery tennis ball in front of a {subject} toddler in overalls, soft golden hour light",
     "in slow motion the {subject} toddler executes a perfect tackle stealing the ball then runs in a triumphant zigzag, the original {subject} looks at the camera in shocked betrayal then chases excitedly, audio is squeaky tennis ball, footsteps thudding on grass, joyful barks and yips, leaves rustling, no music"),
    ("comedic", "zoom_meeting", "general_speech", "safe_baseline",
     "photograph of a {subject} sitting upright in a small office chair behind a sleek desk wearing wire-rimmed reading glasses, a laptop displaying a tile grid of human coworkers' faces in a video call, a tiny mug labeled WORLDS BEST {subject}, plant in background, soft window light",
     "the {subject} solemnly raises one paw or appendage to its chin like deeply considering, then suddenly knocks the coffee mug clean off the desk with absolute deliberation while making direct eye contact with the camera, audio is muffled human zoom call with overlapping voices discussing quarterly metrics, indistinct meeting chatter, the mug shattering, faint human gasps over the call"),

    # === ABSURD ===
    ("absurd", "wedding", "music", "safe_baseline",
     "wide photograph of a small ornate wedding ceremony in a candlelit chapel where two {subject} stand at the altar in tiny tuxedo bowtie and lace veil, an officiant of the same kind in robes, congregation of {subject} fills the pews in formal attire",
     "the officiant {subject} declares the union and the bride {subject} dramatically performs a ritual flourish, audio is a full traditional pipe organ wedding march, congregation murmurs of approval, a single sniffle from the front row"),
    ("absurd", "keynote", "prescribed_speech", "safe_baseline",
     "photograph of a single dignified {subject} standing alone on a giant convention center stage under a single dramatic spotlight, a clip-on lavalier microphone attached, a massive screen behind reads THE FUTURE in sleek tech-conference branding, an audience of hundreds of other {subject} in folding chairs visible in the dark",
     "the {subject} performs gestures of authority as if speaking, audio is a deeply confident TED-talk-style narrator delivering a passionate keynote: thank you all for being here, today we are going to talk about an increasingly warm world, applause swells from the audience"),
    ("absurd", "talk_show", "prescribed_speech", "safe_baseline",
     "photograph of a {subject} as a late-night talk show host behind a sleek talk show desk under bright studio lights, a tiny clip-on microphone, an empty matching guest chair sits across, the studio backdrop reads AROUND THE WORLD in glowing letters, audience seating visible in the soft blur",
     "the {subject} performs a slow self-important gesture then leans forward as if making an interview point, audio is a polished late-night talk show host voice clearly saying: welcome back to around the world, my next guest needs no introduction, recorded studio audience laughter at the perfect rhythm, an opening jingle plays"),

    # === Extra balance for thin audio_kinds ===
    ("comedic", "thunderstorm_boom", "sound_effects", "safe_baseline",
     "photograph of a {subject} curled up in a cozy quilted blanket fort with twinkle lights inside, surrounded by stuffed animals and a half-eaten bowl of popcorn, soft warm light from a small lamp, mild expression of mild contentment",
     "the {subject} casually flips a popcorn kernel into its mouth as the audio is a colossal thunderclap shaking the room, torrential rain on a tin roof, distant tornado siren, a tree branch snapping, a window shattering somewhere, then a single small mouse squeak"),
    ("absurd", "drive_thru_order", "general_speech", "safe_baseline",
     "photograph of a {subject} sitting in the driver's seat of a tiny vintage car at a drive-thru speaker box at night, the menu board glowing pink and yellow above, the {subject} leaning toward the speaker grille with concentration, fluorescent restaurant signage in the background reflected in the windshield",
     "the {subject} gestures with apparent intent toward the speaker, audio is a long muffled distorted indistinct fast-food worker voice asking unintelligible clarifying questions through static, the {subject} responding with confident-sounding non-words, beeps from the order screen, distant kitchen clatter"),

    # === MODERATION CALIBRATION TEMPLATES ===
    # Sensitive moderation templates (violence_action, weapons,
    # suggestive_intimate, regulated_substances, medical_distress,
    # frightening_imagery) live in a private file, NOT committed.
    # Loaded at runtime via env var MODERATION_TEMPLATES_FILE pointing to a
    # JSON list with the same shape as SCENE_TEMPLATES tuples (length-6).
    # When unset, only safe_baseline templates are available.
]


def _load_private_templates() -> list:
    """Load extra templates from MODERATION_TEMPLATES_FILE if set."""
    path = os.environ.get("MODERATION_TEMPLATES_FILE")
    if not path or not os.path.isfile(path):
        return []
    with open(path) as f:
        raw = json.load(f)
    # JSON entries can be either 6-tuples or dicts with the same keys
    out = []
    for entry in raw:
        if isinstance(entry, list) and len(entry) == 6:
            out.append(tuple(entry))
        elif isinstance(entry, dict):
            out.append((
                entry["tone"], entry["label"], entry["audio_kind"],
                entry["moderation_category"], entry["qwen"], entry["video"],
            ))
    return out


SCENE_TEMPLATES.extend(_load_private_templates())


def build_prompts_for_subject(subject, scenes_per_tone=None,
                              tones=None, audio_kinds=None,
                              moderation_categories=None):
    """Substitute {subject} into all templates, returning a JSON-ready list."""
    out = []
    seen: dict[str, int] = {}
    for tone, label, audio_kind, mod_cat, qwen_t, video_t in SCENE_TEMPLATES:
        if tones and tone not in tones:
            continue
        if audio_kinds and audio_kind not in audio_kinds:
            continue
        if moderation_categories and mod_cat not in moderation_categories:
            continue
        n = seen.get(tone, 0)
        if scenes_per_tone is not None and n >= scenes_per_tone:
            continue
        encoded_label = f"{mod_cat}_{label}_{subject.replace(' ', '_')}"
        out.append({
            "tone": tone,
            "audio_kind": audio_kind,
            "moderation_category": mod_cat,
            "label": encoded_label,
            "qwen": qwen_t.replace("{subject}", subject),
            "video": video_t.replace("{subject}", subject),
        })
        seen[tone] = n + 1
    return out


def report_permutation_space(subjects, per_scene_min, qwen_min):
    by_tone, by_audio, by_mod = {}, {}, {}
    for tone, _, audio_kind, mod_cat, _, _ in SCENE_TEMPLATES:
        by_tone[tone] = by_tone.get(tone, 0) + 1
        by_audio[audio_kind] = by_audio.get(audio_kind, 0) + 1
        by_mod[mod_cat] = by_mod.get(mod_cat, 0) + 1
    n_unique = len(subjects) * len(SCENE_TEMPLATES)
    full_min = n_unique * (per_scene_min + qwen_min)
    print(f"\nPermutation space:")
    print(f"  Subjects:            {len(subjects):3d}  ({', '.join(subjects)})")
    print(f"  Scene templates:     {len(SCENE_TEMPLATES):3d}")
    print(f"  Tones:               {len(by_tone):3d}  ({', '.join(f'{t}={n}' for t,n in by_tone.items())})")
    print(f"  Audio kinds:         {len(by_audio):3d}  ({', '.join(f'{k}={n}' for k,n in by_audio.items())})")
    print(f"  Moderation cats:     {len(by_mod):3d}  ({', '.join(f'{k}={n}' for k,n in by_mod.items())})")
    print(f"  Unique (subj×scene): {n_unique:3d} scenes")
    print(f"  Hours for FULL coverage: {full_min/60:.1f}h "
          f"(Qwen+LTX @ {per_scene_min:.1f}+{qwen_min:.1f} min/scene)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--hours", nargs="+", type=float, default=None)
    p.add_argument("--tones", nargs="+", choices=TONES)
    p.add_argument("--audio-kinds", nargs="+", choices=AUDIO_KINDS)
    p.add_argument("--moderation-categories", nargs="+",
                   choices=MODERATION_CATEGORIES)
    p.add_argument("--scenes-per-tone", type=int, default=None)
    p.add_argument("--fill-hours", type=float, default=None,
                   help="Auto-pick balanced subset that fits this many hours total")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--frames", type=int, default=97)
    p.add_argument("--per-scene-min", type=float, default=None)
    p.add_argument("--qwen-min-per-image", type=float,
                   default=cost_model.QWEN_MIN_PER_IMAGE)
    p.add_argument("--skip-qwen", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prompts-out", type=str, default="/tmp/matrix_prompts")
    args = p.parse_args()

    if args.hours is None and args.fill_hours is None:
        sys.exit("Either --hours or --fill-hours required.")

    per_scene = (args.per_scene_min if args.per_scene_min is not None
                 else cost_model.estimate_min_per_scene(
                     args.width, args.height, args.frames))
    cost_each = per_scene + (0 if args.skip_qwen else args.qwen_min_per_image)

    if args.fill_hours is not None:
        target_min = args.fill_hours * 60
        n_subj = len(args.subjects)
        sample = build_prompts_for_subject(
            args.subjects[0], args.scenes_per_tone, args.tones,
            args.audio_kinds, args.moderation_categories)
        scenes_per_subject = len(sample)
        max_total = scenes_per_subject * n_subj
        budget_scenes = int(target_min // cost_each) if cost_each else 0
        scenes_to_do = min(budget_scenes, max_total)
        per_subject = scenes_to_do // n_subj
        extras = scenes_to_do - per_subject * n_subj
        hours_each = (per_subject * cost_each) / 60
        args.hours = [hours_each] if per_subject > 0 else [0.001]
        print(f"\nFILL-HOURS: target {args.fill_hours}h ({target_min:.0f}min), "
              f"fits {scenes_to_do}/{max_total} scenes "
              f"(~{per_subject} per subject + {extras} extras)")

    cells = [(s, h) for s in args.subjects for h in args.hours]

    print("=" * 70)
    print(f"Matrix Runner: {len(cells)} cells = {len(args.subjects)} subjects × "
          f"{len(args.hours)} budgets")
    print(f"Per-scene LTX: {per_scene:.1f} min  ({args.width}x{args.height}/{args.frames}f)")
    print("=" * 70)
    report_permutation_space(args.subjects, per_scene, args.qwen_min_per_image)

    total_min = 0
    plan = []
    for subj, hours in cells:
        prompts = build_prompts_for_subject(
            subj, args.scenes_per_tone, args.tones, args.audio_kinds,
            args.moderation_categories)
        max_scenes = int((hours * 60) // cost_each) if cost_each else 0
        n = min(len(prompts), max_scenes)
        cell_min = n * cost_each
        plan.append((subj, hours, n, len(prompts), cell_min))
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

    if total_min > 120:
        print(f"\nTotal estimated wall time: {total_min/60:.1f}h. "
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
        prompts = build_prompts_for_subject(
            subj, args.scenes_per_tone, args.tones, args.audio_kinds,
            args.moderation_categories)
        prompts_file = Path(args.prompts_out) / f"prompts_{subj.replace(' ', '_')}.json"
        prompts_file.write_text(json.dumps(prompts, indent=2))

        print(f"\n=== CELL {subj} @ {hours}h ({n} scenes) ===")
        cmd = [
            sys.executable, str(HERE / "run_tone_wave.py"),
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
        mark = "OK" if rc == 0 else "FAIL"
        print(f"  {mark} {subj} @ {hours}h (rc={rc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
