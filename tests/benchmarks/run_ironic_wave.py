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
    python tests/run_ironic_wave.py                        # full wave
    python tests/run_ironic_wave.py --scenes 1 2 3         # 1-indexed subset
    python tests/run_ironic_wave.py --skip-qwen            # use cached images
    python tests/run_ironic_wave.py --dry-run              # print plan only
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipelines import Scene, run_wave  # noqa: E402

# (label, qwen, video, width, height, frames)
_RAW = [
    ("heavy_metal_librarian",
     "photorealistic interior of a hushed grand library with tall oak shelves, an elderly librarian in a leather biker vest covered in band patches stands behind the wooden circulation desk, long grey hair, reading glasses, calm expression, patrons silently reading at study tables in the background, warm afternoon light through stained glass windows",
     "the librarian explodes into furious silent headbanging behind the desk while the patrons read on undisturbed, hair whipping around, fist pumping, audio is full speed thrash metal with double-kick drums and shredded electric guitar at maximum volume",
     480, 320, 49),
    ("mime_opera_singer",
     "photograph of a classic white-faced street mime in black and white striped shirt, beret, white gloves, performing on a Parisian sidewalk in front of a small chalk-marked semicircle, cobblestones, an iron lamppost, a few onlookers watching politely, soft overcast daylight",
     "the mime stays completely silent with sealed lips and exaggerated facial expressions while a thunderous Italian operatic tenor aria booms out of nowhere, full orchestra and chorus, audio is Pavarotti-scale Nessun Dorma at concert hall volume, the mime simply tilts head and bows",
     480, 320, 97),
    ("astronaut_city_bus",
     "documentary photograph of an astronaut in a full white NASA spacesuit with helmet on, sitting on a bench seat of a crowded city bus during morning commute, holding a metal coffee thermos, other passengers reading phones and newspapers oblivious, fluorescent overhead lighting, urban morning light through bus windows",
     "the astronaut sips coffee through the closed helmet visor in slow motion as the bus rattles down the street, audio is full NASA mission control radio chatter with countdown commands, beeping telemetry, and Houston we are go for liftoff over crackling comms",
     640, 384, 97),
    ("trex_piano_recital",
     "elegant photograph of a full grown Tyrannosaurus Rex in a black tuxedo and white bow tie seated at a polished black grand piano on a concert hall stage, tiny arms barely reaching the keys, large head leaning down in concentration, audience in formal attire visible in the front rows, dramatic spotlight",
     "the T-Rex carefully picks at the keys with its tiny arms in slow concentrated motion, head sways with the music, audio is full Chopin Nocturne in E-flat major performed at concert quality with rich piano resonance, occasional polite applause swells from the audience",
     768, 432, 97),
    ("cat_evening_news",
     "photograph of a professional television news studio, two well-groomed cats wearing tiny suits and ties seated at a sleek anchor desk with the city skyline backdrop behind them, broadcast graphics, teleprompter visible, studio lighting, papers stacked neatly, the channel logo on the desk",
     "the lead anchor cat looks up at camera with serious gravitas while the co-anchor cat nods solemnly, audio is the full evening news theme music swelling then a deep authoritative male newscaster voice introducing tonight's top stories interrupted by occasional dignified meows",
     768, 512, 121),
    ("knight_orders_latte",
     "photograph of a medieval knight in full polished plate armor including helmet with visor up, standing patiently at the counter of a modern hipster coffee shop, chalkboard menu visible, espresso machine and pastry case, barista in apron taking the order, customers at small tables on laptops, warm pendant lighting",
     "the knight slowly counts coins onto the counter from a leather pouch in metal-clad fingers, audio is full coffee shop ambience: hissing espresso machine, milk steaming, ceramic cups clinking, indie acoustic music, then the barista calls out clearly: oat milk latte for Sir Reginald",
     768, 512, 145),
    ("underwater_chef",
     "photograph of a French chef in a tall white toque and pristine chef's coat working at a fully submerged underwater stainless steel kitchen line, fish swimming between hanging copper pots, coral growing on the prep surfaces, vegetables floating, soft sunbeams from above through the water",
     "the chef performs precise knife work in slow motion as bubbles rise from each cut and ingredients drift weightlessly, audio is full restaurant kitchen chaos: sizzling pans, oil popping, ticket printer, expediter shouting orders, dishes clattering, completely incongruous with the silent underwater scene",
     896, 512, 145),
    ("toddler_ceo_emergency_meeting",
     "photograph of a corporate boardroom with a long mahogany table, executives in suits seated around it looking concerned, at the head of the table a serious-faced toddler in a tiny pinstripe business suit and tie sits in an oversized leather chair, financial charts on the wall display, crayons and a juice box in front of the toddler",
     "the toddler CEO slams a small fist on the table while drawing furiously with crayons across a quarterly report, executives lean in attentively taking notes, audio is full tense corporate orchestral strings building dramatically, gavel bangs, urgent telephones ringing in the background",
     1024, 576, 145),
    ("viking_startup_pitch",
     "photograph of a fierce Viking warrior in full historical regalia, fur cloak, horned helmet, large round wooden shield slung on back, long beard, standing in front of a sleek modern Silicon Valley conference room projector screen displaying a startup pitch deck slide titled Series A, audience of casually-dressed tech investors in chairs",
     "the Viking gestures emphatically at the projector screen with a battle axe, slowly turning to point at key bullet points, audio is upbeat startup pitch narration in TED-talk cadence: our blockchain-enabled longship logistics platform disrupts traditional pillage and tribute markets at a 10x runtime, applause swells",
     1024, 576, 193),
    ("statue_ballet_performance",
     "photograph of three classical white marble Greek sculptures of muscular male figures arranged on a grand ballet stage, soft theatrical lighting, red velvet curtain backdrop, polished wooden stage floor, ornate proscenium visible at the edges, an empty orchestra pit in the foreground",
     "the marble statues glide impossibly into precise synchronized ballet poses in slow motion, holding arabesques and pirouettes, audio is full Tchaikovsky Swan Lake orchestral score at concert quality with sweeping strings and timpani, punctuated by faint stone-on-stone chiseling sounds at every movement",
     1280, 720, 145),
]

SCENES = [
    Scene(label=l, qwen_prompt=q, video_prompt=v, width=w, height=h, frames=f)
    for l, q, v, w, h, f in _RAW
]


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
    for i, s in enumerate(scenes, 1):
        print(f"  {i:2d}. {s.label:30s} {s.width}x{s.height}  {s.frames}f")

    if args.dry_run:
        return 0

    paths = run_wave(
        scenes,
        skip_qwen=args.skip_qwen,
        output_prefix_fn=lambda s: f"wave_{s.label}",
    )
    print("\n" + "=" * 70)
    print(f"Submitted {len(paths)}/{len(scenes)} jobs to ComfyUI queue.")
    return 0 if paths else 1


if __name__ == "__main__":
    sys.exit(main())
