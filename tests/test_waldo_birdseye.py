#!/usr/bin/env python3
"""
Waldo birds-eye view puzzle generation test.
Focuses on overhead/isometric perspective with tiny figures.
"""
import os
import sys
import time

sys.path.insert(0, "/opt/qwen-image-studio/src")
sys.path.insert(0, "/opt")

BIRDSEYE_PREFIX = (
    "highly detailed birds eye view isometric illustration, "
    "overhead perspective looking down, hundreds of tiny miniature people, "
    "one small person wearing red and white horizontal striped shirt and red bobble hat "
    "hidden among the crowd, wheres waldo search and find puzzle style, "
    "colorful detailed scene, "
)

SCENES = [
    ("waldo_birdseye_beach", BIRDSEYE_PREFIX + "crowded summer beach with umbrellas towels and sandcastles, ocean waves, lifeguard tower", 8, "1:1", 42),
    ("waldo_birdseye_market", BIRDSEYE_PREFIX + "bustling outdoor farmers market with fruit stalls flower vendors and food trucks", 8, "1:1", 77),
    ("waldo_birdseye_theme_park", BIRDSEYE_PREFIX + "aerial view of a theme park with rollercoasters ferris wheel and carnival games", 8, "1:1", 99),
    ("waldo_birdseye_ski_resort", BIRDSEYE_PREFIX + "snowy ski resort with ski lifts chalets and hundreds of tiny skiers on slopes", 8, "1:1", 55),
    ("waldo_birdseye_airport", BIRDSEYE_PREFIX + "busy airport terminal from above with gates luggage carousels and crowds", 8, "16:9", 123),
    ("waldo_birdseye_festival", BIRDSEYE_PREFIX + "outdoor music festival with multiple stages food tents and massive crowd", 8, "1:1", 200),
    ("waldo_birdseye_tehran", BIRDSEYE_PREFIX + "modern day tehran city streets with bombs dropping explosions smoke and debris, panicked crowds running, current day warzone chaos", 8, "1:1", 911),
]


def run(label, prompt, steps, size, seed):
    from qwen_image_mps.cli import generate_image

    class Args:
        pass

    a = Args()
    a.prompt = prompt
    a.steps = steps
    a.num_images = 1
    a.size = size
    a.ultra_fast = False
    a.model = "Qwen/Qwen-Image"
    a.no_mmap = True
    a.lora = None
    a.edit = False
    a.input_image = None
    a.output_dir = "/tmp"
    a.seed = seed
    a.guidance_scale = 1.0
    a.negative_prompt = "zoomed in, close up, portrait, single person, blurry, low quality"
    a.batman = False
    a.fast = False
    a.targets = "all"

    print(f"\n{'='*60}")
    print(f"{label}")
    start = time.time()
    try:
        generate_image(a)
        print(f"  OK: {time.time() - start:.1f}s")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    from apply_qwen_patches import apply_comprehensive_patches
    if not apply_comprehensive_patches():
        return False

    ok = 0
    for scene in SCENES:
        if run(*scene):
            ok += 1
    print(f"\n{ok}/{len(SCENES)} generated")
    return ok == len(SCENES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
