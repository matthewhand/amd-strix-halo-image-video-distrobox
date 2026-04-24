#!/usr/bin/env python3
"""
Test Qwen image generation with varying parameters.

Runs a matrix of prompts x settings to compare output quality and timing.
Must be run inside the container with GPU access.
"""
import os
import sys
import time

sys.path.insert(0, "/opt/qwen-image-studio/src")
sys.path.insert(0, "/opt")

VARIATIONS = [
    # (label, prompt, steps, size, guidance_scale, seed)
    ("waldo_city_4step", "wheres waldo in a bustling tokyo street market, detailed illustration, search and find style, crowds of people", 4, "16:9", 1.0, 42),
    ("waldo_city_8step", "wheres waldo in a bustling tokyo street market, detailed illustration, search and find style, crowds of people", 8, "16:9", 1.0, 42),
    ("waldo_beach_4step", "wheres waldo on a crowded beach with umbrellas and sandcastles, detailed illustration, search and find style", 4, "16:9", 1.0, 77),
    ("waldo_space_4step", "wheres waldo on a space station with astronauts floating, detailed illustration, search and find style", 4, "16:9", 1.0, 99),
    ("waldo_square_4step", "wheres waldo at a medieval jousting tournament, detailed illustration, search and find style, crowds", 4, "1:1", 1.0, 55),
    ("waldo_highcfg_4step", "wheres waldo in a crowded ancient roman colosseum, detailed illustration, search and find style", 4, "16:9", 3.5, 42),
    ("waldo_portrait_4step", "wheres waldo climbing the eiffel tower, detailed illustration, search and find style", 4, "9:16", 1.0, 88),
]


def run_variation(label, prompt, steps, size, guidance_scale, seed):
    from qwen_image_mps.cli import generate_image

    class Args:
        pass

    args = Args()
    args.prompt = prompt
    args.steps = steps
    args.num_images = 1
    args.size = size
    args.ultra_fast = (steps <= 4)
    args.model = "Qwen/Qwen-Image"
    args.no_mmap = True
    args.lora = None
    args.edit = False
    args.input_image = None
    args.output_dir = "/tmp"
    args.seed = seed
    args.guidance_scale = guidance_scale
    args.negative_prompt = "blurry, low quality, distorted"
    args.batman = False
    args.fast = (steps <= 4)
    args.targets = "all"

    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Steps: {steps}, Size: {size}, CFG: {guidance_scale}, Seed: {seed}")

    start = time.time()
    try:
        generate_image(args)
        elapsed = time.time() - start
        print(f"  OK: {elapsed:.1f}s")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAIL ({elapsed:.1f}s): {e}")
        return False, elapsed


def main():
    print("Qwen Image Generation - Parameter Variations")
    print("=" * 60)

    # Apply patches once
    from apply_qwen_patches import apply_comprehensive_patches
    if not apply_comprehensive_patches():
        print("FAIL: patches did not apply")
        return False

    results = []
    for v in VARIATIONS:
        ok, elapsed = run_variation(*v)
        results.append((v[0], ok, elapsed))

    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'Label':<30} {'Status':<8} {'Time':>8}")
    print("-" * 50)
    for label, ok, elapsed in results:
        print(f"{label:<30} {'OK' if ok else 'FAIL':<8} {elapsed:>7.1f}s")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    return passed == len(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
