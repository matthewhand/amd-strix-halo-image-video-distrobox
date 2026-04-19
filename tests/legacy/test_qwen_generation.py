#!/usr/bin/env python3
"""
Qwen Image Generation smoke test.

Applies ROCm compatibility patches, runs a 4-step image generation,
and verifies the output file is a valid image.

Must be run inside the distrobox / container with GPU access.
"""
import glob
import os
import sys
import time

# Paths inside the container
sys.path.insert(0, "/opt/qwen-image-studio/src")
sys.path.insert(0, "/opt")


def check_environment():
    """Verify GPU is accessible."""
    import torch

    print(f"PyTorch {torch.__version__}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA/HIP not available")
        return False

    print(f"GPU: {torch.cuda.get_device_name()}")
    x = torch.randn(100, 100, device="cuda", dtype=torch.float16)
    torch.mm(x, x.T)
    print("GPU compute OK")
    return True


def run_generation():
    """Run a 4-step Qwen image generation and validate output."""
    from qwen_image_mps.cli import generate_image

    class Args:
        prompt = "a majestic dragon perched on a crystal mountain, fantasy art, detailed, cinematic lighting"
        steps = 4
        num_images = 1
        size = "16:9"
        ultra_fast = True
        model = "Qwen/Qwen-Image"
        no_mmap = True
        lora = None
        edit = False
        input_image = None
        output_dir = "/tmp"
        seed = 42
        guidance_scale = 1.0
        negative_prompt = "blurry, low quality, distorted, watermark"
        batman = False
        fast = False
        targets = "all"

    print(f"Prompt: {Args.prompt}")
    start = time.time()
    generate_image(Args())
    elapsed = time.time() - start
    print(f"Generation: {elapsed:.1f}s")

    output_dir = os.path.expanduser("~/.qwen-image-studio")
    files = glob.glob(os.path.join(output_dir, "*.png"))
    if not files:
        print(f"FAIL: no output in {output_dir}")
        return False

    latest = max(files, key=os.path.getmtime)
    from PIL import Image

    img = Image.open(latest)
    print(f"OK: {latest} ({img.size[0]}x{img.size[1]}, {os.path.getsize(latest):,} bytes)")
    return True


def main():
    if not check_environment():
        return False

    print("\nApplying patches...")
    from apply_qwen_patches import apply_comprehensive_patches

    if not apply_comprehensive_patches():
        print("FAIL: patches did not apply")
        return False

    print("\nRunning generation...")
    return run_generation()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
