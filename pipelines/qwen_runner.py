"""Single source of truth for Qwen image generation.

Replaces the four near-identical `generate_qwen_image` definitions across the
test wave scripts. The docker-run-rm heredoc lives here exactly once.
"""
import os
import random
import subprocess

from . import config

# Path to apply_qwen_patches.py — resolved at call time so tests can run
# regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PATCHES_PATH = os.path.join(_REPO_ROOT, "scripts", "apply_qwen_patches.py")


def generate_image(prompt, out_path, *, seed=None, steps=8, size="16:9"):
    """Generate one Qwen image to `out_path`. Idempotent — reuses on hit.

    Returns the output path on success, None on failure.
    """
    if os.path.exists(out_path):
        print(f"  Image exists: {out_path}")
        return out_path

    if seed is None:
        seed = random.randint(1, 10000)
    escaped_prompt = prompt.replace("'", "\\'")
    out_dir = os.path.dirname(out_path) or config.OUTPUT_DIR
    out_basename = os.path.basename(out_path)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        *config.DOCKER_GPU, *config.DOCKER_ENV,
        "-v", config.HF_CACHE + ":/root/.cache/huggingface",
        "-v", f"{out_dir}:/output",
        "-v", f"{_PATCHES_PATH}:/opt/apply_qwen_patches.py:ro",
        config.IMAGE,
        "python3", "-c", f"""
import sys, shutil, glob, os
sys.path.insert(0, '/opt/qwen-image-studio/src')
sys.path.insert(0, '/opt')
from apply_qwen_patches import apply_comprehensive_patches
apply_comprehensive_patches()
from qwen_image_mps.cli import generate_image

class Args:
    prompt = '{escaped_prompt}'
    steps = {steps}
    num_images = 1
    size = '{size}'
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
    shutil.copy2(latest, '/output/{out_basename}')
    print(f'Saved: {out_basename}')
""",
    ]
    print("  Generating Qwen image (docker run --rm)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr[-300:]}")
        return None
    if os.path.exists(out_path):
        print(f"  OK: {out_path}")
        return out_path
    print("  FAIL: image not produced")
    return None
