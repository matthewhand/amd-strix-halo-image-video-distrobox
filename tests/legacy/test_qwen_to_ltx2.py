#!/usr/bin/env python3
"""
Chain Qwen image generation → LTX-2 video generation.

1. Stop ComfyUI (free GPU memory)
2. Generate still images with Qwen (docker run --rm, exits after each)
3. Start ComfyUI (loads LTX-2)
4. Submit image-to-video workflows

Strix Halo has 128GB unified memory — Qwen (54GB) and LTX-2 (~70GB)
cannot coexist. This script ensures only one runs at a time.
"""
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

SERVER = "127.0.0.1:8188"
COMFYUI_CONTAINER = "comfyui-test"
IMAGE = "amd-strix-halo-image-video-toolbox:latest"

DOCKER_ENV = [
    "-e", "HSA_OVERRIDE_GFX_VERSION=11.5.1",
    "-e", "LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib",
    "-e", "LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib",
]
DOCKER_GPU = [
    "--device", "/dev/dri", "--device", "/dev/kfd",
    "--security-opt", "seccomp=unconfined",
]


def stop_comfyui():
    """Stop ComfyUI container to free GPU memory for Qwen."""
    subprocess.run(["docker", "kill", COMFYUI_CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", COMFYUI_CONTAINER], capture_output=True)
    print("ComfyUI stopped (GPU memory freed)")
    time.sleep(2)


def start_comfyui():
    """Start ComfyUI container for LTX-2 video generation."""
    subprocess.run(["docker", "rm", "-f", COMFYUI_CONTAINER], capture_output=True)
    cmd = [
        "docker", "run", "-d", "--name", COMFYUI_CONTAINER,
        *DOCKER_GPU, *DOCKER_ENV,
        "-p", "8188:8188",
        "-v", os.path.expanduser("~/comfy-models") + ":/opt/ComfyUI/models",
        "-v", "/tmp/comfy-outputs:/opt/ComfyUI/output",
        IMAGE,
        "bash", "-c",
        "cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output --disable-mmap",
    ]
    subprocess.run(cmd, check=True)
    print("ComfyUI starting...")

    # Wait for it to be ready
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://{SERVER}/system_stats", timeout=2)
            print("ComfyUI ready")
            return True
        except Exception:
            time.sleep(3)
    print("ComfyUI failed to start")
    return False

# Each entry: (label, qwen_prompt, video_prompt, frames)
SCENES = [
    (
        "octopus_accountant",
        "hyperrealistic photograph of an octopus wearing tiny reading glasses sitting at a desk covered in spreadsheets and tax forms, each tentacle holding a different pen or calculator, office cubicle background, fluorescent lighting, mundane corporate setting",
        "the octopus slowly looks up from the spreadsheets directly at the camera with an expression of existential dread, papers rustling, pens moving, calculator buttons clicking, a phone rings in the background",
        97,
    ),
    (
        "medieval_astronaut",
        "oil painting in renaissance style of an astronaut in a full NASA spacesuit sitting for a formal portrait in a medieval castle throne room, golden frame, dramatic chiaroscuro lighting, servants in period costume attending",
        "the astronaut slowly raises the visor revealing a confused medieval knight face underneath, servants gasping, candles flickering, dramatic orchestral music swelling",
        97,
    ),
    (
        "cats_boardroom",
        "corporate photograph of a serious business meeting in a luxury boardroom, but all the executives are cats in tiny suits and ties, one cat standing at a whiteboard with a laser pointer, charts showing fish stock prices, mahogany table",
        "the CEO cat slams its paw on the table and meows loudly, the other cats turn in shock, papers fly everywhere, the stock chart on the screen crashes, dramatic boardroom tension",
        97,
    ),
    (
        "dinosaur_barista",
        "detailed illustration of a friendly T-Rex working as a barista in a modern hipster coffee shop, tiny apron on its massive body, struggling to hold a tiny espresso cup with its small arms, customers waiting patiently in line, chalkboard menu",
        "the T-Rex carefully pours latte art into the tiny cup with trembling small arms, milk splashing everywhere, the cup crumbles in its claws, customers clapping encouragingly, coffee machine steaming",
        145,
    ),
    (
        "underwater_library",
        "photorealistic wide shot of a grand classical library that is completely submerged underwater, fish swimming between the bookshelves, an old librarian octopus organizing books, coral growing on marble columns, shafts of sunlight from above, books floating open with pages drifting",
        "camera slowly glides through the underwater library, pages turning by themselves in the current, small fish dart between shelves, bubbles rising from an open book, whale song echoing through the halls, peaceful and surreal",
        145,
    ),
    (
        "angry_ai_user",
        "photorealistic close up of a frustrated middle aged man in a messy home office, red faced and furious, gripping a keyboard with white knuckles, multiple monitors showing AI chatbot responses, energy drink cans everywhere, dramatic lighting from screen glow",
        "the man slams the keyboard on the desk and yells profanity at the screen, veins bulging on his forehead, a monitor flickers, he grabs his coffee mug and throws it, angry shouting and crashing sounds, keyboard keys flying",
        97,
    ),
]


def generate_qwen_image(label, prompt):
    """Generate image with Qwen via docker run --rm (exits after, frees memory)."""
    output_dir = "/tmp/comfy-outputs"
    output_path = os.path.join(output_dir, f"qwen_input_{label}.png")

    if os.path.exists(output_path):
        print(f"  Image exists: {output_path}")
        return output_path

    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
    seed = random.randint(1, 10000)
    escaped_prompt = prompt.replace("'", "\\'")

    cmd = [
        "docker", "run", "--rm",
        *DOCKER_GPU, *DOCKER_ENV,
        "-v", os.path.expanduser("~/.cache/huggingface") + ":/root/.cache/huggingface",
        "-v", f"{output_dir}:/output",
        "-v", f"{scripts_dir}/apply_qwen_patches.py:/opt/apply_qwen_patches.py:ro",
        IMAGE,
        "python3", "-c", f"""
import sys, shutil, glob, os
sys.path.insert(0, '/opt/qwen-image-studio/src')
sys.path.insert(0, '/opt')
from apply_qwen_patches import apply_comprehensive_patches
apply_comprehensive_patches()
from qwen_image_mps.cli import generate_image

class Args:
    prompt = '{escaped_prompt}'
    steps = 8
    num_images = 1
    size = '16:9'
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
    shutil.copy2(latest, '/output/qwen_input_{label}.png')
    print(f'Saved: qwen_input_{label}.png')
""",
    ]

    print(f"  Generating Qwen image (docker run --rm)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr[-200:]}")
        return None

    if os.path.exists(output_path):
        print(f"  OK: {output_path}")
        return output_path
    print("  FAIL: image not generated")
    return None


def submit_ltx2_i2v(label, image_filename, video_prompt, frames):
    """Submit LTX-2 image-to-video+audio job to ComfyUI."""
    seed = random.randint(1, 10**9)
    workflow = {
        "1": {"class_type": "LowVRAMCheckpointLoader", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        # Audio VAE
        "2": {"class_type": "LowVRAMAudioVAELoader", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        # Text encoder
        "3": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {
            "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
            "ltxv_path": "ltx-2-19b-dev-fp8.safetensors", "max_length": 1024}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": video_prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "low quality, distorted, watermark, static, no motion, silent", "clip": ["3", 0]}},
        # Load the Qwen-generated image
        "6": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        # Image to video latent
        "7": {"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "vae": ["1", 2],
            "image": ["6", 0], "width": 768, "height": 512, "length": frames,
            "batch_size": 1, "strength": 0.85}},
        # Audio latent
        "8": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": frames, "frame_rate": 24, "batch_size": 1, "audio_vae": ["2", 0]}},
        # Concat video + audio latents
        "9": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["7", 2], "audio_latent": ["8", 0]}},
        # Guider (uses conditioning from i2v node)
        "10": {"class_type": "STGGuiderAdvanced", "inputs": {
            "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
            "skip_steps_sigma_threshold": 0.998, "cfg_star_rescale": True,
            "sigmas": "1.0, 0.9933, 0.985, 0.9767, 0.9008, 0.618",
            "cfg_values": "8, 6, 6, 4, 3, 1", "stg_scale_values": "4, 4, 3, 2, 1, 0",
            "stg_rescale_values": "1, 1, 1, 1, 1, 1",
            "stg_layers_indices": "[29], [29], [29], [29], [29], [29]"}},
        # Sampler
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "12": {"class_type": "BasicScheduler", "inputs": {
            "model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["13", 0], "guider": ["10", 0], "sampler": ["11", 0],
            "sigmas": ["12", 0], "latent_image": ["9", 0]}},
        # Separate AV latent
        "15": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["14", 0]}},
        # Video decode (tiled)
        "16": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
            "vae": ["1", 2], "latents": ["15", 0],
            "spatial_tiles": 4, "spatial_overlap": 1,
            "temporal_tile_length": 16, "temporal_overlap": 2,
            "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        # Audio decode
        "17": {"class_type": "LTXVAudioVAEDecode", "inputs": {
            "audio_vae": ["2", 0], "samples": ["15", 1]}},
        # Combine into video with audio
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": 24.0, "audio": ["17", 0]}},
        # Save as mp4
        "19": {"class_type": "SaveVideo", "inputs": {
            "video": ["18", 0], "filename_prefix": f"i2v_{label}", "format": "mp4", "codec": "h264"}},
    }

    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print(f"  Queued LTX-2 i2v: {result.get('prompt_id', '?')}")
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"  FAIL: {e.code}: {body[:300]}")
        return False


def main():
    print("Qwen → LTX-2 Image-to-Video Pipeline")
    print("=" * 60)

    # Step 1: Stop ComfyUI to free GPU memory
    print("\n--- Phase 1: Generate images with Qwen ---")
    stop_comfyui()

    # Step 2: Generate all images with Qwen (sequential, each container exits after)
    images = {}
    for label, qwen_prompt, _, _ in SCENES:
        print(f"\n[{label}] Generating source image...")
        path = generate_qwen_image(label, qwen_prompt)
        if path:
            images[label] = os.path.basename(path)

    if not images:
        print("No images generated. Exiting.")
        return

    # Step 3: Start ComfyUI (loads LTX-2 into GPU)
    print(f"\n--- Phase 2: Animate with LTX-2 ({len(images)} videos) ---")
    if not start_comfyui():
        print("ComfyUI failed to start. Exiting.")
        return

    # Step 4: Copy images into ComfyUI input directory
    for label, filename in images.items():
        src = f"/tmp/comfy-outputs/{filename}"
        os.system(f"docker cp {src} {COMFYUI_CONTAINER}:/opt/ComfyUI/input/{filename}")

    # Step 5: Submit all i2v jobs
    for label, qwen_prompt, video_prompt, frames in SCENES:
        if label in images:
            print(f"\n[{label}]")
            submit_ltx2_i2v(label, images[label], video_prompt, frames)

    print(f"\n{'='*60}")
    print("All jobs submitted. Videos will appear in /tmp/comfy-outputs/")
    print(f"Monitor at http://localhost:8188")


if __name__ == "__main__":
    main()
