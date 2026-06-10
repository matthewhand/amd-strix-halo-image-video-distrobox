#!/usr/bin/env python3
"""
Submit LTX-2 video generation jobs to ComfyUI with varying resolutions and lengths.
Requires ComfyUI running on localhost:8188 with LTX-2 models loaded.
"""
import json
import random
import sys
import time
import urllib.request

SERVER = "127.0.0.1:8188"

VARIATIONS = [
    # (label, width, height, frames, prompt)
    # ~4 seconds (97 frames @ 24fps)
    ("cyberpunk_768x512_97f", 768, 512, 97, "Close-up cinematic shot of a cyberpunk city with neon lights reflecting on wet pavement, camera slowly panning, snowing, 4k, highly detailed"),
    ("dragon_768x512_97f", 768, 512, 97, "A dragon breathing fire while flying over a medieval castle, camera tracking the dragon, epic fantasy, cinematic lighting"),
    ("ocean_848x480_97f", 848, 480, 97, "Calm ocean waves at golden hour, camera slowly dollying forward over the water, sun reflecting, cinematic, 4k"),
    # ~6 seconds (145 frames @ 24fps)
    ("samurai_768x512_145f", 768, 512, 145, "A futuristic samurai walking through a neon-lit rainstorm, camera following from behind, detailed armor, 8k, cinematic"),
    ("jellyfish_768x512_145f", 768, 512, 145, "Bioluminescent jellyfish floating and pulsing in deep ocean, camera slowly descending, glowing tentacles, dark water, 8k macro"),
    # ~8 seconds (193 frames @ 24fps)
    ("forest_768x512_193f", 768, 512, 193, "Aerial flythrough of an ancient misty forest with sunbeams breaking through canopy, birds flying past camera, cinematic"),
    # ~10 seconds (241 frames @ 24fps)
    ("rocket_768x512_241f", 768, 512, 241, "Rocket launch sequence from ignition to liftoff, massive plume of smoke and fire, camera shaking, slow motion, cinematic"),
]


def create_workflow(width, height, frames, prompt, seed=None):
    if seed is None:
        seed = random.randint(1, 10**9)
    return {
        "1": {"class_type": "LowVRAMCheckpointLoader", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        "2": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {
            "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
            "ltxv_path": "ltx-2-19b-dev-fp8.safetensors", "max_length": 1024}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, worst quality, deformed, distorted, watermark", "clip": ["2", 0]}},
        "5": {"class_type": "STGGuiderAdvanced", "inputs": {
            "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
            "skip_steps_sigma_threshold": 0.998, "cfg_star_rescale": True,
            "sigmas": "1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180",
            "cfg_values": "8, 6, 6, 4, 3, 1", "stg_scale_values": "4, 4, 3, 2, 1, 0",
            "stg_rescale_values": "1, 1, 1, 1, 1, 1",
            "stg_layers_indices": "[29], [29], [29], [29], [29], [29]"}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "7": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["8", 0], "guider": ["5", 0], "sampler": ["6", 0], "sigmas": ["7", 0], "latent_image": ["9", 0]}},
        "11": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
            "vae": ["1", 2], "latents": ["10", 0], "spatial_tiles": 4, "spatial_overlap": 1,
            "temporal_tile_length": 16, "temporal_overlap": 2, "last_frame_fix": False,
            "working_device": "auto", "working_dtype": "auto"}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "fps": 24.0}},
        "13": {"class_type": "SaveVideo", "inputs": {"video": ["12", 0], "filename_prefix": "ltx2_output", "format": "mp4", "codec": "h264"}},
    }


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    # Check ComfyUI is up
    try:
        urllib.request.urlopen(f"http://{SERVER}/system_stats")
    except Exception:
        print(f"ComfyUI not reachable at {SERVER}")
        sys.exit(1)

    print(f"Submitting {len(VARIATIONS)} LTX-2 jobs to ComfyUI")
    print("=" * 60)

    for label, w, h, frames, prompt in VARIATIONS:
        wf = create_workflow(w, h, frames, prompt)
        wf["12"]["inputs"]["filename_prefix"] = f"ltx2_{label}"
        try:
            resp = queue_prompt(wf)
            print(f"  Queued: {label} ({w}x{h}, {frames}f) -> {resp.get('prompt_id', '?')}")
        except Exception as e:
            print(f"  FAIL: {label} -> {e}")
        time.sleep(1)

    print("\nAll jobs queued. Monitor at http://localhost:8188")
    print("Output will appear in /tmp/comfy-outputs/")


if __name__ == "__main__":
    main()
