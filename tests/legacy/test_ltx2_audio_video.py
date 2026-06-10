#!/usr/bin/env python3
"""
Submit LTX-2 audio+video generation jobs to ComfyUI.
Generates synchronized audio and video from text prompts.
"""
import json
import random
import sys
import time
import urllib.request

SERVER = "127.0.0.1:8188"

VARIATIONS = [
    # (label, frames, prompt)
    # Visual: intense action / Voice: calmly narrating the wrong thing
    ("warzone_cooking_97f", 97, "Massive tank battle in a destroyed city, explosions and buildings crumbling, a calm british voice narrating a cooking recipe saying and now we fold in the butter gently"),
    ("tornado_meditation_97f", 97, "Category 5 tornado ripping through a town, cars tumbling through the air, a soothing female voice saying breathe in deeply and find your center, you are safe"),
    ("shark_realtor_97f", 97, "Great white shark attacking a fishing boat in a storm, massive waves, a cheerful american realtor voice saying this property features an open floor plan and ocean views"),
    # Visual: calm mundane / Voice: wildly inappropriate commentary
    ("spreadsheet_drill_145f", 145, "Close up of an office worker quietly typing on a spreadsheet in a grey cubicle, a drill sergeant voice screaming move it move it you call that a pivot table soldier"),
    ("cat_nap_sports_145f", 145, "A fluffy cat peacefully sleeping curled up on a windowsill in warm sunlight, an excited sports commentator voice screaming and he goes for the goal incredible"),
    ("grandma_knitting_rap_145f", 145, "An elderly grandmother peacefully knitting in a rocking chair by a fireplace, a deep voiced rapper performing aggressive rap lyrics about the streets"),
]


def create_av_workflow(frames, prompt, seed=None):
    if seed is None:
        seed = random.randint(1, 10**9)
    return {
        # Checkpoint loader
        "1": {"class_type": "LowVRAMCheckpointLoader", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        # Audio VAE loader
        "2": {"class_type": "LowVRAMAudioVAELoader", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        # Text encoder
        "3": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {
            "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
            "ltxv_path": "ltx-2-19b-dev-fp8.safetensors", "max_length": 1024}},
        # Positive prompt
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        # Negative prompt
        "5": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "low quality, distorted audio, static noise, muffled, watermark", "clip": ["3", 0]}},
        # Video latent
        "6": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": 768, "height": 512, "length": frames, "batch_size": 1}},
        # Audio latent
        "7": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": frames, "frame_rate": 24, "batch_size": 1, "audio_vae": ["2", 0]}},
        # Concat AV latent
        "8": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["6", 0], "audio_latent": ["7", 0]}},
        # Guider
        "9": {"class_type": "STGGuiderAdvanced", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "skip_steps_sigma_threshold": 0.998, "cfg_star_rescale": True,
            "sigmas": "1.0, 0.9933, 0.985, 0.9767, 0.9008, 0.618",
            "cfg_values": "8, 6, 6, 4, 3, 1", "stg_scale_values": "4, 4, 3, 2, 1, 0",
            "stg_rescale_values": "1, 1, 1, 1, 1, 1",
            "stg_layers_indices": "[29], [29], [29], [29], [29], [29]"}},
        # Sampler
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {"class_type": "BasicScheduler", "inputs": {
            "model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        # Sample
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["9", 0], "sampler": ["10", 0],
            "sigmas": ["11", 0], "latent_image": ["8", 0]}},
        # Separate AV latent
        "14": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["13", 0]}},
        # Video decode (tiled)
        "15": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
            "vae": ["1", 2], "latents": ["14", 0],
            "spatial_tiles": 4, "spatial_overlap": 1,
            "temporal_tile_length": 16, "temporal_overlap": 2,
            "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        # Audio decode
        "16": {"class_type": "LTXVAudioVAEDecode", "inputs": {
            "audio_vae": ["2", 0], "samples": ["14", 1]}},
        # Create video with audio
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": 24.0, "audio": ["16", 0]}},
        # Save as mp4
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "ltx2_av_output", "format": "mp4", "codec": "h264"}},
    }


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    try:
        urllib.request.urlopen(f"http://{SERVER}/system_stats")
    except Exception:
        print(f"ComfyUI not reachable at {SERVER}")
        sys.exit(1)

    print(f"Submitting {len(VARIATIONS)} LTX-2 Audio+Video jobs")
    print("=" * 60)

    for label, frames, prompt in VARIATIONS:
        wf = create_av_workflow(frames, prompt)
        wf["18"]["inputs"]["filename_prefix"] = f"ltx2_av_{label}"
        try:
            resp = queue_prompt(wf)
            print(f"  Queued: {label} ({frames}f) -> {resp.get('prompt_id', '?')}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            print(f"  FAIL: {label} -> {e.code}: {body[:200]}")
        time.sleep(1)

    print(f"\nOutput: /tmp/comfy-outputs/")


if __name__ == "__main__":
    main()
