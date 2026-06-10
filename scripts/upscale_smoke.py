#!/usr/bin/env python3
"""
Smoke: LTX-2.3 distilled base + spatial upscaler x2 (1280x720 -> 2560x1440).

Stage 1: distilled-fp8 22B generates a base video latent at 1280x720/49f.
Stage 2: LTXVLatentUpsampler doubles the latent to 2560x1440.
Stage 3: tiled VAE decode at 2x tile count.
Stage 4: SaveVideo (no audio mux — verifying the upscaler works at all).

Usage: python scripts/upscale_smoke.py
"""
import json
import os
import random
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import comfyui_api  # noqa: E402

SERVER = "127.0.0.1:8188"
MODEL = "ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA = "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

W, H, F, FPS = 1280, 720, 193, 24  # full-length stress test for option A viability


def main():
    seed = random.randint(1, 10**9)
    wf = {
        # --- shared loaders ---
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODEL}},
        "2": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {
            "gemma_path": GEMMA, "ltxv_path": MODEL, "max_length": 1024,
        }},
        # text encode
        "3": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "a wide cinematic shot of a glistening pink brain hovering serenely "
                    "in a star-filled cosmic void, gentle golden pulse, soft nebulae, "
                    "no dialogue", "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "low quality, distorted, watermark, blurry", "clip": ["2", 0]}},
        "5": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "frame_rate": float(FPS)}},
        # --- Stage 1: base generation at 1280x720 ---
        "6": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": W, "height": H, "length": F, "batch_size": 1}},
        "10": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": MODEL}},
        "11": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "audio_vae": ["10", 0], "frames_number": F, "frame_rate": FPS, "batch_size": 1}},
        "12": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["6", 0], "audio_latent": ["11", 0]}},
        "13": {"class_type": "GuiderParameters", "inputs": {
            "modality": "VIDEO", "cfg": 1.0, "stg": 1.0, "perturb_attn": True,
            "rescale": 0.7, "modality_scale": 1.0, "skip_step": 0, "cross_attn": True}},
        "14": {"class_type": "GuiderParameters", "inputs": {
            "modality": "AUDIO", "cfg": 1.0, "stg": 1.0, "perturb_attn": True,
            "rescale": 0.7, "modality_scale": 1.0, "skip_step": 0, "cross_attn": True,
            "parameters": ["13", 0]}},
        "15": {"class_type": "MultimodalGuider", "inputs": {
            "model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1],
            "parameters": ["14", 0], "skip_blocks": ""}},
        "16": {"class_type": "ManualSigmas", "inputs": {
            "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "19": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["18", 0], "guider": ["15", 0], "sampler": ["17", 0],
            "sigmas": ["16", 0], "latent_image": ["12", 0]}},
        # --- Split AV after stage 1 ---
        "20": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["19", 0]}},
        # --- Stage 2: upscale video latent 2x ---
        "30": {"class_type": "LatentUpscaleModelLoader", "inputs": {
            "model_name": UPSCALER}},
        "31": {"class_type": "LTXVLatentUpsampler", "inputs": {
            "samples": ["20", 0], "upscale_model": ["30", 0], "vae": ["1", 2]}},
        # --- Decode upscaled video latent. 2x res = 4x pixels per tile. Need
        # MANY more tiles to keep peak working memory in bounds. 2x2 OOM'd
        # the host even with 96 GB usable; bumping to 6x6 for ~9x smaller tiles.
        "40": {"class_type": "LTXVTiledVAEDecode", "inputs": {
            "vae": ["1", 2], "latents": ["31", 0],
            "horizontal_tiles": 6, "vertical_tiles": 6, "overlap": 2,
            "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        # --- Save (no audio for this smoke) ---
        "41": {"class_type": "CreateVideo", "inputs": {
            "images": ["40", 0], "fps": float(FPS)}},
        "42": {"class_type": "SaveVideo", "inputs": {
            "video": ["41", 0], "filename_prefix": "upscale_193f",
            "format": "mp4", "codec": "h264"}},
    }
    resp = comfyui_api.submit(wf, SERVER, str(uuid.uuid4()))
    print(f"queued upscale smoke: {resp.get('prompt_id')}")
    print(f"expected output: 2560x1440 / {F}f / {FPS}fps mp4 (no audio)")


if __name__ == "__main__":
    main()
