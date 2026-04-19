#!/usr/bin/env python3
"""
Parametric LTX-2.3 T2V + 2x spatial upscale.

Generates a video at 1280x720/49f (base) then upscales latents via
LTXVLatentUpsampler + tiled VAE decode to 2560x1440/49f mp4 (no audio mux).

49f is the PROVEN-safe frame count for upscale on gfx1151; 193f GPU-hangs.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import comfyui_api  # noqa: E402
from pipelines import config  # noqa: E402

MODEL = "ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA = "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

W, H, F, FPS = 1280, 720, 49, 24


def build_workflow(prompt, output_prefix, seed=None):
    if seed is None:
        seed = random.randint(1, 10**9)
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODEL}},
        "2": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {
            "gemma_path": GEMMA, "ltxv_path": MODEL, "max_length": 1024}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "low quality, distorted, watermark, blurry, static, frozen",
            "clip": ["2", 0]}},
        "5": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "frame_rate": float(FPS)}},
        "6": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": W, "height": H, "length": F, "batch_size": 1}},
        "10": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": MODEL}},
        "11": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "audio_vae": ["10", 0], "frames_number": F,
            "frame_rate": FPS, "batch_size": 1}},
        "12": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["6", 0], "audio_latent": ["11", 0]}},
        "13": {"class_type": "GuiderParameters", "inputs": {
            "modality": "VIDEO", "cfg": 1.0, "stg": 1.0, "perturb_attn": True,
            "rescale": 0.7, "modality_scale": 1.0, "skip_step": 0,
            "cross_attn": True}},
        "14": {"class_type": "GuiderParameters", "inputs": {
            "modality": "AUDIO", "cfg": 1.0, "stg": 1.0, "perturb_attn": True,
            "rescale": 0.7, "modality_scale": 1.0, "skip_step": 0,
            "cross_attn": True, "parameters": ["13", 0]}},
        "15": {"class_type": "MultimodalGuider", "inputs": {
            "model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1],
            "parameters": ["14", 0], "skip_blocks": ""}},
        "16": {"class_type": "ManualSigmas", "inputs": {
            "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, "
                      "0.725, 0.421875, 0.0"}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "19": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["18", 0], "guider": ["15", 0], "sampler": ["17", 0],
            "sigmas": ["16", 0], "latent_image": ["12", 0]}},
        "20": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["19", 0]}},
        "30": {"class_type": "LatentUpscaleModelLoader", "inputs": {
            "model_name": UPSCALER}},
        "31": {"class_type": "LTXVLatentUpsampler", "inputs": {
            "samples": ["20", 0], "upscale_model": ["30", 0], "vae": ["1", 2]}},
        "40": {"class_type": "LTXVTiledVAEDecode", "inputs": {
            "vae": ["1", 2], "latents": ["31", 0],
            "horizontal_tiles": 6, "vertical_tiles": 6, "overlap": 2,
            "last_frame_fix": False,
            "working_device": "auto", "working_dtype": "auto"}},
        "41": {"class_type": "CreateVideo", "inputs": {
            "images": ["40", 0], "fps": float(FPS)}},
        "42": {"class_type": "SaveVideo", "inputs": {
            "video": ["41", 0], "filename_prefix": output_prefix,
            "format": "mp4", "codec": "h264"}},
    }


def submit_and_wait(prompt, output_prefix, server=None, poll=15, timeout=1800):
    server = server or config.SERVER
    wf = build_workflow(prompt, output_prefix)
    try:
        resp = comfyui_api.submit(wf, server, str(uuid.uuid4()))
    except RuntimeError as e:
        return ("error", f"submit: {e}")
    pid = resp.get("prompt_id")
    print(f"  queued {output_prefix}: {pid}")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                f"http://{server}/history/{pid}", timeout=5) as r:
                h = json.loads(r.read().decode())
                if h:
                    st = list(h.values())[0].get("status", {})
                    s = st.get("status_str")
                    if s == "success":
                        elapsed = int(time.time() - start)
                        print(f"  DONE in {elapsed}s")
                        return ("success", None)
                    if s == "error":
                        msgs = st.get("messages", [])
                        for m in msgs:
                            if m[0] == "execution_error":
                                err = (f"{m[1].get('exception_type')} - "
                                       f"{m[1].get('exception_message')}")
                                print(f"  ERROR: {err}")
                                return ("error", err)
                        return ("error", "unknown")
        except Exception:
            pass
        time.sleep(poll)
    return ("error", "timeout")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--prefix", default="upscale_t2v")
    args = p.parse_args()
    status, err = submit_and_wait(args.prompt, args.prefix)
    sys.exit(0 if status == "success" else 1)


if __name__ == "__main__":
    main()
