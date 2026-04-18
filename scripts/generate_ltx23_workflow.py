#!/usr/bin/env python3
"""
Generate an API-format ComfyUI workflow for LTX-2.3 image-to-video + audio.

Based on the official Lightricks example workflow:
https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json

Key differences from LTX-2:
- Uses MultimodalGuider (not STGGuiderAdvanced)
- Uses GuiderParameters for modality-specific config (VIDEO, AUDIO)
- Uses LTXAVTextEncoderLoader (replaces LTXVGemmaCLIPModelLoader)
- Uses LTXVTiledVAEDecode (replaces LTXVSpatioTemporalTiledVAEDecode)
- Uses LTXVImgToVideoConditionOnly (simpler i2v variant)
- 8 inference steps instead of 20
- ManualSigmas with 9-point schedule
"""
import argparse
import json
import random

DEFAULT_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

# Gemma path - ComfyUI-LTXVideo uses text_encoders/ so relative to /opt/ComfyUI/models/
GEMMA_PATH = "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors"


def create_workflow(
    prompt,
    negative_prompt="low quality, distorted, static, frozen, silent, blurry",
    image_filename=None,
    width=768,
    height=512,
    frames=97,
    fps=24,
    seed=None,
    steps=8,
    video_cfg=1.0,
    audio_cfg=1.0,
    modality_scale=1.0,
    model_name="ltx-2.3-22b-distilled-fp8.safetensors",
    output_prefix="ltx23_output",
    include_audio=True,
):
    """Build LTX-2.3 API workflow. Set image_filename=None for T2V, or filename for I2V."""
    if seed is None:
        seed = random.randint(1, 10**9)

    wf = {
        # Checkpoint loader
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model_name}},
        # Text encoder (loads Gemma + LTX encoder stack together)
        "2": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
            "text_encoder": GEMMA_PATH,
            "ckpt_name": model_name,
            "device": "default",
        }},
        # Positive/negative prompts
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
        # LTX conditioning wrapper (adds frame_rate)
        "5": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "frame_rate": float(fps),
        }},
        # Empty video latent
        "6": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width, "height": height, "length": frames, "batch_size": 1,
        }},
    }

    # I2V: add LoadImage + Preprocess + ImgToVideoConditionOnly
    if image_filename:
        wf["7"] = {"class_type": "LoadImage", "inputs": {"image": image_filename}}
        wf["8"] = {"class_type": "LTXVPreprocess", "inputs": {
            "image": ["7", 0], "img_compression": 18,
        }}
        wf["9"] = {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {
            "vae": ["1", 2],
            "image": ["8", 0],
            "latent": ["6", 0],
            "strength": 0.85,
        }}
        video_latent = ["9", 0]
    else:
        video_latent = ["6", 0]

    if include_audio:
        wf["10"] = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": model_name}}
        wf["11"] = {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "audio_vae": ["10", 0],
            "frames_number": frames,
            "frame_rate": fps,
            "batch_size": 1,
        }}
        wf["12"] = {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": video_latent,
            "audio_latent": ["11", 0],
        }}
        sampler_latent = ["12", 0]
    else:
        sampler_latent = video_latent

    # GuiderParameters chain: VIDEO → AUDIO (if audio) → MultimodalGuider
    # The optional `parameters` input lets you stack configs for multiple modalities.
    wf["13"] = {"class_type": "GuiderParameters", "inputs": {
        "modality": "VIDEO",
        "cfg": video_cfg,
        "stg": 1.0,
        "perturb_attn": True,
        "rescale": 0.7,
        "modality_scale": modality_scale,
        "skip_step": 0,
        "cross_attn": True,
    }}
    if include_audio:
        wf["14"] = {"class_type": "GuiderParameters", "inputs": {
            "modality": "AUDIO",
            "cfg": audio_cfg,
            "stg": 1.0,
            "perturb_attn": True,
            "rescale": 0.7,
            "modality_scale": modality_scale,
            "skip_step": 0,
            "cross_attn": True,
            "parameters": ["13", 0],  # chain onto video params
        }}
        guider_params = ["14", 0]
    else:
        guider_params = ["13", 0]

    # Multimodal guider
    wf["15"] = {"class_type": "MultimodalGuider", "inputs": {
        "model": ["1", 0],
        "positive": ["5", 0],
        "negative": ["5", 1],
        "parameters": guider_params,
        "skip_blocks": "",
    }}

    # Manual sigmas + sampler
    wf["16"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": DEFAULT_SIGMAS}}
    wf["17"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["18"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["19"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["18", 0],
        "guider": ["15", 0],
        "sampler": ["17", 0],
        "sigmas": ["16", 0],
        "latent_image": sampler_latent,
    }}

    if include_audio:
        # Separate AV latent back out
        wf["20"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["19", 0]}}
        # Decode video (tiled)
        wf["21"] = {"class_type": "LTXVTiledVAEDecode", "inputs": {
            "vae": ["1", 2],
            "latents": ["20", 0],
            "horizontal_tiles": 4,
            "vertical_tiles": 4,
            "overlap": 1,
            "last_frame_fix": False,
        }}
        # Decode audio
        wf["22"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {
            "audio_vae": ["10", 0],
            "samples": ["20", 1],
        }}
        # Combine into video+audio
        wf["23"] = {"class_type": "CreateVideo", "inputs": {
            "images": ["21", 0],
            "fps": float(fps),
            "audio": ["22", 0],
        }}
    else:
        wf["21"] = {"class_type": "LTXVTiledVAEDecode", "inputs": {
            "vae": ["1", 2],
            "latents": ["19", 0],
            "horizontal_tiles": 4,
            "vertical_tiles": 4,
            "overlap": 1,
            "last_frame_fix": False,
        }}
        wf["23"] = {"class_type": "CreateVideo", "inputs": {
            "images": ["21", 0],
            "fps": float(fps),
        }}

    # Save as mp4
    wf["24"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["23", 0],
        "filename_prefix": output_prefix,
        "format": "mp4",
        "codec": "h264",
    }}
    return wf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LTX-2.3 ComfyUI API workflow")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="low quality, distorted, static, frozen, silent, blurry")
    parser.add_argument("--image", help="Image filename (for i2v)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="ltx23_workflow_api.json")
    parser.add_argument("--prefix", default="ltx23_output")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    wf = create_workflow(
        prompt=args.prompt,
        negative_prompt=args.negative,
        image_filename=args.image,
        width=args.width, height=args.height, frames=args.frames, fps=args.fps,
        seed=args.seed,
        output_prefix=args.prefix,
        include_audio=not args.no_audio,
    )
    with open(args.output, "w") as f:
        json.dump({"prompt": wf}, f, indent=2)
    print(f"Saved: {args.output}")
    print(f"  {len(wf)} nodes, model={wf['1']['inputs']['ckpt_name']}")
    print(f"  i2v={'yes' if args.image else 'no'}, audio={'no' if args.no_audio else 'yes'}")
