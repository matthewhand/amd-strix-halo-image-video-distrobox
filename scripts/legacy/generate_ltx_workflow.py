#!/usr/bin/env python3
"""
Generate a ComfyUI API-format workflow JSON for LTX-2 video generation.

Usage:
    python generate_ltx_workflow.py
    python generate_ltx_workflow.py --prompt "A cat riding a skateboard"
    python generate_ltx_workflow.py --output my_workflow.json
"""
import argparse
import json
import random

DEFAULT_PROMPT = (
    "cinematic shot of a cyberpunk city with neon lights reflecting "
    "on wet pavement, snowing, 4k, highly detailed"
)
DEFAULT_NEGATIVE = "low quality, worst quality, deformed, distorted, watermark"


def create_workflow(prompt, negative_prompt=DEFAULT_NEGATIVE, seed=None):
    if seed is None:
        seed = random.randint(1, 1_000_000_000)

    workflow = {
        "1": {
            "class_type": "LowVRAMCheckpointLoader",
            "inputs": {"ckpt_name": "ltx-2-19b-distilled-fp8.safetensors"},
        },
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {
                "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
                "ltxv_path": "ltx-2-19b-distilled-fp8.safetensors",
                "max_length": 1024,
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "STGGuiderAdvanced",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "skip_steps_sigma_threshold": 0.998,
                "cfg_star_rescale": True,
                "sigmas": "1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180",
                "cfg_values": "8, 6, 6, 4, 3, 1",
                "stg_scale_values": "4, 4, 3, 2, 1, 0",
                "stg_rescale_values": "1, 1, 1, 1, 1, 1",
                "stg_layers_indices": "[29], [29], [29], [29], [29], [29]",
            },
        },
        "6": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "7": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": 20,
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "9": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 512, "length": 49, "batch_size": 1},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["5", 0],
                "sampler": ["6", 0],
                "sigmas": ["7", 0],
                "latent_image": ["9", 0],
            },
        },
        "11": {
            "class_type": "LTXVSpatioTemporalTiledVAEDecode",
            "inputs": {
                "vae": ["1", 2],
                "latents": ["10", 0],
                "spatial_tiles": 4,
                "spatial_overlap": 1,
                "temporal_tile_length": 16,
                "temporal_overlap": 2,
                "last_frame_fix": False,
                "working_device": "auto",
                "working_dtype": "auto",
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "ltx2_output", "images": ["11", 0]},
        },
    }
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LTX-2 ComfyUI workflow JSON")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE, help="Negative prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default="workflow_api.json", help="Output filename")
    args = parser.parse_args()

    wf = create_workflow(args.prompt, args.negative, args.seed)
    with open(args.output, "w") as f:
        json.dump({"prompt": wf}, f, indent=2)
    print(f"{args.output} created")
