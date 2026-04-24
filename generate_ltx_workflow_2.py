import json
import random

# LTX-2 Workflow (API Format) - Fixed Inputs

def create_workflow():
    workflow = {}
    
    seed = random.randint(1, 1000000000)
    
    # Node 1: Helper - Checkpoint Loader
    workflow["1"] = {
        "class_type": "LowVRAMCheckpointLoader",
        "inputs": {
            "ckpt_name": "ltx-2-19b-distilled-fp8.safetensors"
        }
    }
    
    # Node 2: Helper - Gemma Loader
    workflow["2"] = {
        "class_type": "LTXVGemmaCLIPModelLoader",
        "inputs": {
            # Point to first shard as Comfy lists files, not folders
            "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
            "ltxv_path": "ltx-2-19b-distilled-fp8.safetensors",
            "max_length": 1024
        }
    }
    
    # Node 3: Positive Prompt
    workflow["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "A futuristic samurai in a neon-lit rainstorm, detailed armor, 8k, cinematic lighting",
            "clip": ["2", 0]
        }
    }
    
    # Node 4: Negative Prompt
    workflow["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, deformed, distorted, watermark",
            "clip": ["2", 0]
        }
    }
    
    # Node 5: STG Guider Advanced
    workflow["5"] = {
        "class_type": "STGGuiderAdvanced",
        "inputs": {
            "model": ["1", 0],
            "positive": ["3", 0],
            "negative": ["4", 0],
            "skip_steps_sigma_threshold": 0.998,
            "cfg_star_rescale": True,
            # Required STRING inputs with defaults
            "sigmas": "1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180",
            "cfg_values": "8, 6, 6, 4, 3, 1",
            "stg_scale_values": "4, 4, 3, 2, 1, 0",
            "stg_rescale_values": "1, 1, 1, 1, 1, 1",
            "stg_layers_indices": "[29], [29], [29], [29], [29], [29]"
        }
    }
    
    # Node 6: Sampler Selector
    workflow["6"] = {
        "class_type": "KSamplerSelect",
        "inputs": {
            "sampler_name": "euler"
        }
    }
    
    # Node 7: Scheduler (Using simple scheduler as basic fallback)
    # Note: LTX examples use ManualSigmas. If BasicScheduler produces incompatible sigmas, quality might suffer,
    # but it should run.
    workflow["7"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": ["1", 0],
            "scheduler": "simple",
            "steps": 20,
            "denoise": 1.0
        }
    }
    
    # Node 8: Random Noise
    workflow["8"] = {
        "class_type": "RandomNoise",
        "inputs": {
            "noise_seed": seed
        }
    }
    
    # Node 9: Empty Latent
    workflow["9"] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": 768, # Reduced size for speed/safety
            "height": 512,
            "length": 49,
            "batch_size": 1
        }
    }
    
    # Node 10: Sampler Custom Advanced
    workflow["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["8", 0],
            "guider": ["5", 0],
            "sampler": ["6", 0],
            "sigmas": ["7", 0],
            "latent_image": ["9", 0]
        }
    }
    
    # Node 11: VAE Decode using SpatioTemporal
    workflow["11"] = {
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
            "working_dtype": "auto"
        }
    }
    
    # Node 12: Save Image
    workflow["12"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ltx2_test_samurai",
            "images": ["11", 0]
        }
    }
    
    return workflow

if __name__ == "__main__":
    wf = create_workflow()
    # Wrap in "prompt" key for ComfyUI API
    payload = {"prompt": wf}
    with open("workflow_api.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("workflow_api.json created")
