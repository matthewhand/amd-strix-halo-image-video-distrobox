#!/usr/bin/env python3
"""
Test a very simple CPU-only workflow
"""
import requests
import json
import time
import os

def create_ultra_minimal_workflow():
    """Create the most basic possible workflow"""
    return {
        "1": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "text": "cat",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {
                "text": "blurry",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "seed": 42,
                "steps": 1,  # Just 1 step for speed
                "cfg": 1.0,  # Minimal CFG
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "5": {
            "inputs": {
                "width": 64,  # Very small for speed
                "height": 64,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "samples": ["4", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode"
        },
        "7": {
            "inputs": {
                "filename_prefix": "test_cpu",
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        }
    }

def main():
    print("🔬 Ultra-minimal CPU Workflow Test")
    print("=" * 40)
    
    # Check ComfyUI is running
    try:
        response = requests.get("http://localhost:8188/system_stats", timeout=5)
        if response.status_code != 200:
            print("❌ ComfyUI not responding")
            return
        print("✅ ComfyUI is running")
    except:
        print("❌ Cannot connect to ComfyUI")
        return
    
    # Create and submit workflow
    workflow = create_ultra_minimal_workflow()
    
    print("🔍 Submitting ultra-minimal workflow...")
    try:
        response = requests.post("http://localhost:8188/prompt", 
                               json={"prompt": workflow}, 
                               timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Error submitting prompt: {response.text}")
            return
            
        result = response.json()
        prompt_id = result.get("prompt_id")
        print(f"✅ Prompt submitted with ID: {prompt_id}")
        
    except Exception as e:
        print(f"❌ Error during submission: {e}")
        return
    
    # Wait a bit and check
    print("⏳ Waiting 30 seconds for generation...")
    time.sleep(30)
    
    # Check history
    try:
        history_response = requests.get(f"http://localhost:8188/history/{prompt_id}", timeout=5)
        if history_response.status_code == 200:
            history = history_response.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                print(f"📊 Status: {status.get('status_str', 'unknown')}")
                print(f"📊 Completed: {status.get('completed', False)}")
                
                if status.get("completed", False):
                    outputs = history[prompt_id].get("outputs", {})
                    if "7" in outputs and "images" in outputs["7"]:
                        for img in outputs["7"]["images"]:
                            filename = img.get("filename", "unknown.png")
                            print(f"✅ Generated image: {filename}")
                    else:
                        print("❌ No image output found")
                else:
                    print("❌ Generation not completed")
            else:
                print("❌ No history entry found")
        else:
            print(f"❌ Error checking history: {history_response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking history: {e}")

if __name__ == "__main__":
    main()