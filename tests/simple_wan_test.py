#!/usr/bin/env python3
"""
Simple WAN video generation test using ComfyUI API
"""
import requests
import json
import time
import os

def create_simple_workflow():
    """Create a simple WAN workflow for text-to-video generation"""
    workflow = {
        "1": {
            "inputs": {
                "text": "a beautiful mountain landscape with snow",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "2": {
            "inputs": {
                "ckpt_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "3": {
            "inputs": {
                "filename_prefix": "wan_landscape",
                "images": ["4", 0]
            },
            "class_type": "SaveImage"
        },
        "4": {
            "inputs": {
                "seed": 12345,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["1", 0],
                "negative": ["1", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        }
    }
    return workflow

def generate_with_comfyui():
    """Generate video using ComfyUI API"""
    
    # Check if ComfyUI is running
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code != 200:
            print("ComfyUI is not responding correctly")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Cannot connect to ComfyUI: {e}")
        return False
    
    # Create workflow
    workflow = create_simple_workflow()
    prompt = {"prompt": workflow}
    
    print("Sending prompt to ComfyUI...")
    try:
        # Submit prompt
        response = requests.post("http://localhost:8000/prompt", 
                               json=prompt, 
                               headers={"Content-Type": "application/json"})
        
        if response.status_code != 200:
            print(f"Error submitting prompt: {response.text}")
            return False
            
        result = response.json()
        prompt_id = result.get("prompt_id")
        print(f"Prompt submitted with ID: {prompt_id}")
        
        # Monitor progress
        print("Monitoring generation progress...")
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                history_response = requests.get(f"http://localhost:8000/history/{prompt_id}")
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("completed", False):
                            print("Generation completed!")
                            outputs = history[prompt_id].get("outputs", {})
                            if "3" in outputs:  # SaveImage node
                                images = outputs["3"].get("images", [])
                                for img in images:
                                    filename = img.get("filename", "unknown.png")
                                    print(f"Generated image: {filename}")
                            return True
                        
                        # Print progress
                        if "status_str" in status:
                            print(f"Status: {status['status_str']}")
                
                time.sleep(5)  # Wait 5 seconds before checking again
                
            except requests.exceptions.RequestException:
                time.sleep(5)
                continue
        
        print("Generation timed out")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"Error during generation: {e}")
        return False

if __name__ == "__main__":
    print("Starting WAN video generation test via ComfyUI...")
    success = generate_with_comfyui()
    
    if success:
        print("✅ Video generation test completed successfully!")
    else:
        print("❌ Video generation test failed!")