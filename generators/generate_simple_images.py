#!/usr/bin/env python3
"""
Generate simple images using ComfyUI API to test the setup
"""
import requests
import json
import time
import os
import random

def create_simple_workflow(prompt, filename_prefix="test_image"):
    """Create a simple image generation workflow"""
    workflow = {
        "1": {
            "inputs": {
                "text": prompt,
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
                "filename_prefix": filename_prefix,
                "images": ["4", 0]
            },
            "class_type": "SaveImage"
        },
        "4": {
            "inputs": {
                "seed": random.randint(1, 1000000),
                "steps": 10,
                "cfg": 7.0,
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
                "width": 256,
                "height": 256,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        }
    }
    return workflow

def generate_image(prompt, filename_prefix, max_wait=120):
    """Generate a single image using ComfyUI API"""
    
    # Create workflow
    workflow = create_simple_workflow(prompt, filename_prefix)
    prompt_data = {"prompt": workflow}
    
    print(f"🎨 Generating: {prompt}")
    
    try:
        # Submit prompt
        response = requests.post("http://localhost:8188/prompt", 
                               json=prompt_data, 
                               headers={"Content-Type": "application/json"},
                               timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Error submitting prompt: {response.text}")
            return None, None
            
        result = response.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            print(f"❌ No prompt ID in response: {result}")
            return None, None
            
        print(f"   ✅ Prompt ID: {prompt_id}")
        
        # Monitor progress
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                history_response = requests.get(f"http://localhost:8188/history/{prompt_id}", timeout=5)
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("completed", False):
                            print("   ✅ Generation completed!")
                            outputs = history[prompt_id].get("outputs", {})
                            if "3" in outputs:  # SaveImage node
                                images = outputs["3"].get("images", [])
                                for image in images:
                                    filename = image.get("filename", "unknown.png")
                                    # Check multiple possible output directories
                                    output_paths = [
                                        f"/home/matthewh/comfy-outputs/{filename}",
                                        f"/opt/ComfyUI/output/{filename}",
                                        f"/home/matthewh/amd-strix-halo-image-video-toolboxes/output/{filename}"
                                    ]
                                    
                                    for filepath in output_paths:
                                        if os.path.exists(filepath):
                                            size_mb = os.path.getsize(filepath) / (1024 * 1024)
                                            print(f"   📁 File: {filename} ({size_mb:.2f} MB)")
                                            return filepath, size_mb
                        
                        # Print progress
                        if "status_str" in status:
                            print(f"   ⏳ {status['status_str']}")
                
                time.sleep(3)  # Wait 3 seconds before checking again
                
            except requests.exceptions.RequestException:
                time.sleep(3)
                continue
        
        print("   ⏰ Generation timed out")
        return None, None
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return None, None

def main():
    """Generate multiple fun and absurd images"""
    
    # Fun and absurd prompts
    prompts = [
        "a disco dancing pineapple wearing sunglasses",
        "cats in tiny spacesuits on the moon",
        "a giant rubber duck in chocolate milk",
        "robots having a tea party with flowers",
        "a surfing llama riding a spaghetti wave"
    ]
    
    print("🎨 WAN Image Generator - Fun & Absurd Edition")
    print("=" * 50)
    
    # Check ComfyUI connection
    try:
        response = requests.get("http://localhost:8188/", timeout=5)
        if response.status_code != 200:
            print("❌ ComfyUI is not responding correctly on port 8188")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to ComfyUI: {e}")
        return
    
    print("✅ Connected to ComfyUI on port 8188")
    print()
    
    # Generate images
    generated_files = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"🎨 Image {i}/{len(prompts)}")
        filename_prefix = f"absurd_img_{i:02d}"
        
        filepath, size_mb = generate_image(prompt, filename_prefix)
        
        if filepath and size_mb:
            generated_files.append({
                "prompt": prompt,
                "file": filepath,
                "size_mb": size_mb
            })
        
        print()
    
    # Summary
    print("📊 Generation Summary")
    print("=" * 30)
    
    if generated_files:
        total_size = sum(f["size_mb"] for f in generated_files)
        print(f"✅ Successfully generated {len(generated_files)} images")
        print(f"📦 Total size: {total_size:.2f} MB")
        print()
        print("Generated files:")
        for i, f in enumerate(generated_files, 1):
            print(f"  {i}. {os.path.basename(f['file'])} ({f['size_mb']:.2f} MB)")
            print(f"     Prompt: {f['prompt']}")
    else:
        print("❌ No images were generated successfully")
        print("\n💡 Check the ComfyUI web interface at http://localhost:8188 for manual generation")

if __name__ == "__main__":
    main()