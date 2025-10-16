#!/usr/bin/env python3
"""
Generate fun and absurd WAN videos using ComfyUI API on port 8188
"""
import requests
import json
import time
import os
import random

def create_wan_workflow(prompt, filename_prefix="absurd_video"):
    """Create a WAN workflow for text-to-video generation"""
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
            "class_type": "SaveAnimatedWEBP"
        },
        "4": {
            "inputs": {
                "seed": random.randint(1, 1000000),
                "steps": 15,
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

def generate_video(prompt, filename_prefix, max_wait=300):
    """Generate a single video using ComfyUI API"""
    
    # Create workflow
    workflow = create_wan_workflow(prompt, filename_prefix)
    prompt_data = {"prompt": workflow}
    
    print(f"🎬 Generating: {prompt}")
    
    try:
        # Submit prompt to correct port
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
                            if "3" in outputs:  # SaveAnimatedWEBP node
                                videos = outputs["3"].get("videos", [])
                                for video in videos:
                                    filename = video.get("filename", "unknown.webp")
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
                
                time.sleep(5)  # Wait 5 seconds before checking again
                
            except requests.exceptions.RequestException:
                time.sleep(5)
                continue
        
        print("   ⏰ Generation timed out")
        return None, None
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return None, None

def main():
    """Generate multiple fun and absurd videos"""
    
    # Fun and absurd prompts
    prompts = [
        "a disco dancing pineapple wearing sunglasses on a rainbow dance floor",
        "cats in tiny spacesuits exploring a cheese moon",
        "a giant rubber duck swimming in a lake of chocolate milk",
        "robots having a tea party with talking flowers",
        "a surfing llama riding a wave made of spaghetti"
    ]
    
    print("🎭 WAN Video Generator - Fun & Absurd Edition")
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
    
    # Generate videos
    generated_files = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"🎬 Video {i}/{len(prompts)}")
        filename_prefix = f"absurd_{i:02d}"
        
        filepath, size_mb = generate_video(prompt, filename_prefix)
        
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
        print(f"✅ Successfully generated {len(generated_files)} videos")
        print(f"📦 Total size: {total_size:.2f} MB")
        print()
        print("Generated files:")
        for i, f in enumerate(generated_files, 1):
            print(f"  {i}. {os.path.basename(f['file'])} ({f['size_mb']:.2f} MB)")
            print(f"     Prompt: {f['prompt']}")
    else:
        print("❌ No videos were generated successfully")
        print("\n💡 Check the ComfyUI web interface at http://localhost:8188 for manual generation")

if __name__ == "__main__":
    main()