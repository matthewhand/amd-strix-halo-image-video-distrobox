#!/usr/bin/env python3
"""
Generate simple demo content using ComfyUI API with SDXL
"""
import requests
import json
import time
import os
import random

def create_correct_workflow(prompt, filename_prefix="demo"):
    """Create a correct workflow with SDXL"""
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
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "3": {
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        },
        "4": {
            "inputs": {
                "text": "blurry, low quality, distorted, watermark",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {
                "seed": random.randint(1, 1000000),
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["1", 0],
                "negative": ["4", 0],
                "latent_image": ["7", 0]
            },
            "class_type": "KSampler"
        },
        "6": {
            "inputs": {
                "samples": ["5", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEDecode"
        },
        "7": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        }
    }
    return workflow

def submit_workflow(workflow):
    """Submit workflow to ComfyUI"""
    prompt_data = {"prompt": workflow}
    
    try:
        response = requests.post("http://localhost:8188/prompt", 
                               json=prompt_data, 
                               headers={"Content-Type": "application/json"},
                               timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Error submitting prompt: {response.text}")
            return None
            
        result = response.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            print(f"❌ No prompt ID in response: {result}")
            return None
            
        print(f"✅ Prompt ID: {prompt_id}")
        return prompt_id
        
    except Exception as e:
        print(f"❌ Error during submission: {e}")
        return None

def monitor_generation(prompt_id, max_wait=120):
    """Monitor generation progress"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            history_response = requests.get(f"http://localhost:8188/history/{prompt_id}", timeout=5)
            if history_response.status_code == 200:
                history = history_response.json()
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False):
                        print("✅ Generation completed!")
                        outputs = history[prompt_id].get("outputs", {})
                        return outputs
                        
                    # Print progress
                    if "status_str" in status:
                        print(f"⏳ {status['status_str']}")
            
            time.sleep(3)  # Wait 3 seconds before checking again
            
        except requests.exceptions.RequestException:
            time.sleep(3)
            continue
    
    print("⏰ Generation timed out")
    return None

def find_output_files(outputs, node_id):
    """Find output files from ComfyUI outputs"""
    if node_id not in outputs:
        return []
    
    node_output = outputs[node_id]
    files = []
    
    # Check for images
    if "images" in node_output:
        for img in node_output["images"]:
            filename = img.get("filename", "unknown.png")
            # Check multiple possible output directories
            output_paths = [
                f"/home/matthewh/comfy-outputs/{filename}",
                f"/opt/ComfyUI/output/{filename}",
                f"/home/matthewh/amd-strix-halo-image-video-toolboxes/output/{filename}"
            ]
            
            for filepath in output_paths:
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    files.append({
                        "filename": filename,
                        "filepath": filepath,
                        "size_mb": size_mb
                    })
                    break
    
    return files

def generate_demo():
    """Generate demo content"""
    
    print("🎨 Simple SDXL Demo Generator")
    print("=" * 35)
    
    # Check ComfyUI connection
    try:
        response = requests.get("http://localhost:8188/", timeout=5)
        if response.status_code != 200:
            print("❌ ComfyUI is not responding correctly on port 8188")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to ComfyUI: {e}")
        return False
    
    print("✅ Connected to ComfyUI on port 8188")
    print()
    
    # Generate Image 1
    print("🎨 Generating Image 1: Dragon...")
    prompt1 = "a majestic dragon flying over a fantasy castle at sunset, digital art, highly detailed"
    workflow1 = create_correct_workflow(prompt1, "demo_dragon")
    
    prompt_id1 = submit_workflow(workflow1)
    if prompt_id1:
        outputs1 = monitor_generation(prompt_id1)
        if outputs1:
            files1 = find_output_files(outputs1, "3")
            if files1:
                print("📸 Generated Image 1:")
                for img in files1:
                    print(f"  📁 {img['filename']} ({img['size_mb']:.2f} MB)")
                    print(f"     Path: {img['filepath']}")
        else:
            print("❌ Image 1 generation failed")
    else:
        print("❌ Could not submit Image 1")
    
    print()
    
    # Generate Image 2
    print("🎨 Generating Image 2: Unicorn...")
    prompt2 = "a magical unicorn running through a rainbow forest, fantasy art, vibrant colors"
    workflow2 = create_correct_workflow(prompt2, "demo_unicorn")
    
    prompt_id2 = submit_workflow(workflow2)
    if prompt_id2:
        outputs2 = monitor_generation(prompt_id2)
        if outputs2:
            files2 = find_output_files(outputs2, "3")
            if files2:
                print("📸 Generated Image 2:")
                for img in files2:
                    print(f"  📁 {img['filename']} ({img['size_mb']:.2f} MB)")
                    print(f"     Path: {img['filepath']}")
        else:
            print("❌ Image 2 generation failed")
    else:
        print("❌ Could not submit Image 2")
    
    print()
    
    # Generate Image 3
    print("🎨 Generating Image 3: Cyberpunk...")
    prompt3 = "a futuristic city with flying cars and neon lights, cyberpunk style, highly detailed"
    workflow3 = create_correct_workflow(prompt3, "demo_cyberpunk")
    
    prompt_id3 = submit_workflow(workflow3)
    if prompt_id3:
        outputs3 = monitor_generation(prompt_id3)
        if outputs3:
            files3 = find_output_files(outputs3, "3")
            if files3:
                print("📸 Generated Image 3:")
                for img in files3:
                    print(f"  📁 {img['filename']} ({img['size_mb']:.2f} MB)")
                    print(f"     Path: {img['filepath']}")
        else:
            print("❌ Image 3 generation failed")
    else:
        print("❌ Could not submit Image 3")
    
    print()
    print("🎉 Demo generation completed!")
    print("💡 You can access the ComfyUI web interface at http://localhost:8188")
    print("📁 Check the output directories for your generated images!")
    return True

if __name__ == "__main__":
    generate_demo()