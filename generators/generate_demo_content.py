#!/usr/bin/env python3
"""
Generate demo image and video using ComfyUI API
"""
import requests
import json
import time
import os
import random

def create_image_workflow(prompt, filename_prefix="demo_image"):
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

def monitor_generation(prompt_id, max_wait=180):
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
    
    # Check for videos
    if "videos" in node_output:
        for vid in node_output["videos"]:
            filename = vid.get("filename", "unknown.mp4")
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

def generate_content():
    """Generate demo image and video"""
    
    print("🎨 WAN Demo Content Generator")
    print("=" * 40)
    
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
    
    # Generate Image
    print("🎨 Generating Demo Image...")
    image_prompt = "a majestic dragon flying over a fantasy castle at sunset, digital art, highly detailed"
    image_workflow = create_image_workflow(image_prompt, "demo_dragon")
    
    image_prompt_id = submit_workflow(image_workflow)
    if image_prompt_id:
        image_outputs = monitor_generation(image_prompt_id)
        if image_outputs:
            image_files = find_output_files(image_outputs, "3")
            if image_files:
                print("📸 Generated Image:")
                for img in image_files:
                    print(f"  📁 {img['filename']} ({img['size_mb']:.2f} MB)")
                    print(f"     Path: {img['filepath']}")
            else:
                print("❌ No image files found")
        else:
            print("❌ Image generation failed")
    else:
        print("❌ Could not submit image generation")
    
    print()
    
    # Generate Video (using a different workflow for video)
    print("🎬 Generating Demo Video...")
    video_prompt = "a magical unicorn running through a rainbow forest, fantasy animation"
    
    # Simple video workflow (using SaveAnimatedWEBP if available)
    video_workflow = {
        "1": {
            "inputs": {
                "text": video_prompt,
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
                "filename_prefix": "demo_unicorn",
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
    
    video_prompt_id = submit_workflow(video_workflow)
    if video_prompt_id:
        video_outputs = monitor_generation(video_prompt_id, max_wait=300)  # Longer timeout for video
        if video_outputs:
            video_files = find_output_files(video_outputs, "3")
            if video_files:
                print("🎬 Generated Video:")
                for vid in video_files:
                    print(f"  📁 {vid['filename']} ({vid['size_mb']:.2f} MB)")
                    print(f"     Path: {vid['filepath']}")
            else:
                print("❌ No video files found - trying fallback to image generation")
                
                # Fallback: generate another image
                fallback_prompt = "a futuristic city with flying cars and neon lights, cyberpunk style"
                fallback_workflow = create_image_workflow(fallback_prompt, "demo_cyberpunk")
                
                fallback_prompt_id = submit_workflow(fallback_workflow)
                if fallback_prompt_id:
                    fallback_outputs = monitor_generation(fallback_prompt_id)
                    if fallback_outputs:
                        fallback_files = find_output_files(fallback_outputs, "3")
                        if fallback_files:
                            print("🎨 Generated Fallback Image:")
                            for img in fallback_files:
                                print(f"  📁 {img['filename']} ({img['size_mb']:.2f} MB)")
                                print(f"     Path: {img['filepath']}")
        else:
            print("❌ Video generation failed")
    else:
        print("❌ Could not submit video generation")
    
    print()
    print("🎉 Demo generation completed!")
    print("💡 You can access the ComfyUI web interface at http://localhost:8188")
    return True

if __name__ == "__main__":
    generate_content()