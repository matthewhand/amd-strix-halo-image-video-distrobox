#!/usr/bin/env python3
"""
Simple video generation test using direct ComfyUI workflow
"""
import requests
import json
import time
import os

def test_comfyui_endpoints():
    """Test ComfyUI endpoints to find the correct API"""
    
    base_url = "http://localhost:8000"
    
    # Test common endpoints
    endpoints = [
        "/",
        "/prompt",
        "/api/prompt", 
        "/api/v1/prompt",
        "/queue",
        "/history"
    ]
    
    print("Testing ComfyUI endpoints...")
    for endpoint in endpoints:
        try:
            if endpoint == "/":
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            else:
                response = requests.post(f"{base_url}{endpoint}", 
                                        json={}, 
                                        headers={"Content-Type": "application/json"},
                                        timeout=5)
            
            print(f"  {endpoint}: {response.status_code}")
            if response.status_code != 404:
                print(f"    Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"  {endpoint}: Error - {e}")

def create_simple_image_workflow():
    """Create a simple image generation workflow"""
    workflow = {
        "1": {
            "inputs": {
                "text": "a disco dancing pineapple wearing sunglasses",
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
                "filename_prefix": "test_pineapple",
                "images": ["4", 0]
            },
            "class_type": "SaveImage"
        },
        "4": {
            "inputs": {
                "seed": 12345,
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

def try_generate():
    """Try to generate a simple image"""
    
    workflow = create_simple_image_workflow()
    prompt_data = {"prompt": workflow}
    
    # Try different endpoints
    endpoints = [
        "http://localhost:8000/prompt",
        "http://localhost:8000/api/prompt",
        "http://localhost:8000/api/v1/prompt"
    ]
    
    for endpoint in endpoints:
        print(f"\nTrying endpoint: {endpoint}")
        
        try:
            response = requests.post(endpoint, 
                                   json=prompt_data, 
                                   headers={"Content-Type": "application/json"},
                                   timeout=10)
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            
            if response.status_code == 200:
                result = response.json()
                if "prompt_id" in result:
                    print(f"✅ Successfully submitted prompt with ID: {result['prompt_id']}")
                    return result['prompt_id']
                    
        except Exception as e:
            print(f"Error: {e}")
    
    return None

def check_output():
    """Check for any output files"""
    output_dirs = [
        "/home/matthewh/comfy-outputs",
        "/opt/ComfyUI/output",
        "/home/matthewh/amd-strix-halo-image-video-toolboxes/output"
    ]
    
    print("\nChecking for output files...")
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            if files:
                print(f"  {output_dir}: {len(files)} files")
                for file in files[:5]:  # Show first 5 files
                    filepath = os.path.join(output_dir, file)
                    if os.path.isfile(filepath):
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        print(f"    {file} ({size_mb:.2f} MB)")
            else:
                print(f"  {output_dir}: empty")
        else:
            print(f"  {output_dir}: does not exist")

def main():
    print("🔍 ComfyUI API Test")
    print("=" * 30)
    
    # Test endpoints
    test_comfyui_endpoints()
    
    # Check existing outputs
    check_output()
    
    # Try generation
    print("\n🎬 Attempting generation...")
    prompt_id = try_generate()
    
    if prompt_id:
        print(f"✅ Generation submitted! Check ComfyUI interface at http://localhost:8000")
    else:
        print("❌ Could not submit generation request")
        print("\n💡 Suggestions:")
        print("  1. Check ComfyUI is fully loaded in browser")
        print("  2. Try using the web interface directly")
        print("  3. Check if WAN nodes are loaded in ComfyUI")

if __name__ == "__main__":
    main()