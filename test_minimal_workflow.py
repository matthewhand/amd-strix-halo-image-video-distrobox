#!/usr/bin/env python3
"""
Test a minimal workflow to debug ComfyUI issues
"""
import requests
import json
import time
import os

def create_minimal_workflow():
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
                "text": "a simple red circle",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {
                "text": "",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "seed": 12345,
                "steps": 5,
                "cfg": 7.0,
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
                "width": 512,
                "height": 512,
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
                "filename_prefix": "test_minimal",
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        }
    }

def submit_and_monitor(workflow):
    """Submit workflow and monitor with detailed logging"""
    print("🔍 Submitting minimal workflow...")
    
    # Submit the workflow
    try:
        response = requests.post("http://localhost:8188/prompt", 
                               json={"prompt": workflow}, 
                               timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Error submitting prompt: {response.text}")
            return None
            
        result = response.json()
        prompt_id = result.get("prompt_id")
        print(f"✅ Prompt submitted with ID: {prompt_id}")
        return prompt_id
        
    except Exception as e:
        print(f"❌ Error during submission: {e}")
        return None

def monitor_with_details(prompt_id, max_wait=60):
    """Monitor with detailed error information"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            # Check history
            history_response = requests.get(f"http://localhost:8188/history/{prompt_id}", timeout=5)
            if history_response.status_code == 200:
                history = history_response.json()
                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    status = prompt_data.get("status", {})
                    
                    if status.get("completed", False):
                        print("✅ Generation completed!")
                        outputs = prompt_data.get("outputs", {})
                        return outputs
                    
                    # Check for errors
                    if "status_str" in status:
                        status_str = status["status_str"]
                        print(f"⏳ Status: {status_str}")
                        
                        if "error" in status_str.lower():
                            print(f"❌ Error detected: {status_str}")
                            # Try to get more details
                            if "messages" in prompt_data:
                                for msg in prompt_data["messages"]:
                                    if msg[0] == "execution_error":
                                        print(f"🔍 Execution error: {msg[1]}")
                                        print(f"🔍 Node error: {msg[2]}")
            
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error checking status: {e}")
            time.sleep(2)
            continue
    
    print("⏰ Monitoring timed out")
    return None

def check_comfyui_health():
    """Check ComfyUI health and available models"""
    try:
        # Check system stats
        stats_response = requests.get("http://localhost:8188/system_stats", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"✅ ComfyUI version: {stats['system']['comfyui_version']}")
            print(f"✅ PyTorch version: {stats['system']['pytorch_version']}")
        
        # Check available models
        models_response = requests.get("http://localhost:8188/object_info", timeout=5)
        if models_response.status_code == 200:
            models = models_response.json()
            if "CheckpointLoaderSimple" in models:
                checkpoints = models["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                print(f"✅ Available checkpoints: {len(checkpoints)} models")
                for ckpt in checkpoints[:3]:  # Show first 3
                    print(f"  - {ckpt}")
            else:
                print("❌ CheckpointLoaderSimple not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking ComfyUI health: {e}")
        return False

def main():
    print("🔬 ComfyUI Minimal Workflow Test")
    print("=" * 40)
    
    # Check ComfyUI health
    if not check_comfyui_health():
        print("❌ ComfyUI health check failed")
        return
    
    print()
    
    # Create and submit minimal workflow
    workflow = create_minimal_workflow()
    prompt_id = submit_and_monitor(workflow)
    
    if prompt_id:
        print("\n🔍 Monitoring generation...")
        outputs = monitor_with_details(prompt_id)
        
        if outputs:
            print("🎉 Success! Generated files:")
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        filename = img.get("filename", "unknown.png")
                        print(f"  📁 {filename}")
        else:
            print("❌ Generation failed or timed out")
    else:
        print("❌ Failed to submit workflow")

if __name__ == "__main__":
    main()