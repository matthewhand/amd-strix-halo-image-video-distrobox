#!/usr/bin/env python3
"""
Comprehensive E2E Test for AMD Strix Halo Image & Video Toolbox
Tests both Qwen Image Generation and ComfyUI Video Generation workflows.
"""

import sys
import os
import time
import requests
import json
from pathlib import Path
import subprocess
import hashlib

def test_docker_compose_startup():
    """Test Docker Compose service startup for both services"""
    print("🚀 Testing Docker Compose startup...")

    try:
        # Check if container is running
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if 'strix-halo-toolbox' not in result.stdout:
            print("🔄 Starting services...")
            subprocess.run(['docker', 'compose', '--profile', 'qwen', 'up', '--build', '-d'], check=True)
            time.sleep(45)  # Wait for services to start

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose startup failed: {e}")
        return False

def test_service_health():
    """Test if both Qwen Web UI and ComfyUI are responsive"""
    print("🏥 Testing service health...")

    services = {
        'Qwen Image Studio': 'http://localhost:8000',
        'ComfyUI': 'http://localhost:8188'
    }

    results = {}

    for service_name, url in services.items():
        max_retries = 10
        service_healthy = False

        for i in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"✅ {service_name} is responsive")
                    results[service_name] = True
                    service_healthy = True
                    break
            except requests.exceptions.RequestException:
                print(f"⏳ {service_name} attempt {i+1}/{max_retries}: Waiting...")
                time.sleep(5)

        if not service_healthy:
            print(f"❌ {service_name} is not responding")
            results[service_name] = False

    return results

def test_qwen_image_generation():
    """Test Qwen image generation via API"""
    print("🎨 Testing Qwen image generation...")

    generation_data = {
        "prompt": "E2E test: futuristic city with flying cars at sunset",
        "num_images": 1,
        "steps": 4,  # Ultra-fast
        "size": "1:1",
        "seed": 54321,
        "use_fast": True
    }

    try:
        # Submit generation job
        response = requests.post(
            'http://localhost:8000/api/generate',
            json=generation_data,
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ Qwen API request failed: {response.status_code}")
            return False

        job_data = response.json()
        job_id = job_data.get('job_id')
        if not job_id:
            print("❌ No job_id returned from Qwen API")
            return False

        print(f"✅ Qwen generation job submitted: {job_id}")

        # Poll for completion
        max_wait = 180  # 3 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                status_response = requests.get(f'http://localhost:8000/api/job/{job_id}', timeout=10)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status')

                    if status == 'completed':
                        print("✅ Qwen image generation completed!")
                        return True
                    elif status == 'failed':
                        print(f"❌ Qwen generation failed: {status_data.get('error', 'Unknown error')}")
                        return False
                    else:
                        print(f"⏳ Qwen Status: {status}")

                time.sleep(10)
            except requests.exceptions.RequestException:
                time.sleep(5)
                continue

        print("❌ Qwen generation timed out")
        return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Qwen API request failed: {e}")
        return False

def test_comfyui_video_generation():
    """Test ComfyUI video generation (Text-to-Video)"""
    print("🎬 Testing ComfyUI Text-to-Video generation...")

    # Simple T2V workflow for testing
    video_workflow = {
        "prompt": {
            "1": {
                "inputs": {
                    "text": "E2E test: peaceful mountain landscape at sunrise with flowing clouds",
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
                    "filename_prefix": "e2e_test_video_",
                    "format": "video/h264-mp4",
                    "frames_per_second": 8
                },
                "class_type": "SaveAnimatedWEBM"
            }
        }
    }

    try:
        # Submit video generation workflow
        response = requests.post(
            'http://localhost:8188/prompt',
            json={"prompt": video_workflow["prompt"]},
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ ComfyUI API request failed: {response.status_code}")
            return False

        result = response.json()
        prompt_id = result.get('prompt_id')
        if not prompt_id:
            print("❌ No prompt_id returned from ComfyUI")
            return False

        print(f"✅ ComfyUI video generation submitted: {prompt_id}")

        # Poll for completion (T2V should be faster than I2V)
        max_wait = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                history_response = requests.get(f'http://localhost:8188/history/{prompt_id}', timeout=10)
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    if prompt_id in history_data:
                        status = history_data[prompt_id].get('status', {})

                        if status.get('status_str') == 'completed':
                            print("✅ ComfyUI video generation completed!")
                            return True
                        elif status.get('status_str') == 'failed':
                            error_msg = status.get('messages', [])[-1] if status.get('messages') else 'Unknown error'
                            print(f"❌ ComfyUI generation failed: {error_msg}")
                            return False
                        else:
                            print(f"⏳ ComfyUI Status: {status.get('status_str', 'unknown')}")

                time.sleep(15)
            except requests.exceptions.RequestException:
                time.sleep(5)
                continue

        print("❌ ComfyUI video generation timed out")
        return False

    except requests.exceptions.RequestException as e:
        print(f"❌ ComfyUI API request failed: {e}")
        return False

def test_output_file_integrity():
    """Test that generated files have proper integrity and sizes"""
    print("🔍 Testing output file integrity...")

    results = {}

    # Test Qwen image outputs
    qwen_output_dir = Path.home() / '.qwen-image-studio'
    if qwen_output_dir.exists():
        png_files = list(qwen_output_dir.glob('*.png'))
        if png_files:
            latest_image = max(png_files, key=lambda p: p.stat().st_mtime)
            image_size = latest_image.stat().st_size

            # Validate file size (>1MB for real images)
            if image_size > 1000000:
                print(f"✅ Qwen image: {latest_image.name} ({image_size:,} bytes)")
                results['qwen_image'] = True
            else:
                print(f"⚠️  Qwen image too small: {image_size} bytes")
                results['qwen_image'] = False
        else:
            print("❌ No Qwen images found")
            results['qwen_image'] = False
    else:
        print("❌ Qwen output directory not found")
        results['qwen_image'] = False

    # Test ComfyUI video outputs
    comfyui_output_dir = Path.home() / 'comfy-outputs'
    if comfyui_output_dir.exists():
        video_files = []
        # Look for video files and image sequences
        video_files.extend(comfyui_output_dir.glob('*.webm'))
        video_files.extend(comfyui_output_dir.glob('*.mp4'))
        video_files.extend(comfyui_output_dir.glob('frame_*.png'))

        if video_files:
            total_size = sum(f.stat().st_size for f in video_files)
            print(f"✅ ComfyUI outputs: {len(video_files)} files ({total_size:,} bytes total)")
            results['comfyui_video'] = True
        else:
            print("❌ No ComfyUI outputs found")
            results['comfyui_video'] = False
    else:
        print("❌ ComfyUI output directory not found")
        results['comfyui_video'] = False

    return results

def test_gpu_performance():
    """Test GPU utilization and memory during generation"""
    print("🔥 Testing GPU performance...")

    try:
        # Check GPU availability and memory
        result = subprocess.run([
            'docker', 'exec', 'strix-halo-toolbox',
            'python3', '-c',
            '''
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
    print("CUDA Available: True")
else:
    print("CUDA Available: False")
'''
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"✅ GPU Status: {output}")
            return "CUDA Available: True" in output and "128.0 GB" in output
        else:
            print(f"❌ GPU check failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ GPU check timed out")
        return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def test_model_availability():
    """Test that required models are available"""
    print("📦 Testing model availability...")

    models_check = subprocess.run([
        'docker', 'exec', 'strix-halo-toolbox',
        'bash', '-c',
        '''
echo "Qwen Models:"
ls -la /root/.cache/huggingface/hub/models--Qwen--Qwen-Image*/ 2>/dev/null | head -3
echo ""
echo "WAN T2V Models:"
ls -la /opt/ComfyUI/models/diffusion_models/wan2.2_t2v_* 2>/dev/null | wc -l
echo "WAN I2V Models:"
ls -la /opt/ComfyUI/models/diffusion_models/wan2.2_i2v_* 2>/dev/null | wc -l
echo "WAN VAE:"
ls -la /opt/ComfyUI/models/vae/wan2.2_vae.safetensors 2>/dev/null
'''
    ], capture_output=True, text=True, timeout=30)

    if models_check.returncode == 0:
        print("✅ Model availability check completed")
        output = models_check.stdout
        # Check for key models
        has_qwen = "Qwen-Image" in output
        has_wan_t2v = "wan2.2_t2v" in output
        has_wan_vae = "wan2.2_vae.safetensors" in output

        print(f"  Qwen Model: {'✅' if has_qwen else '❌'}")
        print(f"  WAN T2V Models: {'✅' if has_wan_t2v else '❌'}")
        print(f"  WAN VAE: {'✅' if has_wan_vae else '❌'}")

        return has_qwen and has_wan_t2v and has_wan_vae
    else:
        print(f"❌ Model check failed: {models_check.stderr}")
        return False

def cleanup():
    """Clean up test artifacts"""
    print("🧹 Cleaning up...")
    try:
        # Don't stop services as they might be needed
        print("✅ Services left running for manual inspection")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Main test runner"""
    print("🧪 Starting Comprehensive E2E Test")
    print("=" * 60)
    print("Testing: Qwen Image Generation + ComfyUI Video Generation")
    print("=" * 60)

    test_results = []

    # Test 1: Service startup
    test_results.append(("Docker Compose Startup", test_docker_compose_startup()))

    # Test 2: Service health (both services)
    health_results = test_service_health()
    test_results.append(("Qwen Service Health", health_results.get('Qwen Image Studio', False)))
    test_results.append(("ComfyUI Service Health", health_results.get('ComfyUI', False)))

    # Test 3: Model availability
    test_results.append(("Model Availability", test_model_availability()))

    # Test 4: GPU performance
    test_results.append(("GPU Performance", test_gpu_performance()))

    # Test 5: Qwen image generation
    test_results.append(("Qwen Image Generation", test_qwen_image_generation()))

    # Test 6: ComfyUI video generation
    test_results.append(("ComfyUI Video Generation", test_comfyui_video_generation()))

    # Test 7: Output file integrity
    integrity_results = test_output_file_integrity()
    test_results.append(("Qwen Output Integrity", integrity_results.get('qwen_image', False)))
    test_results.append(("ComfyUI Output Integrity", integrity_results.get('comfyui_video', False)))

    # Results summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print(f"\nSummary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is fully functional.")
        cleanup()
        return 0
    else:
        print("💥 Some tests failed! Check logs for details.")
        cleanup()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        cleanup()
        sys.exit(1)