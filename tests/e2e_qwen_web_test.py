#!/usr/bin/env python3
"""
E2E Test for Qwen Image Web UI
Tests complete workflow from service startup to image generation via web interface.
Uses ultra-fast settings (4 steps) to speed up testing.
"""

import sys
import os
import time
import requests
import json
from pathlib import Path
import subprocess

def test_docker_compose_startup():
    """Test Docker Compose service startup"""
    print("🚀 Testing Docker Compose startup...")

    try:
        # Check if container is running
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if 'strix-halo-toolbox' not in result.stdout:
            print("🔄 Starting services...")
            subprocess.run(['docker', 'compose', '--profile', 'qwen-only', 'up', '--build', '-d'], check=True)
            time.sleep(30)  # Wait for services to start

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose startup failed: {e}")
        return False

def test_service_health():
    """Test if Qwen Web UI is responsive"""
    print("🏥 Testing service health...")

    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8000', timeout=10)
            if response.status_code == 200:
                print("✅ Qwen Web UI is responsive")
                return True
        except requests.exceptions.RequestException:
            print(f"⏳ Attempt {i+1}/{max_retries}: Waiting for service...")
            time.sleep(5)

    print("❌ Qwen Web UI is not responding")
    return False

def test_image_generation_via_api():
    """Test image generation via API with ultra-fast settings"""
    print("🎨 Testing image generation via API...")

    generation_data = {
        "prompt": "E2E test: bright blue star on dark background",
        "num_images": 1,
        "steps": 4,  # Ultra-fast
        "size": "1:1",
        "seed": 12345,
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
            print(f"❌ API request failed: {response.status_code}")
            return False

        job_data = response.json()
        job_id = job_data.get('job_id')
        if not job_id:
            print("❌ No job_id returned from API")
            return False

        print(f"✅ Generation job submitted: {job_id}")

        # Poll for completion (ultra-fast should complete quickly)
        max_wait = 180  # 3 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                status_response = requests.get(f'http://localhost:8000/api/job/{job_id}', timeout=10)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status')

                    if status == 'completed':
                        print("✅ Image generation completed!")
                        return job_data
                    elif status == 'failed':
                        print(f"❌ Generation failed: {status_data.get('error', 'Unknown error')}")
                        return False
                    else:
                        print(f"⏳ Status: {status}")

                time.sleep(10)
            except requests.exceptions.RequestException:
                time.sleep(5)
                continue

        print("❌ Generation timed out")
        return False

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False

def test_image_persistence(job_data):
    """Test that generated image persists to host filesystem"""
    print("💾 Testing image persistence...")

    try:
        output_dir = Path.home() / '.qwen-image-studio'

        # Wait for file to appear
        max_wait = 30
        start_time = time.time()

        while time.time() - start_time < max_wait:
            png_files = list(output_dir.glob('*.png'))
            if png_files:
                # Check for most recent file
                latest_file = max(png_files, key=lambda p: p.stat().st_mtime)
                file_size = latest_file.stat().st_size

                if file_size > 1000000:  # > 1MB suggests a real image
                    print(f"✅ Image persisted to: {latest_file}")
                    print(f"📊 File size: {file_size:,} bytes")
                    return True
                else:
                    print(f"⚠️  File too small: {file_size} bytes")

            time.sleep(2)

        print("❌ No persisted image found")
        return False

    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False

def test_gpu_utilization():
    """Test GPU is being utilized during generation"""
    print("🔥 Testing GPU utilization...")

    try:
        # Simple GPU check inside container
        result = subprocess.run([
            'docker', 'exec', 'strix-halo-toolbox',
            'python3', '-c',
            'import torch; print(f"GPU Available: {torch.cuda.is_available()}"); print(f"Device Count: {torch.cuda.device_count()}")'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(f"✅ GPU Status: {result.stdout.strip()}")
            return "GPU Available: True" in result.stdout
        else:
            print(f"❌ GPU check failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ GPU check timed out")
        return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def cleanup():
    """Clean up test artifacts"""
    print("🧹 Cleaning up...")

    try:
        subprocess.run(['docker', 'compose', 'down'], check=True)
    except subprocess.CalledProcessError:
        pass  # Ignore cleanup errors

def main():
    """Main test runner"""
    print("🧪 Starting Qwen Image Web UI E2E Test")
    print("=" * 50)

    test_results = []

    # Test 1: Service startup
    test_results.append(("Docker Compose Startup", test_docker_compose_startup()))

    # Test 2: Service health
    test_results.append(("Service Health", test_service_health()))

    # Test 3: GPU utilization
    test_results.append(("GPU Utilization", test_gpu_utilization()))

    # Test 4: Image generation
    job_data = test_image_generation_via_api()
    test_results.append(("Image Generation", job_data is not False))

    # Test 5: Persistence
    if job_data:
        test_results.append(("Image Persistence", test_image_persistence(job_data)))
    else:
        test_results.append(("Image Persistence", False))

    # Results summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print(f"\nSummary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        cleanup()
        return 0
    else:
        print("💥 Some tests failed!")
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