#!/usr/bin/env python3
"""
Direct WAN video generation test using the WAN CLI
"""
import subprocess
import os
import sys
import time

def run_wan_generation():
    """Run WAN video generation directly"""
    
    # Test parameters
    prompt = "a beautiful mountain landscape with snow"
    size = "1280*704"  # Supported size for ti2v-5B
    frames = 17
    ckpt_dir = "/home/matthewh/comfy-models/diffusion_models"
    save_dir = "/home/matthewh/wan_test_output"
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Build the command
    cmd = [
        "distrobox", "enter", "strix-halo-image-video", "--",
        "/bin/bash", "-c",
        f"source /opt/venv/bin/activate && cd /opt/wan-video-studio && "
        f"python generate.py "
        f"--task ti2v-5B "
        f"--prompt '{prompt}' "
        f"--size {size} "
        f"--frame_num {frames} "
        f"--ckpt_dir {ckpt_dir} "
        f"--save_dir {save_dir} "
        f"--t5_cpu "
        f"--sample_steps 2 "
        f"--offload_model"
    ]
    
    print("Starting WAN video generation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor output
        start_time = time.time()
        timeout = 600  # 10 minutes max
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
            
            # Check timeout
            if time.time() - start_time > timeout:
                print("Generation timed out, terminating...")
                process.terminate()
                return False
        
        # Check result
        if process.returncode == 0:
            print("✅ WAN video generation completed successfully!")
            
            # Check for output files
            output_files = os.listdir(save_dir)
            if output_files:
                print(f"Generated files: {output_files}")
                for file in output_files:
                    file_path = os.path.join(save_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  {file}: {size_mb:.2f} MB")
            else:
                print("No output files found")
                return False
                
            return True
        else:
            print(f"❌ WAN video generation failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running WAN generation: {e}")
        return False

if __name__ == "__main__":
    print("Starting direct WAN video generation test...")
    success = run_wan_generation()
    
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")