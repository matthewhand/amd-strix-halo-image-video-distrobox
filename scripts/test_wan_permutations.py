#!/usr/bin/env python3
import subprocess
import time
import os
import sys

# Configuration
WAN_ROOT = "/opt/wan-video-studio"
MODEL_ROOT = "/workspace/wan-models"
OUTPUT_ROOT = "/workspace/wan-models/test_outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Checkpoints
CKPT_T2V = f"{MODEL_ROOT}/Wan2.2-T2V-A14B"
CKPT_I2V = f"{MODEL_ROOT}/Wan2.2-I2V-A14B"
# Lightning LoRA paths (adjust based on actual download structure)
LORA_T2V = f"{MODEL_ROOT}/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1" 
LORA_I2V = f"{MODEL_ROOT}/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"

# Test Prompt
PROMPT = "A futuristic city with neon lights, cinematic lighting, 4k"

# Test Matrix
VARIANTS = ["t2v"] # Add i2v later if needed or verified
VAE_TILING = [False, True]
ATTENTION = ["sdpa", "flash_attn"]

RESULTS = []

def run_test(variant, tiling, attn):
    task = f"{variant}-A14B"
    ckpt = CKPT_T2V if variant == "t2v" else CKPT_I2V
    lora = LORA_T2V if variant == "t2v" else LORA_I2V
    
    test_id = f"{variant}_{attn}_{'tiled' if tiling else 'untiled'}"
    output_file = f"{OUTPUT_ROOT}/{test_id}.mp4"
    
    # Construct Command
    cmd = [
        "python3", "/opt/wan_launcher.py",
        "--task", task,
        "--size", "480*832", # Small size for speed
        "--frame_num", "5",  # Minimum frames (4n+1) -> 5 frames
        "--ckpt_dir", ckpt,
        "--lora_dir", lora,
        "--offload_model", "False",
        "--prompt", PROMPT,
        "--save_file", output_file
    ]
    
    if tiling:
        cmd.append("--vae_tiling")
        
    env = os.environ.copy()
    env["WAN_ATTENTION_BACKEND"] = attn
    
    print(f"--- Running Test: {test_id} ---")
    print(f"Backend: {attn}, Tiling: {tiling}")
    print(f"Cmd: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600) # 10 min timeout
        duration = time.time() - start_time
        
        if result.returncode == 0:
            status = "PASS"
            print(f"Result: PASS ({duration:.2f}s)")
        else:
            status = "FAIL"
            print(f"Result: FAIL")
            print("Error Output:", result.stderr[-500:])
            
    except Exception as e:
        duration = time.time() - start_time
        status = f"ERROR: {str(e)}"
        print(f"Result: ERROR")

    RESULTS.append({
        "id": test_id,
        "variant": variant,
        "attn": attn,
        "tiling": tiling,
        "status": status,
        "duration": duration if status == "PASS" else 0
    })

# Main Loop
print("Wait for models to populate...")
# Simple check if directories exist
if not os.path.exists(CKPT_T2V):
    print(f"Warning: {CKPT_T2V} not found. Ensure download finishes.")

for variant in VARIANTS:
    # Check if models exist for this variant
    if variant == "t2v" and not os.path.exists(CKPT_T2V): continue
    
    for attn in ATTENTION:
        for tiling in VAE_TILING:
            run_test(variant, tiling, attn)

# Tabulate
print("\n=== Test Results ===")
print(f"{'ID':<25} | {'Status':<5} | {'Time':<6}")
print("-" * 45)
for r in RESULTS:
    print(f"{r['id']:<25} | {r['status']:<5} | {r['duration']:.1f}s")
