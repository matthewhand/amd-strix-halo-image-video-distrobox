import asyncio
import os
import time
from slopfinity.worker_sh import (
    run_image_ernie,
    run_image_qwen,
    run_audio_heartmula,
    run_video_wan
)

EXP_DIR = "/home/matthewh/amd-strix-halo-image-video-toolboxes/comfy-outputs/experiments"
os.makedirs(EXP_DIR, exist_ok=True)

async def render_variety():
    print("🚀 Starting variety render wave...")
    ts = int(time.time())
    
    print("\n1. Ernie Image (Fast)...")
    await run_image_ernie("A cybernetic golden retriever", os.path.join(EXP_DIR, f"variety_{ts}_ernie.png"), steps=8)
    
    print("\n3. Qwen Image (Medium)...")
    await run_image_qwen("A beautiful neon city in the rain", os.path.join(EXP_DIR, f"variety_{ts}_qwen.png"))
    
    print("\n✅ Variety render complete.")

async def render_longer():
    print("\n🚀 Starting longer/larger render wave...")
    ts = int(time.time())
    
    print("\n1. Heartmula Music (Heavy)...")
    await run_audio_heartmula("epic orchestral boss battle music", os.path.join(EXP_DIR, f"larger_{ts}_music.wav"), duration_s=10.0)
    
    # Note: LTX/Wan Video requires a base image. We can use the Qwen one from earlier if we wanted, 
    # but Heartmula is heavy enough for a test.
    print("\n✅ Larger render complete.")

if __name__ == "__main__":
    asyncio.run(render_variety())
    asyncio.run(render_longer())
