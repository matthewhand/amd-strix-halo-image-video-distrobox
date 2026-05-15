import sys
import os
import json
import urllib.request
import time

sys.path.append(os.getcwd())
from run_fleet import generate_video_ltx

try:
    generate_video_ltx(
        image_fn="video_1_base.png",
        prompt="TEST fast video 9 frames",
        out_path="comfy-outputs/experiments/slop_test_video.mp4",
        size_str="1024*1024",
        frames=9
    )
    print("Video generation successful!")
except Exception as e:
    print("Error:", e)
