import asyncio
import os
import sys

# Add repo root to sys.path
ROOT = "/home/matthewh/amd-strix-halo-image-video-toolboxes"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity.llm.probe import ping

def test_speed():
    base_url = "http://10.0.0.107:1234/v1"
    model_id = "llama-3.2"
    provider = "lmstudio"
    
    print(f"Pinging {model_id} at {base_url}...")
    res = ping(base_url, provider, model_id, timeout=30)
    print(f"Result: {res}")

if __name__ == "__main__":
    test_speed()
