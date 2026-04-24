import sys
import os

path = "/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/gemma_encoder.py"

with open(path, "r") as f:
    content = f.read()

target = "return (comfy.sd.CLIP(clip_target),)"
replacement = """
        clip_obj = comfy.sd.CLIP(clip_target)
        # Antigravity Hack: Force CPU for Gemma to avoid ROCm kernel crash
        clip_obj.patcher.load_device = torch.device("cpu")
        clip_obj.patcher.offload_device = torch.device("cpu")
        return (clip_obj,)
"""

if target in content:
    content = content.replace(target, replacement.strip())
    with open(path, "w") as f:
        f.write(content)
    print("Successfully patched gemma_encoder.py loader")
else:
    print("Target string not found or already patched")
