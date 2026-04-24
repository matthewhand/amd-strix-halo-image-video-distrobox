import sys
import os

path = "/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/gemma_encoder.py"

with open(path, "r") as f:
    content = f.read()

# The "bad" code we injected
target = """
        clip_obj = comfy.sd.CLIP(clip_target)
        # Antigravity Hack: Force CPU for Gemma to avoid ROCm kernel crash
        clip_obj.patcher.load_device = torch.device("cpu")
        clip_obj.patcher.offload_device = torch.device("cpu")
        return (clip_obj,)
"""
# The original clean code
replacement = "return (comfy.sd.CLIP(clip_target),)"

if "Antigravity Hack" in content:
    # Use loose matching because of indentation/newlines
    # We will just replace the whole block if we can find a unique substring
    if "clip_obj.patcher.load_device = torch.device(\"cpu\")" in content:
        # Re-read to ensure we get the indentation right, or just do a blunt replace of the hacked section
        # The robust way: locate the start of the hack and the return
        import re
        # Regex to find the hacked block including indentation
        pattern = r"(\s+)(clip_obj = comfy\.sd\.CLIP\(clip_target\).*?return \(clip_obj,\))"
        
        match = re.search(pattern, content, re.DOTALL)
        if match:
             # Keep indentation, restore original return
             indent = match.group(1)
             original_line = f"{indent}return (comfy.sd.CLIP(clip_target),)"
             content = content.replace(match.group(0), original_line)
             
             with open(path, "w") as f:
                f.write(content)
             print("Successfully REVERTED gemma_encoder.py to use GPU!")
        else:
             print("Could not regex match the hack block. Manual intervention needed.")
    else:
         print("Hack string fragment not found.")
else:
    print("Clean: 'Antigravity Hack' not found in file. Already native?")
