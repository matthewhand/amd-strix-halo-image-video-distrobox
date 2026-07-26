"""Patch ComfyUI-LTXVideo pyramid_blending for newer kornia without pad export."""
from pathlib import Path
p = Path("/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py")
if not p.is_file():
    raise SystemExit(0)
text = p.read_text()
old = "from kornia.geometry.transform.pyramid import ("
if "from kornia.geometry.transform.pyramid import (" in text and "pad as kornia_pad" not in text:
    # Replace import block to avoid pad
    import re
    text2 = re.sub(
        r"from kornia\.geometry\.transform\.pyramid import \([^)]+\)",
        "from kornia.geometry.transform.pyramid import (\n"
        "    PyrDown, PyrUp, pyrdown, pyrup,\n"
        ")\n"
        "try:\n"
        "    from kornia.geometry.transform.pyramid import pad as kornia_pad\n"
        "except ImportError:\n"
        "    import torch.nn.functional as F\n"
        "    def kornia_pad(input, padding):\n"
        "        return F.pad(input, padding)\n"
        "    pad = kornia_pad",
        text,
        count=1,
        flags=re.S,
    )
    if text2 != text:
        p.write_text(text2)
        print("patched", p)
    else:
        print("no change")
else:
    print("skip or already")
