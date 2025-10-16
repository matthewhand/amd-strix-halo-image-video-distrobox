#!/usr/bin/env python3
"""
Patch ComfyUI's WanImageToVideo node to apply Strix Halo safe defaults.
Clamps resolution/frame count when WAN22_SAFE_MODE is enabled.
"""
from __future__ import annotations

from pathlib import Path

WAN_NODE_PATH = Path("/opt/ComfyUI/comfy_extras/nodes_wan.py")
MARKER = "WAN_SAFE_LIMIT_APPLIED"


def ensure_marker(text: str) -> bool:
    return MARKER in text


def add_import_os(text: str) -> str:
    if "import os" in text.splitlines()[0:10]:
        return text
    return text.replace("import node_helpers\n", "import node_helpers\nimport os\n", 1)


def patch_execute(text: str) -> str:
    needle = (
        "    @classmethod\n"
        "    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None) -> io.NodeOutput:\n"
    )
    insert = (
        "        safe_mode = str(os.getenv(\"WAN22_SAFE_MODE\", \"0\")).lower() not in (\"0\", \"false\", \"no\")\n"
        "        if safe_mode:\n"
        "            max_side_env = int(os.getenv(\"WAN22_MAX_SIDE\", \"640\"))\n"
        "            max_frames_env = int(os.getenv(\"WAN22_MAX_FRAMES\", \"33\"))\n"
        "            max_latent_env = int(os.getenv(\"WAN22_MAX_LATENT_ELEMENTS\", \"2500000\"))\n"
        "            max_side = max(16, (max_side_env // 16) * 16)\n"
        "            max_frames = max(1, ((max_frames_env - 1) // 4) * 4 + 1)\n"
        "            changed_args = []\n"
        "            def clamp_dim(dim: int) -> int:\n"
        "                target = min(dim, max_side)\n"
        "                target = max(16, (target // 16) * 16)\n"
        "                return target\n"
        "            new_width = clamp_dim(width)\n"
        "            if new_width != width:\n"
        "                changed_args.append(f\"width {width}->{new_width}\")\n"
        "                width = new_width\n"
        "            new_height = clamp_dim(height)\n"
        "            if new_height != height:\n"
        "                changed_args.append(f\"height {height}->{new_height}\")\n"
        "                height = new_height\n"
        "            if length > max_frames:\n"
        "                changed_args.append(f\"frames {length}->{max_frames}\")\n"
        "                length = max_frames\n"
        "            batch_cnt = max(1, batch_size)\n"
        "            def latent_elems(frame_count: int) -> int:\n"
        "                temporal = ((frame_count - 1) // 4) + 1\n"
        "                spatial_h = max(1, height // 8)\n"
        "                spatial_w = max(1, width // 8)\n"
        "                return batch_cnt * 16 * temporal * spatial_h * spatial_w\n"
        "            while latent_elems(length) > max_latent_env and length > 1:\n"
        "                candidate = max(1, ((length - 5) // 4) * 4 + 1)\n"
        "                if candidate == length:\n"
        "                    break\n"
        "                changed_args.append(f\"frames {length}->{candidate}\")\n"
        "                length = candidate\n"
        "            if changed_args:\n"
        "                print(\"[WanImageToVideo] WAN_SAFE_LIMIT_APPLIED: \" + \", \".join(changed_args))\n"
    )
    if needle not in text:
        raise RuntimeError("Unable to locate WanImageToVideo.execute")
    if MARKER in text:
        return text
    return text.replace(needle, needle + insert, 1)


def main() -> None:
    if not WAN_NODE_PATH.exists():
        raise SystemExit(f"Missing {WAN_NODE_PATH}")
    text = WAN_NODE_PATH.read_text()
    if ensure_marker(text):
        return
    text = add_import_os(text)
    text = patch_execute(text)
    WAN_NODE_PATH.write_text(text)


if __name__ == "__main__":
    main()
