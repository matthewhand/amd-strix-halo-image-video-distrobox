#!/usr/bin/env python3
"""Patch ComfyUI-LTXVideo for kornia versions that dropped pyramid.pad.

New kornia removed `pad` from kornia.geometry.transform.pyramid, which
makes `ComfyUI-LTXVideo` fail to import entirely (all LTX nodes missing).
This restores a small pad helper and is safe to re-run (idempotent).
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT = Path("/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py")

PAD_HELPER = '''
def pad(img, padding, mode="constant", value=0.0):
    """Compat shim for older kornia.geometry.transform.pyramid.pad."""
    # padding is (left, right, top, bottom) for spatial dims — same as F.pad NCHW
    if mode == "reflect":
        return F.pad(img, padding, mode="reflect")
    return F.pad(img, padding, mode=mode, value=value)

'''


def patch(path: Path) -> bool:
    if not path.is_file():
        print(f"[patch_ltxvideo_kornia] skip missing {path}", file=sys.stderr)
        return False
    text = path.read_text(encoding="utf-8")
    if "def pad(img, padding" in text:
        print(f"[patch_ltxvideo_kornia] already patched {path}", file=sys.stderr)
        return True
    if "    pad,\n" not in text and "pad," not in text:
        print(f"[patch_ltxvideo_kornia] no pad import in {path}", file=sys.stderr)
        return False
    # Drop pad from kornia import list
    text2 = text.replace("    pad,\n", "")
    text2 = text2.replace(",\n    pad\n", "\n")
    text2 = text2.replace("pad,\n", "")
    # Insert helper after `from torch import Tensor` or after F import
    anchor = "from torch import Tensor\n"
    if anchor in text2 and "def pad(img, padding" not in text2:
        text2 = text2.replace(anchor, anchor + PAD_HELPER, 1)
    else:
        text2 = text2.replace(
            "import torch.nn.functional as F\n",
            "import torch.nn.functional as F\n" + PAD_HELPER,
            1,
        )
    path.write_text(text2, encoding="utf-8")
    print(f"[patch_ltxvideo_kornia] patched {path}", file=sys.stderr)
    return True


def main(argv=None) -> int:
    path = Path(argv[1]) if argv and len(argv) > 1 else DEFAULT
    ok = patch(path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
