#!/usr/bin/env python3
"""
Apply (or revert) ROCm compatibility patches to ComfyUI files.

Each patch works around a specific ROCm kernel crash or missing feature.
Run inside the container where /opt/ComfyUI exists.

Usage:
    python patch_comfyui.py --list              # show available patches
    python patch_comfyui.py --all               # apply all patches
    python patch_comfyui.py gemma-loader        # apply one patch
    python patch_comfyui.py gemma-loader --revert
"""
import argparse
import os
import re
import sys


# --- Patch definitions ---

def patch_gemma_loader(revert=False):
    """Force Gemma CLIP encoder to CPU to avoid ROCm kernel crash."""
    path = "/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/gemma_encoder.py"
    original = "return (comfy.sd.CLIP(clip_target),)"
    patched = """\
        clip_obj = comfy.sd.CLIP(clip_target)
        # Force CPU for Gemma to avoid ROCm kernel crash
        clip_obj.patcher.load_device = torch.device("cpu")
        clip_obj.patcher.offload_device = torch.device("cpu")
        return (clip_obj,)"""
    marker = "Force CPU for Gemma"
    return _apply_or_revert(path, original, patched, marker, revert)


def patch_audio_vae(revert=False):
    """Stub torchaudio import in ComfyUI's audio VAE to prevent crash on ROCm."""
    path = "/opt/ComfyUI/comfy/ldm/lightricks/vae/audio_vae.py"
    original = "import torchaudio"
    patched = """\
try:
    import torchaudio
except (OSError, ImportError):
    class DummyTorchaudio:
        class functional:
            @staticmethod
            def resample(*a, **kw): raise NotImplementedError("torchaudio missing")
        class transforms:
            class MelSpectrogram:
                def __init__(self, *a, **kw): pass
                def to(self, *a, **kw): return self
                def __call__(self, *a, **kw): raise NotImplementedError("torchaudio missing")
    torchaudio = DummyTorchaudio()"""
    marker = "DummyTorchaudio"
    return _apply_or_revert(path, original, patched, marker, revert)


def patch_gemma_embedding(revert=False):
    """Force Gemma3 embedding layer to CPU to bypass ROCm kernel bug."""
    path = "/opt/venv/lib64/python3.13/site-packages/transformers/models/gemma3/modeling_gemma3.py"
    if not os.path.exists(path):
        path = "/opt/venv/lib/python3.13/site-packages/transformers/models/gemma3/modeling_gemma3.py"
    original = "return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)"
    patched = """\
# CPU offload for embedding to bypass ROCm kernel bug
        weight_cpu = self.weight.cpu()
        input_cpu = input_ids.cpu()
        out = torch.nn.functional.embedding(
            input_cpu, weight_cpu, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        return (out * self.embed_scale.cpu().to(out.dtype)).to(self.weight.device)"""
    marker = "CPU offload for embedding"
    return _apply_or_revert(path, original, patched, marker, revert)


def patch_tokenizer(revert=False):
    """Add fallback tokenizer path resolution for Gemma3 in ComfyUI."""
    path = "/opt/ComfyUI/comfy/text_encoders/lt.py"
    original = (
        'class Gemma3_12BTokenizer(sd1_clip.SDTokenizer):\n'
        '    def __init__(self, embedding_directory=None, tokenizer_data={}):\n'
        '        tokenizer = tokenizer_data.get("spiece_model", None)\n'
        '        super().__init__('
    )
    patched = (
        'class Gemma3_12BTokenizer(sd1_clip.SDTokenizer):\n'
        '    def __init__(self, embedding_directory=None, tokenizer_data={}):\n'
        '        tokenizer = tokenizer_data.get("spiece_model", None)\n'
        '        if tokenizer is None:\n'
        '            import folder_paths\n'
        '            t_path = folder_paths.get_full_path("text_encoders", "gemma_3_12B_it/tokenizer.model")\n'
        '            if t_path and os.path.isfile(t_path):\n'
        '                tokenizer = t_path\n'
        '        super().__init__('
    )
    marker = "folder_paths.get_full_path"
    return _apply_or_revert(path, original, patched, marker, revert)


# --- Registry ---

PATCHES = {
    "gemma-loader": ("Gemma CLIP -> CPU (LTXVideo)", patch_gemma_loader),
    "audio-vae": ("Stub torchaudio import", patch_audio_vae),
    "gemma-embedding": ("Gemma3 embedding -> CPU", patch_gemma_embedding),
    "tokenizer": ("Gemma3 tokenizer fallback path", patch_tokenizer),
}


# --- Helpers ---

def _apply_or_revert(path, original, patched, marker, revert):
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return False

    with open(path) as f:
        content = f.read()

    if revert:
        if marker not in content:
            print(f"  OK: not patched")
            return True
        # Simple revert: replace patched block with original
        if patched in content:
            content = content.replace(patched, original)
            with open(path, "w") as f:
                f.write(content)
            print(f"  REVERTED")
            return True
        print(f"  WARN: marker found but exact patch block not matched — manual revert needed")
        return False
    else:
        if marker in content:
            print(f"  OK: already patched")
            return True
        if original not in content:
            print(f"  SKIP: target string not found (upstream may have changed)")
            return False
        content = content.replace(original, patched)
        with open(path, "w") as f:
            f.write(content)
        print(f"  PATCHED")
        return True


def main():
    parser = argparse.ArgumentParser(description="Apply ROCm patches to ComfyUI")
    parser.add_argument("patch", nargs="?", choices=list(PATCHES.keys()),
                        help="Specific patch to apply")
    parser.add_argument("--all", action="store_true", help="Apply all patches")
    parser.add_argument("--revert", action="store_true", help="Revert instead of apply")
    parser.add_argument("--list", action="store_true", help="List available patches")
    args = parser.parse_args()

    if args.list or (not args.patch and not args.all):
        print("Available patches:")
        for name, (desc, _) in PATCHES.items():
            print(f"  {name:20s}  {desc}")
        return

    targets = PATCHES if args.all else {args.patch: PATCHES[args.patch]}
    action = "Reverting" if args.revert else "Applying"

    for name, (desc, fn) in targets.items():
        print(f"{action}: {name} ({desc})")
        fn(revert=args.revert)


if __name__ == "__main__":
    main()
