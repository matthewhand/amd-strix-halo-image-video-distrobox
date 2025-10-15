#!/usr/bin/env python3
"""
Patched CLI runner for Qwen Image Studio.
Applies necessary patches before running the CLI.
"""
import sys
from typing import Tuple


def _check_hip_ready() -> Tuple[bool, str]:
    """
    Verify that a HIP-capable device is visible before importing ROCm modules.
    Returns (ready, message).
    """
    try:
        import torch
    except ImportError as exc:
        return False, f"PyTorch unavailable: {exc}"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned False (no HIP device)"

    try:
        torch.cuda.current_device()  # triggers ROCm initialisation
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"HIP runtime failed to initialise: {exc}"

    return True, ""


def apply_comprehensive_patches():
    """Apply both pipeline and model-level patches"""
    patches_applied = 0

    hip_ready, hip_msg = _check_hip_ready()
    if not hip_ready:
        print(f"⚠️  CLI Runner: Skipping compatibility patches – {hip_msg}")
        return False
    
    # Patch 1: Pipeline-level patch for diffusers
    try:
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline
        original_from_pretrained = DiffusionPipeline.from_pretrained

        @classmethod
        def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            if 'offload_state_dict' in kwargs:
                kwargs.pop('offload_state_dict')
                print(f"[PIPELINE PATCH] Removed offload_state_dict")
            return original_from_pretrained(pretrained_model_name_or_path, **kwargs)

        DiffusionPipeline.from_pretrained = patched_from_pretrained
        print("✅ CLI Runner: Pipeline-level patch applied")
        patches_applied += 1
    except Exception as e:
        print(f"❌ CLI Runner: Pipeline patch failed: {e}")

    # Patch 2: Model-level patch for Qwen2_5_VLForConditionalGeneration
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        original_init = Qwen2_5_VLForConditionalGeneration.__init__

        def patched_init(self, config, *args, **kwargs):
            if 'offload_state_dict' in kwargs:
                kwargs.pop('offload_state_dict')
                print(f"[MODEL PATCH] Removed offload_state_dict from Qwen2_5_VLForConditionalGeneration.__init__")
            original_init(self, config, *args, **kwargs)

        Qwen2_5_VLForConditionalGeneration.__init__ = patched_init
        print("✅ CLI Runner: Model-level patch applied")
        patches_applied += 1
    except ImportError:
        print("⚠️  CLI Runner: Qwen model not available for direct patching")
    except Exception as e:
        print(f"❌ CLI Runner: Model patch failed: {e}")

    print(f"🔧 CLI Runner: Applied {patches_applied}/2 patches successfully")
    return patches_applied > 0

if __name__ == "__main__":
    print("🔧 CLI Runner: Applying patches...")
    if not apply_comprehensive_patches():
        print("💥 CLI Runner: Aborting. Ensure ROCm/HIP devices are accessible before running Qwen.")
        sys.exit(1)
    
    print("🚀 CLI Runner: Starting original CLI...")
    # Add the original CLI's path and run its main function
    sys.path.insert(0, '/opt/qwen-image-studio/src')
    from qwen_image_mps import cli
    cli.main()
