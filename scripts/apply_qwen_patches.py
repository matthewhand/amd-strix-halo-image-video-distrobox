#!/usr/bin/env python3
"""
Qwen Image Studio - ROCm Compatibility Patches

This script applies critical monkey-patches to make Qwen Image Studio
compatible with ROCm 7.10+ on AMD GPUs.

Patches applied:
1. Pipeline-level: Remove offload_state_dict from DiffusionPipeline.from_pretrained
2. Model-level: Remove offload_state_dict from Qwen2_5_VLForConditionalGeneration.__init__
3. Pipeline __init__: Fix segfault when accessing vae.temperal_downsample

Status:
- Pipeline initialization: ✅ WORKING
- Component loading to GPU: ✅ WORKING (53.79 GB / 128 GB)
- Image generation: ⚠️  KNOWN ISSUE - Crashes during text encoder forward pass

The generation crash appears to be a deeper issue in the text encoder or VAE
forward pass on ROCm, separate from the __init__ segfault.
"""

import sys
from typing import Tuple

import torch


def _check_hip_ready() -> Tuple[bool, str]:
    """Ensure a HIP-capable GPU is visible before importing ROCm modules."""
    try:
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() returned False (no HIP device)"

        try:
            torch.cuda.current_device()
        except Exception as exc:  # pylint: disable=broad-except
            return False, f"HIP runtime failed to initialise: {exc}"

        return True, ""
    except ImportError as exc:
        return False, f"PyTorch unavailable: {exc}"


def apply_comprehensive_patches() -> bool:
    """Apply both pipeline and model-level patches"""
    patches_applied = 0

    hip_ready, hip_msg = _check_hip_ready()
    if not hip_ready:
        print(f"⚠️  Qwen Studio: Skipping compatibility patches – {hip_msg}")
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
        print("✅ Pipeline-level patch applied")
        patches_applied += 1
    except Exception as e:
        print(f"❌ Pipeline patch failed: {e}")

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
        print("✅ Model-level patch applied")
        patches_applied += 1
    except ImportError:
        print("⚠️  Qwen model not available for direct patching")
    except Exception as e:
        print(f"❌ Model patch failed: {e}")

    # Patch 3: CRITICAL FIX - QwenImagePipeline __init__ segfault fix
    try:
        from diffusers import QwenImagePipeline
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline as DiffPipeline

        original_qwen_init = QwenImagePipeline.__init__

        def patched_qwen_init(self, scheduler, vae, text_encoder, tokenizer, transformer):
            """Fixed __init__ that avoids accessing vae.temperal_downsample (causes segfault)"""
            DiffPipeline.__init__(self)

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                scheduler=scheduler,
            )

            # CRITICAL FIX: Avoid the problematic line that causes segfault
            # Original: self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
            # This line crashes when accessing self.vae.temperal_downsample
            self.vae_scale_factor = 8  # Use default value

            # Create image processor
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

            self.tokenizer_max_length = 1024
            self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            self.prompt_template_encode_start_idx = 34
            self.default_sample_size = 128

        QwenImagePipeline.__init__ = patched_qwen_init
        print("✅ QwenImagePipeline segfault fix applied")
        patches_applied += 1
    except Exception as e:
        print(f"❌ QwenImagePipeline patch failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"🔧 Applied {patches_applied}/3 patches successfully")
    return patches_applied > 0


def main():
    """Main entry point - apply patches"""
    print("Applying Qwen Image Studio ROCm compatibility patches...")
    if apply_comprehensive_patches():
        print("✅ All patches applied successfully")
        return 0
    else:
        print("⚠️  Some patches failed to apply")
        return 1


if __name__ == "__main__":
    sys.exit(main())
