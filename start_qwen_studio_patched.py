#!/usr/bin/env python3
"""
Patched Qwen Image Studio launcher
Applies necessary patches before starting the web UI
"""
import sys
import os
from typing import Tuple

import uvicorn

# Add Qwen path
sys.path.insert(0, '/opt/qwen-image-studio/src')
# Add the main app directory to the path as well
sys.path.insert(0, '/opt/qwen-image-studio/qwen-image-studio')


def _check_hip_ready() -> Tuple[bool, str]:
    """Ensure a HIP-capable GPU is visible before importing ROCm modules."""
    try:
        import torch
    except ImportError as exc:
        return False, f"PyTorch unavailable: {exc}"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned False (no HIP device)"

    try:
        torch.cuda.current_device()
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"HIP runtime failed to initialise: {exc}"

    return True, ""


def apply_comprehensive_patches():
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
    """Start Qwen Image Studio with patches applied"""

    print("=" * 60)
    print("🎨 Starting Qwen Image Studio (Patched)")
    print("=" * 60)

    # Apply patches
    print("\n🔧 Applying compatibility patches...")
    if not apply_comprehensive_patches():
        print("\n💥 Failed to apply patches - cannot continue")
        print("Hint: ensure ROCm/HIP devices are passed through (e.g. /dev/kfd and /dev/dri).")
        sys.exit(1)

    # Change to the correct directory for Qwen Image Studio
    os.chdir('/opt/qwen-image-studio')

    # Start the web UI by importing and running the app directly
    print("\n🚀 Starting Qwen Image Studio web UI...")
    print("📡 Access at: http://localhost:8000")
    print("=" * 60)

    try:
        # Import the app *after* patches are applied and directory is changed
        import importlib.util

        server_path = os.path.join('/opt/qwen-image-studio', 'qwen-image-studio', 'server.py')
        spec = importlib.util.spec_from_file_location("qwen_image_studio_server", server_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to locate Qwen Image Studio server module at {server_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        app = module.app  # FastAPI app defined in server.py

        # Run uvicorn directly with the imported app to avoid module resolution issues
        uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
