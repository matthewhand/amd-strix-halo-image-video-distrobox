#!/usr/bin/env python3
"""
Direct Qwen Image Studio launcher with module-level patches.
Applies patches at the module level to ensure they're loaded before any imports.
"""
import os
import sys
from pathlib import Path

# Add Qwen paths
sys.path.insert(0, '/opt/qwen-image-studio/src')
sys.path.insert(0, '/opt/qwen-image-studio/qwen-image-studio')
sys.path.insert(0, '/opt/qwen-image-studio')


def apply_module_level_patches() -> None:
    """Apply patches at the module level to ensure they're loaded first."""
    import builtins

    original_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)

        if name == 'diffusers.pipelines.pipeline_utils' and hasattr(module, 'DiffusionPipeline'):
            original_from_pretrained = module.DiffusionPipeline.from_pretrained

            @classmethod
            def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                if 'offload_state_dict' in kwargs:
                    kwargs.pop('offload_state_dict')
                    print("[PIPELINE PATCH] Removed offload_state_dict")
                return original_from_pretrained(pretrained_model_name_or_path, **kwargs)

            module.DiffusionPipeline.from_pretrained = patched_from_pretrained
            print("✅ Pipeline-level patch applied at module level")

        elif name == 'transformers' and hasattr(module, 'Qwen2_5_VLForConditionalGeneration'):
            original_init = module.Qwen2_5_VLForConditionalGeneration.__init__

            def patched_init(self, config, *args, **kwargs):
                if 'offload_state_dict' in kwargs:
                    kwargs.pop('offload_state_dict')
                    print("[MODEL PATCH] Removed offload_state_dict from Qwen2_5_VLForConditionalGeneration")
                original_init(self, config, *args, **kwargs)

            module.Qwen2_5_VLForConditionalGeneration.__init__ = patched_init
            print("✅ Model-level patch applied at module level")

        return module

    builtins.__import__ = patched_import
    print("🔧 Module-level patching system activated")


def main() -> None:
    """Start Qwen Image Studio with module-level patches."""
    print("=" * 60)
    print("🎨 Starting Qwen Image Studio (Direct Module Patches)")
    print("=" * 60)

    print("\n🔧 Applying module-level compatibility patches...")
    apply_module_level_patches()

    os.chdir('/opt/qwen-image-studio')

    print("\n🚀 Starting Qwen Image Studio web UI...")
    print("📡 Access at: http://localhost:8000")
    print("=" * 60)

    try:
        import uvicorn

        uvicorn.run("qwen_image_studio.server:app", host="0.0.0.0", port=8000, reload=False)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Failed to start server: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

