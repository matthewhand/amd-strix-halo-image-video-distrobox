#!/usr/bin/env python3
"""
Patched Qwen Image Studio launcher
Applies necessary patches before starting the web UI
"""
import sys
import os
import uvicorn

# Set the environment variable to use our patched CLI runner
# Use a dynamic path based on the current script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATCHED_CLI_PATH = os.path.join(SCRIPT_DIR, "patched_cli_runner.py")
os.environ["QIM_CLI_PATH"] = PATCHED_CLI_PATH

# Add Qwen path
sys.path.insert(0, '/opt/qwen-image-studio/src')
# Add the main app directory to the path as well
sys.path.insert(0, '/opt/qwen-image-studio/qwen-image-studio')

def apply_comprehensive_patches():
    """Apply both pipeline and model-level patches"""
    patches_applied = 0
    
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

    print(f"🔧 Applied {patches_applied}/2 patches successfully")
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
        return

    # Change to the correct directory for Qwen Image Studio
    import os
    os.chdir('/opt/qwen-image-studio')

    # Start the web UI by importing and running the app directly
    print("\n🚀 Starting Qwen Image Studio web UI...")
    print("📡 Access at: http://localhost:8000")
    print("=" * 60)

    try:
        # Import the app *after* patches are applied and directory is changed
        from qwen_image_studio.server import app
        # Run uvicorn directly with the imported app to avoid module resolution issues
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()