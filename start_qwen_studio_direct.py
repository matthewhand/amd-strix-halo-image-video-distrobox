#!/usr/bin/env python3
"""
Direct Qwen Image Studio launcher with module-level patches
Applies patches at the module level to ensure they're loaded before any imports
"""
import sys
import os

# Add Qwen path
sys.path.insert(0, '/opt/qwen-image-studio/src')
sys.path.insert(0, '/opt/qwen-image-studio/qwen-image-studio')
sys.path.insert(0, '/opt/qwen-image-studio')

# Apply patches BEFORE any imports
def apply_module_level_patches():
    """Apply patches at the module level to ensure they're loaded first"""
    
    # Patch 1: Monkey patch diffusers before it's imported
    import builtins
    original_import = builtins.__import__
    
    def patched_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        
        # Patch diffusers when it's imported
        if name == 'diffusers.pipelines.pipeline_utils':
            if hasattr(module, 'DiffusionPipeline'):
                original_from_pretrained = module.DiffusionPipeline.from_pretrained
                
                @classmethod
                def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                    if 'offload_state_dict' in kwargs:
                        kwargs.pop('offload_state_dict')
                        print("[PIPELINE PATCH] Removed offload_state_dict")
                    return original_from_pretrained(pretrained_model_name_or_path, **kwargs)
                
                module.DiffusionPipeline.from_pretrained = patched_from_pretrained
                print("✅ Pipeline-level patch applied at module level")
        
        # Patch transformers when it's imported
        elif name == 'transformers':
            if hasattr(module, 'Qwen2_5_VLForConditionalGeneration'):
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

def main():
    """Start Qwen Image Studio with module-level patches"""
    
    print("=" * 60)
    print("🎨 Starting Qwen Image Studio (Direct Module Patches)")
    print("=" * 60)
    
    # Apply module-level patches
    print("\n🔧 Applying module-level compatibility patches...")
    apply_module_level_patches()
    
    # Change to the correct directory
    os.chdir('/opt/qwen-image-studio')
    
    # Start the web UI directly with uvicorn
    print("\n🚀 Starting Qwen Image Studio web UI...")
    print("📡 Access at: http://localhost:8000")
    print("=" * 60)
    
    try:
        import uvicorn
        # Use the import string format without reload to avoid process issues
        uvicorn.run("qwen_image_studio.server:app", host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()