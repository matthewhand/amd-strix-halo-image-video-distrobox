#!/usr/bin/env python3
"""
Polished Qwen Image Generation Test - Final Version
Working version with correct output directory checking
"""
import sys
import os
import time
import glob
from pathlib import Path

# Add Qwen path
sys.path.insert(0, '/opt/qwen-image-studio/src')

def apply_comprehensive_patches():
    """Apply both pipeline and model-level patches"""
    
    patches_applied = 0
    
    # Patch 1: Pipeline-level patch
    try:
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline
        original_from_pretrained = DiffusionPipeline.from_pretrained

        @classmethod
        def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            # Remove problematic parameters that cause issues
            problematic_params = ['offload_state_dict']
            removed_params = []
            
            for param in problematic_params:
                if param in kwargs:
                    val = kwargs.pop(param)
                    removed_params.append(f"{param}={val}")
            
            if removed_params:
                print(f"[PIPELINE PATCH] Removed: {', '.join(removed_params)}")
            
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
            # Remove offload_state_dict if present
            if 'offload_state_dict' in kwargs:
                val = kwargs.pop('offload_state_dict')
                print(f"[MODEL PATCH] Removed offload_state_dict={val} from Qwen2_5_VLForConditionalGeneration.__init__")
            return original_init(self, config, *args, **kwargs)

        Qwen2_5_VLForConditionalGeneration.__init__ = patched_init
        print("✅ Model-level patch applied")
        patches_applied += 1
    except ImportError:
        print("⚠️  Qwen model not available for direct patching")
    except Exception as e:
        print(f"❌ Model patch failed: {e}")

    print(f"🔧 Applied {patches_applied}/2 patches successfully")
    return patches_applied > 0

def check_environment():
    """Check GPU and environment setup"""
    
    print("=== Environment Check ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available - cannot proceed")
            return False
            
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test GPU with simple operation
        print("\nTesting GPU operation...")
        x = torch.randn(100, 100, device="cuda", dtype=torch.float16)
        y = torch.mm(x, x.T)
        print("✅ GPU test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        return False

def run_4_step_generation():
    """Run optimized 4-step Qwen image generation"""
    
    print("\n=== 4-Step Qwen Generation ===")
    
    # Optimized parameters for 4-step generation
    class Args:
        prompt = "a majestic dragon perched on a crystal mountain, fantasy art, detailed, cinematic lighting"
        steps = 4  # Ultra-fast generation
        num_images = 1
        size = "16:9" # Widescreen format
        ultra_fast = True  # Use Lightning LoRA for speed
        model = "Qwen/Qwen-Image"
        no_mmap = True
        lora = None
        edit = False
        input_image = None
        output_dir = "/tmp"  # This will be overridden by Qwen's default output dir
        seed = 42 # Reproducible results
        guidance_scale = 1.0  # Optimal for ultra-fast mode
        negative_prompt = "blurry, low quality, distorted, watermark"
        batman = False
        fast = False
        targets = "all"

    args = Args()
    
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"CFG Scale: {args.guidance_scale}")
    print(f"Size: {args.size}")
    print(f"Seed: {args.seed}")
    
    try:
        from qwen_image_mps.cli import generate_image
        
        print("\n🚀 Starting generation...")
        start_time = time.time()
        
        generate_image(args)
        
        elapsed = time.time() - start_time
        print(f"✅ Generation completed in {elapsed:.1f} seconds")
        
        # Check the correct output directory (Qwen's default)
        qwen_output_dir = os.path.expanduser("~/.qwen-image-studio")
        output_files = glob.glob(os.path.join(qwen_output_dir, "*.png"))
        
        if output_files:
            # Get the most recent file
            latest_file = max(output_files, key=os.path.getmtime)
            size = os.path.getsize(latest_file)
            print(f"✅ Generated image: {latest_file} ({size:,} bytes)")
            
            # Verify image is valid
            try:
                from PIL import Image
                img = Image.open(latest_file)
                print(f"✅ Image dimensions: {img.size[0]}x{img.size[1]}")
                print(f"✅ Image format: {img.format}")
                print(f"✅ Image mode: {img.mode}")
                
                # Also copy to /tmp for easier access
                import shutil
                tmp_file = f"/tmp/qwen_test_output.png"
                shutil.copy2(latest_file, tmp_file)
                print(f"✅ Image copied to: {tmp_file}")
                
                return True
            except Exception as e:
                print(f"❌ Invalid image file: {e}")
                return False
        else:
            print(f"❌ No output image found in {qwen_output_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    
    print("=" * 60)
    print("🎨 Polished Qwen Image Generation Test - Final")
    print("AMD Strix Halo (gfx1151) - 4-Step Ultra-Fast Generation")
    print("=" * 60)
    
    # Step 1: Environment check
    if not check_environment():
        print("\n💥 Environment check failed - cannot continue")
        return False
    
    # Step 2: Apply patches
    print("\n🔧 Applying comprehensive patches...")
    if not apply_comprehensive_patches():
        print("\n💥 Failed to apply patches - cannot continue")
        return False
    
    # Step 3: Run generation
    print("\n🎯 Running 4-step image generation...")
    success = run_4_step_generation()
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS! Qwen image generation is working!")
        print("\n📁 Check ~/.qwen-image-studio/ for the generated image")
        print("💡 You can now use Qwen Image Studio web interface")
        print("\nTo start the web UI:")
        print("  start_qwen_studio")
        print("\nTo run this test again:")
        print("  python3 polished_qwen_test_final.py")
    else:
        print("💥 Test failed - check the errors above")
        print("\n🔍 Troubleshooting:")
        print("1. Ensure ROCm is properly installed")
        print("2. Check GPU device permissions")
        print("3. Verify model files are downloaded")
        print("4. Try with more stable settings (SDPA instead of Triton)")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)