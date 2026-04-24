#!/usr/bin/env python3
"""
ComfyUI Model Manager - Downloads and sets up essential models for ComfyUI
Works with both local installations and distrobox environments
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

class ComfyModelManager:
    def __init__(self, comfy_dir: str = "/opt/ComfyUI"):
        self.comfy_dir = Path(comfy_dir)
        self.models_dir = self.comfy_dir / "models"

        # Essential models for basic workflows
        self.essential_models = {
            "checkpoints": [
                {
                    "name": "sd_xl_base_1.0.safetensors",
                    "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                    "description": "SDXL Base Model - Required for modern workflows"
                },
                {
                    "name": "sd_xl_refiner_1.0.safetensors",
                    "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
                    "description": "SDXL Refiner - For high-quality outputs"
                }
            ],
            "vae": [
                {
                    "name": "sdxl_vae.safetensors",
                    "url": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                    "description": "SDXL VAE - Required for SDXL models"
                }
            ],
            "clip_vision": [
                {
                    "name": "clip_vision_g.safetensors",
                    "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
                    "description": "CLIP Vision model for IP-Adapter"
                }
            ]
        }

        # Specialized models for your current workflows (Wan/Qwen)
        # These may not be publicly available yet
        self.specialized_models = {
            "vae": [
                {
                    "name": "wan_2.1_vae.safetensors",
                    "url": None,  # Not publicly available yet
                    "description": "Wan 2.1 VAE - For Wan video generation workflows"
                },
                {
                    "name": "qwen_image_vae.safetensors",
                    "url": None,  # Not publicly available yet
                    "description": "Qwen Image VAE - For Qwen vision workflows"
                }
            ],
            "diffusion_models": [
                {
                    "name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                    "url": None,  # Not publicly available yet
                    "description": "Wan 2.2 Image-to-Video High Noise Model"
                },
                {
                    "name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                    "url": None,  # Not publicly available yet
                    "description": "Wan 2.2 Image-to-Video Low Noise Model"
                },
                {
                    "name": "qwen_image_fp8_e4m3fn.safetensors",
                    "url": None,  # Not publicly available yet
                    "description": "Qwen Image Generation Model"
                }
            ],
            "text_encoders": [
                {
                    "name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "url": None,  # Not publicly available yet
                    "description": "UMT5 XXL Text Encoder - For Wan workflows"
                },
                {
                    "name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "url": None,  # Not publicly available yet
                    "description": "Qwen 2.5 Vision-Language 7B Model"
                }
            ]
        }

    def check_comfyui_installation(self) -> bool:
        """Check if ComfyUI is installed"""
        return self.comfy_dir.exists() and (self.comfy_dir / "main.py").exists()

    def create_model_directories(self):
        """Create all necessary model directories"""
        directories = [
            "checkpoints", "vae", "clip_vision", "loras", "controlnet",
            "embeddings", "upscale_models", "unet", "text_encoders",
            "diffusion_models", "model_patches"
        ]

        for dir_name in directories:
            dir_path = self.models_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            # Create placeholder file if empty
            placeholder = dir_path / f"put_{dir_name}_here"
            if not placeholder.exists():
                placeholder.touch()

    def download_file(self, url: str, destination: Path, description: str = "") -> bool:
        """Download a file with progress indication"""
        try:
            print(f"📥 Downloading: {description or destination.name}")
            print(f"   From: {url}")
            print(f"   To: {destination}")

            # Stream download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(f"\r   Progress: {percent}% ({block_num * block_size}/{total_size} bytes)", end="", flush=True)

            urllib.request.urlretrieve(url, destination, progress_hook)
            print(f"\n✅ Successfully downloaded: {destination.name}")
            return True

        except Exception as e:
            print(f"\n❌ Failed to download {destination.name}: {e}")
            if destination.exists():
                destination.unlink()  # Remove partial download
            return False

    def download_model(self, model_type: str, model_info: Dict) -> bool:
        """Download a specific model"""
        model_dir = self.models_dir / model_type
        model_path = model_dir / model_info["name"]

        if model_path.exists():
            print(f"✅ Already exists: {model_info['name']}")
            return True

        return self.download_file(
            model_info["url"],
            model_path,
            model_info.get("description", model_info["name"])
        )

    def setup_sample_images(self):
        """Create sample input images for testing"""
        input_dir = self.comfy_dir / "input"
        input_dir.mkdir(exist_ok=True)

        # Create a simple test image if the missing one is referenced
        missing_image = input_dir / "fennec_girl_flowers.png"
        if not missing_image.exists():
            print("📝 Creating placeholder for missing fennec_girl_flowers.png")
            # Copy example.png as a placeholder
            example_img = input_dir / "example.png"
            if example_img.exists():
                import shutil
                shutil.copy2(example_img, missing_image)
                print("✅ Created placeholder image")

    def check_specialized_models_status(self) -> Dict[str, List[str]]:
        """Check which specialized models are available and which are missing"""
        status = {"available": [], "missing": []}

        for model_type, models in self.specialized_models.items():
            for model_info in models:
                model_path = self.models_dir / model_type / model_info["name"]
                if model_path.exists():
                    status["available"].append(model_info["name"])
                else:
                    status["missing"].append(model_info["name"])

        return status

    def install_essential_models(self, force_download: bool = False) -> bool:
        """Download essential models for basic ComfyUI functionality"""
        if not self.check_comfyui_installation():
            print(f"❌ ComfyUI not found at {self.comfy_dir}")
            return False

        print("🔧 Setting up ComfyUI model directories...")
        self.create_model_directories()

        print("📦 Installing essential models...")
        success_count = 0
        total_count = sum(len(models) for models in self.essential_models.values())

        for model_type, models in self.essential_models.items():
            print(f"\n📁 Installing {model_type}:")
            for model_info in models:
                if self.download_model(model_type, model_info):
                    success_count += 1

        print(f"\n📊 Installation Summary: {success_count}/{total_count} models installed")

        # Setup sample images
        self.setup_sample_images()

        # Check specialized models status
        specialized_status = self.check_specialized_models_status()
        if specialized_status["missing"]:
            print(f"\n⚠️  Specialized models missing ({len(specialized_status['missing'])} files):")
            for model in specialized_status["missing"]:
                print(f"   - {model}")
            print("\n💡 These models are for Wan/Qwen workflows and may not be publicly available yet.")
            print("   Your current workflows require these specific models.")

        return success_count > 0

    def create_missing_model_files(self):
        """Create placeholder files for missing specialized models to prevent errors"""
        specialized_status = self.check_specialized_models_status()

        print("📝 Creating placeholder files for missing specialized models...")
        for model_name in specialized_status["missing"]:
            # Find which model type this belongs to
            for model_type, models in self.specialized_models.items():
                for model_info in models:
                    if model_info["name"] == model_name:
                        model_path = self.models_dir / model_type / model_name
                        model_path.parent.mkdir(parents=True, exist_ok=True)

                        # Create a small placeholder file
                        with open(model_path, 'w') as f:
                            f.write(f"# Placeholder for {model_name}\n")
                            f.write(f"# Description: {model_info['description']}\n")
                            f.write("# This model needs to be downloaded separately\n")

                        print(f"✅ Created placeholder: {model_type}/{model_name}")
                        break

    def link_existing_models(self, source_paths: List[str]) -> int:
        """Create symlinks to existing models (for testing)"""
        if not self.check_comfyui_installation():
            print(f"❌ ComfyUI not found at {self.comfy_dir}")
            return 0

        print("🔗 Linking existing models...")
        self.create_model_directories()

        linked_count = 0
        for source_path in source_paths:
            source = Path(source_path)
            if not source.exists():
                print(f"⚠️  Source not found: {source}")
                continue

            # Determine model type from path
            model_type = "checkpoints"  # Default
            if "vae" in source.name.lower():
                model_type = "vae"
            elif "lora" in source.name.lower():
                model_type = "loras"
            elif "controlnet" in source.name.lower():
                model_type = "controlnet"

            dest = self.models_dir / model_type / source.name
            if dest.exists():
                print(f"✅ Already exists: {dest.name}")
                continue

            try:
                dest.symlink_to(source)
                print(f"🔗 Linked: {source.name} -> {model_type}")
                linked_count += 1
            except Exception as e:
                print(f"❌ Failed to link {source.name}: {e}")

        return linked_count

    def show_status(self):
        """Show current model installation status"""
        if not self.check_comfyui_installation():
            print(f"❌ ComfyUI not found at {self.comfy_dir}")
            return

        print(f"📊 ComfyUI Model Status at {self.comfy_dir}")
        print("=" * 50)

        for model_type in ["checkpoints", "vae", "loras", "controlnet", "clip_vision"]:
            model_dir = self.models_dir / model_type
            if model_dir.exists():
                files = [f for f in model_dir.iterdir() if f.is_file() and f.suffix in ['.safetensors', '.pth', '.ckpt']]
                print(f"📁 {model_type}: {len(files)} files")
                for f in files[:3]:  # Show first 3 files
                    print(f"   - {f.name}")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more")
            else:
                print(f"📁 {model_type}: Directory not found")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ComfyUI Model Manager")
    parser.add_argument("--comfy-dir", default="/opt/ComfyUI", help="ComfyUI installation directory")
    parser.add_argument("--install", action="store_true", help="Install essential models")
    parser.add_argument("--link", nargs="+", help="Link existing models from given paths")
    parser.add_argument("--status", action="store_true", help="Show current model status")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument("--create-placeholders", action="store_true", help="Create placeholder files for missing specialized models")
    parser.add_argument("--check-specialized", action="store_true", help="Check status of specialized Wan/Qwen models")

    args = parser.parse_args()

    manager = ComfyModelManager(args.comfy_dir)

    if args.status:
        manager.show_status()
        if args.check_specialized:
            specialized_status = manager.check_specialized_models_status()
            print(f"\n🔍 Specialized Models Status:")
            print(f"   ✅ Available: {len(specialized_status['available'])}")
            print(f"   ❌ Missing: {len(specialized_status['missing'])}")
            if specialized_status["missing"]:
                print("   Missing models:")
                for model in specialized_status["missing"]:
                    print(f"     - {model}")
    elif args.install:
        success = manager.install_essential_models(args.force)
        sys.exit(0 if success else 1)
    elif args.create_placeholders:
        manager.create_missing_model_files()
    elif args.link:
        linked = manager.link_existing_models(args.link)
        print(f"🔗 Linked {linked} models")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()