#!/usr/bin/env python3
"""
WAN 2.2 Models Download Script
Downloads all essential WAN 2.2 models for AMD Strix Halo Image & Video Toolbox
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import json
from urllib.parse import urljoin

class WANModelDownloader:
    def __init__(self, base_dir="/home/matthewh"):
        self.base_dir = Path(base_dir)
        self.comfyui_models = self.base_dir / "comfy-models"
        self.wan_lightning = self.base_dir / "Wan2.2-Lightning"
        self.cache_dir = self.base_dir / ".cache/huggingface/hub"

        # Essential WAN 2.2 models
        self.models = {
            # Core T2V models (working)
            "wan2.2_t2v_a14b": {
                "repo": "Wan-AI/Wan2.2-T2V-A14B",
                "files": [
                    "model.safetensors",
                    "config.json",
                    "diffusion_pytorch_model.bin"
                ],
                "target_dir": self.comfyui_models / "diffusion_models",
                "priority": "high"
            },

            # Official ComfyUI repackaged (recommended)
            "wan_comfyui_repack": {
                "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "files": [
                    "wan2.2_i2v_14b.safetensors",
                    "wan2.2_t2v_14b.safetensors",
                    "wan2.2_vae.safetensors"
                ],
                "target_dir": self.comfyui_models / "diffusion_models",
                "priority": "high"
            },

            # Lightning LoRAs (enhanced performance)
            "wan_lightning_loras": {
                "repo": "lightx2v/Wan2.2-Lightning",
                "files": [
                    "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
                ],
                "target_dir": self.cache_dir,
                "priority": "medium"
            },

            # Text-to-Image to Video
            "wan_ti2v_5b": {
                "repo": "Wan-AI/Wan2.2-TI2V-5B",
                "files": [
                    "model.safetensors",
                    "config.json"
                ],
                "target_dir": self.comfyui_models / "diffusion_models",
                "priority": "medium"
            },

            # Animation models
            "wan_animate": {
                "repo": "Wan-AI/Wan2.2-Animate-14B",
                "files": [
                    "model.safetensors",
                    "config.json"
                ],
                "target_dir": self.comfyui_models / "diffusion_models",
                "priority": "low"
            }
        }

    def check_existing_models(self):
        """Check what models are already downloaded"""
        print("🔍 Checking existing WAN models...")

        existing_files = []

        # Check ComfyUI models
        if self.comfyui_models.exists():
            for pattern in ["*wan2.2*", "*WAN2.2*", "*wan_2.2*"]:
                existing_files.extend(self.comfyui_models.rglob(pattern))

        # Check WAN Lightning directory
        if self.wan_lightning.exists():
            existing_files.extend(self.wan_lightning.rglob("*.safetensors"))

        # Check HuggingFace cache
        if self.cache_dir.exists():
            for pattern in ["*wan*", "*Wan*"]:
                existing_files.extend(self.cache_dir.rglob(pattern))

        existing_size = sum(f.stat().st_size for f in existing_files if f.is_file())
        print(f"📊 Found {len(existing_files)} existing WAN files")
        print(f"💾 Total size: {existing_size / 1024**3:.1f} GB")

        return existing_files

    def download_huggingface_model(self, repo, files, target_dir, priority="high"):
        """Download specific files from a HuggingFace repository"""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"⬇️  Downloading {repo} (priority: {priority})")

        downloaded_files = []
        for file_path in files:
            target_file = target_dir / file_path.split('/')[-1]

            # Skip if already exists and is reasonable size
            if target_file.exists() and target_file.stat().st_size > 1000000:  # > 1MB
                print(f"  ✅ {file_path} already exists")
                downloaded_files.append(target_file)
                continue

            try:
                url = f"https://huggingface.co/{repo}/resolve/main/{file_path}"

                print(f"  📥 Downloading {file_path}...")

                # Use curl for large file downloads with resume
                curl_cmd = [
                    "curl", "-L", "-o", str(target_file),
                    "--connect-timeout", "30",
                    "--max-time", "3600",  # 1 hour max per file
                    "--retry", "3",
                    "--retry-delay", "5",
                    url
                ]

                result = subprocess.run(curl_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    size = target_file.stat().st_size
                    print(f"  ✅ {file_path} ({size / 1024**2:.1f} MB)")
                    downloaded_files.append(target_file)
                else:
                    print(f"  ❌ Failed to download {file_path}: {result.stderr}")
                    if target_file.exists():
                        target_file.unlink()  # Remove partial download

            except Exception as e:
                print(f"  ❌ Error downloading {file_path}: {e}")

        return downloaded_files

    def install_huggingface_cli(self):
        """Install huggingface_hub if not available"""
        try:
            import huggingface_hub
            print("✅ huggingface_hub already installed")
            return True
        except ImportError:
            print("📦 Installing huggingface_hub...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "huggingface_hub[hf_transfer]"
            ], check=True)
            return True

    def download_with_huggingface_cli(self, repo, target_dir, include_patterns=None):
        """Download using huggingface-cli for better compatibility"""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"⬇️  Downloading {repo} using huggingface-cli...")

        try:
            cmd = [
                "huggingface-cli", "download", repo,
                "--local-dir", str(target_dir),
                "--resume-download",
                "--include", "*.safetensors",
                "--include", "*.json",
                "--include", "*.bin"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

            if result.returncode == 0:
                # Count downloaded files
                safetensors_files = list(target_dir.glob("*.safetensors"))
                total_size = sum(f.stat().st_size for f in safetensors_files)

                print(f"  ✅ Downloaded {len(safetensors_files)} model files")
                print(f"  💾 Total size: {total_size / 1024**3:.1f} GB")
                return True
            else:
                print(f"  ❌ huggingface-cli failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"  ⏰ Download timed out for {repo}")
            return False
        except Exception as e:
            print(f"  ❌ Error downloading {repo}: {e}")
            return False

    def create_symlinks(self):
        """Create symlinks for ComfyUI model discovery"""
        print("🔗 Creating ComfyUI model symlinks...")

        comfyui_diffusion = self.comfyui_models / "diffusion_models"
        comfyui_diffusion.mkdir(parents=True, exist_ok=True)

        # Create symlinks for commonly expected model names
        model_mappings = [
            # High/low noise model symlinks
            (comfyui_diffusion / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
             "high_noise_model"),
            (comfyui_diffusion / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
             "low_noise_model"),
        ]

        for model_file, link_name in model_mappings:
            link_path = comfyui_diffusion / link_name
            if model_file.exists() and not link_path.exists():
                try:
                    link_path.symlink_to(model_file.name)
                    print(f"  ✅ Created symlink: {link_name} -> {model_file.name}")
                except OSError as e:
                    print(f"  ⚠️  Could not create symlink {link_name}: {e}")

    def verify_downloads(self):
        """Verify downloaded models have reasonable file sizes"""
        print("✅ Verifying downloaded models...")

        all_model_files = []
        for pattern in ["*.safetensors", "*.bin"]:
            all_model_files.extend(self.comfyui_models.rglob(pattern))

        valid_models = []
        for model_file in all_model_files:
            size = model_file.stat().st_size
            if size > 100000000:  # > 100MB for real models
                valid_models.append(model_file)
                print(f"  ✅ {model_file.name} ({size / 1024**3:.1f} GB)")
            else:
                print(f"  ⚠️  {model_file.name} too small: {size} bytes")

        return len(valid_models)

    def run(self):
        """Main download process"""
        print("🚀 WAN 2.2 Models Download Script")
        print("=" * 50)

        # Check existing models
        existing = self.check_existing_models()

        # Install dependencies
        self.install_huggingface_cli()

        # Download models by priority
        high_priority_models = {k: v for k, v in self.models.items() if v["priority"] == "high"}
        medium_priority_models = {k: v for k, v in self.models.items() if v["priority"] == "medium"}
        low_priority_models = {k: v for k, v in self.models.items() if v["priority"] == "low"}

        # Download high priority models first
        print("\n🔥 Downloading HIGH PRIORITY models...")
        for model_id, model_info in high_priority_models.items():
            self.download_with_huggingface_cli(
                model_info["repo"],
                model_info["target_dir"]
            )

        # Ask user about medium/low priority
        print("\n❓ Download medium/low priority models? (y/N): ", end="")
        try:
            response = input().strip().lower()
            if response in ['y', 'yes']:
                print("\n📥 Downloading MEDIUM PRIORITY models...")
                for model_id, model_info in medium_priority_models.items():
                    self.download_with_huggingface_cli(
                        model_info["repo"],
                        model_info["target_dir"]
                    )

                print("\n📥 Downloading LOW PRIORITY models...")
                for model_id, model_info in low_priority_models.items():
                    self.download_with_huggingface_cli(
                        model_info["repo"],
                        model_info["target_dir"]
                    )
        except KeyboardInterrupt:
            print("\n⚠️  Skipping additional downloads")

        # Create symlinks
        self.create_symlinks()

        # Final verification
        valid_count = self.verify_downloads()

        print(f"\n🎉 Download completed!")
        print(f"📊 Successfully verified {valid_count} models")
        print(f"📁 Models located in: {self.comfyui_models}")

def main():
    downloader = WANModelDownloader()
    downloader.run()

if __name__ == "__main__":
    main()