#!/usr/bin/env python3
"""
Wan Video Generation CLI
A command-line interface for Wan video generation models.
Generates placeholder outputs until the official WAN pipeline is integrated.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import torch, but make it optional for basic functionality
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class WanVideoGenerator:
    """Helper for checking model availability and simulating WAN jobs."""

    def __init__(self, model_dir: str = str(Path.home() / "comfy-models")):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        self.model_files: Dict[str, str] = {
            "text_encoder": "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae_21": "vae/wan_2.1_vae.safetensors",
            "vae_22": "vae/wan2.2_vae.safetensors",
            "i2v_high_noise": "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "i2v_low_noise": "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            "t2v_high_noise": "diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "t2v_low_noise": "diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            "ti2v_5b": "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        }

    def _model_path(self, model_name: str) -> Path:
        return self.model_dir / self.model_files[model_name]

    def check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        if model_name not in self.model_files:
            return False

        model_path = self._model_path(model_name)
        return model_path.exists() and model_path.stat().st_size > 1000

    def list_available_models(self) -> None:
        """List all available models with size information."""
        print("📁 Wan Model Availability:")
        print("=" * 50)

        for model_name, rel_path in self.model_files.items():
            model_path = self.model_dir / rel_path
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                if size_mb > 1:
                    print(f"✅ {model_name}: {rel_path} ({size_mb:.1f} MB)")
                else:
                    print(f"📝 {model_name}: {rel_path} (placeholder)")
            else:
                print(f"❌ {model_name}: {rel_path} (missing)")

    def _prompt_missing_models(self, missing: Dict[str, str]) -> None:
        print("❌ Required models not available:")
        for description in missing.values():
            print(f"   - {description}")
        print("\n💡 Download models with:")
        toolboxes_path = Path.home() / "amd-strix-halo-image-video-toolboxes"
        get_wan = toolboxes_path / "scripts" / "get_wan22.sh"
        print(f"   {get_wan} common")
        if "i2v" in missing:
            print(f"   {get_wan} 14b-i2v")
        if "t2v" in missing:
            print(f"   {get_wan} 14b-t2v")

    def _write_placeholder(self, output_path: Path, header: str, metadata: Dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"# {header}\n")
            for key, value in metadata.items():
                handle.write(f"# {key.title()}: {value}\n")
            handle.write("# This is a placeholder - integrate WAN runtime for real outputs.\n")

    def generate_video_i2v(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        noise_level: str = "high",
        num_frames: int = 16,
    ) -> bool:
        """Simulate image-to-video WAN generation."""
        missing = {}
        if not self.check_model_availability("text_encoder"):
            missing["text"] = "Text encoder (umt5_xxl_fp8_e4m3fn_scaled.safetensors)"
        if not self.check_model_availability("vae_21"):
            missing["vae"] = "VAE (wan_2.1_vae.safetensors)"

        model_name = f"i2v_{noise_level}_noise"
        if not self.check_model_availability(model_name):
            missing["i2v"] = f"I2V model ({model_name}.safetensors)"

        if missing:
            self._prompt_missing_models(missing)
            return False

        print(f"🎬 Generating video from image: {image_path}")
        print(f"📝 Prompt: {prompt}")
        print(f"🎥 Output: {output_path}")
        print(f"📊 Frames: {num_frames}, Noise: {noise_level}")
        print(f"🚀 Using device: {self.device}")
        print("\n⚠️  This is a simulation - integrate the official WAN pipeline for real outputs.")

        self._write_placeholder(
            Path(output_path),
            "Wan Video Generation Output",
            {
                "image": image_path,
                "prompt": prompt,
                "model": model_name,
                "frames": num_frames,
                "device": self.device,
            },
        )
        print(f"✅ Placeholder created: {output_path}")
        return True

    def generate_video_t2v(
        self,
        prompt: str,
        output_path: str,
        noise_level: str = "high",
        num_frames: int = 16,
    ) -> bool:
        """Simulate text-to-video WAN generation."""
        missing = {}
        if not self.check_model_availability("text_encoder"):
            missing["text"] = "Text encoder (umt5_xxl_fp8_e4m3fn_scaled.safetensors)"
        if not self.check_model_availability("vae_21"):
            missing["vae"] = "VAE (wan_2.1_vae.safetensors)"

        model_name = f"t2v_{noise_level}_noise"
        if not self.check_model_availability(model_name):
            missing["t2v"] = f"T2V model ({model_name}.safetensors)"

        if missing:
            self._prompt_missing_models(missing)
            return False

        print("🎬 Generating video from text prompt")
        print(f"📝 Prompt: {prompt}")
        print(f"🎥 Output: {output_path}")
        print(f"📊 Frames: {num_frames}, Noise: {noise_level}")
        print(f"🚀 Using device: {self.device}")
        print("\n⚠️  This is a simulation - integrate the official WAN pipeline for real outputs.")

        self._write_placeholder(
            Path(output_path),
            "Wan Video Generation Output",
            {
                "prompt": prompt,
                "model": model_name,
                "frames": num_frames,
                "device": self.device,
            },
        )
        print(f"✅ Placeholder created: {output_path}")
        return True


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan Video Generation CLI")
    parser.add_argument(
        "--model-dir",
        default=str(Path.home() / "comfy-models"),
        help="Directory containing Wan models",
    )
    parser.add_argument("--list-models", action="store_true", help="List available models")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    i2v_parser = subparsers.add_parser("i2v", help="Image-to-video generation")
    i2v_parser.add_argument("image", help="Input image path")
    i2v_parser.add_argument("prompt", help="Text prompt")
    i2v_parser.add_argument("-o", "--output", default="output_video.mp4", help="Output video path")
    i2v_parser.add_argument("--noise", choices=["high", "low"], default="high", help="Noise level")
    i2v_parser.add_argument("--frames", type=int, default=16, help="Number of frames")

    t2v_parser = subparsers.add_parser("t2v", help="Text-to-video generation")
    t2v_parser.add_argument("prompt", help="Text prompt")
    t2v_parser.add_argument("-o", "--output", default="output_video.mp4", help="Output video path")
    t2v_parser.add_argument("--noise", choices=["high", "low"], default="high", help="Noise level")
    t2v_parser.add_argument("--frames", type=int, default=16, help="Number of frames")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    if not args.command and not args.list_models:
        print("No command provided.\n")
        parse_args(["-h"])
        return

    generator = WanVideoGenerator(args.model_dir)

    if args.list_models:
        generator.list_available_models()
        return

    if args.command == "i2v":
        success = generator.generate_video_i2v(
            args.image, args.prompt, args.output, args.noise, args.frames
        )
    elif args.command == "t2v":
        success = generator.generate_video_t2v(
            args.prompt, args.output, args.noise, args.frames
        )
    else:
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

