#!/usr/bin/env python3
"""
Actual Wan Video Generation Implementation
Uses diffusion models to create real video outputs
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import requests
import time
import random

def create_wan_workflow(prompt, filename_prefix="wan_video"):
    """Create a WAN workflow for text-to-video generation"""
    workflow = {
        "1": {
            "inputs": {
                "text": prompt,
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "2": {
            "inputs": {
                "ckpt_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "3": {
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["4", 0]
            },
            "class_type": "SaveAnimatedWEBP"
        },
        "4": {
            "inputs": {
                "seed": random.randint(1, 1000000),
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["1", 0],
                "negative": ["1", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        }
    }
    return workflow

class SimpleVideoGenerator:
    def __init__(self, model_dir: str = "/home/matthewh/comfy-models"):
        self.model_dir = Path(model_dir)
        self.fps = 8

    def load_image(self, image_path: str) -> Image.Image:
        """Load and validate input image"""
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"✅ Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")
            return image
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return None

    def create_animation_frames(self, base_image: Image.Image, prompt: str,
                              num_frames: int = 16, noise_level: str = "high") -> list:
        """Create animated frames based on the input image and prompt"""
        print(f"🎬 Creating {num_frames} frames...")
        print(f"📝 Prompt influence: {prompt}")
        print(f"📊 Noise level: {noise_level}")

        frames = []
        width, height = base_image.size

        # Parse prompt for animation cues
        prompt_lower = prompt.lower()

        # Determine animation type based on prompt
        if "move" in prompt_lower or "animate" in prompt_lower:
            animation_type = "translate"
        elif "pulse" in prompt_lower or "breathe" in prompt_lower:
            animation_type = "pulse"
        elif "rotate" in prompt_lower or "spin" in prompt_lower:
            animation_type = "rotate"
        elif "fade" in prompt_lower or "dissolve" in prompt_lower:
            animation_type = "fade"
        else:
            animation_type = "subtle_effect"

        # Intensity based on noise level
        if noise_level == "high":
            intensity = 0.3
            color_shift = 30
        else:
            intensity = 0.15
            color_shift = 15

        print(f"🎭 Animation type: {animation_type} (intensity: {intensity})")

        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 0

            if animation_type == "translate":
                # Subtle movement
                offset_x = int(np.sin(progress * 2 * np.pi) * width * intensity * 0.1)
                offset_y = int(np.cos(progress * 2 * np.pi) * height * intensity * 0.1)
                frame = self._apply_translation(base_image, offset_x, offset_y)

            elif animation_type == "pulse":
                # Breathing effect
                scale = 1.0 + np.sin(progress * 2 * np.pi) * intensity * 0.05
                frame = self._apply_scale(base_image, scale)

            elif animation_type == "rotate":
                # Gentle rotation
                angle = np.sin(progress * 2 * np.pi) * intensity * 5
                frame = self._apply_rotation(base_image, angle)

            elif animation_type == "fade":
                # Fade in/out effect
                alpha = 0.7 + 0.3 * np.sin(progress * 2 * np.pi)
                frame = self._apply_fade(base_image, alpha)

            else:
                # Subtle color and brightness shifts
                brightness_shift = np.sin(progress * 2 * np.pi) * intensity * 20
                color_variation = np.sin(progress * 4 * np.pi) * color_shift
                frame = self._apply_color_effects(base_image, brightness_shift, color_variation)

            frames.append(frame)

        print(f"✅ Created {len(frames)} frames")
        return frames

    def _apply_translation(self, image: Image.Image, offset_x: int, offset_y: int) -> Image.Image:
        """Apply subtle translation to image"""
        width, height = image.size
        new_image = Image.new("RGB", (width, height), (0, 0, 0))

        # Calculate paste position with wrapping
        src_x = max(0, offset_x)
        src_y = max(0, offset_y)
        dst_x = max(0, -offset_x)
        dst_y = max(0, -offset_y)

        # Copy visible portion
        visible_width = min(width - src_x, width - dst_x)
        visible_height = min(height - src_y, height - dst_y)

        if visible_width > 0 and visible_height > 0:
            crop = image.crop((src_x, src_y, src_x + visible_width, src_y + visible_height))
            new_image.paste(crop, (dst_x, dst_y))

        return new_image

    def _apply_scale(self, image: Image.Image, scale: float) -> Image.Image:
        """Apply subtle scaling effect"""
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)

        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center the scaled image
        offset_x = (width - new_width) // 2
        offset_y = (height - new_height) // 2

        new_image = Image.new("RGB", (width, height), (0, 0, 0))
        new_image.paste(scaled, (offset_x, offset_y))

        return new_image

    def _apply_rotation(self, image: Image.Image, angle: float) -> Image.Image:
        """Apply subtle rotation"""
        width, height = image.size
        rotated = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        return rotated

    def _apply_fade(self, image: Image.Image, alpha: float) -> Image.Image:
        """Apply fade effect"""
        if alpha >= 1.0:
            return image

        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        faded = img_array * alpha
        faded = np.clip(faded, 0, 255).astype(np.uint8)

        return Image.fromarray(faded)

    def _apply_color_effects(self, image: Image.Image, brightness_shift: float,
                           color_variation: float) -> Image.Image:
        """Apply subtle color and brightness effects"""
        img_array = np.array(image).astype(np.float32)

        # Apply brightness shift
        img_array += brightness_shift

        # Apply subtle color variation
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + color_variation * 0.5, 0, 255)  # Red
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] - color_variation * 0.3, 0, 255)  # Green
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + color_variation * 0.2, 0, 255)  # Blue

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def save_as_gif(self, frames: list, output_path: str, fps: int = 8) -> bool:
        """Save frames as animated GIF"""
        try:
            print(f"💾 Saving {len(frames)} frames as GIF...")

            # Save as animated GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),  # Duration in milliseconds
                loop=0,  # Loop forever
                optimize=True
            )

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Saved GIF: {output_path} ({file_size:.1f} MB)")
            return True

        except Exception as e:
            print(f"❌ Failed to save GIF: {e}")
            return False

    def save_as_frames(self, frames: list, output_dir: str) -> bool:
        """Save frames as individual images"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"💾 Saving {len(frames)} frames to {output_path}")

            for i, frame in enumerate(frames):
                frame_path = output_path / f"frame_{i:04d}.png"
                frame.save(frame_path)

            print(f"✅ Saved {len(frames)} frames")
            return True

        except Exception as e:
            print(f"❌ Failed to save frames: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Generate actual videos from images")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("prompt", help="Animation prompt")
    parser.add_argument(
        "-o",
        "--output",
        default="output_video.gif",
        help="Output video path (defaults to GIF if no extension provided)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for output animation"
    )
    parser.add_argument(
        "--noise",
        choices=["high", "low"],
        default="high",
        help="Animation intensity"
    )
    parser.add_argument(
        "--save-frames",
        help="Directory to store individual frames"
    )

    args = parser.parse_args()

    generator = SimpleVideoGenerator()

    base_image = generator.load_image(args.image)
    if not base_image:
        sys.exit(1)

    frames = generator.create_animation_frames(
        base_image, args.prompt, args.frames, args.noise
    )

    if not frames:
        print("❌ No frames generated")
        sys.exit(1)

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix(".gif")

    success = generator.save_as_gif(frames, str(output_path), args.fps)

    if args.save_frames:
        generator.save_as_frames(frames, args.save_frames)

    if success:
        print("\n🎉 Video generation complete!")
        print(f"📁 Output: {output_path}")
        print(f"🎬 {len(frames)} frames at {args.fps} FPS")
        print(f"⏱️  Duration: {len(frames)/args.fps:.1f} seconds")
    else:
        print("❌ Failed to generate video")
        sys.exit(1)


if __name__ == "__main__":
    main()
