#!/usr/bin/env python3
"""
Advanced Video Post-Processing System
Professional video effects, filters, and enhancements
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import json
from datetime import datetime
import colorsys
import math

@dataclass
class VideoEffect:
    """Video effect configuration"""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    preview_frame: Optional[str] = None

@dataclass
class ProcessingConfig:
    """Video processing configuration"""
    input_path: str
    output_path: str
    effects: List[Dict[str, Any]]
    format: str = "gif"
    quality: int = 95
    optimize: bool = True
    fps: Optional[int] = None

class VideoEffects:
    """Collection of video processing effects"""

    @staticmethod
    def brightness_adjustment(frame: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjust brightness of frame"""
        enhancer = ImageEnhance.Brightness(frame)
        return enhancer.enhance(factor)

    @staticmethod
    def contrast_adjustment(frame: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjust contrast of frame"""
        enhancer = ImageEnhance.Contrast(frame)
        return enhancer.enhance(factor)

    @staticmethod
    def saturation_adjustment(frame: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjust color saturation"""
        enhancer = ImageEnhance.Color(frame)
        return enhancer.enhance(factor)

    @staticmethod
    def sharpness_adjustment(frame: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjust image sharpness"""
        enhancer = ImageEnhance.Sharpness(frame)
        return enhancer.enhance(factor)

    @staticmethod
    def gaussian_blur(frame: Image.Image, radius: float = 1.0) -> Image.Image:
        """Apply Gaussian blur"""
        return frame.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def motion_blur(frame: Image.Image, size: int = 5, angle: float = 0) -> Image.Image:
        """Apply motion blur effect"""
        # Create motion blur kernel
        kernel = ImageFilter.Kernel(
            size=(size, 1),
            kernel=[1.0/size] * size,
            scale=1
        )

        # Rotate frame first, apply blur, then rotate back
        if angle != 0:
            frame = frame.rotate(-angle, expand=True)
            blurred = frame.filter(kernel)
            return blurred.rotate(angle, expand=True)
        else:
            return frame.filter(kernel)

    @staticmethod
    def vintage_effect(frame: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Apply vintage film effect"""
        # Convert to numpy array
        img_array = np.array(frame)

        # Add sepia tone
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])

        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255)

        # Blend with original
        result = img_array * (1 - intensity) + sepia_img * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def glitch_effect(frame: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Apply digital glitch effect"""
        img_array = np.array(frame)
        height, width = img_array.shape[:2]

        # Random channel shifts
        if np.random.random() < intensity:
            channel_shift = np.random.randint(-20, 20, size=3)
            for i in range(3):
                if np.random.random() < 0.5:
                    shift = channel_shift[i]
                    img_array[:, :, i] = np.roll(img_array[:, :, i], shift, axis=1)

        # Random pixel blocks
        num_blocks = int(intensity * 10)
        for _ in range(num_blocks):
            x = np.random.randint(0, width - 10)
            y = np.random.randint(0, height - 10)
            block_size = np.random.randint(5, 20)

            # Create random color block
            color = np.random.randint(0, 256, size=3)

            x_end = min(x + block_size, width)
            y_end = min(y + block_size, height)

            img_array[y:y_end, x:x_end] = color

        return Image.fromarray(img_array)

    @staticmethod
    def chromatic_aberration(frame: Image.Image, intensity: float = 5.0) -> Image.Image:
        """Apply chromatic aberration effect"""
        img_array = np.array(frame)
        height, width = img_array.shape[:2]

        # Split RGB channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Shift red and blue channels
        r_shifted = np.roll(r, int(intensity), axis=1)
        b_shifted = np.roll(b, -int(intensity), axis=1)

        # Reconstruct image
        result = np.stack([r_shifted, g, b_shifted], axis=2)
        return Image.fromarray(result)

    @staticmethod
    def rgb_split_effect(frame: Image.Image, intensity: float = 3.0) -> Image.Image:
        """Apply RGB color splitting effect"""
        img_array = np.array(frame)
        height, width = img_array.shape[:2]

        result = np.zeros_like(img_array)

        # Shift each color channel differently
        result[:, :, 0] = np.roll(img_array[:, :, 0], int(intensity), axis=0)  # Red shift down
        result[:, :, 1] = img_array[:, :, 1]  # Green stays
        result[:, :, 2] = np.roll(img_array[:, :, 2], -int(intensity), axis=0)  # Blue shift up

        return Image.fromarray(result)

    @staticmethod
    def film_grain(frame: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Apply film grain effect"""
        img_array = np.array(frame).astype(np.float32)

        # Generate noise
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy = img_array + noise

        # Clip values
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    @staticmethod
    def vignette_effect(frame: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Apply vignette (darkened edges) effect"""
        width, height = frame.size
        center_x, center_y = width // 2, height // 2

        # Create vignette mask
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)

        # Create gradient
        vignette = 1 - (dist_from_center / max_dist) * intensity
        vignette = np.clip(vignette, 0, 1)

        # Apply vignette
        img_array = np.array(frame).astype(np.float32)
        for i in range(3):  # Apply to each RGB channel
            img_array[:, :, i] *= vignette

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    @staticmethod
    def color_grading(frame: Image.Image, shadows: Tuple[float, float, float] = (0, 0, 0),
                     midtones: Tuple[float, float, float] = (0, 0, 0),
                     highlights: Tuple[float, float, float] = (0, 0, 0)) -> Image.Image:
        """Professional color grading with shadows, midtones, highlights"""
        img_array = np.array(frame).astype(np.float32) / 255.0

        # Convert to HSV for better color manipulation
        hsv = colorsys.rgb_to_hsv(img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2])
        h, s, v = hsv

        # Create luminance mask for shadows, midtones, highlights
        luminance = v

        # Shadow mask (dark areas)
        shadow_mask = np.clip(1.0 - luminance * 2.0, 0, 1)

        # Highlight mask (bright areas)
        highlight_mask = np.clip((luminance - 0.5) * 2.0, 0, 1)

        # Midtone mask (everything else)
        midtone_mask = 1.0 - shadow_mask - highlight_mask

        # Apply color adjustments
        for i in range(3):
            img_array[:, :, i] += (shadows[i] * shadow_mask +
                                 midtones[i] * midtone_mask +
                                 highlights[i] * highlight_mask) * 0.1

        return Image.fromarray(np.clip(img_array * 255, 0, 255).astype(np.uint8))

    @staticmethod
    def pixelate_effect(frame: Image.Image, pixel_size: int = 10) -> Image.Image:
        """Apply pixelation effect"""
        width, height = frame.size

        # Small size for pixelation
        small = frame.resize((width // pixel_size, height // pixel_size), Image.Resampling.NEAREST)

        # Scale back up
        pixelated = small.resize((width, height), Image.Resampling.NEAREST)

        return pixelated

    @staticmethod
    def kaleidoscope_effect(frame: Image.Image, segments: int = 6) -> Image.Image:
        """Apply kaleidoscope effect"""
        width, height = frame.size
        center_x, center_y = width // 2, height // 2

        # Create result image
        result = Image.new('RGB', (width, height), 'black')

        # Convert to numpy for easier manipulation
        img_array = np.array(frame)
        result_array = np.array(result)

        angle_step = 2 * math.pi / segments

        for i in range(segments):
            angle = i * angle_step

            # Calculate rotation
            for y in range(height):
                for x in range(width):
                    # Translate to origin
                    tx, ty = x - center_x, y - center_y

                    # Rotate
                    rx = tx * math.cos(angle) - ty * math.sin(angle)
                    ry = tx * math.sin(angle) + ty * math.cos(angle)

                    # Translate back
                    sx, sy = int(rx + center_x), int(ry + center_y)

                    # Check bounds
                    if 0 <= sx < width and 0 <= sy < height:
                        result_array[y, x] = img_array[sy, sx]

        return Image.fromarray(result_array)

    @staticmethod
    def hologram_effect(frame: Image.Image, intensity: float = 0.3) -> Image.Image:
        """Apply hologram projection effect"""
        img_array = np.array(frame)
        height, width = img_array.shape[:2]

        # Create holographic color shift
        result = img_array.copy().astype(np.float32)

        # Add color shifting over time (simulated with position)
        for y in range(height):
            color_shift = int(np.sin(y * 0.05) * intensity * 20)
            result[y, :, 0] = np.clip(result[y, :, 0] + color_shift, 0, 255)  # Red channel
            result[y, :, 2] = np.clip(result[y, :, 2] - color_shift, 0, 255)  # Blue channel

        # Add scan lines
        scan_line_intensity = intensity * 50
        for y in range(0, height, 4):
            result[y, :] *= (1 - scan_line_intensity / 255)

        # Add glow effect
        from scipy.ndimage import gaussian_filter
        try:
            glow = gaussian_filter(result, sigma=2)
            result = result * 0.7 + glow * 0.3
        except ImportError:
            # Fallback if scipy not available
            pass

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

class VideoPostProcessor:
    """Advanced video post-processing system"""

    def __init__(self):
        self.effects = self._initialize_effects()
        self.presets = self._initialize_presets()

    def _initialize_effects(self) -> Dict[str, VideoEffect]:
        """Initialize all available effects"""
        effects = {
            "brightness": VideoEffect(
                name="Brightness",
                description="Adjust overall brightness",
                category="basic",
                parameters={"factor": {"type": "float", "min": 0.1, "max": 3.0, "default": 1.0}}
            ),
            "contrast": VideoEffect(
                name="Contrast",
                description="Adjust image contrast",
                category="basic",
                parameters={"factor": {"type": "float", "min": 0.1, "max": 3.0, "default": 1.0}}
            ),
            "saturation": VideoEffect(
                name="Saturation",
                description="Adjust color saturation",
                category="color",
                parameters={"factor": {"type": "float", "min": 0.0, "max": 2.0, "default": 1.0}}
            ),
            "sharpness": VideoEffect(
                name="Sharpness",
                description="Adjust image sharpness",
                category="basic",
                parameters={"factor": {"type": "float", "min": 0.0, "max": 3.0, "default": 1.0}}
            ),
            "gaussian_blur": VideoEffect(
                name="Gaussian Blur",
                description="Apply smooth blur effect",
                category="blur",
                parameters={"radius": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0}}
            ),
            "motion_blur": VideoEffect(
                name="Motion Blur",
                description="Apply directional motion blur",
                category="blur",
                parameters={
                    "size": {"type": "int", "min": 3, "max": 20, "default": 5},
                    "angle": {"type": "float", "min": 0, "max": 360, "default": 0}
                }
            ),
            "vintage": VideoEffect(
                name="Vintage Film",
                description="Apply vintage sepia tone effect",
                category="stylize",
                parameters={"intensity": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}}
            ),
            "glitch": VideoEffect(
                name="Digital Glitch",
                description="Apply digital glitch artifacts",
                category="stylize",
                parameters={"intensity": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.1}}
            ),
            "chromatic_aberration": VideoEffect(
                name="Chromatic Aberration",
                description="Apply color channel separation",
                category="color",
                parameters={"intensity": {"type": "float", "min": 0.0, "max": 20.0, "default": 5.0}}
            ),
            "rgb_split": VideoEffect(
                name="RGB Split",
                description="Apply RGB color splitting",
                category="color",
                parameters={"intensity": {"type": "float", "min": 0.0, "max": 10.0, "default": 3.0}}
            ),
            "film_grain": VideoEffect(
                name="Film Grain",
                description="Add realistic film grain",
                category="noise",
                parameters={"intensity": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.1}}
            ),
            "vignette": VideoEffect(
                name="Vignette",
                description="Add darkened edges (vignette)",
                category="stylize",
                parameters={"intensity": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}}
            ),
            "color_grading": VideoEffect(
                name="Color Grading",
                description="Professional color grading",
                category="color",
                parameters={
                    "shadows_r": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "shadows_g": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "shadows_b": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "midtones_r": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "midtones_g": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "midtones_b": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "highlights_r": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "highlights_g": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                    "highlights_b": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0}
                }
            ),
            "pixelate": VideoEffect(
                name="Pixelate",
                description="Apply pixelation effect",
                category="stylize",
                parameters={"pixel_size": {"type": "int", "min": 2, "max": 50, "default": 10}}
            ),
            "kaleidoscope": VideoEffect(
                name="Kaleidoscope",
                description="Apply kaleidoscope pattern effect",
                category="stylize",
                parameters={"segments": {"type": "int", "min": 3, "max": 12, "default": 6}}
            ),
            "hologram": VideoEffect(
                name="Hologram",
                description="Apply holographic projection effect",
                category="stylize",
                parameters={"intensity": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3}}
            )
        }
        return effects

    def _initialize_presets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize effect presets"""
        presets = {
            "cinematic": [
                {"effect": "color_grading", "params": {"shadows_r": -0.1, "shadows_g": -0.1, "shadows_b": -0.2}},
                {"effect": "contrast", "params": {"factor": 1.2}},
                {"effect": "saturation", "params": {"factor": 0.9}},
                {"effect": "vignette", "params": {"intensity": 0.3}}
            ],
            "vintage_film": [
                {"effect": "vintage", "params": {"intensity": 0.8}},
                {"effect": "film_grain", "params": {"intensity": 0.15}},
                {"effect": "brightness", "params": {"factor": 0.9}},
                {"effect": "contrast", "params": {"factor": 0.85}}
            ],
            "cyberpunk": [
                {"effect": "color_grading", "params": {"shadows_b": 0.3, "midtones_b": 0.2}},
                {"effect": "chromatic_aberration", "params": {"intensity": 3.0}},
                {"effect": "contrast", "params": {"factor": 1.4}},
                {"effect": "saturation", "params": {"factor": 1.3}}
            ],
            "dreamy": [
                {"effect": "gaussian_blur", "params": {"radius": 0.5}},
                {"effect": "brightness", "params": {"factor": 1.1}},
                {"effect": "saturation", "params": {"factor": 1.2}},
                {"effect": "vignette", "params": {"intensity": 0.2}}
            ],
            "noir": [
                {"effect": "saturation", "params": {"factor": 0.0}},
                {"effect": "contrast", "params": {"factor": 1.5}},
                {"effect": "brightness", "params": {"factor": 0.8}},
                {"effect": "vignette", "params": {"intensity": 0.4}}
            ],
            "glitch_art": [
                {"effect": "glitch", "params": {"intensity": 0.3}},
                {"effect": "rgb_split", "params": {"intensity": 2.0}},
                {"effect": "chromatic_aberration", "params": {"intensity": 5.0}},
                {"effect": "saturation", "params": {"factor": 1.5}}
            ],
            "holographic": [
                {"effect": "hologram", "params": {"intensity": 0.5}},
                {"effect": "chromatic_aberration", "params": {"intensity": 2.0}},
                {"effect": "contrast", "params": {"factor": 1.2}},
                {"effect": "saturation", "params": {"factor": 0.8}}
            ],
            "retro_arcade": [
                {"effect": "pixelate", "params": {"pixel_size": 8}},
                {"effect": "saturation", "params": {"factor": 1.5}},
                {"effect": "contrast", "params": {"factor": 1.3}},
                {"effect": "brightness", "params": {"factor": 1.1}}
            ]
        }
        return presets

    def load_video_frames(self, input_path: str) -> List[Image.Image]:
        """Load video frames from file"""
        frames = []

        if input_path.lower().endswith(('.gif', '.GIF')):
            try:
                with Image.open(input_path) as img:
                    for frame_index in range(getattr(img, 'n_frames', 1)):
                        img.seek(frame_index)
                        frame = img.copy()
                        frames.append(frame.convert('RGB'))
            except Exception as e:
                print(f"Error loading GIF: {e}")
                return []

        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Single image - create animated version
            try:
                frame = Image.open(input_path).convert('RGB')
                frames.append(frame)
            except Exception as e:
                print(f"Error loading image: {e}")
                return []

        else:
            print(f"Unsupported format: {input_path}")
            return []

        if not frames:
            print("No frames loaded")
            return []

        print(f"✅ Loaded {len(frames)} frames from {input_path}")
        return frames

    def apply_effect(self, frame: Image.Image, effect_name: str, params: Dict[str, Any]) -> Image.Image:
        """Apply single effect to frame"""
        effect_map = {
            "brightness": VideoEffects.brightness_adjustment,
            "contrast": VideoEffects.contrast_adjustment,
            "saturation": VideoEffects.saturation_adjustment,
            "sharpness": VideoEffects.sharpness_adjustment,
            "gaussian_blur": VideoEffects.gaussian_blur,
            "motion_blur": VideoEffects.motion_blur,
            "vintage": VideoEffects.vintage_effect,
            "glitch": VideoEffects.glitch_effect,
            "chromatic_aberration": VideoEffects.chromatic_aberration,
            "rgb_split": VideoEffects.rgb_split_effect,
            "film_grain": VideoEffects.film_grain,
            "vignette": VideoEffects.vignette_effect,
            "color_grading": VideoEffects.color_grading,
            "pixelate": VideoEffects.pixelate_effect,
            "kaleidoscope": VideoEffects.kaleidoscope_effect,
            "hologram": VideoEffects.hologram_effect
        }

        if effect_name not in effect_map:
            print(f"⚠️  Unknown effect: {effect_name}")
            return frame

        try:
            effect_func = effect_map[effect_name]

            # Handle special cases for complex parameters
            if effect_name == "color_grading":
                shadows = (params.get("shadows_r", 0), params.get("shadows_g", 0), params.get("shadows_b", 0))
                midtones = (params.get("midtones_r", 0), params.get("midtones_g", 0), params.get("midtones_b", 0))
                highlights = (params.get("highlights_r", 0), params.get("highlights_g", 0), params.get("highlights_b", 0))
                return effect_func(frame, shadows, midtones, highlights)
            else:
                return effect_func(frame, **params)

        except Exception as e:
            print(f"❌ Error applying {effect_name}: {e}")
            return frame

    def process_frames(self, frames: List[Image.Image], effects: List[Dict[str, Any]],
                      progress_callback: Optional[callable] = None) -> List[Image.Image]:
        """Apply multiple effects to all frames"""
        processed_frames = []
        total_frames = len(frames)
        total_effects = len(effects)

        print(f"🎬 Processing {total_frames} frames with {total_effects} effects...")

        for frame_idx, frame in enumerate(frames):
            processed_frame = frame.copy()

            # Apply each effect in sequence
            for effect_idx, effect_config in enumerate(effects):
                effect_name = effect_config["effect"]
                params = effect_config.get("params", {})

                processed_frame = self.apply_effect(processed_frame, effect_name, params)

            processed_frames.append(processed_frame)

            # Progress callback
            if progress_callback:
                progress = ((frame_idx + 1) / total_frames) * 100
                progress_callback(progress, frame_idx + 1, total_frames)

            # Console progress
            if (frame_idx + 1) % max(1, total_frames // 10) == 0:
                progress = ((frame_idx + 1) / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}% ({frame_idx + 1}/{total_frames})")

        print(f"✅ Processed {len(processed_frames)} frames")
        return processed_frames

    def save_video(self, frames: List[Image.Image], output_path: str,
                   format: str = "gif", fps: int = 8, quality: int = 95, optimize: bool = True) -> bool:
        """Save processed frames as video"""
        try:
            if not frames:
                print("❌ No frames to save")
                return False

            print(f"💾 Saving {len(frames)} frames to {output_path}...")

            if format.lower() == "gif":
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / fps),
                    loop=0,
                    optimize=optimize,
                    quality=quality
                )
            else:
                print(f"❌ Unsupported format: {format}")
                return False

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Saved {format.upper()}: {output_path} ({file_size:.1f} MB)")
            return True

        except Exception as e:
            print(f"❌ Failed to save video: {e}")
            return False

    def process_video(self, config: ProcessingConfig, progress_callback: Optional[callable] = None) -> bool:
        """Process video with full pipeline"""
        print(f"🎬 Starting video post-processing...")
        print(f"📁 Input: {config.input_path}")
        print(f"📁 Output: {config.output_path}")
        print(f"🎨 Effects: {len(config.effects)}")

        # Load frames
        frames = self.load_video_frames(config.input_path)
        if not frames:
            return False

        # Process frames
        processed_frames = self.process_frames(frames, config.effects, progress_callback)

        # Save video
        fps = config.fps or self._estimate_fps(config.input_path, frames)
        success = self.save_video(
            processed_frames,
            config.output_path,
            config.format,
            fps,
            config.quality,
            config.optimize
        )

        return success

    def _estimate_fps(self, input_path: str, frames: List[Image.Image]) -> int:
        """Estimate original FPS from input"""
        if input_path.lower().endswith('.gif'):
            try:
                with Image.open(input_path) as img:
                    duration = getattr(img, 'info', {}).get('duration', 100)  # Default 100ms per frame
                    return max(1, int(1000 / duration))
            except:
                pass
        return 8  # Default FPS

    def list_effects(self, category: Optional[str] = None) -> List[VideoEffect]:
        """List available effects, optionally filtered by category"""
        effects = list(self.effects.values())
        if category:
            effects = [e for e in effects if e.category == category]
        return effects

    def list_presets(self) -> Dict[str, List[Dict[str, Any]]]:
        """List available effect presets"""
        return self.presets.copy()

    def get_effect_info(self, effect_name: str) -> Optional[VideoEffect]:
        """Get detailed information about an effect"""
        return self.effects.get(effect_name)

def main():
    parser = argparse.ArgumentParser(description="Advanced Video Post-Processor")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--effects", help="JSON file with effects configuration")
    parser.add_argument("--preset", help="Apply effect preset",
                       choices=["cinematic", "vintage_film", "cyberpunk", "dreamy", "noir",
                                "glitch_art", "holographic", "retro_arcade"])
    parser.add_argument("--list-effects", action="store_true", help="List all available effects")
    parser.add_argument("--list-presets", action="store_true", help="List all presets")
    parser.add_argument("--fps", type=int, help="Output FPS")
    parser.add_argument("--quality", type=int, default=95, help="Output quality (1-100)")
    parser.add_argument("--format", default="gif", choices=["gif"], help="Output format")

    args = parser.parse_args()

    processor = VideoPostProcessor()

    if args.list_effects:
        print("🎨 Available Effects:")
        print("=" * 50)
        categories = {}
        for effect in processor.effects.values():
            if effect.category not in categories:
                categories[effect.category] = []
            categories[effect.category].append(effect)

        for category, effects in sorted(categories.items()):
            print(f"\n📁 {category.title()}:")
            for effect in effects:
                print(f"   • {effect.name}: {effect.description}")

        return

    if args.list_presets:
        print("🎭 Available Presets:")
        print("=" * 50)
        for name, effects in processor.presets.items():
            print(f"\n🎬 {name.replace('_', ' ').title()}:")
            for effect in effects:
                print(f"   • {effect['effect']}: {effect['params']}")

        return

    if not args.input or not args.output:
        print("❌ Input and output files required")
        return

    # Load effects configuration
    effects = []

    if args.preset:
        effects = processor.presets.get(args.preset, [])
        print(f"🎭 Applying preset: {args.preset}")
    elif args.effects:
        try:
            with open(args.effects, 'r') as f:
                effects_config = json.load(f)
                effects = effects_config.get('effects', [])
        except Exception as e:
            print(f"❌ Failed to load effects file: {e}")
            return
    else:
        print("⚠️  No effects specified. Use --preset or --effects")
        return

    if not effects:
        print("❌ No effects to apply")
        return

    # Create processing configuration
    config = ProcessingConfig(
        input_path=args.input,
        output_path=args.output,
        effects=effects,
        format=args.format,
        quality=args.quality,
        fps=args.fps
    )

    # Progress callback
    def progress_callback(progress: float, current: int, total: int):
        print(f"📊 Processing: {progress:.1f}% ({current}/{total})")

    # Process video
    success = processor.process_video(config, progress_callback)

    if success:
        print("\n🎉 Video post-processing complete!")
        print(f"📁 Output: {args.output}")
        print(f"🎨 Effects applied: {len(effects)}")
    else:
        print("\n❌ Video post-processing failed")

if __name__ == "__main__":
    main()