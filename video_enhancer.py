#!/usr/bin/env python3
"""
Automated Video Enhancement System
Intelligent video analysis and enhancement algorithms
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
import json
from datetime import datetime
import cv2
import math

@dataclass
class VideoAnalysis:
    """Video analysis results"""
    brightness: float
    contrast: float
    saturation: float
    sharpness: float
    noise_level: float
    color_balance: Tuple[float, float, float]
    motion_intensity: float
    quality_score: float
    recommendations: List[str]

@dataclass
class EnhancementConfig:
    """Video enhancement configuration"""
    auto_brightness: bool = True
    auto_contrast: bool = True
    auto_saturation: bool = True
    auto_sharpen: bool = True
    noise_reduction: bool = True
    color_correction: bool = True
    stabilize_motion: bool = False
    upscale_factor: float = 1.0
    quality_mode: str = "balanced"  # fast, balanced, quality

class VideoAnalyzer:
    """Intelligent video analysis system"""

    @staticmethod
    def analyze_frame(frame: Image.Image) -> Dict[str, Any]:
        """Analyze single frame properties"""
        frame_array = np.array(frame)

        # Basic statistics
        stat = ImageStat.Stat(frame)
        brightness = sum(stat.mean) / len(stat.mean) / 255.0

        # Contrast (standard deviation)
        contrast = sum(stat.stddev) / len(stat.stddev) / 255.0

        # Saturation analysis
        if frame.mode == 'RGB':
            hsv_frame = frame.convert('HSV')
            hsv_array = np.array(hsv_frame)
            saturation = np.mean(hsv_array[:, :, 1]) / 255.0
        else:
            saturation = 0.5

        # Sharpness (using Laplacian variance)
        gray_frame = frame.convert('L')
        gray_array = np.array(gray_frame)
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000.0  # Normalize

        # Noise level (using high-frequency content)
        noise = cv2.medianBlur(gray_array, 5)
        noise_level = np.mean(np.abs(gray_array.astype(float) - noise.astype(float))) / 255.0

        # Color balance
        if frame.mode == 'RGB':
            r_mean = np.mean(frame_array[:, :, 0]) / 255.0
            g_mean = np.mean(frame_array[:, :, 1]) / 255.0
            b_mean = np.mean(frame_array[:, :, 2]) / 255.0
            color_balance = (r_mean, g_mean, b_mean)
        else:
            color_balance = (0.33, 0.33, 0.34)

        return {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'sharpness': sharpness,
            'noise_level': noise_level,
            'color_balance': color_balance
        }

    @staticmethod
    def analyze_motion(frames: List[Image.Image]) -> float:
        """Analyze motion intensity between frames"""
        if len(frames) < 2:
            return 0.0

        motion_scores = []

        for i in range(len(frames) - 1):
            frame1 = np.array(frames[i].convert('L'))
            frame2 = np.array(frames[i + 1].convert('L'))

            # Calculate optical flow for motion analysis
            flow = cv2.calcOpticalFlowPyrLK(
                frame1.astype(np.uint8),
                frame2.astype(np.uint8),
                None, None
            )[0]

            if flow is not None:
                # Calculate motion magnitude
                magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                motion_score = np.mean(magnitude)
                motion_scores.append(motion_score)

        return np.mean(motion_scores) / 100.0 if motion_scores else 0.0

    @staticmethod
    def calculate_quality_score(analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        brightness_score = 1.0 - abs(analysis['brightness'] - 0.5) * 2  # Ideal ~0.5
        contrast_score = min(analysis['contrast'] * 2, 1.0)  # Higher contrast is better
        saturation_score = 1.0 - abs(analysis['saturation'] - 0.6) * 2  # Ideal ~0.6
        sharpness_score = min(analysis['sharpness'] / 10.0, 1.0)  # Normalize
        noise_score = 1.0 - min(analysis['noise_level'] * 10, 1.0)  # Less noise is better

        # Color balance score (how balanced RGB channels are)
        r, g, b = analysis['color_balance']
        avg_color = (r + g + b) / 3.0
        color_balance_score = 1.0 - (abs(r - avg_color) + abs(g - avg_color) + abs(b - avg_color)) / 3.0

        # Weighted average
        weights = [0.2, 0.2, 0.15, 0.2, 0.15, 0.1]  # brightness, contrast, saturation, sharpness, noise, color
        scores = [brightness_score, contrast_score, saturation_score, sharpness_score, noise_score, color_balance_score]

        quality_score = sum(w * s for w, s in zip(weights, scores))
        return max(0.0, min(1.0, quality_score))

    @staticmethod
    def generate_recommendations(analysis: Dict[str, Any], quality_score: float) -> List[str]:
        """Generate enhancement recommendations based on analysis"""
        recommendations = []

        if quality_score < 0.7:
            recommendations.append("Overall quality needs improvement")

        # Brightness recommendations
        if analysis['brightness'] < 0.3:
            recommendations.append("Increase brightness - video appears too dark")
        elif analysis['brightness'] > 0.7:
            recommendations.append("Decrease brightness - video appears too bright")

        # Contrast recommendations
        if analysis['contrast'] < 0.2:
            recommendations.append("Increase contrast - video lacks depth")
        elif analysis['contrast'] > 0.6:
            recommendations.append("Decrease contrast - video appears harsh")

        # Saturation recommendations
        if analysis['saturation'] < 0.3:
            recommendations.append("Increase saturation - colors appear faded")
        elif analysis['saturation'] > 0.8:
            recommendations.append("Decrease saturation - colors appear oversaturated")

        # Sharpness recommendations
        if analysis['sharpness'] < 1.0:
            recommendations.append("Apply sharpening - video appears soft")
        elif analysis['sharpness'] > 15.0:
            recommendations.append("Apply slight blur - video appears over-sharpened")

        # Noise recommendations
        if analysis['noise_level'] > 0.1:
            recommendations.append("Apply noise reduction - video appears noisy")

        # Color balance recommendations
        r, g, b = analysis['color_balance']
        avg_color = (r + g + b) / 3.0
        if abs(r - avg_color) > 0.1:
            recommendations.append("Correct red channel color balance")
        if abs(g - avg_color) > 0.1:
            recommendations.append("Correct green channel color balance")
        if abs(b - avg_color) > 0.1:
            recommendations.append("Correct blue channel color balance")

        return recommendations

    def analyze_video(self, frames: List[Image.Image]) -> VideoAnalysis:
        """Perform comprehensive video analysis"""
        if not frames:
            return VideoAnalysis(0, 0, 0, 0, 0, (0, 0, 0), 0, 0, [])

        print("🔍 Analyzing video properties...")

        # Analyze all frames
        frame_analyses = []
        for i, frame in enumerate(frames):
            if i % max(1, len(frames) // 10) == 0:
                print(f"📊 Analyzing frame {i + 1}/{len(frames)}")
            frame_analyses.append(self.analyze_frame(frame))

        # Calculate averages
        avg_brightness = np.mean([f['brightness'] for f in frame_analyses])
        avg_contrast = np.mean([f['contrast'] for f in frame_analyses])
        avg_saturation = np.mean([f['saturation'] for f in frame_analyses])
        avg_sharpness = np.mean([f['sharpness'] for f in frame_analyses])
        avg_noise = np.mean([f['noise_level'] for f in frame_analyses])

        # Average color balance
        avg_r = np.mean([f['color_balance'][0] for f in frame_analyses])
        avg_g = np.mean([f['color_balance'][1] for f in frame_analyses])
        avg_b = np.mean([f['color_balance'][2] for f in frame_analyses])
        avg_color_balance = (avg_r, avg_g, avg_b)

        # Motion analysis
        motion_intensity = self.analyze_motion(frames)

        # Create analysis summary
        analysis_dict = {
            'brightness': avg_brightness,
            'contrast': avg_contrast,
            'saturation': avg_saturation,
            'sharpness': avg_sharpness,
            'noise_level': avg_noise,
            'color_balance': avg_color_balance
        }

        # Calculate quality score
        quality_score = self.calculate_quality_score(analysis_dict)

        # Generate recommendations
        recommendations = self.generate_recommendations(analysis_dict, quality_score)

        print(f"✅ Analysis complete - Quality Score: {quality_score:.2f}")

        return VideoAnalysis(
            brightness=avg_brightness,
            contrast=avg_contrast,
            saturation=avg_saturation,
            sharpness=avg_sharpness,
            noise_level=avg_noise,
            color_balance=avg_color_balance,
            motion_intensity=motion_intensity,
            quality_score=quality_score,
            recommendations=recommendations
        )

class VideoEnhancer:
    """Intelligent video enhancement system"""

    def __init__(self, config: EnhancementConfig = None):
        self.config = config or EnhancementConfig()

    def enhance_brightness(self, frame: Image.Image, target_brightness: float = 0.5) -> Image.Image:
        """Enhance frame brightness to target level"""
        current_brightness = np.mean(np.array(frame)) / 255.0
        if abs(current_brightness - target_brightness) < 0.05:
            return frame

        adjustment_factor = target_brightness / current_brightness
        adjustment_factor = max(0.3, min(3.0, adjustment_factor))

        enhancer = ImageEnhance.Brightness(frame)
        return enhancer.enhance(adjustment_factor)

    def enhance_contrast(self, frame: Image.Image, target_contrast: float = 0.4) -> Image.Image:
        """Enhance frame contrast to target level"""
        gray_frame = frame.convert('L')
        current_contrast = np.std(np.array(gray_frame)) / 255.0

        if abs(current_contrast - target_contrast) < 0.05:
            return frame

        adjustment_factor = target_contrast / current_contrast
        adjustment_factor = max(0.5, min(2.5, adjustment_factor))

        enhancer = ImageEnhance.Contrast(frame)
        return enhancer.enhance(adjustment_factor)

    def enhance_saturation(self, frame: Image.Image, target_saturation: float = 0.6) -> Image.Image:
        """Enhance frame saturation to target level"""
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        hsv_frame = frame.convert('HSV')
        hsv_array = np.array(hsv_frame)
        current_saturation = np.mean(hsv_array[:, :, 1]) / 255.0

        if abs(current_saturation - target_saturation) < 0.05:
            return frame

        adjustment_factor = target_saturation / current_saturation
        adjustment_factor = max(0.3, min(2.0, adjustment_factor))

        enhancer = ImageEnhance.Color(frame)
        return enhancer.enhance(adjustment_factor)

    def enhance_sharpness(self, frame: Image.Image, target_sharpness: float = 5.0) -> Image.Image:
        """Enhance frame sharpness to target level"""
        gray_frame = frame.convert('L')
        gray_array = np.array(gray_frame)
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
        current_sharpness = np.var(laplacian) / 1000.0

        if abs(current_sharpness - target_sharpness) < 1.0:
            return frame

        if current_sharpness < target_sharpness:
            # Apply sharpening
            adjustment_factor = min(2.5, target_sharpness / current_sharpness)
        else:
            # Apply slight blur
            adjustment_factor = max(0.7, target_sharpness / current_sharpness)

        enhancer = ImageEnhance.Sharpness(frame)
        return enhancer.enhance(adjustment_factor)

    def reduce_noise(self, frame: Image.Image, strength: float = 0.5) -> Image.Image:
        """Apply noise reduction"""
        if strength < 0.1:
            return frame

        # Convert to numpy array
        img_array = np.array(frame)

        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(img_array, 9, 75, 75)

        # Blend with original based on strength
        result = img_array * (1 - strength) + denoised * strength
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    def correct_color_balance(self, frame: Image.Image, target_balance: Tuple[float, float, float] = (0.33, 0.33, 0.34)) -> Image.Image:
        """Correct color balance"""
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        img_array = np.array(frame).astype(np.float32)

        # Calculate current color balance
        r_mean = np.mean(img_array[:, :, 0]) / 255.0
        g_mean = np.mean(img_array[:, :, 1]) / 255.0
        b_mean = np.mean(img_array[:, :, 2]) / 255.0

        # Calculate correction factors
        r_factor = target_balance[0] / r_mean if r_mean > 0 else 1.0
        g_factor = target_balance[1] / g_mean if g_mean > 0 else 1.0
        b_factor = target_balance[2] / b_mean if b_mean > 0 else 1.0

        # Apply corrections
        img_array[:, :, 0] *= r_factor
        img_array[:, :, 1] *= g_factor
        img_array[:, :, 2] *= b_factor

        # Clip values
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def upscale_frame(self, frame: Image.Image, factor: float) -> Image.Image:
        """Upscale frame using advanced interpolation"""
        if factor <= 1.0:
            return frame

        original_size = frame.size
        new_size = (int(original_size[0] * factor), int(original_size[1] * factor))

        if self.config.quality_mode == "quality":
            # Use LANCZOS for best quality
            return frame.resize(new_size, Image.Resampling.LANCZOS)
        elif self.config.quality_mode == "fast":
            # Use BILINEAR for speed
            return frame.resize(new_size, Image.Resampling.BILINEAR)
        else:
            # Balanced approach
            return frame.resize(new_size, Image.Resampling.BICUBIC)

    def enhance_frame(self, frame: Image.Image, analysis: VideoAnalysis) -> Image.Image:
        """Enhance single frame based on analysis"""
        enhanced = frame.copy()

        # Apply enhancements based on configuration
        if self.config.auto_brightness:
            enhanced = self.enhance_brightness(enhanced, target_brightness=0.5)

        if self.config.auto_contrast:
            enhanced = self.enhance_contrast(enhanced, target_contrast=0.4)

        if self.config.auto_saturation:
            enhanced = self.enhance_saturation(enhanced, target_saturation=0.6)

        if self.config.auto_sharpen:
            enhanced = self.enhance_sharpness(enhanced, target_sharpness=5.0)

        if self.config.noise_reduction and analysis.noise_level > 0.05:
            reduction_strength = min(analysis.noise_level * 2, 0.8)
            enhanced = self.reduce_noise(enhanced, reduction_strength)

        if self.config.color_correction:
            enhanced = self.correct_color_balance(enhanced)

        if self.config.upscale_factor > 1.0:
            enhanced = self.upscale_frame(enhanced, self.config.upscale_factor)

        return enhanced

    def enhance_video(self, frames: List[Image.Image], progress_callback: Optional[callable] = None) -> List[Image.Image]:
        """Enhance entire video"""
        if not frames:
            return []

        print(f"🎬 Enhancing {len(frames)} frames...")

        # Analyze video first
        analyzer = VideoAnalyzer()
        analysis = analyzer.analyze_video(frames)

        print(f"📊 Quality Score: {analysis.quality_score:.2f}")
        if analysis.recommendations:
            print("💡 Recommendations:")
            for rec in analysis.recommendations:
                print(f"   • {rec}")

        enhanced_frames = []

        for i, frame in enumerate(frames):
            enhanced_frame = self.enhance_frame(frame, analysis)
            enhanced_frames.append(enhanced_frame)

            # Progress callback
            if progress_callback:
                progress = ((i + 1) / len(frames)) * 100
                progress_callback(progress, i + 1, len(frames))

            # Console progress
            if (i + 1) % max(1, len(frames) // 10) == 0:
                progress = ((i + 1) / len(frames)) * 100
                print(f"📈 Enhancement: {progress:.1f}% ({i + 1}/{len(frames)})")

        print(f"✅ Enhanced {len(enhanced_frames)} frames")
        return enhanced_frames

    def save_enhanced_video(self, frames: List[Image.Image], output_path: str, fps: int = 8) -> bool:
        """Save enhanced video"""
        try:
            if not frames:
                print("❌ No frames to save")
                return False

            print(f"💾 Saving enhanced video to {output_path}...")

            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=True
            )

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Saved enhanced video: {output_path} ({file_size:.1f} MB)")
            return True

        except Exception as e:
            print(f"❌ Failed to save enhanced video: {e}")
            return False

def load_video_frames(input_path: str) -> List[Image.Image]:
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
        try:
            frame = Image.open(input_path).convert('RGB')
            frames.append(frame)
        except Exception as e:
            print(f"Error loading image: {e}")
            return []

    return frames

def main():
    parser = argparse.ArgumentParser(description="Automated Video Enhancement System")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output enhanced video file")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't enhance")
    parser.add_argument("--quality", choices=["fast", "balanced", "quality"], default="balanced",
                       help="Enhancement quality mode")
    parser.add_argument("--upscale", type=float, default=1.0, help="Upscale factor")
    parser.add_argument("--brightness", action="store_true", help="Enable brightness correction")
    parser.add_argument("--contrast", action="store_true", help="Enable contrast correction")
    parser.add_argument("--saturation", action="store_true", help="Enable saturation correction")
    parser.add_argument("--sharpen", action="store_true", help="Enable sharpening")
    parser.add_argument("--denoise", action="store_true", help="Enable noise reduction")
    parser.add_argument("--color-correct", action="store_true", help="Enable color correction")
    parser.add_argument("--fps", type=int, default=8, help="Output FPS")

    args = parser.parse_args()

    # Load configuration
    config = EnhancementConfig(
        auto_brightness=args.brightness,
        auto_contrast=args.contrast,
        auto_saturation=args.saturation,
        auto_sharpen=args.sharpen,
        noise_reduction=args.denoise,
        color_correction=args.color_correct,
        upscale_factor=args.upscale,
        quality_mode=args.quality
    )

    # If config file provided, load it
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"⚠️  Failed to load config file: {e}")

    # Enable all corrections if none specified
    if not any([args.brightness, args.contrast, args.saturation, args.sharpen, args.denoise, args.color_correct]):
        config.auto_brightness = True
        config.auto_contrast = True
        config.auto_saturation = True
        config.auto_sharpen = True
        config.noise_reduction = True
        config.color_correction = True

    # Load video frames
    print(f"📁 Loading video from {args.input}...")
    frames = load_video_frames(args.input)
    if not frames:
        print("❌ Failed to load video")
        return

    print(f"✅ Loaded {len(frames)} frames")

    # Analyze video
    analyzer = VideoAnalyzer()
    analysis = analyzer.analyze_video(frames)

    print(f"\n📊 Video Analysis Results:")
    print(f"   Quality Score: {analysis.quality_score:.2f}/1.00")
    print(f"   Brightness: {analysis.brightness:.2f}")
    print(f"   Contrast: {analysis.contrast:.2f}")
    print(f"   Saturation: {analysis.saturation:.2f}")
    print(f"   Sharpness: {analysis.sharpness:.2f}")
    print(f"   Noise Level: {analysis.noise_level:.2f}")
    print(f"   Motion Intensity: {analysis.motion_intensity:.2f}")
    print(f"   Color Balance: R={analysis.color_balance[0]:.2f}, G={analysis.color_balance[1]:.2f}, B={analysis.color_balance[2]:.2f}")

    if analysis.recommendations:
        print(f"\n💡 Enhancement Recommendations:")
        for rec in analysis.recommendations:
            print(f"   • {rec}")

    if args.analyze_only:
        print("\n🔍 Analysis complete (no enhancement performed)")
        return

    # Enhance video
    print(f"\n🎬 Starting video enhancement...")
    enhancer = VideoEnhancer(config)

    def progress_callback(progress: float, current: int, total: int):
        print(f"📈 Enhancement: {progress:.1f}% ({current}/{total})")

    enhanced_frames = enhancer.enhance_video(frames, progress_callback)

    if enhanced_frames:
        # Save enhanced video
        success = enhancer.save_enhanced_video(enhanced_frames, args.output, args.fps)

        if success:
            print(f"\n🎉 Video enhancement complete!")
            print(f"📁 Output: {args.output}")
            print(f"📊 Enhanced {len(enhanced_frames)} frames")

            # Show before/after comparison
            before_analysis = analysis
            after_frames = enhanced_frames
            after_analysis = analyzer.analyze_video(after_frames)

            print(f"\n📈 Quality Improvement:")
            print(f"   Before: {before_analysis.quality_score:.2f}")
            print(f"   After:  {after_analysis.quality_score:.2f}")
            print(f"   Gain:   +{(after_analysis.quality_score - before_analysis.quality_score):.2f}")
        else:
            print("\n❌ Video enhancement failed")
    else:
        print("\n❌ No frames enhanced")

if __name__ == "__main__":
    main()