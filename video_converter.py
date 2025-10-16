#!/usr/bin/env python3
"""
Advanced Video Format Converter and Optimizer
Support for multiple formats, resolutions, and optimization profiles
"""
import os
import sys
import argparse
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageSequence
import tempfile
import shutil
from datetime import datetime

@dataclass
class ConversionProfile:
    """Video conversion profile"""
    name: str
    description: str
    format: str
    codec: str
    quality: int
    resolution: Optional[Tuple[int, int]]
    fps: Optional[int]
    bitrate: Optional[str]
    audio_codec: Optional[str]
    optimization: str  # speed, balanced, quality
    max_file_size: Optional[str]  # e.g., "10MB"

@dataclass
class ConversionJob:
    """Single conversion job"""
    input_path: str
    output_path: str
    profile: ConversionProfile
    custom_options: Dict[str, Any]

class VideoConverter:
    """Advanced video conversion and optimization system"""

    def __init__(self):
        self.profiles = self._initialize_profiles()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_converter_"))

    def _initialize_profiles(self) -> Dict[str, ConversionProfile]:
        """Initialize conversion profiles"""
        return {
            # GIF Profiles
            "gif_optimized": ConversionProfile(
                name="GIF Optimized",
                description="Optimized GIF for web use",
                format="gif",
                codec="gif",
                quality=85,
                resolution=None,
                fps=12,
                bitrate=None,
                audio_codec=None,
                optimization="balanced",
                max_file_size="50MB"
            ),
            "gif_high_quality": ConversionProfile(
                name="GIF High Quality",
                description="High quality GIF for desktop",
                format="gif",
                codec="gif",
                quality=95,
                resolution=None,
                fps=24,
                bitrate=None,
                audio_codec=None,
                optimization="quality",
                max_file_size="100MB"
            ),
            "gif_small": ConversionProfile(
                name="GIF Small Size",
                description="Small GIF for mobile/social",
                format="gif",
                codec="gif",
                quality=70,
                resolution=(320, 320),
                fps=8,
                bitrate=None,
                audio_codec=None,
                optimization="speed",
                max_file_size="5MB"
            ),

            # MP4 Profiles
            "mp4_web": ConversionProfile(
                name="MP4 Web Optimized",
                description="MP4 optimized for web streaming",
                format="mp4",
                codec="h264",
                quality=85,
                resolution=(1280, 720),
                fps=30,
                bitrate="2M",
                audio_codec="aac",
                optimization="balanced",
                max_file_size="50MB"
            ),
            "mp4_high_quality": ConversionProfile(
                name="MP4 High Quality",
                description="High quality MP4 for desktop",
                format="mp4",
                codec="h264",
                quality=95,
                resolution=(1920, 1080),
                fps=60,
                bitrate="8M",
                audio_codec="aac",
                optimization="quality",
                max_file_size="200MB"
            ),
            "mp4_mobile": ConversionProfile(
                name="MP4 Mobile",
                description="MP4 optimized for mobile devices",
                format="mp4",
                codec="h264",
                quality=75,
                resolution=(854, 480),
                fps=24,
                bitrate="1M",
                audio_codec="aac",
                optimization="speed",
                max_file_size="20MB"
            ),

            # WebM Profiles
            "webm_modern": ConversionProfile(
                name="WebM Modern",
                description="WebM with VP9 for modern browsers",
                format="webm",
                codec="vp9",
                quality=85,
                resolution=(1280, 720),
                fps=30,
                bitrate="1.5M",
                audio_codec="opus",
                optimization="balanced",
                max_file_size="30MB"
            ),
            "webm_efficient": ConversionProfile(
                name="WebM Efficient",
                description="Highly compressed WebM",
                format="webm",
                codec="av1",
                quality=75,
                resolution=(854, 480),
                fps=24,
                bitrate="800k",
                audio_codec="opus",
                optimization="quality",
                max_file_size="15MB"
            ),

            # Social Media Profiles
            "instagram_story": ConversionProfile(
                name="Instagram Story",
                description="Optimized for Instagram Stories (9:16)",
                format="mp4",
                codec="h264",
                quality=85,
                resolution=(1080, 1920),
                fps=30,
                bitrate="4M",
                audio_codec="aac",
                optimization="balanced",
                max_file_size="100MB"
            ),
            "tiktok_vertical": ConversionProfile(
                name="TikTok Vertical",
                description="Optimized for TikTok (9:16)",
                format="mp4",
                codec="h264",
                quality=80,
                resolution=(1080, 1920),
                fps=30,
                bitrate="3M",
                audio_codec="aac",
                optimization="speed",
                max_file_size="50MB"
            ),
            "youtube_short": ConversionProfile(
                name="YouTube Short",
                description="Optimized for YouTube Shorts (9:16)",
                format="mp4",
                codec="h264",
                quality=90,
                resolution=(1080, 1920),
                fps=30,
                bitrate="6M",
                audio_codec="aac",
                optimization="quality",
                max_file_size="100MB"
            ),

            # Professional Profiles
            "pro_res_4k": ConversionProfile(
                name="Professional 4K",
                description="4K professional quality",
                format="mp4",
                codec="h265",
                quality=95,
                resolution=(3840, 2160),
                fps=60,
                bitrate="20M",
                audio_codec="aac",
                optimization="quality",
                max_file_size="500MB"
            ),
            "cinema_quality": ConversionProfile(
                name="Cinema Quality",
                description="Cinema grade quality",
                format="mp4",
                codec="prores",
                quality=98,
                resolution=(4096, 2160),
                fps=24,
                bitrate="50M",
                audio_codec="pcm",
                optimization="quality",
                max_file_size="1GB"
            )
        }

    def load_video_info(self, input_path: str) -> Dict[str, Any]:
        """Load video information using FFprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-show_entries", "format=size,duration,bit_rate",
                "-of", "json",
                input_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return info
            else:
                print(f"FFprobe error: {result.stderr}")
                return {}

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading video info: {e}")
            return {}

    def convert_gif_to_mp4(self, input_path: str, output_path: str, profile: ConversionProfile) -> bool:
        """Convert GIF to MP4 using FFmpeg"""
        try:
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-c:v", profile.codec,
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p"
            ]

            # Add quality settings
            if profile.codec == "h264":
                cmd.extend(["-crf", str(100 - profile.quality)])

            # Add resolution if specified
            if profile.resolution:
                cmd.extend(["-vf", f"scale={profile.resolution[0]}:{profile.resolution[1]}"])

            # Add FPS if specified
            if profile.fps:
                cmd.extend(["-r", str(profile.fps)])

            # Add bitrate if specified
            if profile.bitrate:
                cmd.extend(["-b:v", profile.bitrate])

            # Add optimization settings
            if profile.optimization == "speed":
                cmd.extend(["-preset", "ultrafast"])
            elif profile.optimization == "quality":
                cmd.extend(["-preset", "slow"])
            else:
                cmd.extend(["-preset", "medium"])

            cmd.extend(["-y", output_path])

            print(f"🎬 Converting GIF to MP4...")
            print(f"📋 Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                return True
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Conversion error: {e}")
            return False

    def convert_mp4_to_gif(self, input_path: str, output_path: str, profile: ConversionProfile) -> bool:
        """Convert MP4 to GIF using FFmpeg and PIL"""
        try:
            # Create temporary frames directory
            frames_dir = self.temp_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            # Extract frames using FFmpeg
            fps = profile.fps or 12
            resolution = profile.resolution or None

            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-vf", f"fps={fps}",
                "-y"
            ]

            if resolution:
                cmd[3] = f"fps={fps},scale={resolution[0]}:{resolution[1]}"

            cmd.extend([
                f"{frames_dir}/frame_%04d.png"
            ])

            print(f"🎬 Extracting frames from MP4...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"❌ Frame extraction error: {result.stderr}")
                return False

            # Load frames and create GIF
            frames = []
            frame_files = sorted(frames_dir.glob("frame_*.png"))

            print(f"📁 Processing {len(frame_files)} frames...")

            for frame_file in frame_files:
                frame = Image.open(frame_file)
                frames.append(frame)

            if not frames:
                print("❌ No frames extracted")
                return False

            # Save as GIF
            print(f"💾 Creating GIF...")
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=True,
                quality=profile.quality
            )

            # Cleanup
            shutil.rmtree(frames_dir, ignore_errors=True)

            return True

        except Exception as e:
            print(f"❌ Conversion error: {e}")
            return False

    def convert_with_ffmpeg(self, input_path: str, output_path: str, profile: ConversionProfile) -> bool:
        """General FFmpeg conversion"""
        try:
            cmd = ["ffmpeg", "-i", input_path]

            # Video codec settings
            if profile.codec in ["h264", "h265"]:
                cmd.extend(["-c:v", profile.codec])
                # CRF value (lower = better quality)
                crf = 50 - (profile.quality // 2)  # Invert and scale
                cmd.extend(["-crf", str(crf)])

                if profile.codec == "h264":
                    cmd.extend(["-preset", "medium"])
                elif profile.codec == "h265":
                    cmd.extend(["-preset", "slow"])

            elif profile.codec == "vp9":
                cmd.extend(["-c:v", "libvpx-vp9"])
                quality_setting = 63 - (profile.quality // 2)
                cmd.extend(["-crf", str(quality_setting), "-b:v", "0"])

            elif profile.codec == "av1":
                cmd.extend(["-c:v", "libaom-av1"])
                quality_setting = 63 - (profile.quality // 2)
                cmd.extend(["-crf", str(quality_setting), "-b:v", "0"])

            elif profile.codec == "prores":
                cmd.extend(["-c:v", "prores_ks"])
                profile_name = ["proxy", "lt", "standard", "hq", "4444"][profile.quality // 20]
                cmd.extend(["-profile:v", profile_name])

            # Audio settings
            if profile.audio_codec:
                cmd.extend(["-c:a", profile.audio_codec])
                if profile.audio_codec == "aac":
                    cmd.extend(["-b:a", "128k"])

            # Resolution
            if profile.resolution:
                cmd.extend(["-vf", f"scale={profile.resolution[0]}:{profile.resolution[1]}"])

            # Frame rate
            if profile.fps:
                cmd.extend(["-r", str(profile.fps)])

            # Bitrate
            if profile.bitrate:
                cmd.extend(["-b:v", profile.bitrate])

            # Format-specific options
            if profile.format == "mp4":
                cmd.extend(["-movflags", "+faststart"])  # Web optimization

            cmd.extend(["-y", output_path])

            print(f"🎬 Converting with FFmpeg...")
            print(f"📋 Profile: {profile.name}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                return True
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Conversion error: {e}")
            return False

    def optimize_gif(self, input_path: str, output_path: str, profile: ConversionProfile) -> bool:
        """Optimize GIF file size"""
        try:
            # Load original GIF
            with Image.open(input_path) as img:
                frames = []
                for frame in ImageSequence.Iterator(img):
                    frames.append(frame.copy())

            if not frames:
                return False

            # Apply optimization based on profile
            optimized_frames = []

            for frame in frames:
                # Resize if needed
                if profile.resolution:
                    frame = frame.resize(profile.resolution, Image.Resampling.LANCZOS)

                # Reduce colors if needed
                if profile.quality < 90:
                    colors = max(2, 256 - (100 - profile.quality) * 2)
                    frame = frame.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

                optimized_frames.append(frame)

            # Save optimized GIF
            fps = profile.fps or 12
            optimized_frames[0].save(
                output_path,
                save_all=True,
                append_images=optimized_frames[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=True
            )

            return True

        except Exception as e:
            print(f"❌ GIF optimization error: {e}")
            return False

    def convert_video(self, job: ConversionJob, progress_callback: Optional[callable] = None) -> bool:
        """Perform video conversion"""
        input_path = job.input_path
        output_path = job.output_path
        profile = job.profile

        print(f"🎬 Starting video conversion...")
        print(f"📁 Input: {input_path}")
        print(f"📁 Output: {output_path}")
        print(f"📋 Profile: {profile.name} ({profile.format.upper()})")

        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ Input file not found: {input_path}")
            return False

        # Get input file info
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        print(f"📊 Input size: {input_size:.1f} MB")

        # Determine conversion method
        success = False

        input_ext = Path(input_path).suffix.lower()
        output_ext = Path(output_path).suffix.lower()

        # Handle specific conversions
        if input_ext == '.gif' and output_ext == '.mp4':
            success = self.convert_gif_to_mp4(input_path, output_path, profile)
        elif input_ext in ['.mp4', '.avi', '.mov', '.mkv'] and output_ext == '.gif':
            success = self.convert_mp4_to_gif(input_path, output_path, profile)
        elif input_ext == '.gif' and output_ext == '.gif':
            success = self.optimize_gif(input_path, output_path, profile)
        else:
            # General conversion
            success = self.convert_with_ffmpeg(input_path, output_path, profile)

        if success:
            # Get output file info
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            compression_ratio = (input_size - output_size) / input_size * 100

            print(f"✅ Conversion complete!")
            print(f"📊 Output size: {output_size:.1f} MB")
            print(f"📈 Compression: {compression_ratio:.1f}%")

            # Check max file size
            if profile.max_file_size:
                max_size_mb = float(profile.max_file_size.replace('MB', ''))
                if output_size > max_size_mb:
                    print(f"⚠️  Warning: Output exceeds max file size of {profile.max_file_size}")

        else:
            print("❌ Conversion failed")

        return success

    def batch_convert(self, jobs: List[ConversionJob], progress_callback: Optional[callable] = None) -> Dict[str, bool]:
        """Convert multiple videos"""
        results = {}

        print(f"🎬 Starting batch conversion of {len(jobs)} videos...")

        for i, job in enumerate(jobs):
            print(f"\n📋 Job {i + 1}/{len(jobs)}: {Path(job.input_path).name}")

            success = self.convert_video(job, progress_callback)
            results[job.input_path] = success

            if progress_callback:
                progress = ((i + 1) / len(jobs)) * 100
                progress_callback(progress, i + 1, len(jobs))

        successful = sum(results.values())
        print(f"\n✅ Batch conversion complete: {successful}/{len(jobs)} successful")

        return results

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported input and output formats"""
        return {
            "input": [".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"],
            "output": [".gif", ".mp4", ".webm", ".avi", ".mov"]
        }

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Advanced Video Format Converter and Optimizer")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--profile", help="Conversion profile",
                       choices=["gif_optimized", "gif_high_quality", "gif_small",
                                "mp4_web", "mp4_high_quality", "mp4_mobile",
                                "webm_modern", "webm_efficient",
                                "instagram_story", "tiktok_vertical", "youtube_short",
                                "pro_res_4k", "cinema_quality"])
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles")
    parser.add_argument("--batch", help="JSON file with batch conversion jobs")
    parser.add_argument("--quality", type=int, choices=range(1, 101), help="Override quality (1-100)")
    parser.add_argument("--resolution", help="Override resolution (WxH)")
    parser.add_argument("--fps", type=int, help="Override FPS")
    parser.add_argument("--bitrate", help="Override bitrate (e.g., 2M)")
    parser.add_argument("--max-size", help="Maximum file size (e.g., 50MB)")

    args = parser.parse_args()

    converter = VideoConverter()

    if args.list_profiles:
        print("🎬 Available Conversion Profiles:")
        print("=" * 60)
        for name, profile in converter.profiles.items():
            print(f"\n📋 {profile.name}")
            print(f"   Description: {profile.description}")
            print(f"   Format: {profile.format.upper()}")
            print(f"   Codec: {profile.codec}")
            print(f"   Quality: {profile.quality}/100")
            if profile.resolution:
                print(f"   Resolution: {profile.resolution[0]}x{profile.resolution[1]}")
            if profile.fps:
                print(f"   FPS: {profile.fps}")
            if profile.bitrate:
                print(f"   Bitrate: {profile.bitrate}")
            if profile.max_file_size:
                print(f"   Max Size: {profile.max_file_size}")
        return

    # Batch conversion
    if args.batch:
        try:
            with open(args.batch, 'r') as f:
                batch_data = json.load(f)

            jobs = []
            for job_data in batch_data.get('jobs', []):
                profile_name = job_data.get('profile', 'mp4_web')
                profile = converter.profiles.get(profile_name)
                if not profile:
                    print(f"⚠️  Unknown profile: {profile_name}")
                    continue

                job = ConversionJob(
                    input_path=job_data['input'],
                    output_path=job_data['output'],
                    profile=profile,
                    custom_options=job_data.get('options', {})
                )
                jobs.append(job)

            if jobs:
                def progress_callback(progress: float, current: int, total: int):
                    print(f"📊 Batch progress: {progress:.1f}% ({current}/{total})")

                results = converter.batch_convert(jobs, progress_callback)

                print(f"\n📊 Batch Results:")
                for input_file, success in results.items():
                    status = "✅" if success else "❌"
                    print(f"   {status} {Path(input_file).name}")

        except Exception as e:
            print(f"❌ Batch processing error: {e}")
        return

    # Single file conversion
    if not args.input or not args.output:
        print("❌ Input and output files required")
        return

    # Determine profile
    if args.profile:
        profile = converter.profiles.get(args.profile)
        if not profile:
            print(f"❌ Unknown profile: {args.profile}")
            return
    else:
        # Auto-detect profile based on output extension
        output_ext = Path(args.output).suffix.lower()
        if output_ext == '.gif':
            profile = converter.profiles['gif_optimized']
        elif output_ext == '.mp4':
            profile = converter.profiles['mp4_web']
        elif output_ext == '.webm':
            profile = converter.profiles['webm_modern']
        else:
            print(f"⚠️  Unknown output format: {output_ext}, using default MP4 web profile")
            profile = converter.profiles['mp4_web']

    # Override profile settings with command line arguments
    if args.quality:
        profile.quality = args.quality
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            profile.resolution = (width, height)
        except:
            print(f"⚠️  Invalid resolution format: {args.resolution}")
    if args.fps:
        profile.fps = args.fps
    if args.bitrate:
        profile.bitrate = args.bitrate
    if args.max_size:
        profile.max_file_size = args.max_size

    # Create conversion job
    job = ConversionJob(
        input_path=args.input,
        output_path=args.output,
        profile=profile,
        custom_options={}
    )

    # Perform conversion
    success = converter.convert_video(job)

    if success:
        print(f"\n🎉 Video conversion complete!")
        print(f"📁 Output: {args.output}")
        print(f"📋 Profile: {profile.name}")
    else:
        print(f"\n❌ Video conversion failed")

    # Cleanup
    converter.cleanup()

if __name__ == "__main__":
    main()