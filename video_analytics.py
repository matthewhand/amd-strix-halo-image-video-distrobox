#!/usr/bin/env python3
"""
Video Analytics and Metrics Dashboard
Comprehensive video analysis, performance tracking, and insights
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image, ImageStat, ImageSequence
import json
from datetime import datetime, timedelta
import cv2
import colorsys
import math
import statistics

@dataclass
class VideoMetrics:
    """Comprehensive video metrics"""
    filename: str
    file_size: int
    duration: float
    frame_count: int
    fps: float
    resolution: Tuple[int, int]
    aspect_ratio: float

    # Quality metrics
    brightness_avg: float
    contrast_avg: float
    saturation_avg: float
    sharpness_avg: float
    noise_level: float

    # Color metrics
    dominant_colors: List[Tuple[int, int, int]]
    color_palette: List[str]
    color_diversity: float
    warm_cool_ratio: float

    # Motion metrics
    motion_intensity: float
    motion_consistency: float
    scene_changes: int

    # Performance metrics
    complexity_score: float
    quality_score: float
    compression_efficiency: float

    # Technical metrics
    bitrate: Optional[float]
    codec: Optional[str]
    format: str

    # Timestamps
    analyzed_at: datetime

@dataclass
class AnalyticsReport:
    """Analytics report for multiple videos"""
    total_videos: int
    total_size: int
    total_duration: float
    average_quality: float
    format_distribution: Dict[str, int]
    resolution_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]
    top_performing_videos: List[VideoMetrics]
    recommendations: List[str]
    generated_at: datetime

class VideoAnalyzer:
    """Advanced video analysis engine"""

    def __init__(self):
        self.supported_formats = ['.gif', '.mp4', '.avi', '.mov', '.mkv', '.webm']

    def analyze_gif(self, file_path: str) -> VideoMetrics:
        """Analyze GIF file"""
        try:
            with Image.open(file_path) as img:
                # Basic properties
                frame_count = getattr(img, 'n_frames', 1)
                width, height = img.size
                resolution = (width, height)
                aspect_ratio = width / height

                # Duration (approximate)
                duration = frame_count * 0.1  # Default 100ms per frame
                fps = frame_count / duration if duration > 0 else 10

                # Load all frames for analysis
                frames = []
                for frame_index in range(frame_count):
                    img.seek(frame_index)
                    frame = img.copy().convert('RGB')
                    frames.append(frame)

                # Analyze frames
                metrics = self._analyze_frames(frames, file_path, resolution, duration, fps, frame_count)
                metrics.format = 'GIF'
                metrics.codec = 'GIF'

                return metrics

        except Exception as e:
            print(f"Error analyzing GIF {file_path}: {e}")
            return self._create_empty_metrics(file_path)

    def analyze_video_file(self, file_path: str) -> VideoMetrics:
        """Analyze video file using OpenCV"""
        try:
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                print(f"Could not open video file: {file_path}")
                return self._create_empty_metrics(file_path)

            # Get basic properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = (width, height)
            aspect_ratio = width / height
            duration = frame_count / fps if fps > 0 else 0

            # Sample frames for analysis (to avoid memory issues)
            sample_interval = max(1, frame_count // 100)  # Sample up to 100 frames
            frames = []

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    frames.append(pil_frame)

                frame_idx += 1

            cap.release()

            # Analyze frames
            metrics = self._analyze_frames(frames, file_path, resolution, duration, fps, len(frames))

            # Determine format and codec
            ext = Path(file_path).suffix.lower()
            metrics.format = ext.upper().replace('.', '')
            metrics.codec = self._detect_codec(file_path)

            return metrics

        except Exception as e:
            print(f"Error analyzing video {file_path}: {e}")
            return self._create_empty_metrics(file_path)

    def _detect_codec(self, file_path: str) -> str:
        """Detect video codec using FFprobe"""
        try:
            import subprocess
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "Unknown"

    def _analyze_frames(self, frames: List[Image.Image], file_path: str,
                       resolution: Tuple[int, int], duration: float,
                       fps: float, frame_count: int) -> VideoMetrics:
        """Analyze video frames"""
        if not frames:
            return self._create_empty_metrics(file_path)

        print(f"🔍 Analyzing {len(frames)} frames from {Path(file_path).name}...")

        # File properties
        file_size = os.path.getsize(file_path)
        filename = Path(file_path).name

        # Initialize metrics
        brightness_values = []
        contrast_values = []
        saturation_values = []
        sharpness_values = []

        # Color analysis
        all_colors = []
        color_hues = []

        # Motion analysis (if we have consecutive frames)
        motion_scores = []

        prev_frame = None
        for i, frame in enumerate(frames):
            # Progress update
            if i % max(1, len(frames) // 10) == 0:
                print(f"📊 Analyzing frame {i + 1}/{len(frames)}")

            # Basic statistics
            stat = ImageStat.Stat(frame)
            brightness = sum(stat.mean) / len(stat.mean) / 255.0
            brightness_values.append(brightness)

            contrast = sum(stat.stddev) / len(stat.stddev) / 255.0
            contrast_values.append(contrast)

            # Saturation
            if frame.mode == 'RGB':
                hsv_frame = frame.convert('HSV')
                hsv_array = np.array(hsv_frame)
                saturation = np.mean(hsv_array[:, :, 1]) / 255.0
                saturation_values.append(saturation)

                # Color collection
                frame_array = np.array(frame)
                pixels = frame_array.reshape(-1, 3)
                all_colors.extend(pixels[::max(1, len(pixels) // 1000)])  # Sample 1000 colors per frame

                # Hue analysis
                for hue in hsv_array[:, :, 0].flatten():
                    color_hues.append(hue)

            # Sharpness
            gray_frame = frame.convert('L')
            gray_array = np.array(gray_frame)
            laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
            sharpness = np.var(laplacian) / 1000.0
            sharpness_values.append(sharpness)

            # Motion analysis
            if prev_frame is not None:
                prev_gray = np.array(prev_frame.convert('L'))
                curr_gray = gray_array

                # Calculate optical flow or frame difference
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(diff) / 255.0
                motion_scores.append(motion_score)

            prev_frame = frame

        # Calculate averages
        brightness_avg = np.mean(brightness_values)
        contrast_avg = np.mean(contrast_values)
        saturation_avg = np.mean(saturation_values) if saturation_values else 0.5
        sharpness_avg = np.mean(sharpness_values)

        # Color analysis
        dominant_colors = self._get_dominant_colors(all_colors, 5)
        color_palette = self._colors_to_hex(dominant_colors)
        color_diversity = self._calculate_color_diversity(all_colors)
        warm_cool_ratio = self._calculate_warm_cool_ratio(color_hues)

        # Noise level
        if frames:
            sample_frame = frames[len(frames) // 2]
            noise_level = self._calculate_noise_level(sample_frame)
        else:
            noise_level = 0.0

        # Motion metrics
        if motion_scores:
            motion_intensity = np.mean(motion_scores)
            motion_consistency = 1.0 - np.std(motion_scores)  # Higher = more consistent
        else:
            motion_intensity = 0.0
            motion_consistency = 1.0

        # Scene changes (simple detection)
        scene_changes = self._detect_scene_changes(frames)

        # Calculate scores
        complexity_score = self._calculate_complexity_score(
            brightness_avg, contrast_avg, saturation_avg, motion_intensity, color_diversity
        )
        quality_score = self._calculate_quality_score(
            brightness_avg, contrast_avg, saturation_avg, sharpness_avg, noise_level
        )
        compression_efficiency = self._calculate_compression_efficiency(
            file_size, resolution, frame_count, fps
        )

        # Calculate bitrate (approximate)
        bitrate = (file_size * 8) / duration if duration > 0 else None

        return VideoMetrics(
            filename=filename,
            file_size=file_size,
            duration=duration,
            frame_count=frame_count,
            fps=fps,
            resolution=resolution,
            aspect_ratio=resolution[0] / resolution[1],
            brightness_avg=brightness_avg,
            contrast_avg=contrast_avg,
            saturation_avg=saturation_avg,
            sharpness_avg=sharpness_avg,
            noise_level=noise_level,
            dominant_colors=dominant_colors,
            color_palette=color_palette,
            color_diversity=color_diversity,
            warm_cool_ratio=warm_cool_ratio,
            motion_intensity=motion_intensity,
            motion_consistency=motion_consistency,
            scene_changes=scene_changes,
            complexity_score=complexity_score,
            quality_score=quality_score,
            compression_efficiency=compression_efficiency,
            bitrate=bitrate,
            codec=None,
            format="Unknown",
            analyzed_at=datetime.now()
        )

    def _get_dominant_colors(self, colors: List[Tuple[int, int, int]], num_colors: int) -> List[Tuple[int, int, int]]:
        """Get dominant colors using K-means clustering"""
        if len(colors) < num_colors:
            return [(int(c[0]), int(c[1]), int(c[2])) for c in colors[:num_colors]]

        try:
            colors_array = np.array(colors)
            # Simple color quantization
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, random_state=42)
            kmeans.fit(colors_array)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in dominant_colors]
        except ImportError:
            # Fallback: simple color binning
            color_bins = {}
            bin_size = 32

            for color in colors:
                binned_color = (
                    (color[0] // bin_size) * bin_size,
                    (color[1] // bin_size) * bin_size,
                    (color[2] // bin_size) * bin_size
                )
                color_bins[binned_color] = color_bins.get(binned_color, 0) + 1

            sorted_colors = sorted(color_bins.items(), key=lambda x: x[1], reverse=True)
            return [color for color, count in sorted_colors[:num_colors]]

    def _colors_to_hex(self, colors: List[Tuple[int, int, int]]) -> List[str]:
        """Convert RGB colors to hex strings"""
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]

    def _calculate_color_diversity(self, colors: List[Tuple[int, int, int]]) -> float:
        """Calculate color diversity score"""
        if len(colors) < 2:
            return 0.0

        # Sample colors for efficiency
        sample_size = min(1000, len(colors))
        sampled_colors = colors[::len(colors) // sample_size]

        # Calculate standard deviation of colors
        colors_array = np.array(sampled_colors)
        std_dev = np.std(colors_array)
        return min(1.0, std_dev / 128.0)  # Normalize to 0-1

    def _calculate_warm_cool_ratio(self, hues: List[float]) -> float:
        """Calculate warm vs cool color ratio"""
        if not hues:
            return 1.0

        warm_hues = [h for h in hues if 0 <= h <= 60 or 300 <= h <= 360]  # Reds, oranges, yellows
        cool_hues = [h for h in hues if 90 <= h <= 270]  # Greens, blues, purples

        total = len(warm_hues) + len(cool_hues)
        if total == 0:
            return 1.0

        return len(warm_hues) / total

    def _calculate_noise_level(self, frame: Image.Image) -> float:
        """Calculate noise level in frame"""
        gray_frame = frame.convert('L')
        gray_array = np.array(gray_frame)

        # Use median filter to estimate noise
        denoised = cv2.medianBlur(gray_array, 5)
        noise = np.mean(np.abs(gray_array.astype(float) - denoised.astype(float)))

        return min(1.0, noise / 64.0)  # Normalize to 0-1

    def _detect_scene_changes(self, frames: List[Image.Image]) -> int:
        """Detect scene changes in video"""
        if len(frames) < 2:
            return 0

        scene_changes = 0
        prev_hist = None

        for frame in frames:
            gray_frame = frame.convert('L')
            hist = cv2.calcHist([np.array(gray_frame)], [0], None, [256], [0, 256])

            if prev_hist is not None:
                # Calculate histogram correlation
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < 0.7:  # Threshold for scene change
                    scene_changes += 1

            prev_hist = hist

        return scene_changes

    def _calculate_complexity_score(self, brightness: float, contrast: float,
                                  saturation: float, motion: float, diversity: float) -> float:
        """Calculate overall complexity score"""
        # Weighted combination of complexity factors
        weights = [0.2, 0.3, 0.2, 0.2, 0.1]  # brightness, contrast, saturation, motion, diversity
        factors = [brightness, contrast, saturation, motion, diversity]

        complexity = sum(w * f for w, f in zip(weights, factors))
        return min(1.0, complexity)

    def _calculate_quality_score(self, brightness: float, contrast: float,
                               saturation: float, sharpness: float, noise: float) -> float:
        """Calculate overall quality score"""
        # Ideal values and scoring
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        contrast_score = min(contrast * 2, 1.0)
        saturation_score = 1.0 - abs(saturation - 0.6) * 2
        sharpness_score = min(sharpness / 10.0, 1.0)
        noise_score = 1.0 - min(noise * 5, 1.0)

        weights = [0.2, 0.25, 0.2, 0.25, 0.1]
        scores = [brightness_score, contrast_score, saturation_score, sharpness_score, noise_score]

        quality = sum(w * s for w, s in zip(weights, scores))
        return max(0.0, min(1.0, quality))

    def _calculate_compression_efficiency(self, file_size: int, resolution: Tuple[int, int],
                                        frame_count: int, fps: float) -> float:
        """Calculate compression efficiency"""
        # Theoretical uncompressed size
        pixels_per_frame = resolution[0] * resolution[1]
        bytes_per_pixel = 3  # RGB
        uncompressed_size = pixels_per_frame * bytes_per_pixel * frame_count

        if uncompressed_size == 0:
            return 1.0

        compression_ratio = file_size / uncompressed_size
        # Higher score for better compression (lower ratio)
        efficiency = max(0.0, 1.0 - compression_ratio * 10)
        return efficiency

    def _create_empty_metrics(self, file_path: str) -> VideoMetrics:
        """Create empty metrics for failed analysis"""
        return VideoMetrics(
            filename=Path(file_path).name,
            file_size=0,
            duration=0,
            frame_count=0,
            fps=0,
            resolution=(0, 0),
            aspect_ratio=0,
            brightness_avg=0,
            contrast_avg=0,
            saturation_avg=0,
            sharpness_avg=0,
            noise_level=0,
            dominant_colors=[],
            color_palette=[],
            color_diversity=0,
            warm_cool_ratio=0,
            motion_intensity=0,
            motion_consistency=0,
            scene_changes=0,
            complexity_score=0,
            quality_score=0,
            compression_efficiency=0,
            bitrate=None,
            codec=None,
            format="Unknown",
            analyzed_at=datetime.now()
        )

class AnalyticsDashboard:
    """Video analytics dashboard and reporting"""

    def __init__(self):
        self.analyzer = VideoAnalyzer()

    def analyze_directory(self, directory: str, recursive: bool = True) -> List[VideoMetrics]:
        """Analyze all videos in directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"❌ Directory not found: {directory}")
            return []

        # Find video files
        video_files = []
        if recursive:
            for ext in self.analyzer.supported_formats:
                video_files.extend(dir_path.rglob(f"*{ext}"))
        else:
            for ext in self.analyzer.supported_formats:
                video_files.extend(dir_path.glob(f"*{ext}"))

        if not video_files:
            print(f"❌ No video files found in {directory}")
            return []

        print(f"🎬 Found {len(video_files)} video files")
        metrics = []

        for video_file in sorted(video_files):
            print(f"\n📁 Analyzing: {video_file.name}")
            try:
                if video_file.suffix.lower() == '.gif':
                    metric = self.analyzer.analyze_gif(str(video_file))
                else:
                    metric = self.analyzer.analyze_video_file(str(video_file))
                metrics.append(metric)
            except Exception as e:
                print(f"❌ Failed to analyze {video_file.name}: {e}")

        return metrics

    def generate_report(self, metrics: List[VideoMetrics]) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        if not metrics:
            return AnalyticsReport(
                total_videos=0, total_size=0, total_duration=0, average_quality=0,
                format_distribution={}, resolution_distribution={}, quality_distribution={},
                top_performing_videos=[], recommendations=[], generated_at=datetime.now()
            )

        # Basic statistics
        total_videos = len(metrics)
        total_size = sum(m.file_size for m in metrics)
        total_duration = sum(m.duration for m in metrics)
        average_quality = sum(m.quality_score for m in metrics) / total_videos

        # Format distribution
        format_distribution = {}
        for metric in metrics:
            format_distribution[metric.format] = format_distribution.get(metric.format, 0) + 1

        # Resolution distribution
        resolution_distribution = {}
        for metric in metrics:
            res_key = f"{metric.resolution[0]}x{metric.resolution[1]}"
            resolution_distribution[res_key] = resolution_distribution.get(res_key, 0) + 1

        # Quality distribution
        quality_distribution = {"High": 0, "Medium": 0, "Low": 0}
        for metric in metrics:
            if metric.quality_score >= 0.7:
                quality_distribution["High"] += 1
            elif metric.quality_score >= 0.4:
                quality_distribution["Medium"] += 1
            else:
                quality_distribution["Low"] += 1

        # Top performing videos (by quality score)
        top_performing = sorted(metrics, key=lambda m: m.quality_score, reverse=True)[:10]

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        return AnalyticsReport(
            total_videos=total_videos,
            total_size=total_size,
            total_duration=total_duration,
            average_quality=average_quality,
            format_distribution=format_distribution,
            resolution_distribution=resolution_distribution,
            quality_distribution=quality_distribution,
            top_performing_videos=top_performing,
            recommendations=recommendations,
            generated_at=datetime.now()
        )

    def _generate_recommendations(self, metrics: List[VideoMetrics]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if not metrics:
            return recommendations

        avg_quality = sum(m.quality_score for m in metrics) / len(metrics)
        avg_compression = sum(m.compression_efficiency for m in metrics) / len(metrics)

        # Quality recommendations
        if avg_quality < 0.6:
            recommendations.append("Consider enhancing video quality - many videos have low quality scores")
        elif avg_quality > 0.9:
            recommendations.append("Excellent video quality across the library")

        # Compression recommendations
        if avg_compression < 0.5:
            recommendations.append("Videos could benefit from better compression to reduce file sizes")

        # Format recommendations
        formats = [m.format for m in metrics]
        if formats.count('GIF') > len(formats) * 0.5:
            recommendations.append("Consider converting some GIFs to MP4 for better compression and quality")

        # Resolution recommendations
        resolutions = [m.resolution for m in metrics]
        avg_resolution = sum(r[0] * r[1] for r in resolutions) / len(resolutions)
        if avg_resolution < 640 * 480:
            recommendations.append("Most videos are low resolution - consider upscaling for better quality")

        # Color recommendations
        avg_saturation = sum(m.saturation_avg for m in metrics) / len(metrics)
        if avg_saturation < 0.3:
            recommendations.append("Videos appear desaturated - consider color enhancement")

        # Motion recommendations
        avg_motion = sum(m.motion_intensity for m in metrics) / len(metrics)
        if avg_motion < 0.1:
            recommendations.append("Most videos have minimal motion - consider adding more dynamic content")
        elif avg_motion > 0.8:
            recommendations.append("High motion intensity detected - ensure frame rates are adequate")

        return recommendations

    def export_metrics(self, metrics: List[VideoMetrics], output_file: str, format: str = "json"):
        """Export metrics to file"""
        try:
            if format.lower() == "json":
                data = {
                    "metrics": [asdict(m) for m in metrics],
                    "generated_at": datetime.now().isoformat(),
                    "total_count": len(metrics)
                }

                # Convert datetime objects to strings
                for metric in data["metrics"]:
                    metric["analyzed_at"] = metric["analyzed_at"].isoformat()

                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)

            elif format.lower() == "csv":
                import csv
                with open(output_file, 'w', newline='') as f:
                    if metrics:
                        fieldnames = asdict(metrics[0]).keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for metric in metrics:
                            row = asdict(metric)
                            row['analyzed_at'] = row['analyzed_at'].isoformat()
                            writer.writerow(row)

            print(f"✅ Exported {len(metrics)} metrics to {output_file}")

        except Exception as e:
            print(f"❌ Failed to export metrics: {e}")

    def print_report(self, report: AnalyticsReport):
        """Print formatted analytics report"""
        print("\n" + "=" * 60)
        print("📊 VIDEO ANALYTICS REPORT")
        print("=" * 60)

        print(f"\n📁 Library Overview:")
        print(f"   Total Videos: {report.total_videos:,}")
        print(f"   Total Size: {report.total_size / (1024**3):.2f} GB")
        print(f"   Total Duration: {report.total_duration / 3600:.1f} hours")
        print(f"   Average Quality: {report.average_quality:.2f}/1.00")

        print(f"\n📊 Format Distribution:")
        for format_name, count in sorted(report.format_distribution.items()):
            percentage = (count / report.total_videos) * 100
            print(f"   {format_name}: {count} ({percentage:.1f}%)")

        print(f"\n📺 Resolution Distribution:")
        for resolution, count in sorted(report.resolution_distribution.items(),
                                     key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / report.total_videos) * 100
            print(f"   {resolution}: {count} ({percentage:.1f}%)")

        print(f"\n🎯 Quality Distribution:")
        for quality_level, count in report.quality_distribution.items():
            percentage = (count / report.total_videos) * 100
            print(f"   {quality_level}: {count} ({percentage:.1f}%)")

        if report.top_performing_videos:
            print(f"\n🏆 Top Performing Videos:")
            for i, video in enumerate(report.top_performing_videos[:5], 1):
                size_mb = video.file_size / (1024 * 1024)
                print(f"   {i}. {video.filename}")
                print(f"      Quality: {video.quality_score:.2f} | Size: {size_mb:.1f}MB | {video.resolution[0]}x{video.resolution[1]}")

        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"   {i}. {rec}")

        print(f"\n📅 Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Video Analytics and Metrics Dashboard")
    parser.add_argument("input", help="Input file or directory to analyze")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format")
    parser.add_argument("--recursive", action="store_true", help="Analyze directories recursively")
    parser.add_argument("--top", type=int, default=10, help="Number of top videos to show")
    parser.add_argument("--sort", choices=["quality", "size", "duration", "complexity"],
                       default="quality", help="Sort videos by metric")

    args = parser.parse_args()

    dashboard = AnalyticsDashboard()

    # Determine if input is file or directory
    input_path = Path(args.input)
    metrics = []

    if input_path.is_file():
        print(f"📁 Analyzing file: {input_path.name}")
        if input_path.suffix.lower() == '.gif':
            metric = dashboard.analyzer.analyze_gif(str(input_path))
        else:
            metric = dashboard.analyzer.analyze_video_file(str(input_path))
        metrics = [metric]
    elif input_path.is_dir():
        print(f"📁 Analyzing directory: {input_path}")
        metrics = dashboard.analyze_directory(str(input_path), args.recursive)
    else:
        print(f"❌ Input not found: {args.input}")
        return

    if not metrics:
        print("❌ No videos analyzed")
        return

    # Sort metrics
    if args.sort == "quality":
        metrics.sort(key=lambda m: m.quality_score, reverse=True)
    elif args.sort == "size":
        metrics.sort(key=lambda m: m.file_size, reverse=True)
    elif args.sort == "duration":
        metrics.sort(key=lambda m: m.duration, reverse=True)
    elif args.sort == "complexity":
        metrics.sort(key=lambda m: m.complexity_score, reverse=True)

    # Generate and print report
    report = dashboard.generate_report(metrics)
    dashboard.print_report(report)

    # Export if requested
    if args.output:
        dashboard.export_metrics(metrics, args.output, args.format)

    # Show top videos
    if len(metrics) > 1:
        print(f"\n🏆 Top {min(args.top, len(metrics))} Videos (by {args.sort}):")
        for i, video in enumerate(metrics[:args.top], 1):
            print(f"   {i}. {video.filename}")
            print(f"      Quality: {video.quality_score:.2f} | Complexity: {video.complexity_score:.2f}")
            print(f"      Size: {video.file_size / (1024*1024):.1f}MB | Duration: {video.duration:.1f}s")
            print(f"      Resolution: {video.resolution[0]}x{video.resolution[1]} | FPS: {video.fps:.1f}")

if __name__ == "__main__":
    main()