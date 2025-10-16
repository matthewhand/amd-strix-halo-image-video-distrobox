#!/usr/bin/env python3
"""
Wan Video Templates and Quality Presets System
Predefined templates for common video generation scenarios
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class VideoTemplate:
    """Video generation template"""
    name: str
    description: str
    prompt_template: str
    frames: int
    fps: int
    noise_level: str
    aspect_ratio: str
    resolution: tuple
    category: str
    tags: List[str]
    estimated_time: float  # seconds
    quality_preset: str = "standard"

@dataclass
class QualityPreset:
    """Quality preset configuration"""
    name: str
    description: str
    frames: int
    fps: int
    noise_level: str
    resolution: tuple
    memory_usage: str  # low, medium, high
    speed_factor: float  # relative to standard
    quality_factor: float  # 1.0 = standard

class VideoTemplateManager:
    """Manage video templates and quality presets"""

    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.templates_file = self.templates_dir / "templates.json"
        self.presets_file = self.templates_dir / "presets.json"

        # Initialize default templates and presets
        self.templates = self._load_default_templates()
        self.presets = self._load_default_presets()

        # Load custom templates if they exist
        self._load_custom_templates()

    def _load_default_templates(self) -> Dict[str, VideoTemplate]:
        """Load default video templates"""
        templates = {}

        # Motion templates
        templates["subtle_breathing"] = VideoTemplate(
            name="Subtle Breathing",
            description="Gentle breathing effect for portraits and objects",
            prompt_template="{subject} with subtle breathing movement, lifelike gentle motion",
            frames=16,
            fps=8,
            noise_level="low",
            aspect_ratio="1:1",
            resolution=(512, 512),
            category="motion",
            tags=["portrait", "breathing", "subtle"],
            estimated_time=15,
            quality_preset="standard"
        )

        templates["gentle_pulse"] = VideoTemplate(
            name="Gentle Pulse",
            description="Soft pulsing animation for highlights and glowing elements",
            prompt_template="{subject} with gentle pulsing glow, soft radiating light waves",
            frames=20,
            fps=10,
            noise_level="low",
            aspect_ratio="16:9",
            resolution=(768, 432),
            category="motion",
            tags=["pulse", "glow", "gentle"],
            estimated_time=20,
            quality_preset="standard"
        )

        templates["slow_pan"] = VideoTemplate(
            name="Slow Pan",
            description="Slow panning motion across landscapes and scenes",
            prompt_template="{subject} with slow cinematic panning movement, smooth camera motion",
            frames=24,
            fps=12,
            noise_level="medium",
            aspect_ratio="16:9",
            resolution=(768, 432),
            category="camera",
            tags=["pan", "cinematic", "landscape"],
            estimated_time=25,
            quality_preset="high"
        )

        # Nature templates
        templates["wind_sway"] = VideoTemplate(
            name="Wind Sway",
            description="Natural swaying motion in wind for trees and plants",
            prompt_template="{subject} swaying gently in the wind, natural movement patterns",
            frames=32,
            fps=16,
            noise_level="medium",
            aspect_ratio="9:16",
            resolution=(432, 768),
            category="nature",
            tags=["wind", "sway", "plants", "trees"],
            estimated_time=30,
            quality_preset="high"
        )

        templates["water_ripple"] = VideoTemplate(
            name="Water Ripple",
            description="Gentle water rippling and wave patterns",
            prompt_template="{subject} with gentle water ripples, soft wave motion patterns",
            frames=24,
            fps=12,
            noise_level="medium",
            aspect_ratio="1:1",
            resolution=(512, 512),
            category="nature",
            tags=["water", "ripples", "waves"],
            estimated_time=25,
            quality_preset="high"
        )

        templates["cloud_drift"] = VideoTemplate(
            name="Cloud Drift",
            description="Slow drifting cloud movement and sky changes",
            prompt_template="{subject} with slowly drifting clouds, atmospheric movement",
            frames=48,
            fps=12,
            noise_level="low",
            aspect_ratio="16:9",
            resolution=(768, 432),
            category="nature",
            tags=["clouds", "sky", "atmosphere"],
            estimated_time=45,
            quality_preset="cinema"
        )

        # Action templates
        templates["action_zoom"] = VideoTemplate(
            name="Action Zoom",
            description="Dynamic zooming motion for action scenes",
            prompt_template="{subject} with dynamic zoom motion, high energy movement",
            frames=16,
            fps=16,
            noise_level="high",
            aspect_ratio="16:9",
            resolution=(768, 432),
            category="action",
            tags=["zoom", "action", "dynamic"],
            estimated_time=15,
            quality_preset="high"
        )

        templates["rotate_spin"] = VideoTemplate(
            name="Rotate Spin",
            description="Spinning rotation motion for objects and characters",
            prompt_template="{subject} with smooth spinning rotation, 360 degree turn",
            frames=32,
            fps=16,
            noise_level="high",
            aspect_ratio="1:1",
            resolution=(512, 512),
            category="action",
            tags=["rotate", "spin", "360"],
            estimated_time=20,
            quality_preset="high"
        )

        # Art templates
        templates["morph_transition"] = VideoTemplate(
            name="Morph Transition",
            description="Smooth morphing between states or transformations",
            prompt_template="{subject} morphing and transforming smoothly, fluid changes",
            frames=48,
            fps=24,
            noise_level="high",
            aspect_ratio="1:1",
            resolution=(512, 512),
            category="art",
            tags=["morph", "transform", "fluid"],
            estimated_time=50,
            quality_preset="cinema"
        )

        templates["color_cycle"] = VideoTemplate(
            name="Color Cycle",
            description="Cycling through different color palettes and moods",
            prompt_template="{subject} with color cycling through different palettes, mood changes",
            frames=32,
            fps=12,
            noise_level="medium",
            aspect_ratio="1:1",
            resolution=(512, 512),
            category="art",
            tags=["colors", "cycle", "palette"],
            estimated_time=30,
            quality_preset="high"
        )

        # Special effects
        templates["glitch_effect"] = VideoTemplate(
            name="Glitch Effect",
            description="Digital glitch and distortion effects",
            prompt_template="{subject} with digital glitch effects, pixel distortion, error artifacts",
            frames=16,
            fps=8,
            noise_level="high",
            aspect_ratio="16:9",
            resolution=(768, 432),
            category="effects",
            tags=["glitch", "digital", "distortion"],
            estimated_time=15,
            quality_preset="standard"
        )

        templates["fade_dissolve"] = VideoTemplate(
            name="Fade Dissolve",
            description="Smooth fading and dissolving transitions",
            prompt_template="{subject} with smooth fade in and out, dissolving effects",
            frames=24,
            fps=12,
            noise_level="low",
            aspect_ratio="1:1",
            resolution=(512, 512),
            category="effects",
            tags=["fade", "dissolve", "transition"],
            estimated_time=20,
            quality_preset="standard"
        )

        return templates

    def _load_default_presets(self) -> Dict[str, QualityPreset]:
        """Load default quality presets"""
        presets = {}

        presets["draft"] = QualityPreset(
            name="draft",
            description="Fast preview quality for testing ideas",
            frames=8,
            fps=6,
            noise_level="low",
            resolution=(256, 256),
            memory_usage="low",
            speed_factor=0.3,
            quality_factor=0.6
        )

        presets["standard"] = QualityPreset(
            name="standard",
            description="Balanced quality for general use",
            frames=16,
            fps=8,
            noise_level="high",
            resolution=(512, 512),
            memory_usage="medium",
            speed_factor=1.0,
            quality_factor=1.0
        )

        presets["high"] = QualityPreset(
            name="high",
            description="High quality for important projects",
            frames=24,
            fps=12,
            noise_level="high",
            resolution=(768, 768),
            memory_usage="high",
            speed_factor=2.0,
            quality_factor=1.4
        )

        presets["ultra"] = QualityPreset(
            name="ultra",
            description="Ultra high quality for final renders",
            frames=32,
            fps=16,
            noise_level="high",
            resolution=(1024, 1024),
            memory_usage="high",
            speed_factor=3.5,
            quality_factor=1.8
        )

        presets["cinema"] = QualityPreset(
            name="cinema",
            description="Cinema quality for professional work",
            frames=48,
            fps=24,
            noise_level="high",
            resolution=(1024, 576),  # 16:9 cinema
            memory_usage="high",
            speed_factor=5.0,
            quality_factor=2.0
        )

        presets["social"] = QualityPreset(
            name="social",
            description="Optimized for social media platforms",
            frames=16,
            fps=8,
            noise_level="medium",
            resolution=(640, 640),  # Square for Instagram
            memory_usage="medium",
            speed_factor=1.2,
            quality_factor=1.1
        )

        presets["mobile"] = QualityPreset(
            name="mobile",
            description="Optimized for mobile viewing and small screens",
            frames=12,
            fps=8,
            noise_level="low",
            resolution=(426, 240),  # Low resolution for mobile
            memory_usage="low",
            speed_factor=0.5,
            quality_factor=0.8
        )

        return presets

    def _load_custom_templates(self):
        """Load custom templates from files"""
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r') as f:
                    custom_data = json.load(f)
                    for name, template_data in custom_data.items():
                        template = VideoTemplate(**template_data)
                        self.templates[name] = template
                print(f"✅ Loaded {len(custom_data)} custom templates")
            except Exception as e:
                print(f"⚠️  Failed to load custom templates: {e}")

    def save_custom_templates(self):
        """Save custom templates to file"""
        custom_templates = {k: v for k, v in self.templates.items()
                          if k not in self._load_default_templates()}

        if custom_templates:
            try:
                with open(self.templates_file, 'w') as f:
                    # Convert to dict for JSON serialization
                    templates_dict = {k: asdict(v) for k, v in custom_templates.items()}
                    json.dump(templates_dict, f, indent=2)
                print(f"✅ Saved {len(custom_templates)} custom templates")
            except Exception as e:
                print(f"❌ Failed to save custom templates: {e}")

    def get_template(self, name: str) -> Optional[VideoTemplate]:
        """Get template by name"""
        return self.templates.get(name)

    def get_preset(self, name: str) -> Optional[QualityPreset]:
        """Get quality preset by name"""
        return self.presets.get(name)

    def list_templates(self, category: Optional[str] = None) -> List[VideoTemplate]:
        """List all templates, optionally filtered by category"""
        templates = list(self.templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def list_presets(self) -> List[QualityPreset]:
        """List all quality presets"""
        return list(self.presets.values())

    def search_templates(self, query: str) -> List[VideoTemplate]:
        """Search templates by name, description, or tags"""
        query = query.lower()
        results = []

        for template in self.templates.values():
            if (query in template.name.lower() or
                query in template.description.lower() or
                any(query in tag.lower() for tag in template.tags)):
                results.append(template)

        return results

    def create_custom_template(self, name: str, description: str,
                             prompt_template: str, frames: int, fps: int,
                             noise_level: str, aspect_ratio: str,
                             resolution: tuple, category: str,
                             tags: List[str]) -> VideoTemplate:
        """Create a new custom template"""
        template = VideoTemplate(
            name=name,
            description=description,
            prompt_template=prompt_template,
            frames=frames,
            fps=fps,
            noise_level=noise_level,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            category=category,
            tags=tags,
            estimated_time=frames / fps * 10  # Rough estimate
        )

        self.templates[name] = template
        self.save_custom_templates()
        return template

    def apply_template(self, template_name: str, subject: str,
                      quality_preset: Optional[str] = None) -> Dict[str, Any]:
        """Apply template and optionally quality preset"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Apply quality preset if specified
        if quality_preset:
            preset = self.get_preset(quality_preset)
            if not preset:
                raise ValueError(f"Quality preset '{quality_preset}' not found")

            # Override template settings with preset
            frames = preset.frames
            fps = preset.fps
            noise_level = preset.noise_level
            resolution = preset.resolution
        else:
            frames = template.frames
            fps = template.fps
            noise_level = template.noise_level
            resolution = template.resolution

        # Generate prompt from template
        prompt = template.prompt_template.format(subject=subject)

        return {
            "prompt": prompt,
            "frames": frames,
            "fps": fps,
            "noise_level": noise_level,
            "resolution": resolution,
            "aspect_ratio": template.aspect_ratio,
            "category": template.category,
            "estimated_time": frames / fps if fps > 0 else 0,
            "template_name": template_name,
            "quality_preset": quality_preset or template.quality_preset
        }

    def generate_template_info(self, template_name: str) -> Dict[str, Any]:
        """Generate detailed information about a template"""
        template = self.get_template(template_name)
        if not template:
            return {"error": f"Template '{template_name}' not found"}

        return {
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "tags": template.tags,
            "frames": template.frames,
            "fps": template.fps,
            "aspect_ratio": template.aspect_ratio,
            "resolution": template.resolution,
            "noise_level": template.noise_level,
            "estimated_time": template.estimated_time,
            "quality_preset": template.quality_preset,
            "example_prompt": template.prompt_template.format(subject="a beautiful landscape")
        }

    def get_categories(self) -> List[str]:
        """Get all available template categories"""
        categories = set(template.category for template in self.templates.values())
        return sorted(list(categories))

def main():
    parser = argparse.ArgumentParser(description="Wan Video Templates Manager")
    parser.add_argument("--list", action="store_true", help="List all templates")
    parser.add_argument("--presets", action="store_true", help="List quality presets")
    parser.add_argument("--categories", action="store_true", help="List template categories")
    parser.add_argument("--search", help="Search templates by query")
    parser.add_argument("--info", help="Show detailed template information")
    parser.add_argument("--apply", help="Apply template with subject")
    parser.add_argument("--subject", help="Subject for template application")
    parser.add_argument("--quality", help="Quality preset to apply")
    parser.add_argument("--category", help="Filter templates by category")

    args = parser.parse_args()

    manager = VideoTemplateManager()

    if args.list:
        templates = manager.list_templates(args.category)
        if templates:
            print(f"\n📋 Available Templates ({len(templates)}):")
            print("=" * 50)
            for template in templates:
                print(f"🎬 {template.name}")
                print(f"   {template.description}")
                print(f"   Category: {template.category} | Frames: {template.frames} | FPS: {template.fps}")
                print(f"   Tags: {', '.join(template.tags)}")
                print()
        else:
            print("No templates found")

    elif args.presets:
        presets = manager.list_presets()
        print(f"\n⚙️  Quality Presets ({len(presets)}):")
        print("=" * 50)
        for preset in presets:
            print(f"🎯 {preset.name}")
            print(f"   {preset.description}")
            print(f"   Frames: {preset.frames} | FPS: {preset.fps} | Resolution: {preset.resolution[0]}x{preset.resolution[1]}")
            print(f"   Memory: {preset.memory_usage} | Speed: {preset.speed_factor}x | Quality: {preset.quality_factor}x")
            print()

    elif args.categories:
        categories = manager.get_categories()
        print(f"\n📂 Template Categories ({len(categories)}):")
        print("=" * 30)
        for category in categories:
            templates = manager.list_templates(category)
            print(f"📁 {category} ({len(templates)} templates)")

    elif args.search:
        results = manager.search_templates(args.search)
        if results:
            print(f"\n🔍 Search Results for '{args.search}' ({len(results)}):")
            print("=" * 50)
            for template in results:
                print(f"🎬 {template.name} ({template.category})")
                print(f"   {template.description}")
                print()
        else:
            print(f"No templates found for '{args.search}'")

    elif args.info:
        info = manager.generate_template_info(args.info)
        if "error" in info:
            print(f"❌ {info['error']}")
        else:
            print(f"\n📋 Template Information: {info['name']}")
            print("=" * 50)
            print(f"Description: {info['description']}")
            print(f"Category: {info['category']}")
            print(f"Tags: {', '.join(info['tags'])}")
            print(f"Frames: {info['frames']} | FPS: {info['fps']} | Aspect Ratio: {info['aspect_ratio']}")
            print(f"Resolution: {info['resolution'][0]}x{info['resolution'][1]}")
            print(f"Noise Level: {info['noise_level']}")
            print(f"Estimated Time: {info['estimated_time']:.1f}s")
            print(f"Quality Preset: {info['quality_preset']}")
            print(f"\nExample Prompt:")
            print(f"   {info['example_prompt']}")

    elif args.apply:
        if not args.subject:
            print("❌ --subject is required when using --apply")
            return

        try:
            config = manager.apply_template(args.apply, args.subject, args.quality)
            print(f"\n🎬 Applied Template: {args.apply}")
            print("=" * 50)
            print(f"Prompt: {config['prompt']}")
            print(f"Frames: {config['frames']} | FPS: {config['fps']}")
            print(f"Resolution: {config['resolution'][0]}x{config['resolution'][1]}")
            print(f"Noise Level: {config['noise_level']}")
            print(f"Category: {config['category']}")
            print(f"Estimated Time: {config['estimated_time']:.1f}s")
            print(f"Quality Preset: {config['quality_preset']}")
        except Exception as e:
            print(f"❌ Error: {e}")

    else:
        print("🎬 Wan Video Templates Manager")
        print("Use --help to see available commands")
        print("\nQuick Examples:")
        print("  wan_video_templates.py --list")
        print("  wan_video_templates.py --presets")
        print("  wan_video_templates.py --info subtle_breathing")
        print("  wan_video_templates.py --apply gentle_pulse --subject 'a glowing orb'")

if __name__ == "__main__":
    main()