#!/usr/bin/env python3
"""
Enhanced Wan Video Generation Server with Web Dashboard
Complete video generation platform with beautiful web interface
"""
import os
import sys
import json
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import argparse
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import logging

# Import our existing modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from wan_video_generator import SimpleVideoGenerator
from gpu_monitor import AMDGPUMonitor
from wan_batch_generator import WanBatchGenerator, VideoJob, BatchConfig
from wan_video_templates import VideoTemplateManager, QualityPresets

app = Flask(__name__,
           template_folder='web_dashboard/templates',
           static_folder='web_dashboard/static')
CORS(app)

# Global state
generation_queue = []
active_jobs = {}
completed_jobs = {}
gpu_monitor = AMDGPUMonitor()
template_manager = VideoTemplateManager()
queue_lock = threading.Lock()
job_lock = threading.Lock()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Enhanced video generation request"""
    prompt: str
    input_image: Optional[str] = None
    output_name: Optional[str] = None
    frames: int = 16
    fps: int = 8
    noise_level: str = "high"
    quality_preset: str = "standard"
    template: Optional[str] = None
    priority: int = 0
    tags: List[str] = None

@dataclass
class JobStatus:
    """Enhanced job status information"""
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float  # 0-100
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    file_size: int = 0
    generation_time: float = 0
    prompt: str = ""
    template: Optional[str] = None
    quality_preset: str = "standard"
    tags: List[str] = None

class EnhancedVideoWorker:
    """Enhanced video generation worker with template support"""

    def __init__(self, job_id: str, request: GenerationRequest):
        self.job_id = job_id
        self.request = request
        self.thread = threading.Thread(target=self._process_job, daemon=True)
        self.generator = SimpleVideoGenerator()

    def start(self):
        """Start worker thread"""
        self.thread.start()

    def _process_job(self):
        """Process video generation job with template support"""
        try:
            # Update job status to processing
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].status = "processing"
                    active_jobs[self.job_id].started_at = datetime.now()
                    active_jobs[self.job_id].progress = 10

            queue.current_workers += 1
            logger.info(f"Processing job {self.job_id}")

            # Apply template if specified
            if self.request.template:
                try:
                    template_config = template_manager.apply_template(
                        self.request.template,
                        self.request.prompt,
                        self.request.quality_preset
                    )
                    prompt = template_config['prompt']
                    frames = template_config['frames']
                    fps = template_config['fps']
                    noise_level = template_config['noise_level']
                    logger.info(f"Applied template '{self.request.template}' to job {self.job_id}")
                except Exception as e:
                    logger.warning(f"Template application failed: {e}, using fallback")
                    prompt = self.request.prompt
                    frames = self.request.frames
                    fps = self.request.fps
                    noise_level = self.request.noise_level
            else:
                # Apply quality preset
                preset = QualityPresets.get_preset(self.request.quality_preset)
                frames = preset.get("frames", self.request.frames)
                fps = preset.get("fps", self.request.fps)
                noise_level = preset.get("noise_level", self.request.noise_level)
                prompt = self.request.prompt

            # Update progress
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = 25

            # Generate output path
            output_dir = Path("web_outputs")
            output_dir.mkdir(exist_ok=True)

            output_name = self.request.output_name or f"video_{self.job_id[:8]}"
            if not output_name.endswith('.gif'):
                output_name += '.gif'
            output_path = output_dir / output_name

            # Update progress
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = 30

            # Load input image
            if self.request.input_image and os.path.exists(self.request.input_image):
                base_image = self.generator.load_image(self.request.input_image)
                if not base_image:
                    raise Exception(f"Failed to load input image: {self.request.input_image}")
            else:
                # Create a default image based on prompt
                base_image = self._create_default_image(prompt)

            # Update progress
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = 40

            # Generate video frames with progress updates
            frames_list = self._generate_frames_with_progress(
                base_image, prompt, frames, noise_level
            )

            if not frames_list:
                raise Exception("Failed to generate video frames")

            # Update progress
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = 85

            # Save video
            success = self.generator.save_as_gif(frames_list, str(output_path), fps)

            if not success:
                raise Exception("Failed to save video")

            # Get file size
            file_size = os.path.getsize(output_path)

            # Mark job as completed
            completion_time = datetime.now()
            generation_time = (completion_time - active_jobs[self.job_id].started_at).total_seconds()

            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].status = "completed"
                    active_jobs[self.job_id].completed_at = completion_time
                    active_jobs[self.job_id].output_path = str(output_path)
                    active_jobs[self.job_id].file_size = file_size
                    active_jobs[self.job_id].progress = 100
                    active_jobs[self.job_id].generation_time = generation_time

                    # Move to completed jobs
                    completed_jobs[self.job_id] = active_jobs.pop(self.job_id)

            logger.info(f"Completed job {self.job_id} - {output_path} ({file_size} bytes)")

        except Exception as e:
            # Mark job as failed
            completion_time = datetime.now()
            generation_time = 0

            if self.job_id in active_jobs and active_jobs[self.job_id].started_at:
                generation_time = (completion_time - active_jobs[self.job_id].started_at).total_seconds()

            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].status = "failed"
                    active_jobs[self.job_id].completed_at = completion_time
                    active_jobs[self.job_id].error_message = str(e)
                    active_jobs[self.job_id].generation_time = generation_time

                    # Move to completed jobs
                    completed_jobs[self.job_id] = active_jobs.pop(self.job_id)

            logger.error(f"Failed job {self.job_id}: {e}")

        finally:
            queue.current_workers -= 1

    def _create_default_image(self, prompt: str) -> 'Image.Image':
        """Create a default image based on prompt"""
        from PIL import Image, ImageDraw
        import numpy as np

        # Create gradient background
        width, height = 512, 512
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)

        # Create gradient based on prompt keywords
        prompt_lower = prompt.lower()

        if 'blue' in prompt_lower or 'sky' in prompt_lower:
            color1 = (0, 100, 200)
            color2 = (0, 50, 150)
        elif 'red' in prompt_lower or 'sunset' in prompt_lower:
            color1 = (255, 100, 50)
            color2 = (200, 50, 30)
        elif 'green' in prompt_lower or 'forest' in prompt_lower:
            color1 = (50, 200, 100)
            color2 = (30, 150, 50)
        elif 'purple' in prompt_lower or 'galaxy' in prompt_lower:
            color1 = (150, 50, 200)
            color2 = (100, 30, 150)
        else:
            color1 = (100, 100, 150)
            color2 = (50, 50, 100)

        # Create vertical gradient
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Add some shapes based on prompt
        if 'circle' in prompt_lower or 'orb' in prompt_lower:
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        fill=(255, 255, 255, 128))

        if 'sun' in prompt_lower:
            center_x, center_y = width // 2, height // 4
            radius = min(width, height) // 8
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        fill=(255, 255, 200))

        return image

    def _generate_frames_with_progress(self, base_image: 'Image.Image', prompt: str,
                                     num_frames: int, noise_level: str) -> list:
        """Generate frames with progress updates"""
        frames = []

        for i in range(num_frames):
            # Simulate frame generation progress
            frame_progress = 40 + (50 * (i + 1) / num_frames)

            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = frame_progress

            # Generate the actual frame
            frame = self._create_single_frame(base_image, prompt, i, num_frames, noise_level)
            frames.append(frame)

        return frames

    def _create_single_frame(self, base_image: 'Image.Image', prompt: str,
                           frame_index: int, total_frames: int, noise_level: str) -> 'Image.Image':
        """Create a single animated frame"""
        import numpy as np
        from PIL import Image, ImageEnhance

        # Calculate animation parameters
        progress = frame_index / (total_frames - 1) if total_frames > 1 else 0

        # Parse prompt for animation type
        prompt_lower = prompt.lower()

        if 'breath' in prompt_lower or 'pulse' in prompt_lower:
            # Breathing/pulsing effect
            scale = 1.0 + np.sin(progress * 2 * np.pi) * 0.1
            frame = self._apply_scale_effect(base_image, scale)
        elif 'rotate' in prompt_lower or 'spin' in prompt_lower:
            # Rotation effect
            angle = progress * 360
            frame = base_image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        elif 'move' in prompt_lower or 'pan' in prompt_lower:
            # Movement effect
            offset_x = int(np.sin(progress * 2 * np.pi) * 20)
            offset_y = int(np.cos(progress * 2 * np.pi) * 10)
            frame = self._apply_translation_effect(base_image, offset_x, offset_y)
        else:
            # Default: subtle color and brightness changes
            brightness_factor = 0.8 + 0.4 * np.sin(progress * 2 * np.pi)
            enhancer = ImageEnhance.Brightness(base_image)
            frame = enhancer.enhance(brightness_factor)

        # Apply noise level
        if noise_level == "high":
            frame = self._apply_noise_effect(frame, 10)
        elif noise_level == "medium":
            frame = self._apply_noise_effect(frame, 5)

        return frame

    def _apply_scale_effect(self, image: 'Image.Image', scale: float) -> 'Image.Image':
        """Apply scaling effect"""
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

    def _apply_translation_effect(self, image: 'Image.Image', offset_x: int, offset_y: int) -> 'Image.Image':
        """Apply translation effect with wrapping"""
        width, height = image.size
        new_image = Image.new("RGB", (width, height), (0, 0, 0))

        # Simple translation (can be enhanced with wrapping)
        src_x = max(0, offset_x)
        src_y = max(0, offset_y)
        dst_x = max(0, -offset_x)
        dst_y = max(0, -offset_y)

        visible_width = min(width - src_x, width - dst_x)
        visible_height = min(height - src_y, height - dst_y)

        if visible_width > 0 and visible_height > 0:
            crop = image.crop((src_x, src_y, src_x + visible_width, src_y + visible_height))
            new_image.paste(crop, (dst_x, dst_y))

        return new_image

    def _apply_noise_effect(self, image: 'Image.Image', intensity: int) -> 'Image.Image':
        """Apply noise effect to image"""
        import numpy as np

        img_array = np.array(image)
        noise = np.random.normal(0, intensity, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_array)

class VideoQueue:
    """Enhanced video generation queue manager"""

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.current_workers = 0
        self.processing = False

    def add_job(self, job_request: GenerationRequest) -> str:
        """Add job to queue"""
        job_id = str(uuid.uuid4())

        with queue_lock:
            generation_queue.append((job_id, job_request))
            # Sort by priority (higher first)
            generation_queue.sort(key=lambda x: x[1].priority, reverse=True)

        logger.info(f"Added job {job_id} to queue (priority: {job_request.priority})")
        return job_id

    def get_next_job(self) -> Optional[tuple]:
        """Get next job from queue"""
        with queue_lock:
            if generation_queue and self.current_workers < self.max_workers:
                return generation_queue.pop(0)
        return None

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with queue_lock:
            return {
                "queued_jobs": len(generation_queue),
                "active_workers": self.current_workers,
                "max_workers": self.max_workers,
                "queue": [{"job_id": job_id, "priority": job.priority, "prompt": job.prompt[:50]}
                         for job_id, job in generation_queue[:10]]  # Show next 10
            }

# Initialize queue
queue = VideoQueue()

def start_queue_processor():
    """Start the enhanced queue processor thread"""
    def process_queue():
        gpu_monitor.start_monitoring()
        time.sleep(2)  # Let GPU monitor initialize

        while True:
            try:
                job_data = queue.get_next_job()
                if job_data:
                    job_id, request = job_data

                    # Create enhanced job status
                    job_status = JobStatus(
                        job_id=job_id,
                        status="queued",
                        progress=0,
                        created_at=datetime.now(),
                        prompt=request.prompt,
                        template=request.template,
                        quality_preset=request.quality_preset,
                        tags=request.tags or []
                    )

                    with job_lock:
                        active_jobs[job_id] = job_status

                    # Start enhanced worker
                    worker = EnhancedVideoWorker(job_id, request)
                    worker.start()

                    # Small delay between starting jobs
                    time.sleep(1)
                else:
                    time.sleep(2)  # Wait for new jobs

            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                time.sleep(5)

    processor_thread = threading.Thread(target=process_queue, daemon=True)
    processor_thread.start()
    logger.info("Enhanced queue processor started")

# Enhanced API Routes
@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": ["web_dashboard", "templates", "gpu_monitoring", "queue_management"]
    })

@app.route('/api/presets', methods=['GET'])
def list_presets():
    """List available quality presets"""
    return jsonify({
        "presets": template_manager.list_presets(),
        "details": QualityPresets.PRESETS
    })

@app.route('/api/templates', methods=['GET'])
def list_templates():
    """List available templates"""
    category = request.args.get('category')
    templates = template_manager.list_templates(category)

    return jsonify({
        "templates": [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "tags": t.tags,
                "frames": t.frames,
                "fps": t.fps,
                "quality_preset": t.quality_preset
            }
            for t in templates
        ],
        "categories": template_manager.get_categories()
    })

@app.route('/api/generate', methods=['POST'])
def generate_video():
    """Enhanced video generation request"""
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('prompt'):
            return jsonify({"error": "Prompt is required"}), 400

        # Create enhanced generation request
        gen_request = GenerationRequest(
            prompt=data['prompt'],
            input_image=data.get('input_image'),
            output_name=data.get('output_name'),
            frames=data.get('frames', 16),
            fps=data.get('fps', 8),
            noise_level=data.get('noise_level', 'high'),
            quality_preset=data.get('quality_preset', 'standard'),
            template=data.get('template'),
            priority=data.get('priority', 0),
            tags=data.get('tags', [])
        )

        # Add to queue
        job_id = queue.add_job(gen_request)

        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "message": "Job added to queue",
            "queue_position": len(generation_queue)
        })

    except Exception as e:
        logger.error(f"Generate request error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get enhanced job status"""
    with job_lock:
        if job_id in active_jobs:
            job = active_jobs[job_id]
        elif job_id in completed_jobs:
            job = completed_jobs[job_id]
        else:
            return jsonify({"error": "Job not found"}), 404

    # Convert datetime objects to ISO format
    job_dict = asdict(job)
    job_dict['created_at'] = job.created_at.isoformat()
    if job.started_at:
        job_dict['started_at'] = job.started_at.isoformat()
    if job.completed_at:
        job_dict['completed_at'] = job.completed_at.isoformat()

    return jsonify(job_dict)

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """Enhanced jobs listing with filtering"""
    limit = request.args.get('limit', 50, type=int)
    status_filter = request.args.get('status')
    template_filter = request.args.get('template')

    with job_lock:
        all_jobs = {**active_jobs, **completed_jobs}

    # Apply filters
    if status_filter:
        all_jobs = {k: v for k, v in all_jobs.items() if v.status == status_filter}
    if template_filter:
        all_jobs = {k: v for k, v in all_jobs.items() if v.template == template_filter}

    # Sort by creation time (newest first) and limit
    sorted_jobs = sorted(all_jobs.items(),
                        key=lambda x: x[1].created_at,
                        reverse=True)[:limit]

    # Convert to response format
    jobs_list = []
    for job_id, job in sorted_jobs:
        job_dict = asdict(job)
        job_dict['created_at'] = job.created_at.isoformat()
        if job.started_at:
            job_dict['started_at'] = job.started_at.isoformat()
        if job.completed_at:
            job_dict['completed_at'] = job.completed_at.isoformat()
        jobs_list.append(job_dict)

    return jsonify({
        "jobs": jobs_list,
        "total": len(jobs_list),
        "filters": {
            "status": status_filter,
            "template": template_filter,
            "limit": limit
        }
    })

@app.route('/api/jobs/<job_id>/download', methods=['GET'])
def download_video(job_id: str):
    """Download generated video"""
    with job_lock:
        if job_id in completed_jobs:
            job = completed_jobs[job_id]
        else:
            return jsonify({"error": "Job not found or not completed"}), 404

    if job.status != "completed" or not job.output_path:
        return jsonify({"error": "Video not available for download"}), 400

    try:
        return send_file(job.output_path,
                        as_attachment=True,
                        download_name=f"video_{job_id[:8]}.gif",
                        mimetype='image/gif')
    except Exception as e:
        return jsonify({"error": f"Download failed: {e}"}), 500

@app.route('/api/queue', methods=['GET'])
def get_queue_status():
    """Get enhanced queue status"""
    return jsonify(queue.get_queue_status())

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Enhanced system statistics"""
    # Job statistics
    with job_lock:
        total_jobs = len(active_jobs) + len(completed_jobs)
        completed_count = sum(1 for job in completed_jobs.values() if job.status == "completed")
        failed_count = sum(1 for job in completed_jobs.values() if job.status == "failed")

        # Template usage statistics
        template_usage = {}
        for job in completed_jobs.values():
            if job.template:
                template_usage[job.template] = template_usage.get(job.template, 0) + 1

    # GPU statistics
    gpu_stats = gpu_monitor.get_current_stats()
    gpu_dict = {}
    if gpu_stats:
        gpu_dict = {
            "memory_used": gpu_stats.memory_used,
            "memory_total": gpu_stats.memory_total,
            "memory_percent": gpu_stats.memory_percent,
            "temperature": gpu_stats.temperature,
            "power_usage": gpu_stats.power_usage,
            "clock_speed": gpu_stats.clock_speed
        }

    return jsonify({
        "jobs": {
            "total": total_jobs,
            "completed": completed_count,
            "failed": failed_count,
            "active": len(active_jobs),
            "success_rate": (completed_count / total_jobs * 100) if total_jobs > 0 else 0
        },
        "gpu": gpu_dict,
        "queue": queue.get_queue_status(),
        "templates": {
            "usage": template_usage,
            "total_available": len(template_manager.templates)
        }
    })

@app.route('/api/batch', methods=['POST'])
def batch_generate():
    """Enhanced batch generation request"""
    try:
        data = request.get_json()

        if not data.get('jobs'):
            return jsonify({"error": "Jobs list is required"}), 400

        batch_id = str(uuid.uuid4())
        job_ids = []

        for job_data in data['jobs']:
            gen_request = GenerationRequest(
                prompt=job_data['prompt'],
                input_image=job_data.get('input_image'),
                output_name=job_data.get('output_name'),
                frames=job_data.get('frames', 16),
                fps=job_data.get('fps', 8),
                noise_level=job_data.get('noise_level', 'high'),
                quality_preset=job_data.get('quality_preset', 'standard'),
                template=job_data.get('template'),
                priority=job_data.get('priority', 0),
                tags=job_data.get('tags', [])
            )

            job_id = queue.add_job(gen_request)
            job_ids.append(job_id)

        return jsonify({
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_jobs": len(job_ids),
            "message": f"Batch of {len(job_ids)} jobs submitted successfully"
        })

    except Exception as e:
        logger.error(f"Batch request error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/docs', methods=['GET'])
def api_docs():
    """Enhanced API documentation"""
    docs = {
        "title": "Wan Video Generation API v2.0",
        "version": "2.0.0",
        "description": "Enhanced video generation platform with web dashboard and template support",
        "endpoints": {
            "GET /": "Web dashboard interface",
            "GET /api/health": "Enhanced health check",
            "GET /api/presets": "List quality presets",
            "GET /api/templates": "List video templates",
            "POST /api/generate": "Submit video generation request",
            "GET /api/jobs": "List all jobs with filtering",
            "GET /api/jobs/<job_id>": "Get detailed job status",
            "GET /api/jobs/<job_id>/download": "Download generated video",
            "GET /api/queue": "Get queue status",
            "GET /api/stats": "Get comprehensive system statistics",
            "POST /api/batch": "Submit batch generation request"
        },
        "features": [
            "Web dashboard interface",
            "Video template system",
            "Quality presets",
            "Real-time GPU monitoring",
            "Queue management with priorities",
            "Batch processing",
            "Job filtering and search"
        ],
        "example_requests": {
            "generate": {
                "prompt": "a beautiful sunset over mountains",
                "template": "cloud_drift",
                "quality_preset": "high",
                "priority": 1,
                "tags": ["nature", "sunset"]
            },
            "batch": {
                "jobs": [
                    {
                        "prompt": "a glowing orb pulsing",
                        "template": "gentle_pulse",
                        "quality_preset": "standard"
                    },
                    {
                        "prompt": "trees swaying in wind",
                        "template": "wind_sway",
                        "quality_preset": "high"
                    }
                ]
            }
        }
    }
    return jsonify(docs)

def main():
    parser = argparse.ArgumentParser(description="Enhanced Wan Video Generation Server with Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--workers", type=int, default=2, help="Max concurrent workers")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dashboard-only", action="store_true", help="Start only the dashboard")

    args = parser.parse_args()

    # Update queue configuration
    queue.max_workers = args.workers

    # Start queue processor
    if not args.dashboard_only:
        start_queue_processor()

    print(f"🚀 Starting Enhanced Wan Video Generation Server")
    print(f"🌐 Web Dashboard: http://{args.host}:{args.port}")
    print(f"📊 Max workers: {args.workers}")
    print(f"📁 Output directory: web_outputs/")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs")
    print(f"🔍 GPU monitoring: Active")
    print(f"🎨 Templates available: {len(template_manager.templates)}")
    print(f"⚙️  Quality presets: {len(template_manager.presets)}")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Enhanced server stopped")
        if not args.dashboard_only:
            gpu_monitor.stop_monitoring()

if __name__ == "__main__":
    main()
