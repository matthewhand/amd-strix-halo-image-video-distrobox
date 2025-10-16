#!/usr/bin/env python3
"""
Wan Video Generation REST API
Remote video generation service with queue management and monitoring
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
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging

# Import our existing modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from wan_video_generator import SimpleVideoGenerator
from gpu_monitor import AMDGPUMonitor
from wan_batch_generator import WanBatchGenerator, VideoJob, BatchConfig

app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend access

# Global state
generation_queue = []
active_jobs = {}
completed_jobs = {}
gpu_monitor = AMDGPUMonitor()
queue_lock = threading.Lock()
job_lock = threading.Lock()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Video generation request"""
    prompt: str
    input_image: Optional[str] = None
    output_name: Optional[str] = None
    frames: int = 16
    fps: int = 8
    noise_level: str = "high"
    quality_preset: str = "standard"
    priority: int = 0  # Higher priority = processed first

@dataclass
class JobStatus:
    """Job status information"""
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

class QualityPresets:
    """Video quality presets"""

    PRESETS = {
        "draft": {
            "frames": 8,
            "fps": 6,
            "noise_level": "low",
            "description": "Fast preview quality"
        },
        "standard": {
            "frames": 16,
            "fps": 8,
            "noise_level": "high",
            "description": "Standard quality"
        },
        "high": {
            "frames": 24,
            "fps": 12,
            "noise_level": "high",
            "description": "High quality"
        },
        "ultra": {
            "frames": 32,
            "fps": 16,
            "noise_level": "high",
            "description": "Ultra high quality"
        },
        "cinema": {
            "frames": 48,
            "fps": 24,
            "noise_level": "high",
            "description": "Cinema quality"
        }
    }

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any]:
        """Get preset by name"""
        return cls.PRESETS.get(name, cls.PRESETS["standard"])

    @classmethod
    def list_presets(cls) -> List[str]:
        """List available presets"""
        return list(cls.PRESETS.keys())

class VideoQueue:
    """Video generation queue manager"""

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
                "queue": [{"job_id": job_id, "priority": job.priority}
                         for job_id, job in generation_queue[:10]]  # Show next 10
            }

class VideoWorker:
    """Video generation worker thread"""

    def __init__(self, job_id: str, request: GenerationRequest):
        self.job_id = job_id
        self.request = request
        self.thread = threading.Thread(target=self._process_job, daemon=True)
        self.generator = SimpleVideoGenerator()

    def start(self):
        """Start worker thread"""
        self.thread.start()

    def _process_job(self):
        """Process video generation job"""
        try:
            # Update job status to processing
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].status = "processing"
                    active_jobs[self.job_id].started_at = datetime.now()
                    active_jobs[self.job_id].progress = 10

            queue.current_workers += 1
            logger.info(f"Processing job {self.job_id}")

            # Apply quality preset
            preset = QualityPresets.get_preset(self.request.quality_preset)
            frames = preset.get("frames", self.request.frames)
            fps = preset.get("fps", self.request.fps)
            noise_level = preset.get("noise_level", self.request.noise_level)

            # Update progress
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = 30

            # Generate output path
            output_dir = Path("api_outputs")
            output_dir.mkdir(exist_ok=True)

            output_name = self.request.output_name or f"video_{self.job_id[:8]}"
            if not output_name.endswith('.gif'):
                output_name += '.gif'
            output_path = output_dir / output_name

            # Load input image
            if self.request.input_image:
                base_image = self.generator.load_image(self.request.input_image)
                if not base_image:
                    raise Exception(f"Failed to load input image: {self.request.input_image}")
            else:
                # Create a default image
                from PIL import Image
                base_image = Image.new('RGB', (512, 512), color='blue')

            # Update progress
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = 50

            # Generate video frames
            frames_list = self.generator.create_animation_frames(
                base_image, self.request.prompt, frames, noise_level
            )

            if not frames_list:
                raise Exception("Failed to generate video frames")

            # Update progress
            with job_lock:
                if self.job_id in active_jobs:
                    active_jobs[self.job_id].progress = 80

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

# Initialize queue
queue = VideoQueue()

def start_queue_processor():
    """Start the queue processor thread"""
    def process_queue():
        gpu_monitor.start_monitoring()
        time.sleep(2)  # Let GPU monitor initialize

        while True:
            try:
                job_data = queue.get_next_job()
                if job_data:
                    job_id, request = job_data

                    # Create job status
                    job_status = JobStatus(
                        job_id=job_id,
                        status="queued",
                        progress=0,
                        created_at=datetime.now()
                    )

                    with job_lock:
                        active_jobs[job_id] = job_status

                    # Start worker
                    worker = VideoWorker(job_id, request)
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
    logger.info("Queue processor started")

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/presets', methods=['GET'])
def list_presets():
    """List available quality presets"""
    return jsonify({
        "presets": QualityPresets.list_presets(),
        "details": QualityPresets.PRESETS
    })

@app.route('/api/generate', methods=['POST'])
def generate_video():
    """Submit video generation request"""
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('prompt'):
            return jsonify({"error": "Prompt is required"}), 400

        # Create generation request
        gen_request = GenerationRequest(
            prompt=data['prompt'],
            input_image=data.get('input_image'),
            output_name=data.get('output_name'),
            frames=data.get('frames', 16),
            fps=data.get('fps', 8),
            noise_level=data.get('noise_level', 'high'),
            quality_preset=data.get('quality_preset', 'standard'),
            priority=data.get('priority', 0)
        )

        # Add to queue
        job_id = queue.add_job(gen_request)

        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "message": "Job added to queue"
        })

    except Exception as e:
        logger.error(f"Generate request error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get job status"""
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
    """List all jobs"""
    limit = request.args.get('limit', 50, type=int)
    status_filter = request.args.get('status')

    with job_lock:
        all_jobs = {**active_jobs, **completed_jobs}

    # Filter by status if specified
    if status_filter:
        all_jobs = {k: v for k, v in all_jobs.items() if v.status == status_filter}

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
        "total": len(jobs_list)
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
    """Get queue status"""
    return jsonify(queue.get_queue_status())

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    # Job statistics
    with job_lock:
        total_jobs = len(active_jobs) + len(completed_jobs)
        completed_count = sum(1 for job in completed_jobs.values() if job.status == "completed")
        failed_count = sum(1 for job in completed_jobs.values() if job.status == "failed")

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
            "active": len(active_jobs)
        },
        "gpu": gpu_dict,
        "queue": queue.get_queue_status()
    })

@app.route('/api/batch', methods=['POST'])
def batch_generate():
    """Submit batch generation request"""
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
                priority=job_data.get('priority', 0)
            )

            job_id = queue.add_job(gen_request)
            job_ids.append(job_id)

        return jsonify({
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_jobs": len(job_ids)
        })

    except Exception as e:
        logger.error(f"Batch request error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation"""
    docs = {
        "title": "Wan Video Generation API",
        "version": "1.0.0",
        "endpoints": {
            "GET /api/health": "Health check",
            "GET /api/presets": "List quality presets",
            "POST /api/generate": "Submit video generation request",
            "GET /api/jobs": "List all jobs",
            "GET /api/jobs/<job_id>": "Get job status",
            "GET /api/jobs/<job_id>/download": "Download generated video",
            "GET /api/queue": "Get queue status",
            "GET /api/stats": "Get system statistics",
            "POST /api/batch": "Submit batch generation request"
        },
        "example_requests": {
            "generate": {
                "prompt": "a sunset over mountains with gentle cloud movement",
                "input_image": "/path/to/image.jpg",
                "quality_preset": "high",
                "priority": 1
            },
            "batch": {
                "jobs": [
                    {"prompt": "cat stretching", "quality_preset": "standard"},
                    {"prompt": "rain falling", "quality_preset": "draft"}
                ]
            }
        }
    }
    return jsonify(docs)

def main():
    parser = argparse.ArgumentParser(description="Wan Video Generation REST API")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--workers", type=int, default=2, help="Max concurrent workers")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Update queue configuration
    queue.max_workers = args.workers

    # Start queue processor
    start_queue_processor()

    print(f"🚀 Starting Wan Video Generation API on {args.host}:{args.port}")
    print(f"📊 Max workers: {args.workers}")
    print(f"📁 Output directory: api_outputs/")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs")
    print(f"🔍 GPU monitoring: Active")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
        gpu_monitor.stop_monitoring()

if __name__ == "__main__":
    main()
