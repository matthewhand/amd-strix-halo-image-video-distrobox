#!/usr/bin/env python3
"""
Wan Video Batch Generator
Process multiple video generation jobs automatically with progress tracking
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

@dataclass
class VideoJob:
    """Single video generation job"""
    job_id: str
    input_image: str
    prompt: str
    output_path: str
    frames: int = 16
    fps: int = 8
    noise_level: str = "high"
    status: str = "pending"
    start_time: float = 0
    end_time: float = 0
    error_message: str = ""
    file_size: int = 0

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_workers: int = 2
    output_dir: str = "batch_outputs"
    log_file: str = "batch_generation.log"
    save_frames: bool = False
    quality_preset: str = "standard"

class GPUMonitor:
    """Monitor GPU usage during generation"""
    def __init__(self):
        self.monitoring = False
        self.stats = {}
        self.lock = threading.Lock()

    def start_monitoring(self):
        """Start GPU monitoring"""
        self.monitoring = True

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        try:
            # Try to get ROCm stats if available
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showtemp", "--showpower"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                # Parse ROCm output
                lines = result.stdout.strip().split('\n')
                stats = {}
                for line in lines:
                    if 'GPU' in line and 'MiB' in line:
                        parts = line.split()
                        stats['memory_usage'] = parts[2]
                    elif 'Temperature' in line and 'C' in line:
                        parts = line.split()
                        stats['temperature'] = parts[-1]
                    elif 'Average Power' in line and 'W' in line:
                        parts = line.split()
                        stats['power'] = parts[-1]
                return stats
        except:
            pass

        return {'memory_usage': 'N/A', 'temperature': 'N/A', 'power': 'N/A'}

class WanBatchGenerator:
    """Enhanced batch video generator with monitoring"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.jobs: List[VideoJob] = []
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.gpu_monitor = GPUMonitor()
        self.start_time = time.time()

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.log_file = Path(config.log_file)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging to file and console"""
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_jobs_from_file(self, jobs_file: str) -> bool:
        """Load jobs from JSON file"""
        try:
            with open(jobs_file, 'r') as f:
                jobs_data = json.load(f)

            for job_data in jobs_data:
                job = VideoJob(**job_data)
                self.jobs.append(job)

            self.logger.info(f"Loaded {len(self.jobs)} jobs from {jobs_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load jobs from {jobs_file}: {e}")
            return False

    def load_jobs_from_csv(self, csv_file: str) -> bool:
        """Load jobs from CSV file (prompt, image_path, output_name)"""
        try:
            import csv
            jobs = []

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('prompt') and row.get('output_name'):
                        job = VideoJob(
                            job_id=row.get('job_id', f"job_{len(jobs)}"),
                            input_image=row.get('image_path', ''),
                            prompt=row['prompt'],
                            output_path=self.output_dir / f"{row['output_name']}.gif",
                            frames=int(row.get('frames', 16)),
                            fps=int(row.get('fps', 8)),
                            noise_level=row.get('noise_level', 'high')
                        )
                        jobs.append(job)

            self.jobs.extend(jobs)
            self.logger.info(f"Loaded {len(jobs)} jobs from {csv_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load jobs from {csv_file}: {e}")
            return False

    def create_sample_jobs(self, count: int = 5) -> None:
        """Create sample jobs for testing"""
        sample_prompts = [
            "a sunset over mountains with gentle cloud movement",
            "a cat stretching lazily in the morning light",
            "rain falling on a city street at night",
            "flowers blooming in a spring garden",
            "waves crashing on a beach at sunset"
        ]

        for i in range(min(count, len(sample_prompts))):
            job = VideoJob(
                job_id=f"sample_job_{i+1}",
                input_image="example.png",  # Use existing example image
                prompt=sample_prompts[i],
                output_path=self.output_dir / f"sample_video_{i+1}.gif",
                frames=16,
                fps=8,
                noise_level="high"
            )
            self.jobs.append(job)

        self.logger.info(f"Created {len(self.jobs)} sample jobs")

    def generate_single_video(self, job: VideoJob) -> VideoJob:
        """Generate a single video"""
        import subprocess
        job.status = "processing"
        job.start_time = time.time()

        try:
            self.logger.info(f"Starting job {job.job_id}: {job.prompt[:50]}...")

            # Use the existing video generator
            generator_script = Path(__file__).resolve().parent / "wan_video_generator.py"
            cmd = [
                sys.executable,
                str(generator_script),
                job.input_image,
                job.prompt,
                "-o", str(job.output_path),
                "--frames", str(job.frames),
                "--fps", str(job.fps),
                "--noise", job.noise_level
            ]

            if self.config.save_frames:
                cmd.extend(["--save-frames", f"frames_{job.job_id}"])

            # Run in distrobox if needed
            if os.path.exists("/opt/ComfyUI"):
                cmd = ["distrobox", "enter", "strix-halo-image-video", "--bash",
                       "cd /opt/ComfyUI && source /opt/venv/bin/activate && " + " ".join(cmd)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                job.status = "completed"
                job.file_size = os.path.getsize(job.output_path) if os.path.exists(job.output_path) else 0
                self.logger.info(f"✅ Completed job {job.job_id} - File: {job.output_path} ({job.file_size} bytes)")
            else:
                job.status = "failed"
                job.error_message = result.stderr
                self.logger.error(f"❌ Failed job {job.job_id}: {job.error_message}")

        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error_message = "Timeout after 5 minutes"
            self.logger.error(f"❌ Timeout for job {job.job_id}")
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            self.logger.error(f"❌ Error in job {job.job_id}: {e}")

        job.end_time = time.time()
        return job

    def process_batch(self) -> Dict[str, Any]:
        """Process all jobs with worker pool"""
        self.logger.info(f"Starting batch processing with {self.config.max_workers} workers")
        self.logger.info(f"Total jobs: {len(self.jobs)}")

        self.gpu_monitor.start_monitoring()
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.generate_single_video, job): job
                for job in self.jobs
            }

            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    completed_job = future.result()
                    if completed_job.status == "completed":
                        self.completed_jobs += 1
                    else:
                        self.failed_jobs += 1
                except Exception as e:
                    self.logger.error(f"Error processing job {job.job_id}: {e}")
                    job.status = "failed"
                    self.failed_jobs += 1

                # Progress update
                progress = (self.completed_jobs + self.failed_jobs) / len(self.jobs) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({self.completed_jobs} completed, {self.failed_jobs} failed)")

        end_time = time.time()
        total_time = end_time - start_time

        self.gpu_monitor.stop_monitoring()

        # Generate report
        report = self._generate_report(total_time)

        # Save report
        report_file = self.output_dir / "batch_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Batch processing completed in {total_time:.1f} seconds")
        self.logger.info(f"Results: {self.completed_jobs} completed, {self.failed_jobs} failed")
        self.logger.info(f"Report saved to: {report_file}")

        return report

    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive report"""
        completed_jobs = [job for job in self.jobs if job.status == "completed"]
        failed_jobs = [job for job in self.jobs if job.status == "failed"]

        total_size = sum(job.file_size for job in completed_jobs)
        avg_time = sum(job.end_time - job.start_time for job in completed_jobs) / len(completed_jobs) if completed_jobs else 0

        return {
            "summary": {
                "total_jobs": len(self.jobs),
                "completed": self.completed_jobs,
                "failed": self.failed_jobs,
                "success_rate": self.completed_jobs / len(self.jobs) * 100 if self.jobs else 0,
                "total_time": total_time,
                "total_file_size": total_size,
                "average_generation_time": avg_time
            },
            "completed_jobs": [
                {
                    "job_id": job.job_id,
                    "prompt": job.prompt,
                    "output_path": str(job.output_path),
                    "file_size": job.file_size,
                    "generation_time": job.end_time - job.start_time
                }
                for job in completed_jobs
            ],
            "failed_jobs": [
                {
                    "job_id": job.job_id,
                    "prompt": job.prompt,
                    "error_message": job.error_message,
                    "generation_time": job.end_time - job.start_time
                }
                for job in failed_jobs
            ],
            "gpu_stats": self.gpu_monitor.get_current_stats()
        }

def main():
    parser = argparse.ArgumentParser(description="Wan Video Batch Generator")
    parser.add_argument("--jobs", help="JSON file with job definitions")
    parser.add_argument("--csv", help="CSV file with jobs (prompt, image_path, output_name)")
    parser.add_argument("--sample", type=int, help="Create N sample jobs", default=0)
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--output-dir", default="batch_outputs", help="Output directory")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames")

    args = parser.parse_args()

    # Configure batch processing
    config = BatchConfig(
        max_workers=args.workers,
        output_dir=args.output_dir,
        log_file="batch_generation.log",
        save_frames=args.save_frames
    )

    # Create generator
    generator = WanBatchGenerator(config)

    # Load jobs
    if args.jobs:
        if not generator.load_jobs_from_file(args.jobs):
            sys.exit(1)
    elif args.csv:
        if not generator.load_jobs_from_csv(args.csv):
            sys.exit(1)

    # Create sample jobs if requested
    if args.sample > 0:
        generator.create_sample_jobs(args.sample)

    if not generator.jobs:
        print("No jobs loaded. Use --jobs, --csv, or --sample to add jobs.")
        sys.exit(1)

    # Process batch
    report = generator.process_batch()

    # Print summary
    print("\n" + "="*50)
    print("BATCH GENERATION COMPLETE")
    print("="*50)
    print(f"✅ Completed: {report['summary']['completed']}")
    print(f"❌ Failed: {report['summary']['failed']}")
    print(f"📊 Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"⏱️  Total Time: {report['summary']['total_time']:.1f}s")
    print(f"💾 Total Size: {report['summary']['total_file_size'] / 1024 / 1024:.1f} MB")
    print(f"📁 Output Directory: {config.output_dir}")

if __name__ == "__main__":
    main()
