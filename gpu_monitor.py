#!/usr/bin/env python3
"""
GPU Monitoring for AMD Strix Halo
Real-time GPU usage tracking for video generation
"""
import os
import sys
import time
import json
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class GPUStats:
    """GPU statistics snapshot"""
    timestamp: float
    memory_used: int
    memory_total: int
    memory_percent: float
    temperature: float
    power_usage: float
    gpu_utilization: float
    clock_speed: str

class GPUHistory:
    """Historical GPU data"""
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.stats_history: list[GPUStats] = []
        self.lock = threading.Lock()

    def add_stat(self, stat: GPUStats) -> None:
        """Add a new stat snapshot"""
        with self.lock:
            self.stats_history.append(stat)
            if len(self.stats_history) > self.max_points:
                self.stats_history.pop(0)

    def get_latest(self) -> Optional[GPUStats]:
        """Get the latest stat"""
        with self.lock:
            return self.stats_history[-1] if self.stats_history else None

    def get_average_memory(self, last_n: int = 10) -> float:
        """Get average memory usage over last N points"""
        with self.lock:
            recent_stats = self.stats_history[-last_n:]
            if not recent_stats:
                return 0.0
            return sum(s.memory_percent for s in recent_stats) / len(recent_stats)

    def get_history_dict(self) -> Dict[str, Any]:
        """Get history as dictionary"""
        with self.lock:
            return {
                "history": [
                    {
                        "timestamp": s.timestamp,
                        "memory_percent": s.memory_percent,
                        "temperature": s.temperature,
                        "power": s.power_usage,
                        "utilization": s.gpu_utilization
                    }
                    for s in self.stats_history
                ]
            }

class AMDGPUMonitor:
    """AMD ROCm GPU monitor"""

    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        self.history = GPUHistory()
        self.monitoring = False
        self.monitor_thread = None
        self.last_stats = None

    def get_rocm_stats(self) -> Optional[GPUStats]:
        """Get GPU stats from ROCm"""
        try:
            # Try rocm-smi first
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showtemp", "--showpower", "--showclock"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                return self._parse_rocm_output(result.stdout)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback to basic system monitoring
        try:
            # Try reading from /sys/class/drm for basic GPU info
            drm_path = Path("/sys/class/drm")
            if drm_path.exists():
                for card_dir in drm_path.iterdir():
                    if card_dir.name.startswith("card"):
                        return self._parse_drm_info(card_dir)
        except:
            pass

        return None

    def _parse_rocm_output(self, output: str) -> GPUStats:
        """Parse ROCm smi output"""
        lines = output.strip().split('\n')
        stats = {}

        for line in lines:
            line = line.strip()
            if 'GPU' in line and 'MiB' in line:
                parts = line.split()
                if len(parts) >= 3:
                    stats['memory_used'] = int(parts[2].replace('MiB', ''))
            elif 'GPU Memory Usage' in line:
                parts = line.split()
                if len(parts) >= 4:
                    stats['memory_percent'] = float(parts[3].rstrip('%'))
            elif 'Temperature' in line and 'C' in line:
                parts = line.split()
                if parts:
                    stats['temperature'] = float(parts[-1].replace('C', ''))
            elif 'Average Power' in line and 'W' in line:
                parts = line.split()
                if parts:
                    stats['power'] = float(parts[-1].replace('W', ''))
            elif 'SCLK' in line and 'MHz' in line:
                parts = line.split()
                if len(parts) >= 2:
                    stats['clock_speed'] = parts[-1]

        # Calculate total memory if we have usage and percentage
        if 'memory_used' in stats and 'memory_percent' in stats:
            stats['memory_total'] = int(stats['memory_used'] / (stats['memory_percent'] / 100))
        else:
            stats['memory_total'] = 0

        # Set defaults for missing values
        stats.setdefault('memory_used', 0)
        stats.setdefault('memory_total', 0)
        stats.setdefault('memory_percent', 0.0)
        stats.setdefault('temperature', 0.0)
        stats.setdefault('power_usage', 0.0)
        stats.setdefault('gpu_utilization', 0.0)
        stats.setdefault('clock_speed', 'N/A')

        return GPUStats(
            timestamp=time.time(),
            **stats
        )

    def _parse_drm_info(self, card_dir: Path) -> GPUStats:
        """Parse basic DRM info"""
        try:
            # Read memory info
            meminfo_file = card_dir / "device/mem_info"
            if meminfo_file.exists():
                with open(meminfo_file, 'r') as f:
                    for line in f:
                        if 'drm_memory' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                total = int(parts[1])
                                if 'used' in line:
                                    used = int(parts[1])
                                    stats['memory_percent'] = (used / total) * 100

        except:
            pass

        return GPUStats(
            timestamp=time.time(),
            memory_used=0,
            memory_total=0,
            memory_percent=0.0,
            temperature=0.0,
            power_usage=0.0,
            gpu_utilization=0.0,
            clock_speed="N/A"
        )

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_rocm_stats()
                if stats:
                    self.history.add_stat(stats)
                    self.last_stats = stats
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"GPU Monitor Error: {e}")
                time.sleep(self.poll_interval)

    def start_monitoring(self):
        """Start GPU monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("🔍 GPU monitoring started")

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            print("🛑 GPU monitoring stopped")

    def get_current_stats(self) -> Optional[GPUStats]:
        """Get current GPU stats"""
        return self.last_stats or self.get_rocm_stats()

    def print_stats(self):
        """Print current GPU stats"""
        stats = self.get_current_stats()
        if stats:
            print(f"📊 GPU Stats:")
            print(f"   Memory: {stats.memory_used}MB / {stats.memory_total}MB ({stats.memory_percent:.1f}%)")
            print(f"   Temperature: {stats.temperature}°C")
            print(f"   Power: {stats.power_usage}W")
            print(f"   Clock: {stats.clock_speed}")
            print(f"   Utilization: {stats.gpu_utilization:.1f}%")
        else:
            print("❌ GPU stats unavailable")

    def save_history(self, filename: str) -> bool:
        """Save monitoring history to file"""
        try:
            history_data = self.history.get_history_dict()
            history_data['metadata'] = {
                'start_time': self.history.stats_history[0].timestamp if self.history.stats_history else time.time(),
                'end_time': time.time(),
                'total_points': len(self.history.stats_history),
                'poll_interval': self.poll_interval
            }

            with open(filename, 'w') as f:
                json.dump(history_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save history: {e}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive GPU report"""
        if not self.history.stats_history:
            return {"error": "No GPU data available"}

        # Calculate statistics
        memory_values = [s.memory_percent for s in self.history.stats_history]
        temp_values = [s.temperature for s in self.history.stats_history if s.temperature > 0]
        power_values = [s.power_usage for s in self.history.stats_history if s.power_usage > 0]

        return {
            "monitoring_summary": {
                "monitoring_duration": time.time() - self.history.stats_history[0].timestamp,
                "total_samples": len(self.history.stats_history),
                "poll_interval": self.poll_interval,
                "start_time": datetime.fromtimestamp(self.history.stats_history[0].timestamp).isoformat(),
                "end_time": datetime.fromtimestamp(self.history.stats_history[-1].timestamp).isoformat()
            },
            "gpu_statistics": {
                "memory": {
                    "peak_usage": max(memory_values),
                    "average_usage": sum(memory_values) / len(memory_values),
                    "current_usage": memory_values[-1] if memory_values else 0,
                    "total_samples": len(memory_values)
                },
                "temperature": {
                    "peak": max(temp_values) if temp_values else "N/A",
                    "average": sum(temp_values) / len(temp_values) if temp_values else "N/A",
                    "current": temp_values[-1] if temp_values else "N/A",
                    "total_samples": len(temp_values)
                },
                "power": {
                    "peak": max(power_values) if power_values else "N/A",
                    "average": sum(power_values) / len(power_values) if power_values else "N/A",
                    "current": power_values[-1] if power_values else "N/A",
                    "total_samples": len(power_values)
                }
            },
            "current_stats": {
                "memory_used_mb": self.history.stats_history[-1].memory_used,
                "memory_total_mb": self.history.stats_history[-1].memory_total,
                "memory_percent": self.history.stats_history[-1].memory_percent,
                "temperature_c": self.history.stats_history[-1].temperature,
                "power_watts": self.history.stats_history[-1].power_usage,
                "clock_speed_mhz": self.history.stats_history[-1].clock_speed
            }
        }

class VideoGenerationMonitor:
    """Monitor video generation performance"""

    def __init__(self):
        self.gpu_monitor = AMDGPUMonitor()
        self.generations = []
        self.current_generation = None

    def start_generation(self, job_id: str, prompt: str):
        """Start monitoring a generation"""
        self.current_generation = {
            "job_id": job_id,
            "prompt": prompt,
            "start_time": time.time(),
            "gpu_stats_start": self.gpu_monitor.get_current_stats(),
            "peak_memory": 0,
            "peak_temperature": 0,
            "peak_power": 0
        }

    def end_generation(self, success: bool = True, error: str = ""):
        """End generation monitoring"""
        if not self.current_generation:
            return

        end_time = time.time()
        duration = end_time - self.current_generation["start_time"]
        end_stats = self.gpu_monitor.get_current_stats()

        generation_info = {
            "job_id": self.current_generation["job_id"],
            "prompt": self.current_generation["prompt"],
            "start_time": self.current_generation["start_time"],
            "end_time": end_time,
            "duration": duration,
            "success": success,
            "error": error,
            "gpu_stats_start": self.current_generation["gpu_stats_start"],
            "gpu_stats_end": end_stats,
            "peak_memory": self.current_generation["peak_memory"],
            "peak_temperature": self.current_generation["peak_temperature"],
            "peak_power": self.current_generation["peak_power"]
        }

        self.generations.append(generation_info)
        self.current_generation = None
        return generation_info

    def update_peaks(self):
        """Update peak values during generation"""
        if self.current_generation and self.gpu_monitor.last_stats:
            stats = self.gpu_monitor.last_stats
            self.current_generation["peak_memory"] = max(
                self.current_generation["peak_memory"], stats.memory_percent
            )
            self.current_generation["peak_temperature"] = max(
                self.current_generation["peak_temperature"], stats.temperature
            )
            self.current_generation["peak_power"] = max(
                self.current_generation["peak_power"], stats.power_usage
            )

def main():
    parser = argparse.ArgumentParser(description="AMD GPU Monitor for Video Generation")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save", help="Save monitoring history to file")
    save_group.add_argument("--load", help="Load monitoring history from file")
    parser.add_argument("--report", action="store_true", help="Generate monitoring report")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    args = parser.parse_args()

    monitor = AMDGPUMonitor()

    if args.load:
        try:
            with open(args.load, 'r') as f:
                data = json.load(f)
            print(f"Loaded monitoring data from {args.load}")
            print(f"Data points: {len(data.get('history', []))}")
        except Exception as e:
            print(f"Failed to load data: {e}")
            return

    if args.monitor:
        print(f"🔍 Starting GPU monitoring for {args.duration} seconds...")
        monitor.start_monitoring()

        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\n⚠️  Monitoring interrupted by user")
        finally:
            monitor.stop_monitoring()

        if args.save:
            monitor.save_history(args.save)
            print(f"💾 Saved monitoring history to {args.save}")

    elif args.report:
        monitor.start_monitoring()
        time.sleep(5)  # Collect some data
        monitor.stop_monitoring()

        report = monitor.generate_report()

        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            print("📊 GPU Monitoring Report")
            print("="*40)
            print(f"Monitoring Duration: {report['monitoring_summary']['monitoring_duration']:.1f}s")
            print(f"Total Samples: {report['monitoring_summary']['total_samples']}")
            print(f"Sample Rate: {1/report['monitoring_summary']['poll_interval']:.1f} Hz")
            print()

            if "gpu_statistics" in report:
                stats = report["gpu_statistics"]
                print("🧠 Memory Statistics:")
                print(f"   Peak Usage: {stats['memory']['peak_usage']:.1f}%")
                print(f"   Average: {stats['memory']['average_usage']:.1f}%")
                print(f"   Current: {stats['memory']['current_usage']:.1f}%")
                print()
                print("🌡 Temperature Statistics:")
                if stats["temperature"]["peak"] != "N/A":
                    print(f"   Peak: {stats['temperature']['peak']}°C")
                    print(f"   Average: {stats['temperature']['average']:.1f}°C")
                    print(f"   Current: {stats['temperature']['current']}°C")
                else:
                    print("   Temperature data not available")
                print()
                print("⚡ Power Statistics:")
                if stats["power"]["peak"] != "N/A":
                    print(f"   Peak: {stats['power']['peak']:.1f}W")
                    print(f"   Average: {stats['power']['average']:.1f}W")
                    print(f"   Current: {stats['power']['current']:.1f}W")
                else:
                    print("   Power data not available")
    else:
        # Just show current stats
        monitor.print_stats()

if __name__ == "__main__":
    main()