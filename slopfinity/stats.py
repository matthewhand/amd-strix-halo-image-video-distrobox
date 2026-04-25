import os
import shutil
import subprocess
from pathlib import Path


def get_sys_stats():
    s = {"gpu": 0, "vram": 0, "ram_u": 0, "ram_t": 0, "load_1m": 0.0, "load_5m": 0.0, "load_15m": 0.0, "load_pct": 0}
    try:
        res = subprocess.run(["rocm-smi", "--showuse", "--showmemuse"], capture_output=True, text=True)
        for l in res.stdout.split('\n'):
            if "GPU use (%)" in l: s["gpu"] = int(l.split(':')[-1].strip())
            if "GPU Memory Allocated (VRAM%)" in l: s["vram"] = int(l.split(':')[-1].strip())
    except: pass
    try:
        # Load average — better than instantaneous CPU% on Linux: captures pending
        # work + IO wait. /proc/loadavg gives 1m/5m/15m running averages. Express
        # it as a percentage of the CPU count so the ticker has a 0-100 scale.
        with open('/proc/loadavg', 'r') as f:
            parts = f.read().split()
        s["load_1m"] = float(parts[0])
        s["load_5m"] = float(parts[1])
        s["load_15m"] = float(parts[2])
        try:
            cpus = max(1, os.cpu_count() or 1)
        except Exception:
            cpus = 1
        s["load_pct"] = min(100, int(round((s["load_1m"] / cpus) * 100)))
    except Exception:
        pass
    try:
        with open('/proc/meminfo', 'r') as f:
            m = {ln.split(':')[0]: ln.split(':')[1].strip() for ln in f}
        s["ram_u"] = round((int(m['MemTotal'].split()[0]) - int(m['MemAvailable'].split()[0])) / (1024 * 1024), 1)
        s["ram_t"] = round(int(m['MemTotal'].split()[0]) / (1024 * 1024), 1)
    except: pass
    return s


def _status_from_pct(pct):
    # Relaxed: only warn close to full. 72 % isn't actionable noise.
    if pct >= 95: return "danger"
    if pct >= 90: return "warn"
    return "ok"


def get_outputs_disk(path):
    """Return {used_gb, total_gb, pct, status} for the filesystem hosting `path`.

    The navbar uses this to surface a single \"Disk\" % without naming the mount,
    since the user just cares whether their slop volume is filling up.
    """
    try:
        du = shutil.disk_usage(path)
        total_gb = round(du.total / (1024 ** 3), 1)
        used_gb = round(du.used / (1024 ** 3), 1)
        pct = round((du.used / du.total) * 100, 1) if du.total else 0
        return {"used_gb": used_gb, "total_gb": total_gb, "pct": pct, "status": _status_from_pct(pct)}
    except Exception:
        return {"used_gb": 0, "total_gb": 0, "pct": 0, "status": "ok", "missing": True}


def get_storage():
    """Return list of {mount, used_gb, total_gb, pct, status}."""
    mounts = ["/", "/mnt/data", "/mnt/downloads"]
    out = []
    for mnt in mounts:
        try:
            du = shutil.disk_usage(mnt)
            total_gb = round(du.total / (1024 ** 3), 1)
            used_gb = round(du.used / (1024 ** 3), 1)
            pct = round((du.used / du.total) * 100, 1) if du.total else 0
            out.append({
                "mount": mnt,
                "used_gb": used_gb,
                "total_gb": total_gb,
                "pct": pct,
                "status": _status_from_pct(pct),
            })
        except Exception:
            out.append({
                "mount": mnt,
                "used_gb": 0,
                "total_gb": 0,
                "pct": 0,
                "status": "ok",
                "missing": True,
            })
    return out


# Reference memory values for Strix Halo unified memory (GB)
_MODEL_GB = {
    # base image
    "qwen": 20,
    "ernie": 12,
    # ltx bases / video
    "ltx-2.3": 28,
    # video only
    "wan2.2": 48,
    "wan2.5": 56,
    # audio
    "heartmula": 10,
    # upscale
    "ltx-spatial": 18,
    # none / empty
    "none": 0,
    "No Audio": 0,
    "No Upscale": 0,
    "": 0,
}

_OVERHEAD_GB = 6


def _lookup(model):
    if model is None:
        return 0
    return _MODEL_GB.get(model, 0)


def get_output_counts(base_dir=None):
    """Return counters for what's been produced: finals, chain clips, base images.

    Counts files in the fleet's output directory. In-container this is
    /workspace; on host it's ./comfy-outputs/experiments. Returns:
        {finals, chains, base_images, total_mp4, total_png, latest_final}
    """
    if base_dir is None:
        for cand in ("/workspace", "./comfy-outputs/experiments"):
            if os.path.isdir(cand):
                base_dir = cand
                break
    if not base_dir or not os.path.isdir(base_dir):
        return {"finals": 0, "chains": 0, "base_images": 0,
                "total_mp4": 0, "total_png": 0, "latest_final": None}
    p = Path(base_dir)
    finals = sorted(p.glob("FINAL_*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
    chains = list(p.glob("v*_c*.mp4"))
    base_imgs = list(p.glob("v*_base.png"))
    latest = finals[0].name if finals else None
    return {
        "finals": len(finals),
        "chains": len(chains),
        "base_images": len(base_imgs),
        "total_mp4": len(list(p.glob("*.mp4"))),
        "total_png": len(list(p.glob("*.png"))),
        "latest_final": latest,
    }


def get_ram_estimate(base_model, video_model, audio_model, upscale_model):
    """Return {estimated_gb, breakdown:[{stage, model, gb}], status}."""
    stages = [
        ("Base Image", base_model),
        ("Video", video_model),
        ("Audio", audio_model),
        ("Upscale", upscale_model),
    ]
    breakdown = []
    total = 0
    for stage, model in stages:
        gb = _lookup(model)
        breakdown.append({"stage": stage, "model": model or "none", "gb": gb})
        total += gb
    breakdown.append({"stage": "Overhead", "model": "OS + ComfyUI", "gb": _OVERHEAD_GB})
    total += _OVERHEAD_GB

    if total >= 100:
        status = "danger"
    elif total >= 80:
        status = "warn"
    else:
        status = "ok"

    return {
        "estimated_gb": round(total, 1),
        "budget_gb": 128,
        "breakdown": breakdown,
        "status": status,
    }
