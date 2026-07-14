import os
import shutil
import subprocess
from pathlib import Path


def _detect_gpu_name():
    """Read the AMD GPU's marketing name once. Cached at module level so we
    don't fork rocm-smi every WS tick. Returns "" on failure — the navbar
    just hides the slot in that case."""
    try:
        res = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True, timeout=5)
        for ln in res.stdout.split('\n'):
            for key in ("Card SKU", "Card Series", "Card series",
                        "Card model", "Card Model"):
                if key in ln:
                    val = ln.split(':', 2)[-1].strip()
                    if val:
                        return val
    except Exception:
        pass
    # Fallback: lspci's first 3D/VGA controller line.
    try:
        res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=2)
        for ln in res.stdout.split('\n'):
            if 'VGA compatible' in ln or '3D controller' in ln:
                # "XX:XX.X VGA compatible controller: AMD/ATI [...]"
                tail = ln.split(':', 2)[-1].strip()
                # Drop the "(rev XX)" suffix if present.
                tail = tail.split('(rev ')[0].strip()
                return tail
    except Exception:
        pass
    return ""


def _detect_cpu_name():
    """Read the CPU's marketing name from /proc/cpuinfo's first 'model name'
    line. Strix Halo lists it as e.g. 'AMD RYZEN AI MAX+ 395 w/ Radeon 8060S'
    — strip the trailing 'w/ <iGPU>' since that GPU is surfaced separately
    under the GPU label. Returns "" on failure."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    name = line.split(':', 1)[-1].strip()
                    # Drop the iGPU suffix to avoid duplication with the
                    # GPU subtitle row above.
                    for sep in (' w/ ', ' with '):
                        if sep in name:
                            name = name.split(sep, 1)[0].strip()
                    return name
    except Exception:
        pass
    return ""


_GPU_NAME = None
_CPU_NAME = None


def get_sys_stats():
    global _GPU_NAME, _CPU_NAME
    if _GPU_NAME is None:
        _GPU_NAME = _detect_gpu_name() or ""
    if _CPU_NAME is None:
        _CPU_NAME = _detect_cpu_name() or ""
    s = {"gpu": 0, "vram": 0, "ram_u": 0, "ram_t": 0,
         "load_1m": 0.0, "load_5m": 0.0, "load_15m": 0.0, "load_pct": 0,
         "gpu_name": _GPU_NAME, "cpu_name": _CPU_NAME}
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
    # Relaxed further: only flag the disk when it is genuinely close to full.
    # warn at 92 %, danger at 97 % — anything below is operational noise.
    if pct >= 97: return "danger"
    if pct >= 92: return "warn"
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
        free_gb = round(du.free / (1024 ** 3), 1)
        return {"used_gb": used_gb, "total_gb": total_gb, "free_gb": free_gb, "pct": pct, "status": _status_from_pct(pct)}
    except Exception:
        return {"used_gb": 0, "total_gb": 0, "pct": 0, "status": "ok", "missing": True}


def check_disk_guard():
    """Return (ok, reason) — False when the outputs partition is below
    the user-configured low-water marks. Two thresholds; either trips
    the guard. Setting either to 0 disables that check.

    Lives here (rather than server.py) so every router that needs it
    (queue.py /inject, runner.py /disk/guard) can import it without a
    circular dependency on server.py, which itself imports all routers.
    """
    # Lazy imports: config + paths are cheap but importing them at module
    # top would tangle the import graph (paths is imported widely).
    import slopfinity.config as cfg
    from slopfinity.paths import EXP_DIR

    config = cfg.load_config()
    min_pct = float(config.get("disk_min_pct") or 0)
    min_gb = float(config.get("disk_min_gb") or 0)
    if min_pct <= 0 and min_gb <= 0:
        return True, ""
    try:
        d = get_outputs_disk(EXP_DIR)
        free_gb = d.get("free_gb")
        if free_gb is None:
            free_gb = (d.get("total_gb") or 0) - (d.get("used_gb") or 0)
        free_pct = 100 - (d.get("pct") or 0)
    except Exception:
        return True, ""  # fail open if we can't read disk stats
    if min_pct > 0 and free_pct <= min_pct:
        return False, f"only {free_pct:.1f}% free (threshold ≤ {min_pct}%)"
    if min_gb > 0 and free_gb <= min_gb:
        return False, f"only {free_gb:.1f} GB free (threshold ≤ {min_gb} GB)"
    return True, ""


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


# Peak resident GB for Strix Halo UMA — aligned with stage_gate.STAGE_BUDGETS
# (conservative; prefer over-estimate over OOM). Used for UI WILL-USE and
# serial headroom math: free_after_load must stay ≥ SAFETY_FREE_GB.
_MODEL_GB = {
    # base image
    "qwen": 28,
    "qwen-image": 28,
    "ernie": 18,
    # ltx: default to video peak when role unknown; role-aware overrides below
    "ltx-2.3": 48,
    # video only
    "wan2.2": 84,
    "wan2.5": 96,
    # audio (music)
    "heartmula": 14,
    # voice (TTS)
    "qwen-tts": 10,
    "kokoro": 8,
    "dramabox": 18,
    # upscale
    "ltx-spatial": 30,
    # none / empty
    "none": 0,
    "No Audio": 0,
    "No Upscale": 0,
    "": 0,
}

# Role-aware peaks when the same model name means different footprints
_ROLE_MODEL_GB = {
    ("image", "ltx-2.3"): 38,
    ("video", "ltx-2.3"): 48,
    ("image", "qwen"): 28,
    ("image", "ernie"): 18,
    ("tts", "dramabox"): 18,
    ("tts", "kokoro"): 8,
    ("tts", "qwen-tts"): 10,
    ("audio", "heartmula"): 14,
}

SAFETY_FREE_GB = 10  # must remain free after model load (stage_gate floor)

# Pretty labels for the WILL-USE breakdown (mirrors _modelDisplayName in app.js).
_MODEL_LABEL = {
    "qwen": "Qwen Image",
    "qwen-image": "Qwen Image",
    "ernie": "Ernie Image",
    "ltx-2.3": "LTX-2.3",
    "wan2.2": "Wan 2.2",
    "wan2.5": "Wan 2.5",
    "heartmula": "Heartmula",
    "qwen-tts": "Qwen-TTS",
    "kokoro": "Kokoro-TTS",
    "dramabox": "DramaBox",
    "ltx-spatial": "LTX Spatial x2",
}


def _pretty(model):
    if not model or model == "none":
        return "—"
    if isinstance(model, str) and model.startswith("slopped:"):
        return "Slopped (" + model.split(":", 1)[1] + ")"
    return _MODEL_LABEL.get(model, model)

_OVERHEAD_GB = 6


def _lookup(model, role=None):
    if model is None:
        return 0
    # `slopped:<file>` placeholders contribute the same RAM as the role's
    # default model would — we don't actually load a fresh checkpoint, so
    # treat them as zero incremental cost. This keeps the WILL-USE numbers
    # honest even when the user picks an existing asset for a role.
    if isinstance(model, str) and model.startswith("slopped:"):
        return 0
    # Config abstract: scheduler.stage_budget_overrides (wan2.x → 0 by default)
    try:
        from .stage_gate import need_gb as _need_gb

        if role:
            return float(_need_gb(str(role), str(model)))
        # bare model: try video role first (wan lives there), else table
        if str(model).lower().startswith("wan"):
            return float(_need_gb("video", str(model)))
    except Exception:
        pass
    if role:
        rg = _ROLE_MODEL_GB.get((str(role).lower(), str(model).lower()))
        if rg is not None:
            return rg
    return _MODEL_GB.get(model, 0)


def get_output_counts(base_dir=None):
    """Return counters for what's been produced: finals, chain clips, base images.

    Counts files in the fleet's output directory. In-container this is
    /workspace; on host it's ./comfy-outputs/experiments. Returns:
        {finals, chains, base_images, total_mp4, total_png, latest_final}

    Single directory pass instead of multiple globs (broadcast/tick hot path).
    """
    if base_dir is None:
        for cand in ("/workspace", "./comfy-outputs/experiments"):
            if os.path.isdir(cand):
                base_dir = cand
                break
    if not base_dir or not os.path.isdir(base_dir):
        return {"finals": 0, "chains": 0, "base_images": 0,
                "total_mp4": 0, "total_png": 0, "latest_final": None}
    import fnmatch
    p = Path(base_dir)
    finals = []
    chains = 0
    base_imgs = 0
    total_mp4 = 0
    total_png = 0
    try:
        entries = list(p.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        name = entry.name
        if fnmatch.fnmatchcase(name, "*.mp4"):
            total_mp4 += 1
            if fnmatch.fnmatchcase(name, "FINAL_*.mp4"):
                finals.append(entry)
            # Match both current "slop_<idx>_" prefix and legacy "v<idx>_" form.
            elif (fnmatch.fnmatchcase(name, "slop_*_c*.mp4")
                  or fnmatch.fnmatchcase(name, "v*_c*.mp4")):
                chains += 1
        elif fnmatch.fnmatchcase(name, "*.png"):
            total_png += 1
            if (fnmatch.fnmatchcase(name, "slop_*_base.png")
                    or fnmatch.fnmatchcase(name, "v*_base.png")):
                base_imgs += 1
    try:
        finals.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    except OSError:
        pass
    latest = finals[0].name if finals else None
    return {
        "finals": len(finals),
        "chains": chains,
        "base_images": base_imgs,
        "total_mp4": total_mp4,
        "total_png": total_png,
        "latest_final": latest,
    }


def get_ram_estimate(base_model, video_model, audio_model, upscale_model, tts_model=None):
    """Return {estimated_gb, breakdown:[{role, stage, model, label, gb}], status}.

    Each breakdown entry now carries a pretty `label` and a stable `role` key
    so the UI can render a per-model WILL-USE table with friendly names. The
    final `Overhead` row keeps role=`overhead` for the same reason.
    """
    stages = [
        ("image",   "Image",   base_model),
        ("video",   "Video",   video_model),
        ("audio",   "Music",   audio_model),
        ("tts",     "Voice",   tts_model),
        ("upscale", "Upscale", upscale_model),
    ]
    breakdown = []
    naive_total = 0  # all warm at once (legacy sum — OOM risk)
    serial_peak = 0  # max single-stage peak if stages are gated/serial
    for role, stage, model in stages:
        gb = _lookup(model, role=role)
        breakdown.append({
            "role":  role,
            "stage": stage,
            "model": model or "none",
            "label": _pretty(model),
            "gb":    gb,
        })
        naive_total += gb
        if gb > serial_peak:
            serial_peak = gb
    breakdown.append({
        "role":  "overhead",
        "stage": "Overhead",
        "model": "OS + ComfyUI",
        "label": "OS + ComfyUI",
        "gb":    _OVERHEAD_GB,
    })
    naive_total += _OVERHEAD_GB
    # Serial path: peak stage + overhead + free floor after load
    serial_with_floor = serial_peak + _OVERHEAD_GB + SAFETY_FREE_GB
    # Prefer serial estimate for "will use" when stage_gate is the runtime path
    total = serial_with_floor

    # Status reflects serial (gated) need; naive all-warm only raises warn
    # so the UI still flags stacked-load risk without claiming serial is unsafe.
    if serial_with_floor >= 100:
        status = "danger"
    elif serial_with_floor >= 80 or naive_total >= 100:
        status = "warn"
    else:
        status = "ok"

    return {
        "estimated_gb": round(total, 1),
        "serial_peak_gb": round(serial_peak, 1),
        "serial_need_gb": round(serial_with_floor, 1),  # peak + oh + keep 10 free
        "naive_all_warm_gb": round(naive_total, 1),
        "safety_free_gb": SAFETY_FREE_GB,
        "budget_gb": 128,
        "breakdown": breakdown,
        "status": status,
    }
