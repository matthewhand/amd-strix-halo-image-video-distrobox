import os
import json
import asyncio
import subprocess
import time
import urllib.request
import urllib.error
from typing import List

from fastapi import APIRouter, Form, Request, UploadFile, File, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from slopfinity.paths import EXP_DIR, TTS_OUT_DIR, TEMPLATES_DIR
import slopfinity.config as cfg
from slopfinity.stats import get_outputs_disk, get_ram_estimate
import slopfinity.scheduler as sched
from slopfinity.workers import ffmpeg_mux as _ffmpeg_mux

templates = Jinja2Templates(directory=TEMPLATES_DIR)

TTS_WORKER_URL = os.environ.get("TTS_WORKER_URL", "http://localhost:8010/tts")

_SLOPPED_EXTS = {
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "audio": (".wav", ".mp3", ".flac", ".ogg"),
    "tts": (".wav", ".mp3", ".flac", ".ogg"),
}

router = APIRouter()

def _check_disk_guard():
    """Return (ok, reason) — False when the outputs partition is below
    the user-configured low-water marks."""
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
        return True, ""
    if min_pct > 0 and free_pct <= min_pct:
        return False, f"only {free_pct:.1f}% free (threshold ≤ {min_pct}%)"
    if min_gb > 0 and free_gb <= min_gb:
        return False, f"only {free_gb:.1f} GB free (threshold ≤ {min_gb} GB)"
    return True, ""

def _find_pids_by_cmdline(needle: str) -> list[int]:
    """Scan /proc for processes whose argv contains a leaf matching `needle`."""
    pids: list[int] = []
    if os.path.isdir("/proc"):
        try:
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry.name}/cmdline", "rb") as f:
                        raw = f.read()
                    if not raw:
                        continue
                    args = [
                        a.decode("utf-8", errors="replace")
                        for a in raw.split(b"\x00")
                        if a
                    ]
                    if any(os.path.basename(a) == needle for a in args):
                        pids.append(int(entry.name))
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
            return pids
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["pgrep", "-f", needle], capture_output=True, text=True, timeout=5
        ).stdout
        return [int(p) for p in out.split() if p.isdigit()]
    except Exception:
        return []

def _resolve_tts_worker_url() -> str:
    """Pick the TTS worker URL: settings config > env > hardcoded default."""
    cfg_url = (cfg.load_config().get("tts_worker_url") or "").strip()
    if cfg_url:
        return cfg_url
    return TTS_WORKER_URL

def _call_tts_worker(text: str, voice: str, timeout: float = 600.0,
                     engine: str | None = None, lang: str | None = None,
                     speed: float | None = None) -> dict:
    """POST to the TTS worker (kokoro / qwen multi-engine since v337)."""
    body = {"text": text, "voice": voice}
    if engine: body["engine"] = engine
    if lang: body["lang"] = lang
    if speed is not None: body["speed"] = speed
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _resolve_tts_worker_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)

@router.get("/pipeline/slopped")
async def pipeline_slopped(role: str):
    exts = _SLOPPED_EXTS.get(role)
    if not exts:
        return {"role": role, "files": []}
    files = []
    try:
        for name in os.listdir(EXP_DIR):
            if not name.lower().endswith(exts):
                continue
            path = os.path.join(EXP_DIR, name)
            if not os.path.isfile(path):
                continue
            files.append((name, os.path.getmtime(path)))
    except Exception:
        pass
    files.sort(key=lambda x: x[1], reverse=True)
    return {"role": role, "files": [n for n, _ in files[:60]]}

@router.get("/disk/guard")
async def disk_guard_endpoint():
    ok, reason = _check_disk_guard()
    config = cfg.load_config()
    try:
        d = get_outputs_disk(EXP_DIR)
        free_gb = d.get("free_gb")
        if free_gb is None:
            free_gb = (d.get("total_gb") or 0) - (d.get("used_gb") or 0)
        free_pct = round(100 - (d.get("pct") or 0), 1)
    except Exception:
        free_gb = 0
        free_pct = 0
    return {
        "ok": ok,
        "reason": reason,
        "free_pct": free_pct,
        "free_gb": round(float(free_gb or 0), 1),
        "threshold_pct": float(config.get("disk_min_pct") or 0),
        "threshold_gb": float(config.get("disk_min_gb") or 0),
    }

@router.get("/runner/status")
async def runner_status():
    pids = _find_pids_by_cmdline("run_fleet.py")
    info = []
    now = time.time()
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat = f.read().split()
            starttime_ticks = int(stat[21])
            clk_tck = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", 100))
            with open("/proc/uptime", "r") as f:
                uptime_s = float(f.read().split()[0])
            age_s = uptime_s - (starttime_ticks / clk_tck)
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            info.append({"pid": pid, "age_s": round(age_s, 1), "cmdline": cmd[:200]})
        except Exception:
            info.append({"pid": pid, "age_s": None, "cmdline": "(unreadable)"})
    return {"ok": True, "running": len(pids) > 0, "pids": info, "wall_time": now}

@router.post("/runner/terminate")
async def runner_terminate():
    import signal
    flag_path = os.path.join(EXP_DIR, "terminate.flag")
    flag_written = False
    try:
        with open(flag_path, "w") as f:
            f.write(str(time.time()))
        flag_written = True
    except Exception:
        pass
    pids = _find_pids_by_cmdline("run_fleet.py")
    killed: list[int] = []
    perm_errs: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            perm_errs.append(pid)
    await asyncio.sleep(2.0)
    escalated: list[int] = []
    for pid in killed:
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            escalated.append(pid)
        except Exception:
            pass
    return {
        "ok": True,
        "flag_written": flag_written,
        "flag_path": flag_path,
        "matched_pids": pids,
        "killed": killed,
        "escalated_to_sigkill": escalated,
        "permission_denied_pids": perm_errs,
    }

@router.post("/runner/terminate-clear")
async def runner_terminate_clear():
    flag_path = os.path.join(EXP_DIR, "terminate.flag")
    existed = os.path.exists(flag_path)
    try:
        os.remove(flag_path)
    except FileNotFoundError:
        pass
    return {"ok": True, "existed": existed, "flag_path": flag_path}

@router.get("/tts/voices")
async def tts_voices():
    """List voices the configured TTS worker exposes."""
    fallback = {
        "ok": True,
        "engines": {
            "kokoro": {
                "default_voice": "af_heart",
                "voices": [
                    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
                    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
                    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                    "am_michael", "am_onyx", "am_puck", "am_santa",
                    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
                    "ef_dora", "em_alex", "em_santa",
                    "ff_siwis",
                    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
                    "if_sara", "im_nicola",
                    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
                    "pf_dora", "pm_alex", "pm_santa",
                    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
                ],
            },
            "qwen": {
                "default_voice": "ryan",
                "voices": [
                    "aiden", "dylan", "eric", "ono_anna", "ryan",
                    "serena", "sohee", "uncle_fu", "vivian",
                ],
            },
        },
        "source": "fallback (worker unreachable)",
    }
    base_url = _resolve_tts_worker_url()
    voices_url = base_url.rstrip("/")
    if voices_url.endswith("/tts"):
        voices_url = voices_url[:-4]
    voices_url = voices_url.rstrip("/") + "/voices"
    try:
        req = urllib.request.Request(voices_url, method="GET",
                                     headers={"accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            body = json.loads(r.read().decode("utf-8"))
            body.setdefault("source", "worker")
            return body
    except Exception:
        return fallback

@router.post("/tts")
async def tts(data: dict = Body(...)):
    """Proxy to the TTS worker."""
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "af_heart"
    engine = data.get("engine")
    lang = data.get("lang")
    speed_raw = data.get("speed")
    try:
        speed = float(speed_raw) if speed_raw is not None else None
    except (TypeError, ValueError):
        speed = None

    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)
    MAX_TEXT_CHARS = 5000
    if len(text) > MAX_TEXT_CHARS:
        return JSONResponse({"ok": False, "error": f"text too long"}, status_code=413)
    
    try:
        async with sched.acquire_gpu("TTS", "qwen-tts", safety_gb=4):
            result = await asyncio.to_thread(
                _call_tts_worker, text, voice,
                engine=engine, lang=lang, speed=speed,
            )
    except urllib.error.URLError as e:
        return JSONResponse(
            {"ok": False, "status": "worker-unreachable", "error": str(e), "voice": voice},
            status_code=503,
        )
    except Exception as e:
        return JSONResponse(
            {"ok": False, "status": "worker-error", "error": str(e), "voice": voice},
            status_code=502,
        )
    url = result.get("url") or result.get("audio_path")
    return {
        "ok": bool(result.get("ok")),
        "status": result.get("status", "ok" if result.get("ok") else "error"),
        "url": url,
        "audio_path": url,
        "engine": result.get("engine") or engine,
        "voice": result.get("voice", voice),
        "error": result.get("error"),
    }

@router.post("/mux")
async def mux(data: dict = Body(...)):
    vrel = data.get("video_path") or ""
    arel = data.get("audio_path") or ""
    out_name = data.get("out_name") or f"muxed_{int(time.time() * 1000)}.mp4"
    if not vrel or not arel:
        return JSONResponse({"ok": False, "error": "video_path and audio_path required"}, status_code=400)

    def _resolve(p: str) -> str:
        if os.path.isabs(p): return p
        return os.path.join(EXP_DIR, p.lstrip("/"))

    video = _resolve(vrel)
    audio = _resolve(arel)
    out_path = os.path.join(EXP_DIR, out_name)

    try:
        ok = _ffmpeg_mux.mux(
            video, audio, out_path,
            loop_audio=bool(data.get("loop_audio")),
            pad_to_video=bool(data.get("pad_to_video", True)),
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    if not ok:
        return JSONResponse({"ok": False, "error": "ffmpeg mux failed"}, status_code=500)
    return {"ok": True, "url": f"/files/{out_name}"}

@router.get("/ram_estimate")
async def ram_estimate(base: str = "", video: str = "", audio: str = "", upscale: str = "", tts: str = ""):
    return get_ram_estimate(base or None, video or None, audio or None, upscale or None, tts or None)

@router.get("/system/ram")
async def system_ram():
    available = sched._mem_available_gb()
    safety = float(sched.SAFETY_GB)
    return {"available_gb": available, "safety_gb": safety, "tight": available < safety}

@router.get("/pipeline/plan")
async def pipeline_plan(lookahead: int = 2):
    from .memory_planner import (
        build_sequence_for_job,
        plan_resident_set,
        naive_load_count,
        planned_load_count,
    )
    config = cfg.load_config()
    queue = cfg.get_queue() or []

    def _job_models(job: dict | None) -> tuple:
        j = job or {}
        return (
            j.get("base_model")    or config.get("base_model"),
            j.get("video_model")   or config.get("video_model"),
            j.get("audio_model")   or config.get("audio_model"),
            j.get("tts_model")     or config.get("tts_model"),
            j.get("upscale_model") or config.get("upscale_model"),
        )

    pending = [j for j in queue if (j.get("status") in (None, "pending"))]
    jobs_to_plan = [None] + pending[: max(0, int(lookahead))]
    sequence = []
    flat_for_planner = []
    for ji, job in enumerate(jobs_to_plan):
        base, video, audio, tts_, upscale = _job_models(job)
        steps = build_sequence_for_job(base, video, audio, tts_, upscale)
        for s in steps:
            flat_for_planner.append(s)
            sequence.append({"stage": s.stage, "role": s.role, "model": s.model, "gb": s.gb, "job_index": ji})

    mem_avail = sched._mem_available_gb()
    budget = max(1.0, mem_avail - sched.SAFETY_GB - sched.OVERHEAD_GB)
    decisions_raw = plan_resident_set(flat_for_planner, budget_gb=budget)
    decisions = [{"step": {"stage": d.step.stage, "role": d.step.role, "model": d.step.model, "gb": d.step.gb},
                  "load": d.load, "keep": d.keep, "evict": d.evict, "resident_after": d.resident_after}
                 for d in decisions_raw]

    naive = naive_load_count(flat_for_planner)
    planned = planned_load_count(decisions_raw)
    est_saved_seconds = max(0, (naive - planned)) * 180
    return {"budget_gb": round(budget, 1), "mem_available_gb": mem_avail, "sequence": sequence, "decisions": decisions,
            "savings": {"naive_loads": naive, "planned_loads": planned, "saved_loads": max(0, naive - planned), "est_saved_seconds": est_saved_seconds}}

@router.delete("/asset/{filename}")
async def asset_delete(filename: str):
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    path = os.path.join(EXP_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    try:
        os.remove(path)
        sidecar = os.path.join(EXP_DIR, filename + ".json")
        if os.path.isfile(sidecar): os.remove(sidecar)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "filename": filename}
