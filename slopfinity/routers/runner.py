import os
import json
import asyncio
import subprocess
from fastapi import APIRouter, Form, Request, UploadFile, File, Body
from fastapi.responses import JSONResponse, HTMLResponse
from typing import List
from slopfinity.paths import EXP_DIR, TTS_OUT_DIR
import slopfinity.config as cfg
from slopfinity.stats import get_sys_stats, get_outputs_disk, get_ram_estimate
from fastapi.templating import Jinja2Templates
from slopfinity.paths import TEMPLATES_DIR
templates = Jinja2Templates(directory=TEMPLATES_DIR)
import os
TTS_WORKER_URL = os.environ.get("TTS_WORKER_URL", "http://localhost:8010/tts")
from slopfinity.workers import ffmpeg_mux as _ffmpeg_mux
import urllib.request
import urllib.error
import slopfinity.scheduler as sched


router = APIRouter()

@router.get("/pipeline/slopped")
async def pipeline_slopped(role: str):
    """List existing assets in EXP_DIR matching the given role's extensions.

    Used by the pipeline popup to populate the small `<select>` shown beneath
    a model dropdown when the user picks `Slopped`. Returns up to 200 entries,
    newest first.
    """
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
    # Cap to 60 most-recent: the image role renders these as thumbnails and
    # we want the modal to stay snappy. Audio/tts also share this cap.
    return {"role": role, "files": [n for n, _ in files[:60]]}

@router.get("/disk/guard")
async def disk_guard_endpoint():
    """Live disk-guard check. Used by the dashboard to pre-warn the user
    before they click Queue Slop instead of failing the /inject call.
    Returns {ok, reason, free_pct, free_gb, threshold_pct, threshold_gb}."""
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
    """Inspect run_fleet.py orchestrator state without sending signals.

    Returns the pids currently matching, plus per-pid age + cmdline so
    callers can verify _whether_ a runner is alive before deciding to
    terminate. Used by e2e tests + the dashboard's pause-button retry
    logic to distinguish "runner stuck" from "runner not running".
    """
    pids = _find_pids_by_cmdline("run_fleet.py")
    info = []
    now = time.time()
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat = f.read().split()
            # field 22 = starttime in clock ticks since boot
            starttime_ticks = int(stat[21])
            clk_tck = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", 100))
            with open("/proc/uptime", "r") as f:
                uptime_s = float(f.read().split()[0])
            age_s = uptime_s - (starttime_ticks / clk_tck)
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            info.append({"pid": pid, "age_s": round(age_s, 1), "cmdline": cmd[:200]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            info.append({"pid": pid, "age_s": None, "cmdline": "(unreadable)"})
    return {"ok": True, "running": len(pids) > 0, "pids": info, "wall_time": now}

@router.post("/runner/terminate")
async def runner_terminate():
    """Stop the run_fleet.py orchestrator running on the host.

    TWO-LAYER strategy:
      1. Write `terminate.flag` to EXP_DIR. run_fleet.py checks this at
         the top of every iter and exits cleanly. Works regardless of
         host/container PID namespace + capability boundaries (the
         dashboard sometimes runs in a container that can't SIGTERM
         host processes due to default Docker security profiles, even
         with pid:host).
      2. Best-effort SIGTERM/SIGKILL. If the dashboard CAN see + signal
         the runner, we hard-stop it for hung-LLM-HTTP scenarios past
         any in-loop flag check. PermissionError is swallowed — the
         flag is the canonical mechanism, the signal is bonus.

    Returns the flag path + the pids touched (signal + escalation).
    Both succeed independently."""
    import signal
    flag_path = os.path.join(EXP_DIR, "terminate.flag")
    flag_written = False
    try:
        with open(flag_path, "w") as f:
            f.write(str(time.time()))
        flag_written = True
    except Exception as e:
        flag_err = repr(e)
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
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    return {
        "ok": True,
        "flag_written": flag_written,
        "flag_path": flag_path,
        "matched_pids": pids,
        "killed": killed,
        "escalated_to_sigkill": escalated,
        "permission_denied_pids": perm_errs,
        "note": "flag is the canonical mechanism; runner exits at next iter top. "
                "Signals are best-effort — may fail with PermissionError under "
                "container security profiles but that does not invalidate the flag.",
    }

@router.post("/runner/terminate-clear")
async def runner_terminate_clear():
    """Remove terminate.flag so a fresh run_fleet.py launch can proceed.

    Without this, any newly-launched runner reads the stale flag and
    exits immediately. Called automatically by start-runner workflows
    (and manually if the user wants to clear without restarting)."""
    flag_path = os.path.join(EXP_DIR, "terminate.flag")
    existed = os.path.exists(flag_path)
    try:
        os.remove(flag_path)
    except FileNotFoundError:
        pass
    return {"ok": True, "existed": existed, "flag_path": flag_path}

@router.get("/seeds/list")
async def seeds_list():
    """Return uploaded seed images (filenames matching ``seed_*``) sorted
    by mtime desc. Powers the Subjects-card seed picker modal so users
    can stage one or more uploads as starting frames for the next inject.
    """
    items = []
    try:
        for f in os.listdir(EXP_DIR):
            if not f.startswith("seed_"):
                continue
            if not f.lower().endswith(_SEED_IMAGE_EXTS):
                continue
            try:
                mtime = os.path.getmtime(os.path.join(EXP_DIR, f))
            except OSError:
                continue
            items.append({"file": f, "mtime": mtime})
    except OSError:
        pass
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"items": items}

@router.post("/upload")
async def upload_seed_assets(files: list[UploadFile] = File(...)):
    """Accept user-uploaded image files and drop them into EXP_DIR
    so they surface in the slop gallery via the existing /assets path.

    Filename pattern: ``seed_{ts}_{slug}.{ext}`` — the ``seed_`` prefix
    distinguishes user uploads from generator output for any future
    consume-as-input pipeline branch.
    """
    saved = []
    skipped = []
    ts = int(time.time())
    for idx, uf in enumerate(files or []):
        original = (uf.filename or "upload").strip()
        ext = os.path.splitext(original)[1].lower()
        if ext not in _SEED_IMAGE_EXTS:
            skipped.append({"name": original, "reason": "non-image extension"})
            continue
        # slugify: keep alnum + dash, replace anything else with _
        stem = os.path.splitext(os.path.basename(original))[0] or "upload"
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:64].strip("_") or "upload"
        out_name = f"seed_{ts}_{idx:02d}_{slug}{ext}"
        out_path = os.path.join(EXP_DIR, out_name)
        size = 0
        too_big = False
        try:
            with open(out_path, "wb") as fh:
                while True:
                    chunk = await uf.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > _SEED_MAX_BYTES:
                        too_big = True
                        break
                    fh.write(chunk)
            if too_big:
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                skipped.append({"name": original, "reason": "exceeds 25MB cap"})
            else:
                saved.append(out_name)
        except OSError as exc:
            skipped.append({"name": original, "reason": f"write failed: {exc}"})
            try:
                os.remove(out_path)
            except OSError:
                pass
    return {"ok": True, "saved": saved, "skipped": skipped}

@router.get("/vae_grid")
async def vae_grid_check(file: str):
    """Return the VAE-grid detector's result for `file`. Reads the
    persisted ``<file>.grid.json`` sidecar when present; otherwise
    runs the FFT detector lazily and writes the sidecar so subsequent
    requests are instant. `file` is resolved relative to EXP_DIR and
    must not contain ``..``.
    """
    if not file or ".." in file or file.startswith("/"):
        return JSONResponse({"ok": False, "error": "bad_path"}, status_code=400)
    abs_path = os.path.join(EXP_DIR, file)
    if not os.path.isfile(abs_path):
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    from . import vae_grid as _vg
    cached = _vg.read_sidecar(abs_path)
    if cached:
        return {"ok": True, "cached": True, **cached}
    # Lazy compute. Run in a thread so a 50ms FFT doesn't stall the
    # event loop while the dashboard is busy with WS broadcasts.
    result = await asyncio.to_thread(_vg.detect_grid, abs_path)
    _vg.write_sidecar(abs_path, result)
    return {"ok": True, "cached": False, **result}

@router.post("/tts")
async def tts(data: dict = Body(...)):
    """Proxy to the Qwen3-TTS worker on :8010.

    Preserves the JS contract: response contains {ok, status, url, audio_path,
    voice}. On worker-unreachable, returns a clear error — NEVER falls back
    to a sine-wave stub.
    """
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "ryan"
    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)
    # Manual TTS preview — route through acquire_gpu with a TTS-shaped budget
    # so a mid-fleet click queues correctly and LM Studio gets suspended
    # (Qwen-TTS shares the GPU). safety_gb=4: the worker already lives in
    # its own process holding ~10 GB, this lock just gates concurrent demand.
    try:
        async with sched.acquire_gpu("TTS", "qwen-tts", safety_gb=4):
            result = await asyncio.to_thread(_call_tts_worker, text, voice)
    except urllib.error.URLError as e:
        return JSONResponse(
            {
                "ok": False,
                "status": "worker-unreachable",
                "error": "qwen-tts-service not running — enable profile qwen-tts "
                         f"(docker compose --profile qwen-tts up -d qwen-tts-service): {e}",
                "voice": voice,
            },
            status_code=503,
        )
    except Exception as e:
        return JSONResponse(
            {"ok": False, "status": "worker-error", "error": str(e), "voice": voice},
            status_code=502,
        )
    # Back-compat shape for slopfinity/static/app.js generateTts().
    url = result.get("url") or result.get("audio_path")
    return {
        "ok": bool(result.get("ok")),
        "status": result.get("status", "ok" if result.get("ok") else "error"),
        "url": url,
        "audio_path": url,
        "voice": result.get("voice", voice),
        "error": result.get("error"),
    }

@router.post("/mux")
async def mux(data: dict = Body(...)):
    """Mux audio onto video using ffmpeg_mux.

    Body: {video_path, audio_path, out_name, [loop_audio], [pad_to_video]}
    Paths are treated as relative to /workspace (EXP_DIR) if not absolute.
    """
    vrel = data.get("video_path") or ""
    arel = data.get("audio_path") or ""
    out_name = data.get("out_name") or f"muxed_{int(time.time() * 1000)}.mp4"
    if not vrel or not arel:
        return JSONResponse(
            {"ok": False, "error": "video_path and audio_path required"},
            status_code=400,
        )

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(EXP_DIR, p.lstrip("/"))

    video = _resolve(vrel)
    audio = _resolve(arel)
    out_path = os.path.join(EXP_DIR, out_name)

    try:
        ok = _ffmpeg_mux.mux(
            video,
            audio,
            out_path,
            loop_audio=bool(data.get("loop_audio")),
            pad_to_video=bool(data.get("pad_to_video", True)),
        )
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": f"missing input: {e}"}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    if not ok:
        return JSONResponse({"ok": False, "error": "ffmpeg mux failed"}, status_code=500)
    return {"ok": True, "url": f"/files/{out_name}"}

@router.get("/ram_estimate")
async def ram_estimate(base: str = "", video: str = "", audio: str = "", upscale: str = "", tts: str = ""):
    return get_ram_estimate(
        base or None,
        video or None,
        audio or None,
        upscale or None,
        tts or None,
    )

@router.get("/system/ram")
async def system_ram():
    """Live MemAvailable (GB) plus the scheduler's safety threshold.

    Used by the client-side RAM-tight warning modal to gate manual AI buttons
    (🎲 Suggest, /enhance, /enhance/distribute, /tts). When `tight=True` the
    UI prompts the user before firing the request; the user can still proceed.
    """
    available = sched._mem_available_gb()
    safety = float(sched.SAFETY_GB)
    return {
        "available_gb": available,
        "safety_gb": safety,
        "tight": available < safety,
    }

@router.get("/pipeline/plan")
async def pipeline_plan(lookahead: int = 2):
    """Compute the Belady-MIN resident-set plan for the active job + first
    `lookahead` queued jobs. Advisory only — the scheduler does not yet honour
    this plan (see docs/memory-stage-planner-design.md).

    Response shape:
      {
        budget_gb: float,
        mem_available_gb: float,
        sequence: [{stage, role, model, gb, job_index}, ...],
        decisions: [{step, load, keep, evict, resident_after}, ...],
        savings: {naive_loads, planned_loads, saved_loads, est_saved_seconds},
      }
    """
    from .memory_planner import (
        build_sequence_for_job,
        plan_resident_set,
        naive_load_count,
        planned_load_count,
    )

    config = cfg.load_config()
    queue = cfg.get_queue() or []

    # Active job uses the current config selections; queued items may override
    # base/video/audio/tts/upscale per item, falling back to config defaults.
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
            sequence.append({
                "stage":     s.stage,
                "role":      s.role,
                "model":     s.model,
                "gb":        s.gb,
                "job_index": ji,
            })

    # Budget: MEM_AVAILABLE - SAFETY - OVERHEAD. Floor at 1 GB so a totally
    # starved host still produces a (degraded) plan rather than crashing.
    mem_avail = sched._mem_available_gb()
    budget = max(1.0, mem_avail - sched.SAFETY_GB - sched.OVERHEAD_GB)

    decisions_raw = plan_resident_set(flat_for_planner, budget_gb=budget)
    decisions = [
        {
            "step":           {"stage": d.step.stage, "role": d.step.role,
                               "model": d.step.model, "gb": d.step.gb},
            "load":           d.load,
            "keep":           d.keep,
            "evict":          d.evict,
            "resident_after": d.resident_after,
        }
        for d in decisions_raw
    ]

    naive = naive_load_count(flat_for_planner)
    planned = planned_load_count(decisions_raw)
    # Rough cost per cold-load: ~90 s aiter JIT + ~90 s checkpoint load ≈ 180 s
    # for a freshly-loaded model. Used purely to translate "loads saved" into
    # a human-readable wall-clock figure for the UI.
    est_saved_seconds = max(0, (naive - planned)) * 180

    return {
        "budget_gb":         round(budget, 1),
        "mem_available_gb":  mem_avail,
        "lookahead":         int(lookahead),
        "queued_jobs_planned": len(jobs_to_plan) - 1,
        "sequence":          sequence,
        "decisions":         decisions,
        "savings": {
            "naive_loads":       naive,
            "planned_loads":     planned,
            "saved_loads":       max(0, naive - planned),
            "est_saved_seconds": est_saved_seconds,
        },
    }

@router.delete("/asset/{filename}")
async def asset_delete(filename: str):
    """Delete an asset file (and its sidecar JSON if present) from EXP_DIR.

    Filename safety mirrors /asset/ GET — leaf name only. Returns 404 if the
    file is gone (idempotent in spirit but explicit to surface UI bugs).
    """
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    path = os.path.join(EXP_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    try:
        os.remove(path)
        sidecar = os.path.join(EXP_DIR, filename + ".json")
        if os.path.isfile(sidecar):
            try:
                os.remove(sidecar)
            except Exception:
                pass
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "filename": filename}
