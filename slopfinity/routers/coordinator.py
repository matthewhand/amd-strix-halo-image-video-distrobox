import json
import subprocess
import os
import time
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
import slopfinity.config as cfg
import slopfinity.scheduler as sched
from slopfinity.paths import EXP_DIR


router = APIRouter()

@router.post("/pause")
async def pause_scheduler():
    """Clear the scheduler's `paused` event — new stages will wait."""
    await sched.pause()
    return {"paused": True}

@router.post("/resume")
async def resume_scheduler():
    """Set the scheduler's `paused` event — stages may proceed."""
    await sched.resume()
    return {"paused": False}

@router.post("/free")
async def free_endpoint():
    """Trigger ComfyUI /free + gc. Returns freed_gb when measurable."""
    result = await sched.free_between()
    return {"ok": result.get("ok", False), **result}

@router.post("/story/stitch")
async def story_stitch(data: dict = Body(...)):
    """Concatenate a list of FINAL_*.mp4 clips into one combined story
    video using ffmpeg's concat demuxer (stream copy, no re-encode).

    Body:
      filenames: ["FINAL_19_dragon.mp4", "FINAL_20_lighthouse.mp4", ...]
      output_name: optional — defaults to STORY_<ts>.mp4

    Returns {ok, output, error?}. Output lands in EXP_DIR alongside
    the source clips so it surfaces in the gallery via /assets.

    Used by the endless-story flow: the user accepts N suggestion chips
    over the course of a story, each becomes a queued iter, each iter
    produces a FINAL_*.mp4. Once they're all done the user clicks
    Stitch in the story-pane footer to concat them into the actual
    final story video. Stream-copy (no re-encode) is fast and lossless
    when the source clips share codec params (which they do —
    same model + settings produced them).
    """
    raw_files = data.get("filenames") or []
    if not isinstance(raw_files, list) or not raw_files:
        return JSONResponse({"ok": False, "error": "filenames must be a non-empty list"}, status_code=400)

    # Validate every file exists in EXP_DIR + is a FINAL_*.mp4 (basename
    # only — strip any path components a malicious client might supply).
    abs_paths = []
    for raw in raw_files:
        if not isinstance(raw, str):
            return JSONResponse({"ok": False, "error": f"filename not a string: {raw!r}"}, status_code=400)
        name = os.path.basename(raw)
        if not (name.startswith("FINAL_") and name.lower().endswith(".mp4")):
            return JSONResponse({"ok": False, "error": f"not a FINAL_*.mp4: {name}"}, status_code=400)
        p = os.path.join(EXP_DIR, name)
        if not os.path.isfile(p):
            return JSONResponse({"ok": False, "error": f"missing file: {name}"}, status_code=404)
        abs_paths.append(p)

    output_name = (data.get("output_name") or "").strip()
    if not output_name:
        output_name = f"STORY_{int(time.time())}.mp4"
    output_name = os.path.basename(output_name)
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"
    output_path = os.path.join(EXP_DIR, output_name)
    if os.path.exists(output_path):
        return JSONResponse({"ok": False, "error": f"output already exists: {output_name}"}, status_code=409)

    # ffmpeg concat demuxer needs a temp list file with `file '<path>'`
    # entries — one per clip in order. We write it next to the output so
    # debugging is easy if the concat fails.
    list_path = output_path + ".concat.txt"
    try:
        with open(list_path, "w") as f:
            for p in abs_paths:
                # Single-quote the path with ffmpeg's escape rule (replace
                # ' with '\'') so spaces / quotes in filenames don't break
                # the demuxer's parser.
                escaped = p.replace("'", r"'\''")
                f.write(f"file '{escaped}'\n")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-y",                    # overwrite (we already 409'd above; safety)
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",            # stream copy — fast + lossless
            output_path,
        ]
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "ffmpeg failed"
            return JSONResponse(
                {"ok": False, "error": f"ffmpeg: {err}", "cmd": " ".join(cmd)},
                status_code=500,
            )
    finally:
        # Clean up the concat list — keep the output, drop the helper.
        try:
            os.remove(list_path)
        except Exception:
            pass

    return {
        "ok": True,
        "output": output_name,
        "url": f"/files/{output_name}",
        "n_inputs": len(abs_paths),
    }

@router.post("/emergency_free")
async def emergency_free_endpoint():
    """ComfyUI /free + pkill stray model launchers."""
    result = await sched.emergency_free()
    return {"ok": True, **result}

@router.get("/scheduler/status")
async def scheduler_status():
    """Snapshot of the scheduler: pause state + queue depth."""
    return {
        "paused": sched.is_paused(),
        "pending_events": sched.SchedulerEvents.qsize(),
    }

@router.post("/coordinator/start")
async def coordinator_start():
    """Spawn the Phase-4 Coordinator (concurrent StageWorker loops).

    Idempotent — calling while already running returns the current status.
    The legacy fleet runner remains independent; running both at once is
    not recommended (they would race on the same queue).
    """
    if _coordinator is None:
        return JSONResponse(
            {"ok": False, "error": "coordinator module unavailable",
             "detail": _coord_imp_err_repr},
            status_code=500,
        )
    co = _coordinator.get_coordinator()
    try:
        await co.start()
    except RuntimeError as e:
        # Phases 1-3 may not be merged yet — surface clearly.
        return JSONResponse(
            {"ok": False, "error": str(e), **co.status()},
            status_code=503,
        )
    return {"ok": True, **co.status()}

@router.post("/coordinator/stop")
async def coordinator_stop():
    """Cancel the Coordinator's worker tasks and clear the running flag."""
    if _coordinator is None:
        return JSONResponse(
            {"ok": False, "error": "coordinator module unavailable",
             "detail": _coord_imp_err_repr},
            status_code=500,
        )
    co = _coordinator.get_coordinator()
    await co.stop()
    return {"ok": True, **co.status()}

@router.get("/coordinator/status")
async def coordinator_status():
    """Snapshot of the Coordinator: running flag + worker list + import health."""
    if _coordinator is None:
        return {"ok": False, "error": "coordinator module unavailable",
                "detail": _coord_imp_err_repr,
                "running": False, "workers": []}
    co = _coordinator.get_coordinator()
    return {"ok": True, "persisted_running": _coordinator.is_running(), **co.status()}
