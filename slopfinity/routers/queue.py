import os
import json
import uuid
import time
from typing import Optional, List

from fastapi import APIRouter, Body, Form
from fastapi.responses import JSONResponse

from slopfinity.paths import EXP_DIR
import slopfinity.config as cfg
import slopfinity.scheduler as sched


router = APIRouter()

@router.post("/inject")
async def inject(
    prompt: str = Form(...),
    priority: str = Form(...),
    stage_prompts: str = Form(default=""),
    terminate: str = Form(default=""),
    concurrent: str = Form(default=""),
    infinity: str = Form(default=""),
    when_idle: str = Form(default=""),
    chaos: str = Form(default=""),
    image_only: str = Form(default=""),
    fast_track: str = Form(default=""),
    seed_images: str = Form(default=""),
    seeds_mode: str = Form(default=""),
):
    # Disk-low guard — bail early when the outputs partition is below the
    # configured threshold so the queue doesn't pile up against a wall.
    # User can lift the guard in Settings → General if they really want
    # to push past it. terminate=1 still works (it cancels rather than
    # creates work).
    if not terminate:
        ok, reason = _check_disk_guard()
        if not ok:
            return JSONResponse(
                {"status": "blocked", "reason": f"disk-low guard: {reason}",
                 "hint": "raise the threshold in Settings → General → Disk guard"},
                status_code=409,
            )
    q = cfg.get_queue()
    if terminate:
        # Mark every pending and in-flight item cancelled (so the user
        # can see what got killed) and write a flag the fleet runner
        # watches for. The `working` sentinel is included so the active
        # item's infinity loop also gets cleared.
        now_ts = time.time()
        for item in q:
            if item.get("status") in (None, "pending", "working"):
                item["status"] = "cancelled"
                item["cancelled_ts"] = now_ts
                item["infinity"] = False
        try:
            with open(os.path.join(EXP_DIR, "cancel.flag"), "w") as f:
                f.write(str(now_ts))
        except Exception:
            pass
    task = {
        "prompt": prompt,
        "priority": priority,
        "status": "pending",
        "ts": time.time(),
        "concurrent": bool(concurrent),
        "infinity": bool(infinity),
        "when_idle": bool(when_idle),
        "chaos": bool(chaos),
        "image_only": bool(image_only),
        # Fast Track — orchestrator overrides chains/frames/tier and
        # skips audio/tts for THIS iter only when set. Use the dashboard's
        # 🏃 button (Subjects card) to flip this on per-injection.
        "fast_track": bool(fast_track),
    }
    if stage_prompts:
        try:
            task["stage_prompts"] = json.loads(stage_prompts)
        except Exception:
            task["stage_prompts_raw"] = stage_prompts

    # Seed-image staging (user uploads via /upload, then picks via the
    # Subjects-card seed picker). seed_images is a JSON-encoded list of
    # filenames living in EXP_DIR; seeds_mode picks consumption strategy:
    #   per-task   → fan out to N tasks, one seed each; each iter copies
    #                the seed to comfy-input as the chain-0 base image.
    #   per-chain  → keep one task with all seeds; run_fleet uses LTX FLF2V
    #                to span seed[i] → seed[i+1] per chain (N-1 chains).
    seeds = []
    if seed_images:
        try:
            raw = json.loads(seed_images)
            if isinstance(raw, list):
                # Sanitize: keep only basename, must start with seed_, must exist.
                for s in raw:
                    if not isinstance(s, str):
                        continue
                    name = os.path.basename(s)
                    if not name.startswith("seed_"):
                        continue
                    if not os.path.exists(os.path.join(EXP_DIR, name)):
                        continue
                    seeds.append(name)
        except Exception:
            seeds = []
    mode = (seeds_mode or "").strip().lower()
    if mode not in ("per-task", "per-chain"):
        mode = "per-task"

    tasks_to_queue: list = []
    if seeds and mode == "per-task":
        # Fan out: one task per seed, each carrying a single seed_image.
        # Each spawned task gets a unique ts so the queue UI shows them as
        # distinct rows. The parent prompt + flags propagate verbatim.
        for idx, s in enumerate(seeds):
            t = dict(task)
            t["ts"] = task["ts"] + idx * 1e-6  # nudge so timestamps stay sortable + unique
            t["seed_image"] = s
            t["seeds_mode"] = "per-task"
            tasks_to_queue.append(t)
    elif seeds and mode == "per-chain":
        task["seed_images"] = seeds
        task["seeds_mode"] = "per-chain"
        tasks_to_queue.append(task)
    else:
        tasks_to_queue.append(task)

    pending = [x for x in q if x.get("status") in (None, "pending")]
    working = [x for x in q if x.get("status") == "working"]
    done = [x for x in q if x.get("status") == "done"]
    cancelled = [x for x in q if x.get("status") == "cancelled"]
    # `now` and `next` both front-insert so the task runs immediately after
    # the currently-active job. Terminate is a separate flag (handled above)
    # which cancels the active job; pairing terminate + next/now means
    # "kill what's running and start this in its place".
    if priority in ("now", "next"):
        # Reverse so first-fanned task ends up at the front.
        for t in reversed(tasks_to_queue):
            pending.insert(0, t)
    else:
        for t in tasks_to_queue:
            pending.append(t)
    # Order on disk: working (active job sentinel) → pending (queued work) →
    # done (history) → cancelled. Newly-injected work always sits BEFORE
    # done records so the fleet's pop-from-front consumes pending items first.
    cfg.save_queue(working + pending + done + cancelled)
    return {"status": "ok"}

@router.post("/queue/pause")
async def queue_pause():
    """Pause new iter starts in run_fleet. Writes pause.flag in EXP_DIR;
    the orchestrator polls the flag and skips its iter loop body while
    it exists. The currently-running iter (if any) finishes naturally —
    pause is a SOFT stop, not a kill.

    Resume via POST /queue/resume."""
    try:
        with open(os.path.join(EXP_DIR, "pause.flag"), "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "paused": True}

@router.post("/queue/resume")
async def queue_resume():
    """Remove pause.flag — fleet returns to its iter loop on next poll."""
    flag = os.path.join(EXP_DIR, "pause.flag")
    try:
        if os.path.exists(flag):
            os.remove(flag)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "paused": False}

@router.get("/queue/pause-state")
async def queue_pause_state():
    """Quick poll for the dashboard. Returns {paused: bool}."""
    return {"paused": os.path.exists(os.path.join(EXP_DIR, "pause.flag"))}

@router.post("/cancel-all")
async def cancel_all():
    """Mark every pending or in-flight queue item as cancelled and
    signal the fleet runner.

    Cancelling the in-flight (`working`) sentinel also disables its
    requeue — the runner re-reads the working record at requeue time.
    """
    q = cfg.get_queue()
    now_ts = time.time()
    n = 0
    for item in q:
        if item.get("status") in (None, "pending", "working"):
            item["status"] = "cancelled"
            item["cancelled_ts"] = now_ts
            item["infinity"] = False
            n += 1
    try:
        with open(os.path.join(EXP_DIR, "cancel.flag"), "w") as f:
            f.write(str(now_ts))
    except Exception:
        pass
    cfg.save_queue(q)
    return {"status": "ok", "cancelled": n}

@router.post("/queue/cancel")
async def queue_cancel(data: dict = Body(...)):
    """Cancel a single queue item by ts. If it's the active job (matched by
    `current` flag in the future, or just the first pending item today), also
    write a cancel.flag so the fleet runner aborts gracefully."""
    target_ts = data.get("ts")
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    q = cfg.get_queue()
    found = False
    is_first_pending = True
    for item in q:
        # Match pending OR the in-flight `working` sentinel — cancelling
        # a working item flips its requeue off via the same
        # status=cancelled marker the fleet runner checks at requeue time.
        if item.get("ts") == target_ts and item.get("status") in (None, "pending", "working"):
            was_working = item.get("status") == "working"
            item["status"] = "cancelled"
            item["cancelled_ts"] = time.time()
            # Strip infinity so it doesn't re-loop after cancellation.
            item["infinity"] = False
            if is_first_pending or was_working:
                try:
                    with open(os.path.join(EXP_DIR, "cancel.flag"), "w") as f:
                        f.write(str(time.time()))
                except Exception:
                    pass
            found = True
            break
        if item.get("status") in (None, "pending"):
            is_first_pending = False
    if not found:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    cfg.save_queue(q)
    return {"ok": True}

@router.post("/queue/edit")
async def queue_edit(data: dict = Body(...)):
    """Replace the prompt text of a pending or in-flight queue item by ts.

    Editing a `working` item updates the seed_prompt the fleet runner
    will use for the NEXT cycle (the in-flight cycle uses the prompt
    captured at pop-time and isn't interrupted).
    """
    target_ts = data.get("ts")
    new_prompt = (data.get("prompt") or "").strip()
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    if not new_prompt:
        return JSONResponse({"ok": False, "error": "empty prompt"}, status_code=400)
    q = cfg.get_queue()
    found = False
    for item in q:
        if item.get("ts") == target_ts and item.get("status") in (None, "pending", "working"):
            item["prompt"] = new_prompt
            item["seed_prompt"] = new_prompt
            found = True
            break
    if not found:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    cfg.save_queue(q)
    return {"ok": True}

@router.post("/queue/toggle-infinity")
async def queue_toggle_infinity(data: dict = Body(...)):
    """Flip the `infinity` flag on a queued or in-flight item by ts.

    Also matches `working` rows — the fleet runner stamps a working
    sentinel for the in-flight item, and toggling that sentinel lets the
    user disable the requeue loop mid-flight (the runner re-reads the
    record at requeue time and skips re-appending if `infinity` is now
    False).
    """
    target_ts = data.get("ts")
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    q = cfg.get_queue()
    new_val = None
    for item in q:
        if item.get("ts") == target_ts and item.get("status") in (None, "pending", "working"):
            item["infinity"] = not item.get("infinity", False)
            new_val = item["infinity"]
            break
    if new_val is None:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    cfg.save_queue(q)
    return {"ok": True, "infinity": new_val}

@router.post("/queue/toggle-polymorphic")
async def queue_toggle_polymorphic(data: dict = Body(...)):
    """Flip the `chaos` (polymorphic) flag on a queued or in-flight item by ts.

    Mirrors the new value into both `chaos` and `polymorphic` so the
    fleet runner — which reads either field — picks up the change
    consistently at requeue time.
    """
    target_ts = data.get("ts")
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    q = cfg.get_queue()
    new_val = None
    for item in q:
        if item.get("ts") == target_ts and item.get("status") in (None, "pending", "working"):
            item["chaos"] = not item.get("chaos", False)
            item["polymorphic"] = item["chaos"]
            new_val = item["chaos"]
            break
    if new_val is None:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    cfg.save_queue(q)
    return {"ok": True, "chaos": new_val}

@router.get("/queue/paginated")
async def queue_paginated(offset: int = 0, limit: int = 25, filter: str = "all"):
    """Paginated, filtered view of the persisted queue. Newest first.

    Used by the "View all" drawer so the client doesn't have to render
    1000+ done items in one shot. Filters:
      - all: every item
      - pending / done / cancelled: status match
      - failed: status==done AND succeeded is False
    """
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = max(1, min(500, int(limit)))
    except (TypeError, ValueError):
        limit = 25
    q = cfg.get_queue() or []
    if filter == "failed":
        q = [it for it in q if it.get("status") == "done" and it.get("succeeded") is False]
    elif filter in ("done", "pending", "cancelled"):
        q = [it for it in q if it.get("status") == filter]
    # Newest first — completed_ts for done items, ts for everything else.
    q = sorted(q, key=lambda x: x.get("completed_ts") or x.get("ts") or 0, reverse=True)
    total = len(q)
    page = q[offset:offset + limit]
    return {
        "items": page,
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": offset + limit < total,
        "filter": filter,
    }

@router.post("/queue/requeue")
async def queue_requeue(data: dict = Body(...)):
    """Re-pend a queue item identified by ts.

    Accepts BOTH cancelled items AND done-but-failed items — the
    per-row ↻ Re-queue button is a generic "try this again" affordance.
    Cancelled items get flipped back in place. Failed items get a fresh
    pending entry appended (mirroring /queue/requeue-failed) and the
    original failed record is dropped so the queue doesn't grow stale
    duplicates over time.
    """
    target_ts = data.get("ts")
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    q = cfg.get_queue()
    new_q = []
    requeued = False
    base_ts = time.time()
    for item in q:
        if item.get("ts") == target_ts:
            if item.get("status") == "cancelled":
                item["status"] = "pending"
                item.pop("cancelled_ts", None)
                new_q.append(item)
                requeued = True
                continue
            if item.get("status") == "done" and item.get("succeeded") is False:
                # Drop the failed record; append a fresh pending entry.
                fresh = item.copy()
                fresh.update({
                    "status": "pending",
                    "ts": base_ts,
                    "requeued_from_ts": item.get("ts"),
                })
                # Remove fields that represent the RESULT of the failed run.
                # We keep 'times' as requested ("carry over times").
                fresh.pop("completed_ts", None)
                fresh.pop("succeeded", None)
                fresh.pop("error", None)
                fresh.pop("asset_paths", None)
                fresh.pop("logs", None)
                new_q.append(fresh)
                requeued = True
                continue
        new_q.append(item)
    if not requeued:
        return JSONResponse({"ok": False, "error": "not requeueable (must be cancelled or done-failed)"}, status_code=404)
    cfg.save_queue(new_q)
    return {"ok": True}

@router.post("/queue/clear-failed")
async def queue_clear_failed():
    """Drop all done-but-failed items from the queue history.

    Keeps pending, running, succeeded-done, and cancelled items intact.
    """
    q = cfg.get_queue()
    before = len(q)
    kept = [
        item for item in q
        if not (item.get("status") == "done" and item.get("succeeded") is False)
    ]
    removed = before - len(kept)
    if removed:
        cfg.save_queue(kept)
    return {"ok": True, "removed": removed}

@router.post("/queue/clear-completed")
async def queue_clear_completed():
    """Drop all successfully-completed items from the queue history.

    Mirror of /queue/clear-failed. Keeps pending, running, failed, and
    cancelled items intact — only succeeded-done entries are pruned.
    """
    q = cfg.get_queue()
    before = len(q)
    kept = [
        item for item in q
        if not (item.get("status") == "done" and item.get("succeeded") is not False)
    ]
    removed = before - len(kept)
    if removed:
        cfg.save_queue(kept)
    return {"ok": True, "removed": removed}

@router.post("/queue/requeue-failed")
async def queue_requeue_failed():
    """Re-add every done-but-failed item as a fresh pending entry; drop the
    failed records.

    The fresh entry preserves prompt + the per-item toggles + config_snapshot,
    and resets status/ts so the scheduler picks it up on the next sweep.
    """
    q = cfg.get_queue()
    requeued = 0
    new_q = []
    base_ts = time.time()
    for item in q:
        if item.get("status") == "done" and item.get("succeeded") is False:
            fresh = item.copy()
            fresh.update({
                "status": "pending",
                # Disambiguate ts within the same second so multiple
                # requeued items don't collide on the (ts) primary key.
                "ts": base_ts + (requeued * 1e-6),
                "requeued_from_ts": item.get("ts"),
            })
            fresh.pop("completed_ts", None)
            fresh.pop("succeeded", None)
            fresh.pop("error", None)
            fresh.pop("asset_paths", None)
            fresh.pop("logs", None)
            new_q.append(fresh)
            requeued += 1
            # original failed entry is dropped (not appended to new_q)
        else:
            new_q.append(item)
    if requeued:
        cfg.save_queue(new_q)
    return {"ok": True, "requeued": requeued}
