import os
import json
import time
import asyncio
import logging
import subprocess
from typing import List
from . import config as cfg
from .stats import (
    get_storage,
    get_outputs_disk,
    get_ram_estimate,
    get_sys_stats,
    get_output_counts
)
from .paths import EXP_DIR
from .llm import _LLM_LOCK, lmstudio_call
from .ws_manager import clients
from . import scheduler as sched

_log = logging.getLogger(__name__)
# Throttle: the broadcast loop ticks every 2s, so a persistent fault would spam
# the log. Only emit when the error signature changes (and re-emit on change).
_last_bcast_err = {"sig": None}


def _log_broadcast_error(e):
    sig = repr(e)
    if sig != _last_bcast_err["sig"]:
        _last_bcast_err["sig"] = sig
        _log.warning("broadcast tick error: %s", sig, exc_info=True)


# Idle ComfyUI VRAM reclaim: tracks the last time the slopfinity pipeline was
# active so the tick loop can POST ComfyUI /free once after a quiet period.
# `freed` latches so we don't hammer /free every 2s while idle.
_comfy_idle = {"last_active": None, "freed": False}


def _idle_free_due(active, last_active, now, idle_min, already_freed):
    """Pure: should ComfyUI be freed this tick? True only when the pipeline is
    inactive, the feature is enabled (idle_min > 0), we haven't already freed
    this idle period, and the idle threshold has been crossed."""
    if active or already_freed or idle_min <= 0 or last_active is None:
        return False
    return (now - last_active) >= idle_min * 60


_RECENT_EVENTS_MAX = 20
_recent_events = []

def _fleet_is_running() -> bool:
    """Return True if at least one run_fleet.py process is alive.
    Uses the same /proc scan as runner.py so it works without psutil."""
    try:
        if os.path.isdir("/proc"):
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry.name}/cmdline", "rb") as f:
                        raw = f.read()
                    if not raw:
                        continue
                    args = [a.decode("utf-8", errors="replace") for a in raw.split(b"\x00") if a]
                    if any(os.path.basename(a) == "run_fleet.py" for a in args):
                        return True
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
        else:
            out = subprocess.run(["pgrep", "-f", "run_fleet.py"],
                                 capture_output=True, text=True, timeout=2).stdout
            return bool(out.strip())
    except Exception:
        pass
    return False

def _reset_state_to_idle() -> None:
    """Write an Idle sentinel to state.json to self-heal a stale non-Idle mode."""
    try:
        state_path = os.path.join(EXP_DIR, "state.json")
        idle = {"mode": "Idle", "step": None, "video_index": None,
                "total_videos": 0, "chain_index": None, "total_chains": 0,
                "current_prompt": ""}
        with open(state_path, "w") as f:
            import json as _json
            _json.dump(idle, f)
    except Exception:
        pass



async def broadcast():
    try:
        known = set(os.listdir(EXP_DIR))
    except Exception:
        known = set()
        
    # Stage/job timers — persisted across slopfinity restarts
    _stage_track = {"step": None, "since": time.time()}
    _job_track = {"video_index": None, "since": time.time()}
    _last_completed = None
    _job_stage_actuals = {}
    _prev_state = None
    
    while True:
        try:
            state = cfg.get_state()
            now_ts = time.time()
            cur_step = state.get("step")
            cur_v = state.get("video_index")
            
            if cur_step != _stage_track["step"]:
                outgoing = _stage_track["step"]
                if outgoing and cur_v:
                    _job_stage_actuals.setdefault(cur_v, {})[outgoing] = {
                        "duration_s": now_ts - _stage_track["since"],
                        "ended_ts": now_ts,
                    }
                _stage_track["step"] = cur_step
                _stage_track["since"] = now_ts
                
            if cur_v != _job_track["video_index"]:
                if _prev_state and _prev_state.get("video_index"):
                    _last_completed = {
                        "video_index": _prev_state.get("video_index"),
                        "prompt": _prev_state.get("current_prompt"),
                        "completed_ts": now_ts,
                        "started_ts": _job_track["since"],
                    }
                _job_stage_actuals = {cur_v: _job_stage_actuals.get(cur_v, {})} if cur_v else {}
                _job_track["video_index"] = cur_v
                _job_track["since"] = now_ts
                
            state["stage_started_ts"] = _stage_track["since"]
            state["job_started_ts"] = _job_track["since"]
            state["last_completed"] = _last_completed
            state["stage_actuals"] = _job_stage_actuals.get(cur_v, {}) if cur_v else {}
            _prev_state = state
            
            stats = get_sys_stats()
            
            # Record GPU usage for the scheduler's guard
            from . import scheduler
            _sched_conf = scheduler._load_scheduler_config()
            _max_samples = int(_sched_conf.get("pause_idle_samples", 5))
            gpu = scheduler.get_gpu()
            gpu.record_gpu_usage(stats.get("gpu", 0), max_samples=_max_samples)
            await scheduler.notify_gpu_usage()

            queue = cfg.get_queue()
            cutoff = time.time() - 48 * 3600

            def _is_stale(x):
                return x.get("status") == "cancelled" and (x.get("cancelled_ts") or 0) < cutoff

            # This tick runs frequently; only take the cross-process lock + write
            # when there's actually something to prune (the unlocked read above is
            # just a cheap pre-check — the mutator re-filters fresh under the lock,
            # so a concurrent writer can't be clobbered).
            if any(_is_stale(x) for x in queue):
                queue = await asyncio.to_thread(
                    cfg.mutate_queue, lambda q: [x for x in q if not _is_stale(x)]
                )

            # ── Idle ComfyUI VRAM reclaim ────────────────────────────────────
            # ComfyUI keeps model weights resident between renders, and neither
            # the live run_fleet path nor an idle dashboard ever unloads them —
            # so an idle box can sit on tens of GB of GPU memory indefinitely
            # (a real problem when other AI workloads share the machine). When
            # the slopfinity pipeline has been idle (no active job, no
            # pending/working queue items, no in-flight dashboard GPU stage) for
            # `comfy_idle_free_minutes`, POST ComfyUI /free once to reclaim it;
            # ComfyUI transparently reloads on the next prompt. Scoped to
            # ComfyUI only — never touches the user's other GPU processes.
            # Default 10 min; set to 0 to disable.
            try:
                _now = time.time()
                _active = (
                    (state.get("mode") not in ("Idle", "Completed", "Terminated"))
                    or any(x.get("status") in (None, "pending", "working") for x in queue)
                    or bool(gpu.in_flight)
                )
                if _comfy_idle["last_active"] is None:
                    _comfy_idle["last_active"] = _now
                _idle_min = float(_sched_conf.get("comfy_idle_free_minutes", 10) or 0)
                if _active:
                    _comfy_idle["last_active"] = _now
                    _comfy_idle["freed"] = False
                elif _idle_free_due(_active, _comfy_idle["last_active"], _now,
                                    _idle_min, _comfy_idle["freed"]):
                    res = await scheduler.free_between()
                    _comfy_idle["freed"] = True
                    _log.info("pipeline idle %.0f min → freed ComfyUI VRAM: %s",
                              _idle_min, res)
            except Exception as e:
                _log_broadcast_error(e)

            config = cfg.load_config()
            storage = get_storage()
            ram = get_ram_estimate(
                config.get("base_model"),
                config.get("video_model"),
                config.get("audio_model"),
                config.get("upscale_model"),
                config.get("tts_model"),
            )
            outputs = get_output_counts(EXP_DIR)
            outputs_disk = get_outputs_disk(EXP_DIR)
            safe_config = cfg.redact(config)
            
            drained: List[dict] = []
            while True:
                try:
                    ev = sched.SchedulerEvents.get_nowait()
                except asyncio.QueueEmpty:
                    break
                drained.append(ev)
            if drained:
                _recent_events.extend(drained)
                del _recent_events[:-_RECENT_EVENTS_MAX]
                
            msg = {
                "type": "state",
                "state": state,
                "stats": stats,
                "queue": queue,
                "storage": storage,
                "outputs_disk": outputs_disk,
                "ram": ram,
                "config": safe_config,
                "outputs": outputs,
                "scheduler": {
                    "paused": sched.is_paused(),
                    "events": list(_recent_events[-5:]),
                },
                "events": drained,
            }
            
            for c in list(clients):
                try:
                    await c.send_json(msg)
                except Exception:
                    pass
                    
            try:
                if state.get('mode') != 'Idle' and state.get('step'):
                    # Dead-man's check: if the state says we're rendering but
                    # no run_fleet.py process is actually alive, self-heal by
                    # resetting state to Idle and clearing the animation.
                    fleet_alive = _fleet_is_running()
                    if not fleet_alive:
                        _reset_state_to_idle()
                        # Broadcast a clear so clients kill the animation immediately
                        _clear_msg = {"type": "render_heartbeat_clear"}
                        for c in list(clients):
                            try:
                                await c.send_json(_clear_msg)
                            except Exception:
                                pass
                    else:
                        _step_text_map = {
                            'Concept': 'rewriting prompt',
                            'Base Image': 'rendering image',
                            'Video Chains': 'rendering video',
                            'Audio': 'composing music',
                            'TTS': 'recording voiceover',
                            'Post Process': 'upscaling',
                            'Final Merge': 'merging final',
                        }
                        _hb_text = _step_text_map.get(state['step'], 'working')
                        
                        # Handle pausing/paused overrides for the header animation
                        if sched.is_paused():
                            # If anything is holding a GPU reservation, we are still "pausing"
                            # the pipeline (waiting for current stage to finish).
                            _gpu = sched.get_gpu()
                            if _gpu.resident_gb > 0 or _gpu.in_flight:
                                _hb_text = 'pausing'
                            else:
                                _hb_text = 'paused'
                        
                        _hb_msg = {
                            "type": "render_heartbeat",
                            "text": _hb_text,
                            "expires_ts": time.time() + 15,
                        }
                        for c in list(clients):
                            try:
                                await c.send_json(_hb_msg)
                            except Exception:
                                pass
            except Exception: pass
            
            try:
                curr = set(os.listdir(EXP_DIR))
            except Exception: curr = known
            new = curr - known
            for f in new:
                if f.endswith('.mp4') or f.endswith('.png'):
                    for c in list(clients):
                        try:
                            await c.send_json({"type": "new_file", "file": f})
                        except Exception: pass
            known = curr
        except Exception as e:
            _log_broadcast_error(e)
        await asyncio.sleep(2)

async def chaos_rotator():
    last_seen_index = None
    while True:
        try:
            config = cfg.load_config()
            if not config.get("chaos_mode"):
                last_seen_index = None
                await asyncio.sleep(15)
                continue
            state = cfg.get_state()
            cur_idx = state.get("video_index") if isinstance(state, dict) else None
            if last_seen_index is None:
                last_seen_index = cur_idx
                await asyncio.sleep(10)
                continue
            if cur_idx is None or cur_idx == last_seen_index:
                await asyncio.sleep(10)
                continue
            last_seen_index = cur_idx
            current_subjects = config.get("infinity_themes") or []
            sample = ", ".join(current_subjects[:8])
            tmpl = cfg.get_chaos_suggest_system_prompt(config)
            try:
                sys_p = tmpl.format(subjects_csv=sample)
            except Exception:
                sys_p = tmpl
            async with _LLM_LOCK:
                raw = await asyncio.to_thread(lmstudio_call, sys_p, "Give me 8 tangentially-related subject ideas.")
            arr = []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list): arr = parsed
            except Exception:
                a, b = raw.find('['), raw.rfind(']')
                if a != -1 and b > a:
                    try: arr = json.loads(raw[a:b + 1])
                    except Exception: arr = []
            arr = [str(x).strip() for x in arr if str(x).strip()][:8]
            if arr:
                # Locked RMW so run_fleet's infinity-index increment can't
                # revert this fresh theme list (and vice-versa).
                def _set_themes(c):
                    c["infinity_themes"] = arr
                    c["infinity_index"] = 0
                    return c
                config = await asyncio.to_thread(cfg.mutate_config, _set_themes)
            await asyncio.sleep(5)
        except Exception:
            await asyncio.sleep(30)
