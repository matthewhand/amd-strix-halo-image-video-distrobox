import os
import json
import time
import asyncio
from fastapi import APIRouter, Request, Body
from fastapi.responses import JSONResponse
import slopfinity.config as cfg
from slopfinity.routers.assets import _list_outputs
from slopfinity.paths import EXP_DIR
from slopfinity.llm import _LLM_LOCK
from slopfinity.llm import lmstudio_chat_raw


router = APIRouter()

def _chat_tool_queue_clip(args: dict) -> dict:
    """Execute the queue_clip tool. Mirrors POST /inject's logic without
    going through HTTP — direct queue mutation."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    q = cfg.get_queue()
    task = {
        "prompt": prompt,
        "priority": "next",
        "status": "pending",
        "ts": time.time(),
        "chaos": True,
        "infinity": bool(args.get("infinity")),
        "fast_track": bool(args.get("fast_track")),
    }
    # Per-task overrides land in config_snapshot so run_fleet's snapshot
    # readers pick them up for this iter only. Falls through to global
    # config when keys are absent.
    snap = {}
    if args.get("chains"):
        try: snap["chains"] = max(1, min(30, int(args["chains"])))
        except (TypeError, ValueError): pass
    if args.get("frames"):
        try: snap["frames"] = max(9, min(241, int(args["frames"])))
        except (TypeError, ValueError): pass
    if args.get("tier") in ("low", "med", "high"):
        snap["tier"] = args["tier"]
    if snap:
        task["config_snapshot"] = snap
    pending = [x for x in q if x.get("status") in (None, "pending")]
    working = [x for x in q if x.get("status") == "working"]
    done = [x for x in q if x.get("status") == "done"]
    cancelled = [x for x in q if x.get("status") == "cancelled"]
    pending.insert(0, task)
    cfg.save_queue(working + pending + done + cancelled)
    return {"ok": True, "ts": task["ts"], "prompt": prompt, "overrides": snap}

def _chat_tool_list_queue(_args: dict) -> dict:
    q = cfg.get_queue()
    pending = [x for x in q if x.get("status") in (None, "pending")]
    working = [x for x in q if x.get("status") == "working"]
    done = [x for x in q if x.get("status") == "done"]
    cancelled = [x for x in q if x.get("status") == "cancelled"]
    next_5 = [{"ts": x.get("ts"), "prompt": (x.get("prompt") or "")[:80]} for x in pending[:5]]
    return {
        "pending": len(pending),
        "running": len(working),
        "done": len(done),
        "cancelled": len(cancelled),
        "next_5": next_5,
    }

def _chat_tool_get_status(_args: dict) -> dict:
    s = cfg.get_state()
    return {
        "mode": s.get("mode", "Idle"),
        "step": s.get("step", "Waiting"),
        "current_prompt": s.get("current_prompt", "") or "",
        "chain": s.get("chain_index", 0),
        "total_chains": s.get("total_chains", 0),
        "video": s.get("video", 0),
        "started_at": s.get("started_at", 0),
    }

def _chat_tool_cancel_item(args: dict) -> dict:
    ts = args.get("ts")
    try: ts_f = float(ts)
    except (TypeError, ValueError):
        return {"ok": False, "error": "ts must be a number"}
    q = cfg.get_queue()
    hit = None
    for item in q:
        if abs(float(item.get("ts") or 0) - ts_f) < 0.001:
            if item.get("status") in (None, "pending", "working"):
                item["status"] = "cancelled"
                item["cancelled_ts"] = time.time()
                hit = item
                break
    if not hit:
        return {"ok": False, "error": "no matching pending/running item"}
    cfg.save_queue(q)
    return {"ok": True, "cancelled_prompt": (hit.get("prompt") or "")[:80]}

def _chat_tool_recent_finals(args: dict) -> dict:
    n = max(1, min(20, int(args.get("n") or 5)))
    items = []
    try:
        for f in os.listdir(EXP_DIR):
            if not f.startswith("FINAL_") or not f.lower().endswith(".mp4"):
                continue
            try:
                mtime = os.path.getmtime(os.path.join(EXP_DIR, f))
            except OSError:
                continue
            # Look up the sidecar for the prompt if present.
            prompt = ""
            sidecar = os.path.join(EXP_DIR, f + ".json")
            if os.path.exists(sidecar):
                try:
                    with open(sidecar) as sh:
                        sd = json.load(sh)
                        prompt = (sd.get("prompt") or "")[:200]
                except Exception:
                    pass
            items.append({"file": f, "mtime": mtime, "prompt": prompt})
    except OSError:
        pass
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"items": items[:n]}

def _chat_tool_describe_config(_args: dict) -> dict:
    c = cfg.load_config()
    keys = ("base_model", "video_model", "audio_model", "tts_model",
            "size", "frames", "chains", "tier")
    return {k: c.get(k) for k in keys}


_CHAT_TOOLS_MANIFEST = [
    {
        "type": "function",
        "function": {
            "name": "queue_clip",
            "description": "Queue a single video clip with the given prompt. Optional knobs override the global pipeline config for that one task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What to render. Describe the visual subject in plain text."},
                    "chains": {"type": "integer", "description": "Number of chained video parts. Default uses the user's pipeline config (typically 10). Pass 1-3 for short smoke clips."},
                    "frames": {"type": "integer", "description": "Frames per chain. Default 49. Use 17 for very fast smoke clips."},
                    "tier": {"type": "string", "enum": ["low", "med", "high"], "description": "Quality tier — low=8 steps, med=20, high=50."},
                    "fast_track": {"type": "boolean", "description": "When true, applies tier=low + chains=2 + frames=17 + skips audio/tts. Targets ~3 min/clip on Strix Halo."},
                    "infinity": {"type": "boolean", "description": "Auto-requeue this prompt forever (until the user cancels)."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_queue",
            "description": "Return a summary of the queue: counts of pending / running / done / cancelled, plus the prompts of the next 5 pending items.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Return what the fleet runner is currently doing: mode, current step, current prompt, chain N of M, elapsed seconds.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_item",
            "description": "Cancel a single pending or running queue item by its timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts": {"type": "number", "description": "The queue item's ts (returned by list_queue)."},
                },
                "required": ["ts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_finals",
            "description": "List the most recently completed FINAL_*.mp4 clips with their prompts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many recent finals to return. Default 5, max 20."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_config",
            "description": "Return the active pipeline configuration: base_model, video_model, audio_model, tts_model, size, frames, chains, tier.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]














_CHAT_TOOL_HANDLERS = {
    "queue_clip": _chat_tool_queue_clip,
    "list_queue": _chat_tool_list_queue,
    "get_status": _chat_tool_get_status,
    "cancel_item": _chat_tool_cancel_item,
    "recent_finals": _chat_tool_recent_finals,
    "describe_config": _chat_tool_describe_config,
}

_CHAT_SYSTEM_PROMPT = (
    "You are the slopfinity assistant — a tool-using helper for a self-hosted "
    "AMD Strix Halo video generation fleet. The user is in front of a dashboard. "
    "When they want to render something, call queue_clip. When they ask about "
    "what's happening, call get_status or list_queue. When they ask about past "
    "outputs, call recent_finals. Be concise — answer in 1-3 short paragraphs. "
    "Don't ask for confirmation before calling tools; just do the thing and "
    "report back. After tool calls, summarize what happened in plain English."
)

_CHAT_MAX_TURNS = 6  # bounded tool-call recursion


@router.post("/chat")
async def chat_endpoint(payload: dict = Body(...)):
    """Multi-turn chat with tool-calling. Client sends the conversation
    history; server prepends the system prompt, calls the LLM with the
    tools manifest, executes any tool_calls server-side, loops until the
    model returns a content-only response (or hits the turn cap), and
    returns the full updated message history.
    """
    history = payload.get("messages") or []
    if not isinstance(history, list):
        return JSONResponse({"ok": False, "error": "messages must be a list"}, status_code=400)
    # Keep server-side history bounded too — the client should already trim.
    history = history[-50:]
    # User message has landed — kick off the thought bubble immediately
    # so the UI shows cogs even before we acquire the GPU lock (which
    # may block behind a busy queue).
    await _broadcast_chat_thinking("received")
    config = cfg.load_config()
    # Provide the LLM with current pipeline state so its tools usage is informed.
    # Exclude heavy/private blocks (auto_suspend, api_keys).
    public_config = {k: v for k, v in config.items() if k not in cfg.SENSITIVE_KEYS and k != "auto_suspend"}
    sys_msg = f"{_CHAT_SYSTEM_PROMPT}\n\nCurrent pipeline configuration:\n{json.dumps(public_config, indent=2)}"
    messages = [{"role": "system", "content": sys_msg}] + history

    tool_audit = []  # surfaced to the client for UI rendering
    try:
        async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4), _LLM_LOCK:
            for _ in range(_CHAT_MAX_TURNS):
                # Heartbeat the bubble while the LLM call runs. The
                # dead-man timeout in the client (8s) needs us to keep
                # broadcasting `calling` at <8s intervals or it auto-hides.
                _hb_stop = asyncio.Event()
                _hb_task = asyncio.create_task(_chat_thinking_heartbeat(_hb_stop))
                try:
                    msg = await asyncio.to_thread(
                        lmstudio_chat_raw, messages, _CHAT_TOOLS_MANIFEST,
                    )
                finally:
                    _hb_stop.set()
                    try:
                        await _hb_task
                    except Exception:
                        pass
                messages.append(msg)
                calls = msg.get("tool_calls") or []
                if not calls:
                    break
                for call in calls:
                    fn = (call.get("function") or {})
                    name = fn.get("name") or ""
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {}
                    handler = _CHAT_TOOL_HANDLERS.get(name)
                    if not handler:
                        result = {"ok": False, "error": f"unknown tool: {name}"}
                    else:
                        try:
                            result = handler(args)
                        except Exception as e:
                            result = {"ok": False, "error": f"{name} raised: {e!r}"}
                    tool_audit.append({"name": name, "args": args, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or "",
                        "name": name,
                        "content": json.dumps(result),
                    })
            else:
                # Hit the turn cap without a content-only response. Append a
                # graceful fallback so the client doesn't render an empty turn.
                messages.append({
                    "role": "assistant",
                    "content": "(stopped: tool-call loop hit the turn cap — try rephrasing.)",
                })
    finally:
        # Always emit `done` — even on exception — so the bubble doesn't
        # hang. Client also has an 8s dead-man timer as a safety net.
        await _broadcast_chat_thinking("done")

    # Strip the system prompt before returning; client doesn't need to see it.
    out_messages = [m for m in messages if m.get("role") != "system"]
    return {"ok": True, "messages": out_messages, "tool_audit": tool_audit}
