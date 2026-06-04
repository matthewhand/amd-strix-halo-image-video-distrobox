import os
import json
import time
import asyncio
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Request, Body
from fastapi.responses import JSONResponse

import slopfinity.config as cfg
import slopfinity.scheduler as sched
from slopfinity.routers.assets import _list_outputs
from slopfinity.paths import EXP_DIR
from slopfinity.llm import _LLM_LOCK, lmstudio_chat_raw
from slopfinity.workers import ffmpeg_mux as _ffmpeg_mux

from sqlmodel import Session, select
from slopfinity.db import engine
from slopfinity.models import ChatSession, ChatMessage

router = APIRouter()

@router.get("/chat/history")
async def get_history():
    """Retrieve the active chat history from the database."""
    with Session(engine) as session:
        active_session = session.exec(
            select(ChatSession).where(ChatSession.is_active == True)
        ).first()
        if not active_session:
            return {"ok": True, "messages": []}
        
        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == active_session.id)
            .order_by(ChatMessage.ts)
        ).all()
        
        return {
            "ok": True, 
            "messages": [m.model_dump() for m in messages],
            "session_id": active_session.id
        }

@router.post("/chat/history")
async def update_history(payload: dict = Body(...)):
    """Wholesale replace the active chat history (sync from client)."""
    messages_data = payload.get("messages") or []
    with Session(engine) as session:
        active_session = session.exec(
            select(ChatSession).where(ChatSession.is_active == True)
        ).first()
        if not active_session:
            active_session = ChatSession()
            session.add(active_session)
            session.commit()
            session.refresh(active_session)
        
        # Delete old messages in this session
        old_msgs = session.exec(
            select(ChatMessage).where(ChatMessage.session_id == active_session.id)
        ).all()
        for m in old_msgs:
            session.delete(m)
        
        # Add new ones
        for m in messages_data:
            msg = ChatMessage(
                session_id=active_session.id,
                role=m.get("role"),
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                ts=m.get("ts") or time.time()
            )
            session.add(msg)
        
        active_session.updated_at = datetime.utcnow()
        session.add(active_session)
        session.commit()
        return {"ok": True}

@router.delete("/chat/history")
async def archive_history():
    """Archive the active chat session and start a fresh one."""
    with Session(engine) as session:
        active_session = session.exec(
            select(ChatSession).where(ChatSession.is_active == True)
        ).first()
        if active_session:
            active_session.is_active = False
            active_session.updated_at = datetime.utcnow()
            session.add(active_session)
        
        # Create new active session
        new_session = ChatSession()
        session.add(new_session)
        session.commit()
        return {"ok": True, "session_id": new_session.id}

@router.get("/chat/archive")
async def list_archive():
    """List archived chat sessions from the last week."""
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    with Session(engine) as session:
        archived = session.exec(
            select(ChatSession)
            .where(ChatSession.is_active == False)
            .where(ChatSession.created_at > one_week_ago)
            .order_by(ChatSession.created_at.desc())
        ).all()
        return {"ok": True, "sessions": [s.model_dump() for s in archived]}

@router.post("/chat/tokens")
async def count_tokens(payload: dict = Body(...)):
    """Calculate the total tokens for a given set of messages."""
    messages = payload.get("messages") or []
    try:
        import tiktoken
        # Default to o200k_base (GPT-4o) or cl100k_base (GPT-4/3.5)
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        import math
        # Fallback to simple heuristic if tiktoken fails to load encoding
        text = json.dumps(messages)
        return {"ok": True, "tokens": math.ceil(len(text) / 4) + 1500}
    
    total = 1500 # Base overhead for system prompt + tools
    for m in messages:
        content = m.get("content") or ""
        total += len(enc.encode(content))
        if m.get("tool_calls"):
            total += len(enc.encode(json.dumps(m["tool_calls"])))
    
    return {"ok": True, "tokens": total}


def _resolve_tts_worker_url() -> str:
    """Pick the TTS worker URL: settings config > env > hardcoded default."""
    cfg_url = (cfg.load_config().get("tts_worker_url") or "").strip()
    if cfg_url:
        return cfg_url
    return os.environ.get("TTS_WORKER_URL", "http://localhost:8010/tts")


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


def _resolve_files_path(p: str) -> str:
    """Map /files/<x> URLs back to absolute disk paths under EXP_DIR.
    Bare basenames also resolve. Refuses any path that escapes EXP_DIR
    (no `..` or absolute paths outside)."""
    if not p:
        return ""
    rel = p[len("/files/"):] if p.startswith("/files/") else p
    rel = rel.lstrip("/")
    if ".." in rel.split("/"):
        raise ValueError("path traversal not allowed")
    abs_path = os.path.realpath(os.path.join(EXP_DIR, rel))
    real_root = os.path.realpath(EXP_DIR)
    if not abs_path.startswith(real_root + os.sep) and abs_path != real_root:
        raise ValueError(f"path escapes EXP_DIR: {p}")
    return abs_path


def _chat_tool_queue_clip(args: dict) -> dict:
    """Execute the queue_clip tool. Mirrors POST /inject's logic without
    going through HTTP — direct queue mutation."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    task = {
        "prompt": prompt,
        "priority": "next",
        "status": "pending",
        "ts": time.time(),
        "chaos": True,
        "infinity": bool(args.get("infinity")),
        "fast_track": bool(args.get("fast_track")),
    }
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

    def _queue_clip(q):
        pending = [x for x in q if x.get("status") in (None, "pending")]
        working = [x for x in q if x.get("status") == "working"]
        done = [x for x in q if x.get("status") == "done"]
        cancelled = [x for x in q if x.get("status") == "cancelled"]
        pending.insert(0, task)
        return working + pending + done + cancelled

    cfg.mutate_queue(_queue_clip)
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
    hit: list = []

    def _cancel(q):
        for item in q:
            if abs(float(item.get("ts") or 0) - ts_f) < 0.001:
                if item.get("status") in (None, "pending", "working"):
                    item["status"] = "cancelled"
                    item["cancelled_ts"] = time.time()
                    hit.append(item)
                    break
        return q

    cfg.mutate_queue(_cancel)
    if not hit:
        return {"ok": False, "error": "no matching pending/running item"}
    return {"ok": True, "cancelled_prompt": (hit[0].get("prompt") or "")[:80]}


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


def _chat_tool_generate_image(args: dict) -> dict:
    """Queue an image-only task. Reuses queue_clip's plumbing but pins
    image_only=True + chains=1 so the fleet skips video/music/tts."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    task = {
        "prompt": prompt,
        "priority": "next",
        "status": "pending",
        "ts": time.time(),
        "image_only": True,
        "chaos": False,
    }

    def _gen_image(q):
        pending = [x for x in q if x.get("status") in (None, "pending")]
        working = [x for x in q if x.get("status") == "working"]
        done = [x for x in q if x.get("status") == "done"]
        cancelled = [x for x in q if x.get("status") == "cancelled"]
        pending.insert(0, task)
        return working + pending + done + cancelled

    cfg.mutate_queue(_gen_image)
    return {"ok": True, "ts": task["ts"], "prompt": prompt, "kind": "image_only"}


def _chat_tool_synthesize_tts(args: dict) -> dict:
    """Direct passthrough to /tts. Same validation as the HTTP path:
    1-5000 chars, sane speed, optional lang/voice override."""
    text = (args.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    if len(text) > 5000:
        return {"ok": False, "error": f"text too long ({len(text)} > 5000 chars)"}
    voice = args.get("voice") or "af_heart"
    lang = args.get("lang")
    speed_raw = args.get("speed")
    try:
        speed = float(speed_raw) if speed_raw is not None else None
    except (TypeError, ValueError):
        speed = None
    if speed is not None and not (0.5 <= speed <= 2.0):
        return {"ok": False, "error": "speed must be 0.5-2.0"}
    try:
        result = _call_tts_worker(text, voice, engine=None, lang=lang, speed=speed)
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"TTS worker unreachable: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"TTS error: {type(e).__name__}: {e}"}
    return {
        "ok": bool(result.get("ok")),
        "url": result.get("url") or result.get("audio_path"),
        "voice": voice,
        "engine": result.get("engine"),
    }


def _chat_tool_list_tts_voices(_args: dict) -> dict:
    """Mirror /tts/voices for chat consumption — same fallback shape if
    the worker is unreachable so the LLM always sees a list."""
    base = _resolve_tts_worker_url().rstrip("/")
    if base.endswith("/tts"):
        base = base[:-4]
    voices_url = base.rstrip("/") + "/voices"
    try:
        req = urllib.request.Request(voices_url, method="GET",
                                     headers={"accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {
            "ok": False,
            "error": f"voices endpoint unreachable ({e}); try af_heart for kokoro or ryan for qwen",
        }


def _chat_tool_mux_assets(args: dict) -> dict:
    """ffmpeg mux without -shortest. Audio shorter than video → pad
    with apad. Audio longer → trim to video duration via -t. Either
    way the output matches the VIDEO duration exactly so a 30s music
    on a 20s video produces 20s of muxed video (the music gets
    trimmed to fit, no truncation to whichever ends first)."""
    try:
        video_abs = _resolve_files_path(args.get("video_path") or "")
        audio_abs = _resolve_files_path(args.get("audio_path") or "")
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if not (os.path.isfile(video_abs) and os.path.isfile(audio_abs)):
        return {"ok": False, "error": f"missing input file(s): video={video_abs} audio={audio_abs}"}
    out_name = (args.get("out_name") or "").strip() or f"muxed_{int(time.time())}.mp4"
    out_name = os.path.basename(out_name)  # paranoia
    out_abs = os.path.join(EXP_DIR, out_name)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_abs,
        "-i", audio_abs,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-af", "apad",
        "-shortest",
        out_abs,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffmpeg timeout (120s)"}
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        return {"ok": False, "error": "ffmpeg failed", "stderr": "\n".join(tail)}
    return {"ok": True, "url": f"/files/{out_name}", "size": os.path.getsize(out_abs)}


def _chat_tool_concat_videos(args: dict) -> dict:
    """ffmpeg concat demuxer. All inputs must share codec/size/fps —
    fast stream copy, no re-encode. Writes a temp concat list file
    inside EXP_DIR/.concat-lists/ then runs ffmpeg."""
    paths = args.get("paths") or []
    if not isinstance(paths, list) or len(paths) < 2:
        return {"ok": False, "error": "need at least 2 paths"}
    try:
        abs_paths = [_resolve_files_path(p) for p in paths]
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    missing = [p for p in abs_paths if not os.path.isfile(p)]
    if missing:
        return {"ok": False, "error": f"missing files: {missing}"}
    out_name = (args.get("out_name") or "").strip() or f"concat_{int(time.time())}.mp4"
    out_name = os.path.basename(out_name)
    out_abs = os.path.join(EXP_DIR, out_name)
    list_dir = os.path.join(EXP_DIR, ".concat-lists")
    os.makedirs(list_dir, exist_ok=True)
    list_path = os.path.join(list_dir, f"concat_{int(time.time())}.txt")
    with open(list_path, "w") as f:
        for p in abs_paths:
            f.write(f"file '{p.replace(chr(39), chr(92) + chr(39))}'\n")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_abs,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffmpeg timeout (120s)"}
    finally:
        try: os.unlink(list_path)
        except OSError: pass
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        return {"ok": False, "error": "ffmpeg concat failed", "stderr": "\n".join(tail)}
    return {"ok": True, "url": f"/files/{out_name}", "n_clips": len(abs_paths), "size": os.path.getsize(out_abs)}


def _chat_tool_not_yet_wired(args: dict) -> dict:
    """Stub for tools whose backend integration is planned but not yet shipped."""
    return {
        "ok": False,
        "status": "not-yet-wired",
        "error": "Standalone music/video generation isn't wired yet. "
                 "Use queue_clip for the full pipeline (image+video+music+tts), "
                 "or wait for a follow-up PR that adds run_fleet.py task_opts "
                 "for audio_only / video_only flows.",
        "alternative_tools": ["queue_clip", "generate_image", "synthesize_tts"],
    }


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
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate ONE image from a prompt. Queues a task with image_only=True so the fleet runs concept+image and skips video/music/tts. Returns the queue ts; poll with list_queue or recent_finals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Visual subject in plain text."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_tts",
            "description": "Synthesize speech from text via the configured TTS worker. Returns a /files/tts/<name>.wav URL on success. Default voice af_heart (Kokoro) — use list_tts_voices to see the full set across both engines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "1-5000 chars to speak."},
                    "voice": {"type": "string", "description": "Voice id (e.g. af_heart, am_eric, jf_alpha). Defaults to af_heart."},
                    "lang": {"type": "string", "description": "Optional language override (en-us, ja, cmn, etc.). Default: auto-pick from voice prefix."},
                    "speed": {"type": "number", "description": "Speech speed 0.5-2.0. Default 1.0."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tts_voices",
            "description": "List voices available across the kokoro + qwen TTS engines. Use this when the user asks 'what voices can you use?'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mux_assets",
            "description": "Combine a video file and an audio file into a single MP4 via ffmpeg. Pass /files/<…>.mp4 and /files/<…>.wav paths from previous tool outputs. Pads/trims audio to match video duration so neither stream is wasted (no -shortest truncation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path or /files URL of the video stream (mp4)."},
                    "audio_path": {"type": "string", "description": "Path or /files URL of the audio stream (wav/mp3)."},
                    "out_name": {"type": "string", "description": "Output basename (no path). Default 'muxed_<ts>.mp4'."},
                },
                "required": ["video_path", "audio_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "concat_videos",
            "description": "Concatenate multiple MP4 clips into one output via ffmpeg concat demuxer (stream copy, fast, no re-encode). All inputs must share codec / size / fps. Used to stitch story-mode FINAL_*.mp4 clips into one continuous video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Ordered list of /files paths or basenames to concat."},
                    "out_name": {"type": "string", "description": "Output basename. Default 'concat_<ts>.mp4'."},
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_music",
            "description": "[NOT YET WIRED] Generate a music WAV via heartmula. Currently invoked via queue_clip with audio bundled into the full pipeline; standalone music gen is a planned follow-up. Returns a structured pointer to the open issue rather than firing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Tags/description (genre, instruments, mood)."},
                    "duration_s": {"type": "number", "description": "Target length in seconds (3-180)."},
                },
                "required": ["prompt", "duration_s"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": "[NOT YET WIRED] Generate ONE video clip without music/tts via LTX-2. Currently invoked via queue_clip with the full pipeline; standalone video gen is a planned follow-up requiring run_fleet.py task_opt extensions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Visual subject."},
                    "duration_s": {"type": "number", "description": "Target length in seconds; converted to chains internally."},
                },
                "required": ["prompt", "duration_s"],
            },
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
    "generate_image": _chat_tool_generate_image,
    "synthesize_tts": _chat_tool_synthesize_tts,
    "list_tts_voices": _chat_tool_list_tts_voices,
    "mux_assets": _chat_tool_mux_assets,
    "concat_videos": _chat_tool_concat_videos,
    "generate_music": _chat_tool_not_yet_wired,
    "generate_video": _chat_tool_not_yet_wired,
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


async def _broadcast_chat_thinking(phase: str) -> None:
    """Emit a `chat_thinking` signal to every connected WS client."""
    from slopfinity.server import clients
    msg = {"type": "chat_thinking", "phase": phase, "ts": time.time()}
    for c in list(clients):
        try:
            await c.send_json(msg)
        except Exception:
            pass


async def _chat_thinking_heartbeat(stop_evt: asyncio.Event) -> None:
    """Repeatedly emit `chat_thinking: calling` every ~2s while the LLM
    call is in flight. Stops when `stop_evt` is set."""
    try:
        while not stop_evt.is_set():
            await _broadcast_chat_thinking("calling")
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
    except Exception:
        pass


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
    history = history[-50:]
    await _broadcast_chat_thinking("received")
    config = cfg.load_config()
    public_config = cfg.redact(config)
    if isinstance(public_config, dict):
        public_config.pop("auto_suspend", None)
    sys_msg = f"{_CHAT_SYSTEM_PROMPT}\n\nCurrent pipeline configuration:\n{json.dumps(public_config, indent=2)}"
    messages = [{"role": "system", "content": sys_msg}] + history

    tool_audit = []
    try:
        async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4), _LLM_LOCK:
            for _ in range(_CHAT_MAX_TURNS):
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
                messages.append({
                    "role": "assistant",
                    "content": "(stopped: tool-call loop hit the turn cap — try rephrasing.)",
                })
    finally:
        await _broadcast_chat_thinking("done")

    out_messages = [m for m in messages if m.get("role") != "system"]
    return {"ok": True, "messages": out_messages, "tool_audit": tool_audit}
