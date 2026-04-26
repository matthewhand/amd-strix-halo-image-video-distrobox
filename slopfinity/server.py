"""Slopfinity FastAPI dashboard — packaged entry point."""
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, Body, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import random
import re
import subprocess
import time
import asyncio
import urllib.request
import urllib.error
from typing import List

from . import config as cfg
from . import branding as _branding
from .stats import get_sys_stats, get_storage, get_outputs_disk, get_ram_estimate, get_output_counts
from .llm import lmstudio_call, lmstudio_chat_raw, DEFAULT_LLM_CONFIG, list_providers
from .llm.probe import discover as llm_discover, ping as llm_ping
from . import scheduler as sched
from . import fanout as _fanout
from .workers import ffmpeg_mux as _ffmpeg_mux

TTS_WORKER_URL = os.environ.get("TTS_WORKER_URL", "http://localhost:8010/tts")


def _load_branding():
    active = (cfg.load_config().get("branding") or {}).get("active") or "slopfinity"
    return _branding.load(active)


PKG_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(PKG_DIR, "templates")
STATIC_DIR = os.path.join(PKG_DIR, "static")

# Where generated artifacts live. In the container /workspace is mounted;
# in local dev we fall back to the experiments directory.
EXP_DIR = "/workspace"
if not os.path.isdir(EXP_DIR):
    EXP_DIR = os.path.abspath("./comfy-outputs/experiments")
    os.makedirs(EXP_DIR, exist_ok=True)

TTS_OUT_DIR = os.path.join(EXP_DIR, "tts")
os.makedirs(TTS_OUT_DIR, exist_ok=True)

app = FastAPI(title=_load_branding()["app"]["name"] + " Dashboard")

app.mount("/files", StaticFiles(directory=EXP_DIR), name="files")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the multi-res favicon. Browsers fetch /favicon.ico unconditionally
    so without this route every page load logs a 404."""
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))


@app.middleware("http")
async def _sw_allowed_header(request: Request, call_next):
    """Inject Service-Worker-Allowed: / on sw.js responses so it can scope to root."""
    response = await call_next(request)
    path = request.url.path
    if path == "/sw.js" or path == "/static/sw.js":
        response.headers["Service-Worker-Allowed"] = "/"
        response.headers["Cache-Control"] = "no-cache"
    return response

templates = Jinja2Templates(directory=TEMPLATES_DIR)

clients: List[WebSocket] = []

# Rolling buffer of recent scheduler events (last 20). Drained from
# sched.SchedulerEvents each broadcast tick.
_recent_events: List[dict] = []
_RECENT_EVENTS_MAX = 20


_ASSET_EXTS = ('.mp4', '.png', '.wav', '.webm', '.mov', '.mp3', '.ogg', '.flac')


def _kind_of(f: str) -> str:
    """Classify an asset filename into 'video', 'audio', or 'image'."""
    fl = f.lower()
    if fl.endswith(('.mp4', '.webm', '.mov')):
        return 'video'
    if fl.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        return 'audio'
    return 'image'


def _list_outputs():
    """Return four lists sorted newest-first:
        finals  — FINAL_*.mp4 (curated keepers → Completed Gallery)
        live    — everything else (chain mp4s, base pngs, bridges, test images)
                  mixed and sorted by mtime → Live Gallery
        legacy_pngs — all pngs (for back-compat templates that still branch on imgs)
        mixed   — finals + live interleaved by mtime so a FINAL's component
                  pieces (its base.png, chain mp4s) sit right next to it in
                  the gallery instead of being banished below all finals.
    """
    try:
        entries = [
            (f, os.path.getmtime(os.path.join(EXP_DIR, f)))
            for f in os.listdir(EXP_DIR)
            if f.endswith('.mp4') or f.endswith('.png')
        ]
    except Exception:
        entries = []
    entries.sort(key=lambda x: x[1], reverse=True)
    finals = [f for f, _ in entries if f.endswith('.mp4') and f.startswith('FINAL_')]
    live = [f for f, _ in entries if not (f.endswith('.mp4') and f.startswith('FINAL_'))]
    legacy_pngs = [f for f, _ in entries if f.endswith('.png')]
    mixed = [f for f, _ in entries]
    return finals, live, legacy_pngs, mixed


@app.get("/assets")
async def assets(offset: int = 0, limit: int = 48):
    """Return assets ordered by mtime desc, paginated.

    Used by the client-side infinite-scroll loader on the slop view to fetch
    older content as the user scrolls toward the bottom of the lower pane.
    The initial 64 cards are server-side rendered by `index()`; this endpoint
    serves offset >= 64 typically.

    EXP_DIR may mutate mid-request (the fleet writes new files every few
    minutes); we tolerate that by guarding os.listdir / getmtime with
    try/except so a vanished file at stat-time doesn't 500 the page.
    """
    try:
        names = [
            f for f in os.listdir(EXP_DIR)
            if f.lower().endswith(_ASSET_EXTS)
        ]
    except OSError:
        names = []
    pairs = []
    for f in names:
        try:
            pairs.append((f, os.path.getmtime(os.path.join(EXP_DIR, f))))
        except OSError:
            # File vanished between listdir and stat; skip it.
            continue
    pairs.sort(key=lambda x: x[1], reverse=True)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 256))
    page = pairs[offset:offset + limit]
    return {
        "items": [
            {"file": f, "mtime": ts, "kind": _kind_of(f)}
            for f, ts in page
        ],
        "offset": offset,
        "limit": limit,
        "total": len(pairs),
        "has_more": offset + limit < len(pairs),
    }


@app.get("/assets/by-vidx")
async def assets_by_vidx(v_idx: int):
    """Resolve actual on-disk filenames for a given video index.

    The fleet runner uses slug-based filenames
    (e.g. ``slop_1_sterile_chrome_corridors_algorithms_shep_base.png``)
    rather than the legacy ``v{N}_base.png`` shape that the dashboard
    used to synthesize. This endpoint maps a v_idx to whatever real
    filenames currently exist on disk so the client can build correct
    `/files/<name>` links instead of guessing — the previous synthesis
    would 404 against fresh slugged outputs, or worse, match a stale
    file from a previous run that happens to still be on disk under
    the old un-slugged name. Both the legacy ``v<idx>_`` and current
    ``slop_<idx>_`` prefixes are matched so historic outputs keep
    showing in the slop feed after the rename landed.
    """
    try:
        files = os.listdir(EXP_DIR)
    except OSError:
        files = []
    result: dict = {}
    legacy_prefix = f"v{v_idx}_"
    current_prefix = f"slop_{v_idx}_"
    prefix = current_prefix  # primary; legacy_prefix tested as fallback below
    # Track newest mtime per role so we prefer the most recent file when
    # the directory contains multiple matches (e.g. several video chains
    # for the same v_idx — keep the latest one for the `video` slot).
    best_mtime: dict[str, float] = {}

    def _consider(role: str, name: str) -> None:
        try:
            mt = os.path.getmtime(os.path.join(EXP_DIR, name))
        except OSError:
            return
        if role not in best_mtime or mt > best_mtime[role]:
            best_mtime[role] = mt
            result[role] = name

    # ffmpeg bridge frames: slop_{N}_<slug>_f{M}.png (or legacy
    # v{N}_<slug>_f{M}.png). Surfaced as a "bridges" {idx: filename}
    # sub-map so the dashboard can render the per-chain last-frame
    # extracts inline. The regex matches both prefixes for back-compat.
    bridge_re = re.compile(rf"^(?:slop_{v_idx}|v{v_idx})(?:_.+)?_f(\d+)\.png$")

    for f in files:
        if f.startswith(prefix) or f.startswith(legacy_prefix):
            mb = bridge_re.match(f)
            if mb:
                idx = int(mb.group(1))
                bridges = result.setdefault("bridges", {})
                # Prefer the slugged form if both forms ever exist (matches
                # whatever the runner currently writes); otherwise newest mtime.
                try:
                    mt = os.path.getmtime(os.path.join(EXP_DIR, f))
                except OSError:
                    mt = 0
                prev = bridges.get(idx)
                if prev is None:
                    bridges[idx] = f
                else:
                    try:
                        prev_mt = os.path.getmtime(os.path.join(EXP_DIR, prev))
                    except OSError:
                        prev_mt = 0
                    if mt > prev_mt:
                        bridges[idx] = f
                continue
            if f.endswith("_base.png"):
                _consider("base", f)
            elif f.endswith(".mp4"):
                # Chain segments: v{N}_c{M}.mp4 (also covers slugged
                # variants like v{N}_<slug>_c{M}.mp4 if they appear).
                _consider("video", f)
            elif f.endswith(".wav"):
                # Heuristic: TTS lines often live alongside chain audio.
                if "tts" in f.lower():
                    _consider("tts", f)
                else:
                    _consider("audio", f)
        # Final merge has its own naming convention (FINAL_{N}*.mp4)
        # and isn't prefixed with v{N}_.
        if f == f"FINAL_{v_idx}.mp4":
            _consider("final", f)
        elif f.startswith(f"FINAL_{v_idx}.") and f.endswith(".mp4"):
            _consider("final", f)
        elif f.startswith(f"FINAL_{v_idx}_") and f.endswith(".mp4"):
            _consider("final", f)
    return {"v_idx": v_idx, "assets": result}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    finals, live, imgs, mixed = _list_outputs()
    vids = finals  # alias preserved for Jinja template back-compat (v-grid)
    state = cfg.get_state()
    config = cfg.load_config()
    queue = cfg.get_queue()
    storage = get_storage()
    outputs_disk = get_outputs_disk(EXP_DIR)
    ram = get_ram_estimate(
        config.get("base_model"),
        config.get("video_model"),
        config.get("audio_model"),
        config.get("upscale_model"),
        config.get("tts_model"),
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "config": config,
            "state": state,
            "queue": queue,
            "vids": vids[:16],       # Completed Gallery (FINAL_*.mp4)
            "live": live[:64],        # Live Gallery initial page (chain mp4s + pngs); older via GET /assets
            "imgs": imgs[:10],        # back-compat (hidden i-grid)
            "mixed": mixed[:64],      # Finals interleaved with components by mtime — render path uses this when assets-filter is ON
            "storage": storage,
            "outputs_disk": outputs_disk,
            "ram": ram,
            "branding": _load_branding(),
            "branding_profiles": _branding.list_profiles(),
        },
    )


@app.post("/enhance")
async def enhance(data: dict = Body(...)):
    config = cfg.load_config()
    prompt = data.get("prompt", "")
    distribute = bool(data.get("distribute"))
    if distribute:
        sys_p = (
            "You are a master multi-stage cinematic director. Given a single user idea, "
            "produce STRICT JSON with keys 'image', 'video', 'music', 'tts'. "
            "'image' = a vivid still-frame prompt, 'video' = a motion/camera prompt, "
            "'music' = a short mood/genre description for a music generator, "
            "'tts' = a one or two sentence voiceover line. Return ONLY JSON, no prose."
        )
        # Manual LLM call — route through acquire_gpu so it participates in the
        # auto-suspend dance (PR #75): if a fleet stage is mid-flight, this
        # queues; if LM Studio was suspended, it gets resumed for this call.
        # safety_gb=4 since this is just an LLM ping, not a 60 GB diffusion.
        async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4):
            raw = await asyncio.to_thread(lmstudio_call, sys_p, prompt)
        # Best-effort JSON extraction
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            parsed = {"image": prompt, "video": prompt, "music": "", "tts": ""}
        return {
            "suggestion": raw,
            "distribute": True,
            "stages": {
                "image": parsed.get("image", ""),
                "video": parsed.get("video", ""),
                "music": parsed.get("music", ""),
                "tts": parsed.get("tts", ""),
            },
        }
    async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4):
        suggestion = await asyncio.to_thread(lmstudio_call, config["enhancer_prompt"], prompt)
    return {"suggestion": suggestion}


@app.post("/enhance/distribute")
async def enhance_distribute(data: dict = Body(...)):
    """Single-idea fan-out with preserve-tokens and lock support.

    Accepts: {core, stages: {image, video, music, tts}, locked: [...],
              preserve_tokens: [...], persist: bool=True}
    Returns: {ok, stages, preserved_ok, preserved_dropped, persisted}

    When `persist` is True (the default), the resulting per-stage prompts
    are written back to `config.{image,video,music,tts}_prompt` so they
    show up — and are editable — in the "prompts →" modal. The fleet
    runner re-reads config.json on each stage entry, so a fan-out followed
    by an immediate run will pick up the freshly persisted overrides.
    """
    core = (data.get("core") or "").strip()
    stages_in = data.get("stages") or {}
    locked = data.get("locked") or []
    preserve_tokens = data.get("preserve_tokens") or []
    persist = data.get("persist")
    if persist is None:
        persist = True
    # Fan-out makes multiple LLM calls. Hold acquire_gpu across the whole
    # batch so we suspend/resume LM Studio just once, not per call.
    # safety_gb=4 since this is just LLM rewrites, not a 60 GB diffusion stage.
    async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4):
        result = await asyncio.to_thread(
            _fanout.fanout,
            core,
            stages_in,
            locked,
            preserve_tokens,
            lmstudio_call,
        )
    persisted = False
    if persist:
        out_stages = result.get("stages") or {}
        try:
            config = cfg.load_config()
            if out_stages.get("image"):
                config["image_prompt"] = out_stages["image"]
            if out_stages.get("video"):
                config["video_prompt"] = out_stages["video"]
            if out_stages.get("music"):
                config["music_prompt"] = out_stages["music"]
            if out_stages.get("tts"):
                config["tts_prompt"] = out_stages["tts"]
            cfg.save_config(config)
            persisted = True
        except Exception as e:
            # Persistence is best-effort — never fail the fan-out itself
            # because of a transient config write hiccup.
            print(f"[enhance/distribute] config persist failed: {e}")
    return {
        "ok": result["ok"],
        "stages": result["stages"],
        "preserved_ok": result["preserved_ok"],
        "preserved_dropped": result["preserved_dropped"],
        "persisted": persisted,
    }


def _default_suggest_system_prompt(n: int) -> str:
    # Default prompt is intentionally GENERIC — no editorial tone — so a
    # fresh install doesn't bias prompts toward any particular aesthetic.
    # Users can override via the SLOPFINITY_SUGGEST_CUSTOM_PROMPT env var
    # or Settings → Prompts → "Subjects-suggest system prompt".
    return (
        "You are a concept artist for an AI video fleet. "
        f"Output exactly {n} short visual subject ideas, one per line. "
        "Each line must be 3-8 words, plain text, no numbering, no bullets, "
        "no quotes, no JSON, no markdown — just the phrase. "
        "Variety across themes; visually rich."
    )


@app.get("/subjects/suggest")
async def subjects_suggest(n: int = 6, subjects: str = "", endless: int = 0, opener: int = 0,
                            fresh: int = 0):
    """Generate N short visual subject ideas via the configured local LLM.

    Cache key includes both N and (subjects, settings flags) so toggling the
    "derive from subjects" switch or editing Subjects invalidates the cache.

    Settings honoured:
      * `suggest_use_subjects` (default True): when True, the user message
        seeds the LLM with the current Subjects textarea content for
        style/theme matching. The `subjects` query parameter carries that
        content from the client.
      * `suggest_custom_prompt` (default ""): when non-empty, replaces the
        built-in suggestion system prompt verbatim.

    `fresh=1` bypasses the cache AND injects a random salt into the user
    message so the LLM produces a different batch — used by the marquee
    drip-feed loop that fills rows 2..N (otherwise rows 2/3/4 hit the
    same cache_key as row 1 and all show identical chips).
    """
    import time
    config = cfg.load_config()
    use_subjects = bool(config.get("suggest_use_subjects", cfg.DEFAULT_SUGGEST_USE_SUBJECTS))
    # Env override wins over Settings → Prompts. Lets a user pin their
    # personal tone (e.g. "cynical philosophical") in `.env` so a fresh
    # install/checkout doesn't lose it. Empty/unset → fall back to the
    # Settings field, then the built-in default.
    env_override = (os.environ.get("SLOPFINITY_SUGGEST_CUSTOM_PROMPT") or "").strip()
    custom_prompt = env_override or (config.get("suggest_custom_prompt") or "").strip()
    subjects_in = (subjects or "").strip() if use_subjects else ""
    cache_key = (n, use_subjects, custom_prompt, subjects_in, bool(endless), bool(opener))
    cache = getattr(subjects_suggest, "_cache", None)
    now = time.time()
    # Cache persists indefinitely while the cache_key is unchanged — page
    # reloads and re-renders should NEVER re-fire the LLM unless the user
    # actually changed something (Subjects text, custom prompt, n, the
    # use_subjects toggle). The previous 30-second TTL caused every reload
    # past ~30 s to burn an unnecessary LLM call.
    # Exceptions that always re-fire fresh:
    #   * endless mode — every tick is a NEW story beat
    #   * opener — single-shot lucky-dip
    #   * fresh=1 — caller explicitly asked for variation (marquee drip-feed)
    if cache and cache[1] == cache_key and not endless and not opener and not fresh:
        return {"suggestions": cache[2], "cached": True}
    sys_p = custom_prompt if custom_prompt else _default_suggest_system_prompt(n)
    # "I'm Feeling Lucky" — a single random story-opener used to seed an
    # Endless run when the textarea is empty. We override the system
    # prompt with one that asks for ONE evocative opening scene; the
    # rest of the cycle then asks for continuations off that seed.
    if opener:
        sys_p = (
            "You are a concept artist for an AI video fleet. "
            "Output exactly ONE short visual subject — 3-8 words, plain text, "
            "no numbering, no bullets, no quotes, no JSON, no markdown — "
            "an evocative opening scene that could anchor a longer story."
        )
        user_msg = "Give me one story-opening scene."
    elif endless and subjects_in:
        # Endless Story mode (Subjects card "Endless Story" toggle on).
        # Treat the existing chips as the story-so-far and ask for the
        # NEXT chapter — explicitly forbid restating earlier scenes.
        user_msg = (
            "These are the story beats already on screen, in chronological order:\n"
            f"{subjects_in}\n\n"
            f"Continue the story from where it leaves off. Generate {n} short "
            "next-scene subject ideas that build on the trajectory above. "
            "Each line must move the story forward; do NOT repeat or paraphrase "
            "any line above. One scene per line, plain text, 3-8 words each."
        )
    elif use_subjects and subjects_in:
        user_msg = (
            "Match the style/theme of these existing subjects:\n"
            f"{subjects_in}\n\n"
            f"Now generate {n} more in the same vein."
        )
    else:
        user_msg = f"Give me {n} subject ideas."

    # When fresh=1 we want EACH call to give different ideas (the marquee
    # drip-feed asks for rows 2..N and they'd otherwise be identical to
    # row 1 since the system+user prompts are the same). Append a small
    # randomized salt that the LLM will treat as a free-association nudge.
    if fresh:
        salt_themes = [
            "atmospheric weather", "industrial decay", "biological mutation",
            "celestial phenomena", "domestic surrealism", "geometric impossibility",
            "kinetic machinery", "underwater architecture", "nocturnal wildlife",
            "bureaucratic absurdity", "fungal growth", "crystalline structures",
            "deep-sea bioluminescence", "post-industrial landscapes", "neon arcades",
        ]
        chosen = random.choice(salt_themes)
        user_msg += f"\n\nNudge: lean toward {chosen}. Avoid repeating any earlier batch."
    # Run the (blocking, network-bound) LLM call in a thread so it doesn't
    # stall FastAPI's event loop — without this, the WS state broadcast and
    # other endpoints (Settings open, etc.) freeze for the duration.
    # Wrap in acquire_gpu so manual 🎲 Suggest participates in auto-suspend
    # (resumes LM Studio if a fleet stage paused it). safety_gb=4 since
    # this is just an LLM ping, not a multi-GB diffusion stage.
    async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4):
        raw = await asyncio.to_thread(lmstudio_call, sys_p, user_msg)
    # Parse plain-text, newline-separated lines. Strip common LLM
    # decorations the prompt asks it not to use but sometimes still
    # produces — leading numbers, bullets, quotes, surrounding markdown
    # code fences. Falls back to legacy JSON-array parsing if the model
    # ignored the plain-text instruction (so cached behaviour from older
    # clients still works).
    suggestions = []
    text = (raw or "").strip()
    # Strip ``` fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Plain-text path: line-by-line.
    if text and not text.startswith("[") and not text.startswith("{"):
        for line in text.splitlines():
            t = line.strip()
            if not t:
                continue
            # Strip leading bullets / numbering: "1. ", "1) ", "- ", "* "
            t = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s+", "", t)
            # Trim wrapping quotes.
            if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                t = t[1:-1].strip()
            if t:
                suggestions.append(t)
    # JSON-array fallback (legacy / model regression).
    if not suggestions:
        try:
            s = json.loads(raw)
            if isinstance(s, list):
                suggestions = s
        except Exception:
            start, end = raw.find('['), raw.rfind(']')
            if start != -1 and end > start:
                try:
                    suggestions = json.loads(raw[start:end + 1])
                except Exception:
                    pass
    suggestions = [str(s).strip() for s in suggestions if str(s).strip()][:n]
    # Don't store endless / opener results in the cache — we want each
    # tick fresh, and openers are one-shot by design.
    if suggestions and not endless and not opener:
        subjects_suggest._cache = (now, cache_key, suggestions)
    return {"suggestions": suggestions, "cached": False}


# Chat mode — tool-using assistant. Replaces the prior Variations mode.
# The LLM (configured local provider, OpenAI-compat) gets a tools manifest
# describing actions it can take. When the model emits tool_calls in its
# response, we execute each one server-side, append the result as a tool
# message, and re-call the LLM. Loop is bounded so a confused model can't
# run away. Returns the full updated message list to the client; the
# client keeps history in localStorage and renders tool-call chips inline.

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


@app.post("/chat")
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
    messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}] + history

    tool_audit = []  # surfaced to the client for UI rendering
    async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4):
        for _ in range(_CHAT_MAX_TURNS):
            msg = await asyncio.to_thread(
                lmstudio_chat_raw, messages, _CHAT_TOOLS_MANIFEST,
            )
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

    # Strip the system prompt before returning; client doesn't need to see it.
    out_messages = [m for m in messages if m.get("role") != "system"]
    return {"ok": True, "messages": out_messages, "tool_audit": tool_audit}


# Real-model candidate pools per role. `__random__` picks uniformly from
# the role's pool when /config arrives. Keep these in sync with the option
# lists in templates/index.html.
_RANDOM_CANDIDATES = {
    "base_model":  ["qwen", "ernie"],
    "audio_model": ["heartmula"],
    "tts_model":   ["qwen-tts", "kokoro"],
}


@app.post("/config")
async def update_config(data: dict = Body(...)):
    config = cfg.load_config()
    # Resolve `__random__` placeholders to a concrete model from the role's
    # candidate pool. We persist the chosen model (not the sentinel) so the
    # rest of the pipeline never has to know about pseudo-models. `__slopped__`
    # without a concrete file selection is treated the same as `__random__`
    # (fall back to a real model) — the UI is responsible for sending
    # `slopped:<filename>` once the user picks one.
    for role, pool in _RANDOM_CANDIDATES.items():
        v = data.get(role)
        if v == "__random__" or v == "__slopped__":
            data[role] = random.choice(pool) if pool else "none"
    config.update(data)
    cfg.save_config(config)
    return {"status": "ok"}


# File-extension filters for the slopped sub-select per role. Voice (TTS) and
# music both produce WAVs in the current pipeline so they share the same set.
_SLOPPED_EXTS = {
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "audio": (".wav", ".mp3", ".flac", ".ogg"),
    "tts":   (".wav", ".mp3", ".flac", ".ogg"),
}


@app.get("/pipeline/slopped")
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


@app.post("/inject")
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


@app.post("/runner/terminate")
async def runner_terminate():
    """Hard-stop the run_fleet.py orchestrator on the host.

    Sends SIGTERM (then SIGKILL on a 5 s budget) to every process whose
    cmdline contains 'run_fleet.py'. For when /cancel-all isn't enough —
    e.g. the runner is stuck inside a hung LLM HTTP call past any
    in-loop cancel-flag check. Returns the pids it touched.

    Safety: only matches the run_fleet.py basename; doesn't pkill on
    arbitrary patterns. The runner is meant to be relaunched manually
    after a terminate (it isn't supervised by anything yet)."""
    import signal
    pids = []
    try:
        out = subprocess.run(["pgrep", "-f", "run_fleet.py"],
                             capture_output=True, text=True, timeout=5).stdout
        pids = [int(p) for p in out.split() if p.isdigit()]
    except Exception as e:
        return {"ok": False, "error": f"pgrep failed: {e}"}
    if not pids:
        return {"ok": True, "killed": [], "note": "no run_fleet.py process running"}
    killed = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            return {"ok": False, "error": f"PermissionError sending SIGTERM to {pid}"}
    # Brief grace + SIGKILL fallback.
    await asyncio.sleep(2.0)
    for pid in killed:
        try:
            os.kill(pid, 0)
            # Still alive — escalate.
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"ok": True, "killed": killed}


@app.post("/cancel-all")
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


@app.post("/queue/cancel")
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


@app.post("/queue/edit")
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


@app.post("/queue/toggle-infinity")
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


@app.post("/queue/toggle-polymorphic")
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


@app.get("/queue/paginated")
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


@app.post("/queue/requeue")
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


_SEED_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.gif')
_SEED_MAX_BYTES = 25 * 1024 * 1024  # 25MB per file — generous for camera RAW-ish PNGs


@app.get("/seeds/list")
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


@app.post("/upload")
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


@app.get("/vae_grid")
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


@app.post("/queue/clear-failed")
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


@app.post("/queue/clear-completed")
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


@app.post("/queue/requeue-failed")
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


def _call_tts_worker(text: str, voice: str, timeout: float = 600.0) -> dict:
    """POST to the Qwen3-TTS worker at TTS_WORKER_URL. Raises on transport error.

    Isolated for test mocking.
    """
    payload = json.dumps({"text": text, "voice": voice}).encode("utf-8")
    req = urllib.request.Request(
        TTS_WORKER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


@app.post("/tts")
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


@app.post("/mux")
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


@app.get("/ram_estimate")
async def ram_estimate(base: str = "", video: str = "", audio: str = "", upscale: str = "", tts: str = ""):
    return get_ram_estimate(
        base or None,
        video or None,
        audio or None,
        upscale or None,
        tts or None,
    )


@app.get("/system/ram")
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


@app.get("/pipeline/plan")
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


@app.get("/asset/{filename}")
async def asset_info(filename: str):
    """Return metadata about a single asset file: kind, model, size, mtime,
    and best-effort prompt (if a sidecar .json exists or state matches).

    Filename should be the leaf name only (no path), living under EXP_DIR.
    """
    import re
    # basic name safety — no path traversal
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    path = os.path.join(EXP_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    st = os.stat(path)
    # kind
    if filename.endswith(".mp4"):
        kind = "final" if filename.startswith("FINAL_") else "chain"
    elif filename.endswith(".wav"):
        kind = "audio"
    elif filename.endswith(".png") or filename.endswith(".jpg"):
        kind = "image"
    else:
        kind = "other"
    # model (mirror template / app.js logic)
    model = None
    m = re.match(r"^test_([A-Za-z0-9.-]+)_", filename)
    if m:
        model = m.group(1)
    elif filename.startswith("ltx_base_") or filename.endswith(".mp4"):
        model = "ltx-2.3"
    elif re.match(r"^v\d+_f\d+\.png$", filename):
        model = "ltx-bridge"
    # best-effort prompt — look for a sibling JSON sidecar OR the running state.json
    prompt = None
    sidecar = os.path.join(EXP_DIR, filename + ".json")
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as f:
                prompt = json.load(f).get("prompt")
        except Exception:
            pass
    if not prompt:
        # fallback: if this file's mtime is within 60 s of the current state, use current prompt
        state = cfg.get_state()
        if state.get("ts") and abs(st.st_mtime - state["ts"]) < 60:
            prompt = state.get("current_prompt")
    return {
        "ok": True,
        "filename": filename,
        "kind": kind,
        "model": model,
        "size_bytes": st.st_size,
        "size_human": (
            f"{st.st_size/1e9:.2f} GB" if st.st_size > 1e9
            else f"{st.st_size/1e6:.1f} MB" if st.st_size > 1e6
            else f"{st.st_size/1e3:.0f} KB"
        ),
        "mtime": st.st_mtime,
        "mtime_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "age_seconds": int(time.time() - st.st_mtime),
        "prompt": prompt,
        "url": f"/files/{filename}",
    }


@app.get("/asset/components/{filename}")
async def asset_components(filename: str):
    """For a FINAL_*.mp4, list the source components that were concatenated to
    produce it (chain mp4s, base png, music wav, optional tts wav).

    Lookup strategy:
      1. Pattern-match `FINAL_<v_idx>_<slug>.mp4` (and `_audio` variant).
      2. Prefer the concat list `_concat_<v_idx>.txt` if still on disk
         (run_fleet removes it post-mux, so usually missing).
      3. Otherwise glob `slop_<v_idx>_*` to reconstruct components by sidecar
         + filename pattern (`_c\\d+.mp4`, `_base.png`, `.wav`).

    Each component row includes filename, kind, model, part/of (if known),
    size + mtime, and a `/files/<name>` URL for direct linking. Sidecar
    fields are merged in best-effort.

    Returns `{ok: True, v_idx: int, components: [...]}` on success.
    Graceful: missing sidecars or missing concat list are non-fatal — the
    endpoint returns whatever components it could reconstruct.
    """
    import re
    import glob
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    if not filename.startswith("FINAL_") or not filename.endswith(".mp4"):
        return JSONResponse(
            {"ok": False, "error": "components only apply to FINAL_*.mp4"},
            status_code=400,
        )
    m = re.match(r"^FINAL_(\d+)_(.+?)(?:_audio)?\.mp4$", filename)
    if not m:
        return JSONResponse(
            {"ok": False, "error": "could not parse v_idx/slug from filename"},
            status_code=400,
        )
    v_idx = int(m.group(1))
    slug = m.group(2)

    def _component_row(name: str) -> dict:
        """Build a single component descriptor from a leaf filename."""
        p = os.path.join(EXP_DIR, name)
        try:
            s = os.stat(p)
            size = s.st_size
            mt = s.st_mtime
        except OSError:
            size = 0
            mt = 0.0
        # Best-effort sidecar merge — sidecars carry kind/model/part/of plus
        # FLF2V/cont fields like kf_start/kf_end/handoff_k.
        side = {}
        sidecar = os.path.join(EXP_DIR, name + ".json")
        if os.path.isfile(sidecar):
            try:
                with open(sidecar) as f:
                    side = json.load(f) or {}
            except Exception:
                side = {}
        # Derive kind from filename if sidecar didn't say.
        if name.endswith(".wav"):
            inferred_kind = "audio"
        elif name.endswith(".png") or name.endswith(".jpg"):
            inferred_kind = "image"
        elif re.search(r"_c\d+\.mp4$", name):
            inferred_kind = "chain"
        elif name.endswith(".mp4"):
            inferred_kind = "video"
        else:
            inferred_kind = "other"
        # Pull part index from filename when sidecar omitted it.
        part = side.get("part")
        if part is None:
            mc = re.search(r"_c(\d+)\.mp4$", name)
            if mc:
                part = int(mc.group(1))
        return {
            "file": name,
            "url": f"/files/{name}",
            "kind": side.get("kind") or inferred_kind,
            "model": side.get("model"),
            "prompt": side.get("prompt"),
            "part": part,
            "of": side.get("of"),
            "kf_start": side.get("kf_start"),
            "kf_end": side.get("kf_end"),
            "handoff_k": side.get("handoff_k"),
            "size_bytes": size,
            "mtime": mt,
        }

    components: list = []
    seen: set = set()

    # 1. Prefer concat list if still on disk (rare — run_fleet rm's it).
    concat_path = os.path.join(EXP_DIR, f"_concat_{v_idx}.txt")
    if os.path.isfile(concat_path):
        try:
            with open(concat_path) as f:
                for line in f:
                    line = line.strip()
                    cm = re.match(r"^file\s+'(.+)'\s*$", line)
                    if not cm:
                        continue
                    nm = os.path.basename(cm.group(1))
                    if nm in seen:
                        continue
                    if not os.path.isfile(os.path.join(EXP_DIR, nm)):
                        continue
                    components.append(_component_row(nm))
                    seen.add(nm)
        except Exception:
            pass

    # 2. Glob fallback — picks up chain mp4s in numeric order, plus base/wav.
    prefix = f"slop_{v_idx}_"
    candidates = []
    for p in glob.glob(os.path.join(EXP_DIR, f"{prefix}*")):
        nm = os.path.basename(p)
        if nm.endswith(".json"):
            continue  # sidecars are handled inline
        # Only count this component if it shares the slug — guards against
        # accidental v_idx collisions across different prompts.
        if not nm.startswith(f"{prefix}{slug}"):
            continue
        # Skip handoff/bridge frames `_f\d+.png` — they're FLF2V intermediates
        # consumed during chain assembly, not part of the final concat.
        if re.search(r"_f\d+\.png$", nm):
            continue
        candidates.append(nm)

    # Order: chain segments first (by part), then base.png, then any wavs.
    def _sort_key(nm: str):
        mc = re.search(r"_c(\d+)\.mp4$", nm)
        if mc:
            return (0, int(mc.group(1)))
        if nm.endswith("_base.png"):
            return (1, 0)
        if nm.endswith(".wav"):
            return (2, nm)
        return (3, nm)

    for nm in sorted(candidates, key=_sort_key):
        if nm in seen:
            continue
        components.append(_component_row(nm))
        seen.add(nm)

    return {
        "ok": True,
        "filename": filename,
        "v_idx": v_idx,
        "slug": slug,
        "concat_list_present": os.path.isfile(concat_path),
        "components": components,
    }


@app.delete("/asset/{filename}")
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


@app.get("/outputs")
async def outputs():
    """Return counters for produced artifacts: final mp4s, chain clips, base images."""
    return get_output_counts(EXP_DIR)


@app.get("/manifest.webmanifest")
async def manifest_webmanifest():
    """Dynamic PWA manifest using the active branding profile."""
    b = _load_branding()
    app_block = b.get("app") or {}
    colors = b.get("colors") or {}
    name = app_block.get("name") or "Slopfinity"
    short_name = app_block.get("short_name") or name
    theme_color = colors.get("primary") or "#ff79c6"
    background_color = "#282a36"
    manifest = {
        "name": name,
        "short_name": short_name,
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": background_color,
        "theme_color": theme_color,
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }
    return JSONResponse(manifest, media_type="application/manifest+json")


@app.get("/sw.js")
async def service_worker():
    """Serve the service worker with Service-Worker-Allowed: / so it can scope to root."""
    sw_path = os.path.join(STATIC_DIR, "sw.js")
    return FileResponse(
        sw_path,
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.get("/branding")
async def branding_endpoint():
    """Return the active branding profile + the list of available profiles."""
    return {
        "active": (cfg.load_config().get("branding") or {}).get("active") or "slopfinity",
        "profiles": _branding.list_profiles(),
        "resolved": _load_branding(),
    }


@app.post("/branding")
async def branding_switch(data: dict = Body(...)):
    """Switch active branding profile by name. Persisted to config.json."""
    name = (data.get("active") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "missing 'active'"}, status_code=400)
    if name not in _branding.list_profiles():
        return JSONResponse({"ok": False, "error": f"unknown profile '{name}'"}, status_code=404)
    config = cfg.load_config()
    config.setdefault("branding", {})["active"] = name
    cfg.save_config(config)
    return {"ok": True, "active": name, "resolved": _branding.load(name)}


def _current_llm_settings() -> dict:
    c = cfg.load_config()
    llm = dict(DEFAULT_LLM_CONFIG)
    llm.update(c.get("llm") or {})
    return llm


@app.get("/settings")
async def settings_get():
    """Return current settings. `api_key` is masked in transit."""
    c = cfg.load_config()
    llm = dict(DEFAULT_LLM_CONFIG)
    llm.update(c.get("llm") or {})
    has_key = bool(llm.get("api_key"))
    llm_safe = dict(llm)
    llm_safe["api_key"] = "***" if has_key else ""
    branding_cfg = c.get("branding") or {}
    return {
        "llm": llm_safe,
        "llm_has_api_key": has_key,
        "providers": list_providers(),
        "branding": {
            "active": branding_cfg.get("active") or "slopfinity",
            "profiles": _branding.list_profiles(),
        },
        "philosophical_prompt": c.get("philosophical_prompt") or "",
        "philosophical_prompt_default": cfg.DEFAULT_PHILOSOPHICAL_PROMPT,
        # Prompts tab — overrides + their built-in defaults (so the UI can
        # pre-fill placeholders and offer "Reset to default" without a second
        # round-trip). Empty string = "use default" semantics, same as the
        # philosophical_prompt pattern above.
        "enhancer_prompt": c.get("enhancer_prompt") or "",
        "enhancer_prompt_default": cfg.DEFAULT_CONFIG["enhancer_prompt"],
        "fanout_system_prompt": c.get("fanout_system_prompt") or "",
        "fanout_system_prompt_default": cfg.DEFAULT_FANOUT_SYSTEM_PROMPT,
        "fleet_user_prompt_template": c.get("fleet_user_prompt_template") or "",
        "fleet_user_prompt_template_default": cfg.DEFAULT_FLEET_USER_PROMPT_TEMPLATE,
        "infinity_user_prompt_template": c.get("infinity_user_prompt_template") or "",
        "infinity_user_prompt_template_default": cfg.DEFAULT_INFINITY_USER_PROMPT_TEMPLATE,
        "chaos_suggest_system_prompt": c.get("chaos_suggest_system_prompt") or "",
        "chaos_suggest_system_prompt_default": cfg.DEFAULT_CHAOS_SUGGEST_SYSTEM_PROMPT,
        "void_fallback_template": c.get("void_fallback_template") or "",
        "void_fallback_template_default": cfg.DEFAULT_VOID_FALLBACK_TEMPLATE,
        "suggest_use_subjects": bool(c.get("suggest_use_subjects", cfg.DEFAULT_SUGGEST_USE_SUBJECTS)),
        "suggest_custom_prompt": c.get("suggest_custom_prompt") or "",
        "suggest_auto_disabled": bool(c.get("suggest_auto_disabled", cfg.DEFAULT_SUGGEST_AUTO_DISABLED)),
        "auto_suspend": c.get("auto_suspend") or list(cfg.DEFAULT_AUTO_SUSPEND),
        # Per-model loading prefs (Settings → Scheduler → "Per-model loading
        # preferences"). Both lists default to empty; the hydrator on the
        # client toggles checkboxes based on membership.
        "model_loading": {
            "sticky": list((c.get("model_loading") or {}).get("sticky") or []),
            "eager_unload": list((c.get("model_loading") or {}).get("eager_unload") or []),
        },
        "scheduler": {
            "memory_safety_gb": (c.get("scheduler") or {}).get("memory_safety_gb", 10),
            "use_planner": bool((c.get("scheduler") or {}).get("use_planner", False)),
            # CPU offload prefs default ON — Strix Halo iGPU is the
            # bottleneck for image/video stages, so giving the LLM + TTS
            # to CPU lets them run alongside without competing for GPU.
            "llm_cpu_only": bool((c.get("scheduler") or {}).get("llm_cpu_only", True)),
            "tts_cpu_only": bool((c.get("scheduler") or {}).get("tts_cpu_only", True)),
        },
    }


@app.post("/settings")
async def settings_post(data: dict = Body(...)):
    """Partial update of settings. Writes to `config.json` under `llm.*`.

    - `api_key == ""` is treated as "no change" (mask token) — strip it.
    - `api_key == "***"` means the client echoed back the mask — also strip.
    - Any explicit non-empty value is persisted.
    """
    c = cfg.load_config()
    llm_in = data.get("llm") or {}
    if isinstance(llm_in, dict):
        current_llm = dict(DEFAULT_LLM_CONFIG)
        current_llm.update(c.get("llm") or {})
        for k, v in llm_in.items():
            if k == "api_key":
                if v in ("", "***", None):
                    continue
                current_llm[k] = v
            else:
                current_llm[k] = v
        # Coerce numerics defensively
        try:
            current_llm["temperature"] = float(current_llm.get("temperature", 0.7))
        except Exception:
            current_llm["temperature"] = 0.7
        try:
            current_llm["max_retries"] = max(0, min(5, int(current_llm.get("max_retries", 2))))
        except Exception:
            current_llm["max_retries"] = 2
        try:
            current_llm["timeout_s"] = max(1, int(current_llm.get("timeout_s", 60)))
        except Exception:
            current_llm["timeout_s"] = 60
        current_llm["auto_suspend"] = bool(current_llm.get("auto_suspend", False))
        c["llm"] = current_llm
    # Allow pass-through updates for a few other top-level buckets (e.g.
    # scheduler, model_loading). model_loading.{sticky,eager_unload} are
    # consumed by the memory_planner / scheduler to bias which model
    # checkpoints are evicted/retained across stages.
    for bucket in ("scheduler", "model_loading"):
        if bucket in data and isinstance(data[bucket], dict):
            existing = c.get(bucket) or {}
            existing.update(data[bucket])
            c[bucket] = existing
    # Fleet system prompt override. Empty string -> None ("use built-in default")
    # so the runner's loader can fall back without a sentinel check.
    if "philosophical_prompt" in data:
        v = data.get("philosophical_prompt")
        if v is None or (isinstance(v, str) and v.strip() == ""):
            c["philosophical_prompt"] = None
        elif isinstance(v, str):
            c["philosophical_prompt"] = v
    # Prompts tab — every other surfaced override. Same null-on-blank pattern
    # as philosophical_prompt: empty string => None => falls back to default.
    # `enhancer_prompt` is special: concept.py errors on empty, so reset-to-
    # default writes the canonical default rather than None to keep that
    # codepath honest.
    _PROMPT_OVERRIDE_KEYS = (
        "fanout_system_prompt",
        "fleet_user_prompt_template",
        "infinity_user_prompt_template",
        "chaos_suggest_system_prompt",
        "void_fallback_template",
    )
    for key in _PROMPT_OVERRIDE_KEYS:
        if key in data:
            v = data.get(key)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                c[key] = None
            elif isinstance(v, str):
                c[key] = v
    if "enhancer_prompt" in data:
        v = data.get("enhancer_prompt")
        if v is None or (isinstance(v, str) and v.strip() == ""):
            c["enhancer_prompt"] = cfg.DEFAULT_CONFIG["enhancer_prompt"]
        elif isinstance(v, str):
            c["enhancer_prompt"] = v
    # Auto-suggest LLM controls (Settings → LLM → Generation).
    if "suggest_use_subjects" in data:
        c["suggest_use_subjects"] = bool(data.get("suggest_use_subjects"))
    if "suggest_custom_prompt" in data:
        v = data.get("suggest_custom_prompt")
        c["suggest_custom_prompt"] = v if isinstance(v, str) else ""
    if "suggest_auto_disabled" in data:
        c["suggest_auto_disabled"] = bool(data.get("suggest_auto_disabled"))
    # Auto-suspend list (Settings → LLM → Auto-suspend during GPU inference).
    # Stored as a top-level list of {id, label, enabled, method, ...} entries.
    # Each entry's method-specific fields are preserved verbatim.
    if "auto_suspend" in data:
        v = data.get("auto_suspend")
        if isinstance(v, list):
            cleaned: list[dict] = []
            for e in v:
                if not isinstance(e, dict):
                    continue
                ce = {
                    "id": str(e.get("id") or "").strip() or None,
                    "label": str(e.get("label") or "").strip() or None,
                    "enabled": bool(e.get("enabled")),
                    "method": str(e.get("method") or "sigstop"),
                }
                # Pass through whitelisted method-specific fields.
                for f in ("process_name", "endpoint", "container", "body"):
                    if f in e and e[f] is not None:
                        ce[f] = e[f]
                if ce["id"]:
                    # Drop None label so the canonical merge can fill it back in.
                    if not ce["label"]:
                        ce.pop("label")
                    cleaned.append(ce)
            c["auto_suspend"] = cleaned
    cfg.save_config(c)
    return {"ok": True}


@app.get("/settings/models")
async def settings_models(base_url: str = "", provider: str = "lmstudio", api_key: str = ""):
    """Proxy list_models to the chosen local provider (never call from browser)."""
    from .llm.providers import get_provider
    if not base_url:
        return JSONResponse({"ok": False, "error": "missing base_url"}, status_code=400)
    # If the api_key arrives as the mask, resolve it from stored config.
    if api_key in ("***",):
        api_key = (cfg.load_config().get("llm") or {}).get("api_key") or ""
    p = get_provider(provider)
    try:
        models = p.list_models(base_url, api_key=api_key or None, timeout=5)
        return {"ok": True, "models": [m["id"] for m in models]}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "models": []}, status_code=200)


@app.post("/settings/test")
async def settings_test(data: dict = Body(...)):
    base_url = (data.get("base_url") or "").strip()
    provider = (data.get("provider") or "lmstudio").strip()
    model_id = (data.get("model_id") or "").strip()
    api_key = data.get("api_key") or ""
    if api_key in ("***",):
        api_key = (cfg.load_config().get("llm") or {}).get("api_key") or ""
    if not base_url or not model_id:
        return {"ok": False, "error": "base_url and model_id required", "latency_ms": 0}
    # Also count models to enrich the ✓ badge
    from .llm.providers import get_provider
    count = None
    try:
        count = len(get_provider(provider).list_models(base_url, api_key=api_key or None, timeout=3))
    except Exception:
        count = None
    res = llm_ping(base_url, provider, model_id, api_key=api_key or None, timeout=15)
    res["model_count"] = count
    return res


@app.get("/settings/probe")
async def settings_probe():
    """Async scan of well-known local LLM ports."""
    found = await llm_discover(timeout=1.0)
    return {"ok": True, "endpoints": found}


@app.post("/pause")
async def pause_scheduler():
    """Clear the scheduler's `paused` event — new stages will wait."""
    await sched.pause()
    return {"paused": True}


@app.post("/resume")
async def resume_scheduler():
    """Set the scheduler's `paused` event — stages may proceed."""
    await sched.resume()
    return {"paused": False}


@app.post("/free")
async def free_endpoint():
    """Trigger ComfyUI /free + gc. Returns freed_gb when measurable."""
    result = await sched.free_between()
    return {"ok": result.get("ok", False), **result}


@app.post("/emergency_free")
async def emergency_free_endpoint():
    """ComfyUI /free + pkill stray model launchers."""
    result = await sched.emergency_free()
    return {"ok": True, **result}


@app.get("/llm/health")
async def llm_health_endpoint():
    """Quick reachability probe for the configured LLM provider. Returns
    {ok: bool, provider, model_id, error?}. Used by the dashboard to gate
    LLM-dependent Subjects modes (Endless / Simple / Chat) — when the
    provider is down, only Raw mode remains usable.
    """
    config = cfg.load_config()
    llm = config.get("llm") or {}
    provider = llm.get("provider") or "lmstudio"
    base_url = llm.get("base_url") or "http://localhost:1234/v1"
    model_id = llm.get("model_id") or ""
    api_key = llm.get("api_key") or None
    try:
        result = await asyncio.to_thread(
            llm_ping, base_url, provider, model_id, api_key, 5
        )
        ok = bool(result and result.get("ok"))
        return {"ok": ok, "provider": provider, "model_id": model_id,
                "error": (result or {}).get("error") if not ok else None}
    except Exception as e:
        return {"ok": False, "provider": provider, "model_id": model_id,
                "error": str(e)}


@app.post("/llm/suspend")
async def llm_suspend_endpoint():
    """Manually SIGSTOP any running local LLM (LM Studio / Ollama).

    Independent of the `llm.auto_suspend` toggle — gives the user a one-shot
    pause for ad-hoc memory triage. Resume via POST /llm/resume.
    """
    result = await sched.suspend_llm_async()
    return {"ok": True, **result}


@app.post("/llm/resume")
async def llm_resume_endpoint():
    """Manually SIGCONT any suspended local LLM process."""
    result = await sched.resume_llm_async()
    return {"ok": True, **result}


@app.get("/scheduler/status")
async def scheduler_status():
    """Snapshot of the scheduler: pause state + queue depth."""
    return {
        "paused": sched.is_paused(),
        "pending_events": sched.SchedulerEvents.qsize(),
    }


# ---------------------------------------------------------------------------
# Coordinator (Phase 4) start/stop endpoints. The legacy fleet runner is
# still supported in parallel; users can run either path during transition.
# ---------------------------------------------------------------------------
try:
    from . import coordinator as _coordinator  # noqa: E402
except Exception as _coord_imp_err:  # pragma: no cover
    _coordinator = None
    _coord_imp_err_repr = repr(_coord_imp_err)
else:
    _coord_imp_err_repr = None


@app.post("/coordinator/start")
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


@app.post("/coordinator/stop")
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


@app.get("/coordinator/status")
async def coordinator_status():
    """Snapshot of the Coordinator: running flag + worker list + import health."""
    if _coordinator is None:
        return {"ok": False, "error": "coordinator module unavailable",
                "detail": _coord_imp_err_repr,
                "running": False, "workers": []}
    co = _coordinator.get_coordinator()
    return {"ok": True, "persisted_running": _coordinator.is_running(), **co.status()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)


async def broadcast():
    try:
        known = set(os.listdir(EXP_DIR))
    except Exception:
        known = set()
    # Stage/job timers — persisted across slopfinity restarts so refreshing
    # the dashboard doesn't reset the user's view of how long the active
    # job has been running. Keyed on (step, video_index) transitions.
    _stage_track = {"step": None, "since": time.time()}
    _job_track = {"video_index": None, "since": time.time()}
    # `_last_completed` holds the most recently finished job so the queue
    # panel can keep it visible (greyed) until the NEXT job completes —
    # see UI request: "completed items stay until the next one completes".
    _last_completed = None
    # Per-current-job stage actuals: { video_index: { 'Concept': {duration_s, ended_ts}, ... } }
    # Resets when video_index changes. Lets the frontend show timing for
    # past stages (Text, Image, Video, …) on the active row even after
    # the user refreshes the page — JS-local _jobActuals doesn't survive.
    _job_stage_actuals = {}
    _prev_state = None
    while True:
        try:
            state = cfg.get_state()
            now_ts = time.time()
            cur_step = state.get("step")
            cur_v = state.get("video_index")
            if cur_step != _stage_track["step"]:
                # Stage transition — record the OUTGOING stage's actual duration
                # against the current job before flipping.
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
                # Drop stale per-job actuals for old jobs (keep only the new one).
                _job_stage_actuals = {cur_v: _job_stage_actuals.get(cur_v, {})} if cur_v else {}
                _job_track["video_index"] = cur_v
                _job_track["since"] = now_ts
            state["stage_started_ts"] = _stage_track["since"]
            state["job_started_ts"] = _job_track["since"]
            state["last_completed"] = _last_completed
            state["stage_actuals"] = _job_stage_actuals.get(cur_v, {}) if cur_v else {}
            _prev_state = state
            stats = get_sys_stats()
            queue = cfg.get_queue()
            # Auto-rotate: cancelled items older than 48 h drop out of the
            # visible queue. Cheap to compute on each tick.
            cutoff = time.time() - 48 * 3600
            kept = [
                x for x in queue
                if not (x.get("status") == "cancelled" and (x.get("cancelled_ts") or 0) < cutoff)
            ]
            if len(kept) != len(queue):
                cfg.save_queue(kept)
                queue = kept
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
            # Never broadcast api_key / other sensitive fields.
            safe_config = cfg.redact(config)
            # Drain any pending scheduler events into the rolling buffer.
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
            # Render heartbeat — single source of truth for the queue-header
            # activity label. Backend-driven so a stalled scheduler can no
            # longer leave the spinner stuck on. Client treats `expires_ts`
            # as a TTL: if the next heartbeat doesn't arrive within ~15 s
            # the label hides itself client-side.
            try:
                if state.get('mode') != 'Idle' and state.get('step'):
                    # No trailing "…" — the per-character assembly-line animation
                    # in the queue header is its own activity indicator; an
                    # ellipsis on top of bouncing letters reads as redundant.
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
            except Exception:
                pass
            try:
                curr = set(os.listdir(EXP_DIR))
            except Exception:
                curr = known
            new = curr - known
            for f in new:
                if f.endswith('.mp4') or f.endswith('.png'):
                    for c in list(clients):
                        try:
                            await c.send_json({"type": "new_file", "file": f})
                        except Exception:
                            pass
            known = curr
        except Exception:
            pass
        await asyncio.sleep(2)


async def chaos_rotator():
    """Background task: when config.chaos_mode is on, every time a job
    completes (signalled by state.video_index incrementing) ask the LLM for
    a fresh batch of subjects that are *tangentially related* to the current
    list, then overwrite config.infinity_themes. The fleet runner picks the
    new list up on its next subject roll-over.
    """
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
                # Nothing has completed since we last looked.
                await asyncio.sleep(10)
                continue
            last_seen_index = cur_idx
            current_subjects = config.get("infinity_themes") or []
            sample = ", ".join(current_subjects[:8])
            # Honour the Settings → Prompts override (`chaos_suggest_system_prompt`).
            # Template uses {subjects_csv} so users can rearrange the prose.
            tmpl = cfg.get_chaos_suggest_system_prompt(config)
            try:
                sys_p = tmpl.format(subjects_csv=sample)
            except Exception:
                sys_p = tmpl  # malformed user template — fall back to raw
            raw = await asyncio.to_thread(lmstudio_call, sys_p, "Give me 8 tangentially-related subject ideas.")
            arr = []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    arr = parsed
            except Exception:
                a, b = raw.find('['), raw.rfind(']')
                if a != -1 and b > a:
                    try:
                        arr = json.loads(raw[a:b + 1])
                    except Exception:
                        arr = []
            arr = [str(x).strip() for x in arr if str(x).strip()][:8]
            if arr:
                config["infinity_themes"] = arr
                config["infinity_index"] = 0
                cfg.save_config(config)
            await asyncio.sleep(5)
        except Exception:
            await asyncio.sleep(30)


@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast())
    asyncio.create_task(chaos_rotator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9099)
