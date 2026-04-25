"""Slopfinity FastAPI dashboard — packaged entry point."""
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import subprocess
import time
import asyncio
import urllib.request
import urllib.error
from typing import List

from . import config as cfg
from . import branding as _branding
from .stats import get_sys_stats, get_storage, get_outputs_disk, get_ram_estimate, get_output_counts
from .llm import lmstudio_call, DEFAULT_LLM_CONFIG, list_providers
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


def _list_outputs():
    """Return three lists sorted newest-first:
        finals  — FINAL_*.mp4 (curated keepers → Completed Gallery)
        live    — everything else (chain mp4s, base pngs, bridges, test images)
                  mixed and sorted by mtime → Live Gallery
        legacy_pngs — all pngs (for back-compat templates that still branch on imgs)
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
    return finals, live, legacy_pngs


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    finals, live, imgs = _list_outputs()
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
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "config": config,
            "state": state,
            "queue": queue,
            "vids": vids[:16],       # Completed Gallery (FINAL_*.mp4)
            "live": live[:48],        # Live Gallery (chain mp4s + pngs, newest first)
            "imgs": imgs[:10],        # back-compat (hidden i-grid)
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
    suggestion = await asyncio.to_thread(lmstudio_call, config["enhancer_prompt"], prompt)
    return {"suggestion": suggestion}


@app.post("/enhance/distribute")
async def enhance_distribute(data: dict = Body(...)):
    """Single-idea fan-out with preserve-tokens and lock support.

    Accepts: {core, stages: {image, video, music, tts}, locked: [...],
              preserve_tokens: [...]}
    Returns: {ok, stages, preserved_ok, preserved_dropped}
    """
    core = (data.get("core") or "").strip()
    stages = data.get("stages") or {}
    locked = data.get("locked") or []
    preserve_tokens = data.get("preserve_tokens") or []
    result = _fanout.fanout(
        core=core,
        stages=stages,
        locked=locked,
        preserve_tokens=preserve_tokens,
        llm_call=lmstudio_call,
    )
    return {
        "ok": result["ok"],
        "stages": result["stages"],
        "preserved_ok": result["preserved_ok"],
        "preserved_dropped": result["preserved_dropped"],
    }


@app.get("/subjects/suggest")
async def subjects_suggest(n: int = 6):
    """Generate N short visual subject ideas via the configured local LLM.

    Results are cached for 30 seconds keyed on N to avoid hammering the LLM
    if the UI chip-refresh is spammed.
    """
    import time
    cache = getattr(subjects_suggest, "_cache", None)
    now = time.time()
    if cache and now - cache[0] < 30 and cache[1] == n:
        return {"suggestions": cache[2], "cached": True}
    sys_p = (
        "You are a concept artist for an AI video fleet. "
        f"Output ONLY a JSON array of exactly {n} short visual subject ideas "
        "(3-8 words each). Cynical, philosophical, visually rich. Just the array."
    )
    # Run the (blocking, network-bound) LLM call in a thread so it doesn't
    # stall FastAPI's event loop — without this, the WS state broadcast and
    # other endpoints (Settings open, etc.) freeze for the duration.
    raw = await asyncio.to_thread(lmstudio_call, sys_p, f"Give me {n} subject ideas.")
    suggestions = []
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
    if suggestions:
        subjects_suggest._cache = (now, n, suggestions)
    return {"suggestions": suggestions, "cached": False}


@app.post("/config")
async def update_config(data: dict = Body(...)):
    config = cfg.load_config()
    config.update(data)
    cfg.save_config(config)
    return {"status": "ok"}


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
):
    q = cfg.get_queue()
    if terminate:
        # Mark every pending item cancelled (so the user can see what got
        # killed) and write a flag the fleet runner watches for.
        now_ts = time.time()
        for item in q:
            if item.get("status") in (None, "pending"):
                item["status"] = "cancelled"
                item["cancelled_ts"] = now_ts
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
    }
    if stage_prompts:
        try:
            task["stage_prompts"] = json.loads(stage_prompts)
        except Exception:
            task["stage_prompts_raw"] = stage_prompts
    pending = [x for x in q if x.get("status") in (None, "pending")]
    cancelled = [x for x in q if x.get("status") == "cancelled"]
    if priority == "now":
        pending.insert(0, task)
    else:
        pending.append(task)
    cfg.save_queue(pending + cancelled)
    return {"status": "ok"}


@app.post("/cancel-all")
async def cancel_all():
    """Mark every pending queue item as cancelled and signal the fleet runner."""
    q = cfg.get_queue()
    now_ts = time.time()
    n = 0
    for item in q:
        if item.get("status") in (None, "pending"):
            item["status"] = "cancelled"
            item["cancelled_ts"] = now_ts
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
        if item.get("ts") == target_ts and item.get("status") in (None, "pending"):
            item["status"] = "cancelled"
            item["cancelled_ts"] = time.time()
            # Strip infinity so it doesn't re-loop after cancellation.
            item["infinity"] = False
            if is_first_pending:
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
    """Replace the prompt text of a pending/active queue item by ts."""
    target_ts = data.get("ts")
    new_prompt = (data.get("prompt") or "").strip()
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    if not new_prompt:
        return JSONResponse({"ok": False, "error": "empty prompt"}, status_code=400)
    q = cfg.get_queue()
    found = False
    for item in q:
        if item.get("ts") == target_ts and item.get("status") in (None, "pending"):
            item["prompt"] = new_prompt
            found = True
            break
    if not found:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    cfg.save_queue(q)
    return {"ok": True}


@app.post("/queue/toggle-infinity")
async def queue_toggle_infinity(data: dict = Body(...)):
    """Flip the `infinity` flag on a queued (or active) item by ts."""
    target_ts = data.get("ts")
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    q = cfg.get_queue()
    new_val = None
    for item in q:
        if item.get("ts") == target_ts and item.get("status") in (None, "pending"):
            item["infinity"] = not item.get("infinity", False)
            new_val = item["infinity"]
            break
    if new_val is None:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    cfg.save_queue(q)
    return {"ok": True, "infinity": new_val}


@app.post("/queue/requeue")
async def queue_requeue(data: dict = Body(...)):
    """Flip a cancelled queue item back to pending. Identified by ts."""
    target_ts = data.get("ts")
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    q = cfg.get_queue()
    found = False
    for item in q:
        if item.get("ts") == target_ts and item.get("status") == "cancelled":
            item["status"] = "pending"
            item.pop("cancelled_ts", None)
            found = True
            break
    if not found:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    cfg.save_queue(q)
    return {"ok": True}


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
    try:
        result = _call_tts_worker(text, voice)
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
async def ram_estimate(base: str = "", video: str = "", audio: str = "", upscale: str = ""):
    return get_ram_estimate(base or None, video or None, audio or None, upscale or None)


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
        c["llm"] = current_llm
    # Allow pass-through updates for a few other top-level buckets (e.g. scheduler)
    for bucket in ("scheduler",):
        if bucket in data and isinstance(data[bucket], dict):
            existing = c.get(bucket) or {}
            existing.update(data[bucket])
            c[bucket] = existing
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


@app.get("/scheduler/status")
async def scheduler_status():
    """Snapshot of the scheduler: pause state + queue depth."""
    return {
        "paused": sched.is_paused(),
        "pending_events": sched.SchedulerEvents.qsize(),
    }



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
    while True:
        try:
            state = cfg.get_state()
            # Stamp stage_started_ts / job_started_ts on the fly so the
            # frontend can derive elapsed without losing it on refresh.
            now_ts = time.time()
            cur_step = state.get("step")
            cur_v = state.get("video_index")
            if cur_step != _stage_track["step"]:
                _stage_track["step"] = cur_step
                _stage_track["since"] = now_ts
            if cur_v != _job_track["video_index"]:
                _job_track["video_index"] = cur_v
                _job_track["since"] = now_ts
            state["stage_started_ts"] = _stage_track["since"]
            state["job_started_ts"] = _job_track["since"]
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
            sys_p = (
                "You are a concept artist for an AI video fleet. The user is currently "
                "working with these subjects: [" + sample + "]. Generate 8 NEW visual "
                "subject ideas that are TANGENTIALLY related to those — riff on the "
                "themes, motifs, mood, or vibe, but introduce fresh angles. 3-8 words "
                "each. Cynical, philosophical, surreal, visually rich. Output ONLY a "
                "JSON array of strings, no prose."
            )
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
