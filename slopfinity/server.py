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
from typing import List

from . import config as cfg
from . import branding as _branding
from .stats import get_sys_stats, get_storage, get_ram_estimate
from .llm import lmstudio_call
from . import fanout as _fanout


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


def _list_outputs():
    try:
        files = os.listdir(EXP_DIR)
    except Exception:
        files = []
    vids = sorted([f for f in files if f.endswith('.mp4')], reverse=True)
    imgs = sorted([f for f in files if f.endswith('.png')], reverse=True)
    return vids, imgs


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    vids, imgs = _list_outputs()
    state = cfg.get_state()
    config = cfg.load_config()
    queue = cfg.get_queue()
    storage = get_storage()
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
            "vids": vids[:8],
            "imgs": imgs[:10],
            "storage": storage,
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
        raw = lmstudio_call(sys_p, prompt)
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
    suggestion = lmstudio_call(config["enhancer_prompt"], prompt)
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
):
    q = cfg.get_queue()
    task = {"prompt": prompt, "priority": priority, "ts": time.time()}
    if stage_prompts:
        try:
            task["stage_prompts"] = json.loads(stage_prompts)
        except Exception:
            task["stage_prompts_raw"] = stage_prompts
    if priority == "now":
        q.insert(0, task)
    else:
        q.append(task)
    cfg.save_queue(q)
    return {"status": "ok"}


@app.post("/tts")
async def tts(data: dict = Body(...)):
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "ryan"
    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)
    fname = f"tts_{voice}_{int(time.time() * 1000)}.wav"
    out_path = os.path.join(TTS_OUT_DIR, fname)
    try:
        subprocess.run(
            [
                "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-y", out_path,
            ],
            capture_output=True, check=False,
        )
        ok = os.path.exists(out_path)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    url = f"/files/tts/{fname}"
    return {"ok": ok, "status": "ok" if ok else "stub-failed", "audio_path": url, "url": url, "voice": voice}


@app.get("/ram_estimate")
async def ram_estimate(base: str = "", video: str = "", audio: str = "", upscale: str = ""):
    return get_ram_estimate(base or None, video or None, audio or None, upscale or None)


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
    while True:
        try:
            state = cfg.get_state()
            stats = get_sys_stats()
            queue = cfg.get_queue()
            config = cfg.load_config()
            storage = get_storage()
            ram = get_ram_estimate(
                config.get("base_model"),
                config.get("video_model"),
                config.get("audio_model"),
                config.get("upscale_model"),
            )
            msg = {
                "type": "state",
                "state": state,
                "stats": stats,
                "queue": queue,
                "storage": storage,
                "ram": ram,
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


@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9099)
