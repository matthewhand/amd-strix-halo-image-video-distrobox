"""Slopfinity FastAPI dashboard — packaged entry point."""

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Body,
)
from slopfinity.routers.assets import _list_outputs, router as assets_router
from slopfinity.routers.queue import router as queue_router
from slopfinity.routers.chat import router as chat_router
from slopfinity.routers.suggest import router as suggest_router
from slopfinity.routers.config import router as config_router
from slopfinity.routers.runner import router as runner_router
from slopfinity.routers.llm import router as llm_router
from slopfinity.routers.coordinator import router as coordinator_router
from slopfinity.routers.story import router as story_router
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import time
import asyncio
from typing import List

from . import config as cfg
from . import branding as _branding
from .stats import (
    get_storage,
    get_outputs_disk,
    get_ram_estimate,
)
from .ws_manager import clients
from .broadcaster import broadcast, chaos_rotator

def _load_branding():
    active = (cfg.load_config().get("branding") or {}).get("active") or "slopfinity"
    return _branding.load(active)


from slopfinity.paths import EXP_DIR, STATIC_DIR, TEMPLATES_DIR

app = FastAPI(title=_load_branding()["app"]["name"] + " Dashboard")
app.include_router(assets_router)
app.include_router(queue_router)
app.include_router(chat_router)
app.include_router(story_router)
app.include_router(suggest_router)
app.include_router(config_router)
app.include_router(runner_router)
app.include_router(llm_router)
app.include_router(coordinator_router)

app.mount("/files", StaticFiles(directory=EXP_DIR), name="files")


@app.get("/static/sw.js", include_in_schema=False)
async def serve_sw_js():
    return FileResponse(
        _sw_js_with_hash(),
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache",
            "Service-Worker-Allowed": "/",
        },
    )


_sw_cache = {"hash": None, "ts": 0.0, "tmp_path": None}


def _sw_js_with_hash() -> str:
    import hashlib
    import tempfile

    src_path = os.path.join(STATIC_DIR, "sw.js")
    now = time.time()
    if (
        _sw_cache["tmp_path"]
        and (now - _sw_cache["ts"]) < 5.0
        and os.path.exists(_sw_cache["tmp_path"])
    ):
        return _sw_cache["tmp_path"]

    h = hashlib.sha256()
    candidates = [
        os.path.join(STATIC_DIR, "app.js"),
        os.path.join(STATIC_DIR, "app.css"),
        os.path.join(STATIC_DIR, "manifest.webmanifest"),
        os.path.join(TEMPLATES_DIR, "index.html"),
    ]
    icons_dir = os.path.join(STATIC_DIR, "icons")
    if os.path.isdir(icons_dir):
        for n in sorted(os.listdir(icons_dir)):
            candidates.append(os.path.join(icons_dir, n))
    for p in candidates:
        try:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
        except FileNotFoundError:
            continue
    digest = h.hexdigest()[:12]
    cache_name = f"slopfinity-shell-{digest}"

    if (
        _sw_cache["hash"] == digest
        and _sw_cache["tmp_path"]
        and os.path.exists(_sw_cache["tmp_path"])
    ):
        _sw_cache["ts"] = now
        return _sw_cache["tmp_path"]

    try:
        with open(src_path, "r") as f:
            body = f.read()
    except Exception:
        return src_path
    body = body.replace("__CACHE_VERSION__", cache_name)

    tmp_path = _sw_cache["tmp_path"]
    if not tmp_path:
        fd, tmp_path = tempfile.mkstemp(prefix="slopfinity-sw-", suffix=".js")
        os.close(fd)
    with open(tmp_path, "w") as f:
        f.write(body)
    _sw_cache.update(hash=digest, ts=now, tmp_path=tmp_path)
    return tmp_path


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"ok": True, "service": "slopfinity"}


@app.get("/readyz", include_in_schema=False)
async def readyz():
    checks = {}
    try:
        cfg.get_queue()
        checks["queue_readable"] = True
    except Exception as e:
        checks["queue_readable"] = False
        checks["queue_error"] = str(e)[:120]
    try:
        exp_dir = os.environ.get("EXP_DIR") or os.path.join(
            os.getcwd(), "comfy-outputs", "experiments"
        )
        os.makedirs(exp_dir, exist_ok=True)
        probe = os.path.join(exp_dir, ".readyz-probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)
        checks["exp_dir_writable"] = True
    except Exception as e:
        checks["exp_dir_writable"] = False
        checks["exp_dir_error"] = str(e)[:120]
    try:
        _llm = cfg.load_config().get("llm") or {}
        _bu = _llm.get("base_url") or "http://localhost:1234/v1"
        from urllib.parse import urlparse
        u = urlparse(_bu)
        checks["llm_url_ok"] = bool(u.scheme and u.hostname)
    except Exception as e:
        checks["llm_url_ok"] = False
        checks["llm_error"] = str(e)[:120]
    ok = checks.get("queue_readable") and checks.get("exp_dir_writable")
    return JSONResponse(
        {"ok": bool(ok), "checks": checks}, status_code=200 if ok else 503
    )


@app.middleware("http")
async def _sw_allowed_header(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/sw.js" or path == "/static/sw.js":
        response.headers["Service-Worker-Allowed"] = "/"
        response.headers["Cache-Control"] = "no-cache"
    return response


_TRUSTED_ORIGINS_ENV = "SLOPFINITY_TRUSTED_ORIGINS"
_CSRF_DISABLE_ENV = "SLOPFINITY_DISABLE_CSRF"
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _trusted_origin_set(request_host_header: str) -> set[str]:
    allow = set()
    if request_host_header:
        for scheme in ("http", "https"):
            allow.add(f"{scheme}://{request_host_header}")
    extra = os.environ.get(_TRUSTED_ORIGINS_ENV, "").strip()
    if extra:
        for piece in extra.split(","):
            piece = piece.strip().rstrip("/")
            if piece:
                allow.add(piece)
    return allow


@app.middleware("http")
async def _csrf_origin_check(request: Request, call_next):
    if os.environ.get(_CSRF_DISABLE_ENV) == "1":
        return await call_next(request)
    if request.method.upper() not in _MUTATING_METHODS:
        return await call_next(request)

    host = request.headers.get("host", "")
    allow = _trusted_origin_set(host)
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")

    def _origin_match(value: str) -> bool:
        if not value: return False
        try:
            from urllib.parse import urlparse
            u = urlparse(value)
            if not u.scheme or not u.netloc: return False
            return f"{u.scheme}://{u.netloc}" in allow
        except Exception: return False

    if _origin_match(origin) or _origin_match(referer):
        return await call_next(request)

    if not origin and not referer:
        host_lower = host.split(":", 1)[0].lower()
        if host_lower in ("127.0.0.1", "localhost", "::1"):
            return await call_next(request)
        return JSONResponse(
            {"ok": False, "error": "csrf: missing Origin/Referer header"},
            status_code=403,
        )

    return JSONResponse(
        {"ok": False, "error": "csrf: cross-origin mutating request rejected"},
        status_code=403,
    )


templates = Jinja2Templates(directory=TEMPLATES_DIR)

import re as _re_for_jinja

def _jinja_regex_match(s, pattern):
    if s is None: return False
    try: return bool(_re_for_jinja.search(pattern, s))
    except _re_for_jinja.error: return False

templates.env.filters["regex_match"] = _jinja_regex_match

@app.on_event("startup")
async def startup():
    if os.environ.get("SLOPFINITY_TEST_MODE") == "1":
        print("🧪 SLOPFINITY_TEST_MODE=1 — skipping background tasks (broadcast/chaos)")
        return
    asyncio.create_task(broadcast())
    asyncio.create_task(chaos_rotator())

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    finals, live, imgs, mixed = _list_outputs()
    vids = finals
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
            "vids": vids[:16],
            "live": live[:64],
            "imgs": imgs[:10],
            "mixed": mixed[:64],
            "storage": storage,
            "outputs_disk": outputs_disk,
            "ram": ram,
            "branding": _load_branding(),
            "branding_profiles": _branding.list_profiles(),
        },
    )

@app.get("/manifest.webmanifest")
async def manifest_webmanifest():
    b = _load_branding()
    app_block = b.get("app") or {}
    colors = b.get("colors") or {}
    name = app_block.get("name") or "Slopfinity"
    short_name = app_block.get("short_name") or name
    theme_color = colors.get("primary") or "#ff79c6"
    background_color = "#282a36"
    manifest = {
        "name": name, "short_name": short_name, "start_url": "/", "scope": "/",
        "display": "standalone", "background_color": background_color, "theme_color": theme_color,
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }
    return JSONResponse(manifest, media_type="application/manifest+json")

@app.get("/sw.js")
async def service_worker():
    sw_path = os.path.join(STATIC_DIR, "sw.js")
    return FileResponse(
        sw_path,
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )

@app.get("/branding")
async def branding_endpoint():
    return {
        "active": (cfg.load_config().get("branding") or {}).get("active") or "slopfinity",
        "profiles": _branding.list_profiles(),
        "resolved": _load_branding(),
    }

@app.post("/branding")
async def branding_switch(data: dict = Body(...)):
    from fastapi import Body
    name = (data.get("active") or "").strip()
    if not name: return JSONResponse({"ok": False, "error": "missing 'active'"}, status_code=400)
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
        clients.remove(websocket)
    except Exception:
        if websocket in clients: clients.remove(websocket)
