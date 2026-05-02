"""Slopfinity FastAPI dashboard — packaged entry point."""

from fastapi import (
    FastAPI,
    Request,
    Form,
    WebSocket,
    WebSocketDisconnect,
    Body,
    UploadFile,
    File,
)
from slopfinity.routers.assets import _list_outputs, router as assets_router
from slopfinity.routers.queue import router as queue_router
from slopfinity.routers.chat import router as chat_router
from slopfinity.routers.suggest import router as suggest_router
from slopfinity.routers.config import router as config_router
from slopfinity.routers.runner import router as runner_router
from slopfinity.routers.llm import router as llm_router
from slopfinity.routers.coordinator import router as coordinator_router
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
from .stats import (
    get_sys_stats,
    get_storage,
    get_outputs_disk,
    get_ram_estimate,
    get_output_counts,
)
from .llm import lmstudio_call, lmstudio_chat_raw, DEFAULT_LLM_CONFIG, list_providers
from .llm.probe import discover as llm_discover, ping as llm_ping
from . import scheduler as sched
from . import fanout as _fanout
from .workers import ffmpeg_mux as _ffmpeg_mux

TTS_WORKER_URL = os.environ.get("TTS_WORKER_URL", "http://localhost:8010/tts")


def _load_branding():
    active = (cfg.load_config().get("branding") or {}).get("active") or "slopfinity"
    return _branding.load(active)


from slopfinity.paths import EXP_DIR, TTS_OUT_DIR, STATIC_DIR, TEMPLATES_DIR

# Module-level mutex serializing LLM calls. The local providers
# (LM Studio / llama.cpp / Ollama) are typically single-GPU and a single
# in-flight request; firing concurrent requests at them either rejects
# the second one outright or silently degrades latency for both. Wrapping
# every lmstudio_call / lmstudio_chat_raw to_thread invocation in this
# lock turns the dashboard's parallel auto-suggest + interactive-chat +
# enhance traffic into a sequential queue from the model's POV. Each
# call still runs off-loop (asyncio.to_thread), so FastAPI's event loop
# stays responsive — only the LLM-backed coroutines wait their turn.
_LLM_LOCK = asyncio.Lock()


app = FastAPI(title=_load_branding()["app"]["name"] + " Dashboard")
app.include_router(assets_router)
app.include_router(queue_router)
app.include_router(chat_router)
app.include_router(suggest_router)
app.include_router(config_router)
app.include_router(runner_router)
app.include_router(llm_router)
app.include_router(coordinator_router)

app.mount("/files", StaticFiles(directory=EXP_DIR), name="files")


@app.get("/static/sw.js", include_in_schema=False)
async def serve_sw_js():
    """Serve the service worker with a content-hash cache version
    substituted into the `__CACHE_VERSION__` sentinel. Eliminates the
    manual `slopfinity-shell-vNNN` bump that every UI PR has had to
    remember (the audit found this in PR-D — bumped 6× during the
    visual-polish train alone).

    Hash is computed from the shell assets that, if changed, MUST
    invalidate the user's cache: app.js, app.css, the rendered
    template (index.html source), manifest, and the icon set. Cached
    in-process with a 5s TTL so the per-request cost is one stat call
    when assets are stable, one read+sha256 when they change."""
    return FileResponse(
        _sw_js_with_hash(),
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache",  # browser MUST revalidate the SW itself
            "Service-Worker-Allowed": "/",
        },
    )


_sw_cache = {"hash": None, "ts": 0.0, "tmp_path": None}


def _sw_js_with_hash() -> str:
    """Return a path to a temp file containing the SW source with the
    `__CACHE_VERSION__` sentinel replaced by the current shell hash.
    The temp file is rewritten in-place when the hash changes so we
    don't litter /tmp with one file per change."""
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
        return src_path  # fall back to the static file as-is
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
    """Serve the multi-res favicon. Browsers fetch /favicon.ico unconditionally
    so without this route every page load logs a 404."""
    from fastapi.responses import FileResponse

    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))


# ---------------------------------------------------------------------------
# Health probes — `/healthz` and `/readyz` so monitoring / orchestrators
# / future docker-compose healthchecks can probe the dashboard without
# parsing the full /runner/status payload. Audit's #2 high-priority
# observability gap.
#
#   /healthz — process liveness. Always 200 if the FastAPI worker is
#              answering. No deps. Cheap. For a load balancer.
#   /readyz  — service readiness. 200 only if the queue file is
#              readable, EXP_DIR is writable, and the LLM probe is
#              live. 503 with a body listing the failing checks
#              otherwise. For startup gating + alerting.
#
# Both endpoints are unauthenticated (consistent with the rest of the
# dashboard's read endpoints) and skipped by the CSRF middleware
# (GET methods always pass-through).
# ---------------------------------------------------------------------------
@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"ok": True, "service": "slopfinity"}


@app.get("/readyz", include_in_schema=False)
async def readyz():
    checks = {}
    # Queue file readable
    try:
        cfg.get_queue()
        checks["queue_readable"] = True
    except Exception as e:
        checks["queue_readable"] = False
        checks["queue_error"] = str(e)[:120]
    # EXP_DIR writable
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
    # LLM probe — soft, mark unhealthy but still 200 on read failures
    # if everything else is OK (LLM is optional for dashboard liveness)
    try:
        from .llm.probe import discover as _llm_discover

        _llm = cfg.load_config().get("llm") or {}
        _bu = _llm.get("base_url") or "http://localhost:1234/v1"
        # Don't actually call out — just confirm the URL is well-formed.
        # An end-to-end LLM call belongs in /llm/health, not /readyz.
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
    """Inject Service-Worker-Allowed: / on sw.js responses so it can scope to root."""
    response = await call_next(request)
    path = request.url.path
    if path == "/sw.js" or path == "/static/sw.js":
        response.headers["Service-Worker-Allowed"] = "/"
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------------------------------------------------------------------
# CSRF / cross-site mitigations
#
# The dashboard ships zero auth and a wide mutating API surface (queue,
# settings, /inject, /upload, /runner/terminate, etc.). Without an Origin
# / Referer check, ANY page the user browses while the dashboard is open
# can drive the dashboard via cross-site form-POST or fetch (no preflight
# is required for `text/plain` bodies; FastAPI accepts those into JSON
# handlers via `Body(...)`). Combined with the persistent `auto_suspend`
# settings path (which can run shell commands), that's a CSRF→RCE chain
# reachable from a drive-by browse.
#
# This middleware rejects mutating requests whose Origin/Referer host
# isn't in the allowlist. By default the allowlist is just the bind
# host:port (same-origin). Operators who proxy the dashboard or embed
# it can add hosts via SLOPFINITY_TRUSTED_ORIGINS (comma-separated, full
# scheme://host[:port]).
#
# Read endpoints (GET/HEAD/OPTIONS) are not gated — they're either
# public-by-design (favicon, static) or already exposed to any visitor.
# ---------------------------------------------------------------------------
_TRUSTED_ORIGINS_ENV = "SLOPFINITY_TRUSTED_ORIGINS"
_CSRF_DISABLE_ENV = "SLOPFINITY_DISABLE_CSRF"  # opt-out for legacy automation
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _trusted_origin_set(request_host_header: str) -> set[str]:
    """Build the allow-set on each request: same-origin always, plus
    any explicit additions from the env var. Cheap (env read + split)."""
    allow = set()
    # Same-origin: derive from the Host header. We accept both http and
    # https variants since the dashboard runs http on localhost in dev
    # and may sit behind a TLS-terminating proxy in deploys.
    if request_host_header:
        for scheme in ("http", "https"):
            allow.add(f"{scheme}://{request_host_header}")
    # Env-var additions
    extra = os.environ.get(_TRUSTED_ORIGINS_ENV, "").strip()
    if extra:
        for piece in extra.split(","):
            piece = piece.strip().rstrip("/")
            if piece:
                allow.add(piece)
    return allow


@app.middleware("http")
async def _csrf_origin_check(request: Request, call_next):
    """Reject cross-site mutating requests by Origin/Referer mismatch.

    Allows the request if:
      - Method is GET/HEAD/OPTIONS, OR
      - SLOPFINITY_DISABLE_CSRF=1 is set (escape hatch), OR
      - Origin (preferred) or Referer (fallback) matches the same host
        the request came in on, OR an entry in SLOPFINITY_TRUSTED_ORIGINS.

    Rejects with 403 otherwise. The error body is JSON so frontend
    fetch wrappers can surface a useful toast.
    """
    if os.environ.get(_CSRF_DISABLE_ENV) == "1":
        return await call_next(request)
    if request.method.upper() not in _MUTATING_METHODS:
        return await call_next(request)

    host = request.headers.get("host", "")
    allow = _trusted_origin_set(host)

    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")

    def _origin_match(value: str) -> bool:
        if not value:
            return False
        # Strip path/query from referer; keep scheme://host[:port]
        try:
            from urllib.parse import urlparse

            u = urlparse(value)
            if not u.scheme or not u.netloc:
                return False
            return f"{u.scheme}://{u.netloc}" in allow
        except Exception:
            return False

    if _origin_match(origin) or _origin_match(referer):
        return await call_next(request)

    # No Origin AND no Referer: rare for browser-driven traffic, common
    # for curl / scripts / pytest harnesses on the loopback. Allow when
    # the request is hitting loopback (the attacker model is "drive-by
    # cross-site browse" — those attacks ALWAYS carry an Origin header,
    # so absent-Origin from loopback is safe). For non-loopback hosts
    # (LAN bind, behind a proxy) the absent header is suspicious and
    # we still reject.
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

# Custom Jinja filter — `{{ s | regex_match(pattern) }}` returns truthy
# when `pattern` matches anywhere in `s` (re.search semantics, not full
# match). Used by the slop-card SSR loop to flag bridge-frame PNGs
# (`slop_<N>_<slug>_f<M>.png` / `v<N>_f<M>.png`) so the dashboard's
# secondary 'frames' filter chip can hide them by default. Re-using
# Python's compiled re cache means repeated calls are cheap.
import re as _re_for_jinja


def _jinja_regex_match(s, pattern):
    if s is None:
        return False
    try:
        return bool(_re_for_jinja.search(pattern, s))
    except _re_for_jinja.error:
        return False


templates.env.filters["regex_match"] = _jinja_regex_match

clients: List[WebSocket] = []

# Rolling buffer of recent scheduler events (last 20). Drained from
# sched.SchedulerEvents each broadcast tick.
_recent_events: List[dict] = []
_RECENT_EVENTS_MAX = 20


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
            "vids": vids[:16],  # Completed Gallery (FINAL_*.mp4)
            "live": live[
                :64
            ],  # Live Gallery initial page (chain mp4s + pngs); older via GET /assets
            "imgs": imgs[:10],  # back-compat (hidden i-grid)
            "mixed": mixed[
                :64
            ],  # Finals interleaved with components by mtime — render path uses this when assets-filter is ON
            "storage": storage,
            "outputs_disk": outputs_disk,
            "ram": ram,
            "branding": _load_branding(),
            "branding_profiles": _branding.list_profiles(),
        },
    )


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


# Chat mode — tool-using assistant. Replaces the prior Variations mode.
# The LLM (configured local provider, OpenAI-compat) gets a tools manifest
# describing actions it can take. When the model emits tool_calls in its
# response, we execute each one server-side, append the result as a tool
# message, and re-call the LLM. Loop is bounded so a confused model can't
# run away. Returns the full updated message list to the client; the
# client keeps history in localStorage and renders tool-call chips inline.


async def _broadcast_chat_thinking(phase: str) -> None:
    """Emit a `chat_thinking` signal to every connected WS client.

    Phases: 'received' (user message landed), 'calling' (LLM call in flight,
    sent on a heartbeat cadence), 'done' (response ready, hide bubble).
    Client uses this as a dead-man switch — see `_chatThinkingExpiresAt`
    in app.js. Best-effort: dropped clients are silently pruned.
    """
    msg = {"type": "chat_thinking", "phase": phase, "ts": time.time()}
    for c in list(clients):
        try:
            await c.send_json(msg)
        except Exception:
            pass


async def _chat_thinking_heartbeat(stop_evt: asyncio.Event) -> None:
    """Repeatedly emit `chat_thinking: calling` every ~2s while the LLM
    call is in flight. Stops when `stop_evt` is set. Pairs with the 8s
    dead-man timeout client-side: as long as heartbeats arrive, the
    cogs keep spinning."""
    try:
        while not stop_evt.is_set():
            await _broadcast_chat_thinking("calling")
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
    except Exception:
        pass


# Real-model candidate pools per role. `__random__` picks uniformly from
# the role's pool when /config arrives. Keep these in sync with the option
# lists in templates/index.html.
_RANDOM_CANDIDATES = {
    "base_model": ["qwen", "ernie"],
    "audio_model": ["heartmula"],
    "tts_model": ["qwen-tts", "kokoro"],
}


# File-extension filters for the slopped sub-select per role. Voice (TTS) and
# music both produce WAVs in the current pipeline so they share the same set.
_SLOPPED_EXTS = {
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "audio": (".wav", ".mp3", ".flac", ".ogg"),
    "tts": (".wav", ".mp3", ".flac", ".ogg"),
}


def _check_disk_guard():
    """Return (ok, reason) — False when the outputs partition is below
    the user-configured low-water marks. Two thresholds; either trips
    the guard. Setting either to 0 disables that check."""
    config = cfg.load_config()
    min_pct = float(config.get("disk_min_pct") or 0)
    min_gb = float(config.get("disk_min_gb") or 0)
    if min_pct <= 0 and min_gb <= 0:
        return True, ""
    try:
        d = get_outputs_disk(EXP_DIR)
        free_gb = d.get("free_gb")
        if free_gb is None:
            free_gb = (d.get("total_gb") or 0) - (d.get("used_gb") or 0)
        free_pct = 100 - (d.get("pct") or 0)
    except Exception:
        return True, ""  # fail open if we can't read disk stats
    if min_pct > 0 and free_pct <= min_pct:
        return False, f"only {free_pct:.1f}% free (threshold ≤ {min_pct}%)"
    if min_gb > 0 and free_gb <= min_gb:
        return False, f"only {free_gb:.1f} GB free (threshold ≤ {min_gb} GB)"
    return True, ""


def _find_pids_by_cmdline(needle: str) -> list[int]:
    """Scan /proc for processes whose argv contains a leaf matching `needle`.

    Pure-Python replacement for `pgrep -f` — the dashboard's runtime
    environment doesn't always have procps. Match precision: an arg's
    BASENAME must equal `needle`. So `python3 /path/to/run_fleet.py` matches
    (basename of arg = 'run_fleet.py') but `bash -c 'echo run_fleet.py'`
    does NOT (basename of arg = 'echo'/'-c'/the literal echo string —
    none equal 'run_fleet.py'). The original `if needle in cmd` was too
    greedy and matched bash wrappers that quoted the literal.
    """
    pids: list[int] = []
    if os.path.isdir("/proc"):
        try:
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry.name}/cmdline", "rb") as f:
                        raw = f.read()
                    if not raw:
                        continue  # kernel thread
                    args = [
                        a.decode("utf-8", errors="replace")
                        for a in raw.split(b"\x00")
                        if a
                    ]
                    if any(os.path.basename(a) == needle for a in args):
                        pids.append(int(entry.name))
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
            return pids
        except Exception:
            pass  # fall through to pgrep
    try:
        # pgrep with -x matches whole-word command names; combined with
        # -f to match against full argv it still risks the `bash wrapper
        # quoting needle` false positive, but it's the best fallback we
        # can do without /proc.
        out = subprocess.run(
            ["pgrep", "-f", needle], capture_output=True, text=True, timeout=5
        ).stdout
        return [int(p) for p in out.split() if p.isdigit()]
    except FileNotFoundError:
        return []
    except Exception:
        return []


_SEED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_SEED_MAX_BYTES = 25 * 1024 * 1024  # 25MB per file — generous for camera RAW-ish PNGs


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
            {
                "src": "/static/icons/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable",
            },
            {
                "src": "/static/icons/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable",
            },
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
        "active": (cfg.load_config().get("branding") or {}).get("active")
        or "slopfinity",
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
        return JSONResponse(
            {"ok": False, "error": f"unknown profile '{name}'"}, status_code=404
        )
    config = cfg.load_config()
    config.setdefault("branding", {})["active"] = name
    cfg.save_config(config)
    return {"ok": True, "active": name, "resolved": _branding.load(name)}


def _coerce_cpu_mode(raw) -> str:
    """Normalise any stored CPU-mode value to a canonical string.

    Accepts:
      - "smart" / "cpu" / "gpu"  -- returned as-is
      - True (old boolean)        -- "cpu"
      - False (old boolean)       -- "gpu"
      - None / missing            -- "smart"  (new default)
    """
    if raw is True:
        return "cpu"
    if raw is False:
        return "gpu"
    if isinstance(raw, str) and raw in ("smart", "cpu", "gpu"):
        return raw
    return "smart"


def _cpu_mode_to_bool(raw):
    """Return a backward-compat boolean for the old llm_cpu_only field.

    "cpu"   -- True
    "gpu"   -- False
    "smart" -- None  (callers should call _resolve_cpu_mode for a live decision)
    """
    if isinstance(raw, bool):
        return raw
    mode = _coerce_cpu_mode(raw)
    if mode == "cpu":
        return True
    if mode == "gpu":
        return False
    return None  # smart -- resolved at runtime


def _resolve_cpu_mode(mode: str) -> bool:
    """Resolve a cpu_mode string to a concrete True/False at call-time.

    For "smart" mode, reads live GPU utilisation from rocm-smi:
      - GPU at 0% -- use GPU (return False)
      - GPU  > 0% -- use CPU (return True, avoid contention)
    """
    mode = _coerce_cpu_mode(mode)
    if mode == "cpu":
        return True
    if mode == "gpu":
        return False
    # smart -- read live GPU %
    try:
        from .stats import get_sys_stats

        stats = get_sys_stats()
        gpu_pct = int(stats.get("gpu") or 0)
        return gpu_pct > 0
    except Exception:
        return True  # fallback to CPU on any error


def _current_llm_settings() -> dict:
    c = cfg.load_config()
    llm = dict(DEFAULT_LLM_CONFIG)
    llm.update(c.get("llm") or {})
    has_key = bool(llm.get("api_key"))
    llm_safe = dict(llm)
    llm_safe["api_key"] = "***" if has_key else ""
    return llm_safe


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
        # SSRF guard: validate base_url BEFORE persisting. Without this,
        # an attacker who lands a single same-origin POST (e.g. via XSS
        # bypassing the CSRF middleware in #142) can flip llm.base_url
        # to http://169.254.169.254/ and have the dashboard exfil cloud
        # metadata on the next /chat call.
        nb = (llm_in.get("base_url") or "").strip()
        if nb:
            ok, err = _validate_llm_base_url(nb)
            if not ok:
                return JSONResponse(
                    {"ok": False, "error": f"llm.base_url: {err}"}, status_code=400
                )
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
            current_llm["max_retries"] = max(
                0, min(5, int(current_llm.get("max_retries", 2)))
            )
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
    # Cloud-endpoints gate (Settings → LLM). When False (default) the
    # provider dropdown only shows local providers. The registry itself
    # is still local-only today; this just persists the user's choice
    # so adding a cloud provider later doesn't require a UI hop.
    if "allow_cloud_endpoints" in data:
        c["allow_cloud_endpoints"] = bool(data.get("allow_cloud_endpoints"))
    # Auto-suggest LLM controls (Settings → LLM → Generation).
    if "suggest_use_subjects" in data:
        c["suggest_use_subjects"] = bool(data.get("suggest_use_subjects"))
    if "suggest_custom_prompt" in data:
        v = data.get("suggest_custom_prompt")
        c["suggest_custom_prompt"] = v if isinstance(v, str) else ""
    if "suggest_auto_disabled" in data:
        c["suggest_auto_disabled"] = bool(data.get("suggest_auto_disabled"))
    # Spiffy mode — per-row prompt-pill cluster in simple mode. Default
    # False. Plumbed in v316 but the settings POST branch was missing
    # (Agent C's e2e spec flagged this with a .fixme).
    if "suggest_per_row_prompts" in data:
        c["suggest_per_row_prompts"] = bool(data.get("suggest_per_row_prompts"))
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
                # `command` is the script-method override (added 2026-04);
                # empty string is preserved so the UI keeps an empty input.
                for f in ("process_name", "endpoint", "container", "body", "command"):
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


def _validate_llm_base_url(base_url: str) -> tuple[bool, str]:
    """Reject SSRF-prone llm.base_url values BEFORE the URL hits urlopen.

    The llm.base_url is user-controlled via /settings POST. Without
    validation, a hostile setter (or attacker who slipped past the
    CSRF gate via a same-origin XSS) can repoint LLM calls at:
      * cloud-metadata services (http://169.254.169.254/)
      * RFC1918 / link-local internal admin panels
      * file:// or gopher:// or ftp:// schemes that return readable
        bytes Python's urlopen happily handles
    The dashboard's own integrations only ever target localhost/loopback
    LLM endpoints (LM Studio, Ollama, etc.) or — when explicitly opted
    in via `allow_cloud_endpoints=true` — public cloud APIs.

    Returns (ok, error_msg). Empty error_msg on ok=True.
    """
    from urllib.parse import urlparse

    if not base_url:
        return False, "missing base_url"
    try:
        u = urlparse(base_url)
    except Exception as e:
        return False, f"unparseable: {e}"
    if u.scheme not in ("http", "https"):
        return False, f"scheme '{u.scheme}' not allowed (use http or https)"
    host = (u.hostname or "").lower()
    if not host:
        return False, "missing host"
    # When cloud endpoints are NOT explicitly enabled, restrict to
    # loopback hostnames. This is the safe default.
    cfg_dict = cfg.load_config()
    if not bool(cfg_dict.get("allow_cloud_endpoints", False)):
        if host not in ("localhost", "127.0.0.1", "::1"):
            return False, (
                f"host '{host}' blocked: enable Settings → Allow Cloud Endpoints "
                "to permit non-loopback LLM endpoints"
            )
    # Even with cloud endpoints enabled, block link-local and metadata
    # endpoints that no legitimate LLM provider uses but are common
    # SSRF probe targets.
    if host in (
        "169.254.169.254",  # AWS / GCP / Azure metadata
        "metadata.google.internal",
        "100.100.100.200",  # Alibaba metadata
    ):
        return (
            False,
            f"host '{host}' is a cloud-metadata endpoint and is always blocked",
        )
    return True, ""


@app.get("/settings/models")
async def settings_models(
    base_url: str = "", provider: str = "lmstudio", api_key: str = ""
):
    """Proxy list_models to the chosen local provider (never call from browser)."""
    from .llm.providers import get_provider

    if not base_url:
        return JSONResponse({"ok": False, "error": "missing base_url"}, status_code=400)
    ok, err = _validate_llm_base_url(base_url)
    if not ok:
        return JSONResponse({"ok": False, "error": err, "models": []}, status_code=400)
    # If the api_key arrives as the mask, resolve it from stored config.
    if api_key in ("***",):
        api_key = (cfg.load_config().get("llm") or {}).get("api_key") or ""
    p = get_provider(provider)
    try:
        models = p.list_models(base_url, api_key=api_key or None, timeout=5)
        return {"ok": True, "models": [m["id"] for m in models]}
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e), "models": []}, status_code=200
        )


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
    ok, err = _validate_llm_base_url(base_url)
    if not ok:
        return {"ok": False, "error": err, "latency_ms": 0}
    # Also count models to enrich the ✓ badge
    from .llm.providers import get_provider

    count = None
    try:
        count = len(
            get_provider(provider).list_models(
                base_url, api_key=api_key or None, timeout=3
            )
        )
    except Exception:
        count = None
    res = llm_ping(base_url, provider, model_id, api_key=api_key or None, timeout=15)
    res["model_count"] = count
    return res


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Origin check — the WebSocket /ws fan-outs full state (prompts,
    # queue, chat replies). Without an origin check, any web page the
    # user browses can `new WebSocket("ws://localhost:9099/ws")` and
    # harvest those events. Mirror the http CSRF middleware: allow
    # same-origin + the env-var allowlist; reject everything else.
    if os.environ.get(_CSRF_DISABLE_ENV) != "1":
        host = websocket.headers.get("host", "")
        allow = _trusted_origin_set(host)
        origin = websocket.headers.get("origin", "")
        if origin:
            try:
                from urllib.parse import urlparse

                u = urlparse(origin)
                if f"{u.scheme}://{u.netloc}" not in allow:
                    await websocket.close(code=4403)
                    return
            except Exception:
                await websocket.close(code=4403)
                return
        # Permit no-Origin for non-browser clients only when the host
        # header is loopback. A LAN cross-origin browser always sends
        # Origin; absent + non-loopback means we should reject.
        elif not (host.startswith("127.0.0.1") or host.startswith("localhost")):
            await websocket.close(code=4403)
            return

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
                _job_stage_actuals = (
                    {cur_v: _job_stage_actuals.get(cur_v, {})} if cur_v else {}
                )
                _job_track["video_index"] = cur_v
                _job_track["since"] = now_ts
            state["stage_started_ts"] = _stage_track["since"]
            state["job_started_ts"] = _job_track["since"]
            state["last_completed"] = _last_completed
            state["stage_actuals"] = _job_stage_actuals.get(cur_v, {}) if cur_v else {}
            _prev_state = state
            stats = get_sys_stats()
            # Record GPU usage for the scheduler's guard.
            from . import scheduler

            _sched_conf = scheduler._load_scheduler_config()
            _max_samples = int(_sched_conf.get("pause_idle_samples", 5))
            scheduler.GPU.record_gpu_usage(
                stats.get("gpu", 0), max_samples=_max_samples
            )
            # Notify the scheduler condition variable so it can re-check the idle guard.
            async with scheduler.GPU.cond:
                scheduler.GPU.cond.notify_all()

            queue = cfg.get_queue()
            # Auto-rotate: cancelled items older than 48 h drop out of the
            # visible queue. Cheap to compute on each tick.
            cutoff = time.time() - 48 * 3600
            kept = [
                x
                for x in queue
                if not (
                    x.get("status") == "cancelled"
                    and (x.get("cancelled_ts") or 0) < cutoff
                )
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
                if state.get("mode") != "Idle" and state.get("step"):
                    # No trailing "…" — the per-character assembly-line animation
                    # in the queue header is its own activity indicator; an
                    # ellipsis on top of bouncing letters reads as redundant.
                    _step_text_map = {
                        "Concept": "rewriting prompt",
                        "Base Image": "rendering image",
                        "Video Chains": "rendering video",
                        "Audio": "composing music",
                        "TTS": "recording voiceover",
                        "Post Process": "upscaling",
                        "Final Merge": "merging final",
                    }
                    _hb_text = _step_text_map.get(state["step"], "working")
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
                if f.endswith(".mp4") or f.endswith(".png"):
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
            async with _LLM_LOCK:
                raw = await asyncio.to_thread(
                    lmstudio_call,
                    sys_p,
                    "Give me 8 tangentially-related subject ideas.",
                )
            arr = []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    arr = parsed
            except Exception:
                a, b = raw.find("["), raw.rfind("]")
                if a != -1 and b > a:
                    try:
                        arr = json.loads(raw[a : b + 1])
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

    # Default to loopback so the dashboard (which has no auth and a wide
    # mutating API surface) is not exposed to the LAN by default. Operators
    # who explicitly want LAN / docker / proxy access set
    # `SLOPFINITY_BIND_HOST=0.0.0.0` (or the specific interface) — and
    # should pair that with `SLOPFINITY_TRUSTED_ORIGINS` so the CSRF
    # middleware accepts the new host.
    _bind_host = os.environ.get("SLOPFINITY_BIND_HOST", "127.0.0.1")
    _bind_port = int(os.environ.get("SLOPFINITY_BIND_PORT", "9099"))
    uvicorn.run(app, host=_bind_host, port=_bind_port)
