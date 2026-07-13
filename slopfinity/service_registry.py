"""Network service registry — probe / ensure_up / ensure_down for toolbox workers.

Lifecycle only (compose/scripts/docker start|stop). Generation stays on HTTP URLs.
See docs/network-service-lifecycle-design.md.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Union

Cmd = Union[str, Sequence[str]]

# Display / legacy stage labels → registry stage keys
_STAGE_ALIASES = {
    "base image": "image",
    "base_image": "image",
    "video chains": "video",
    "video_chains": "video",
    "post process": "upscale",
    "post_process": "upscale",
    "post": "upscale",
    "tts": "tts",
    "image": "image",
    "video": "video",
    "audio": "audio",
    "upscale": "upscale",
    "merge": "merge",
    "concept": "concept",
}

# One-shot GPU jobs with no long-lived HTTP service — free uma-heavy peers first.
_ONESHOT_UMA_STAGES = frozenset({"image", "video", "upscale"})


def _compose_cwd() -> str:
    return os.environ.get("SLOPFINITY_COMPOSE_DIR") or os.getcwd()


def normalize_stage(stage: str) -> str:
    """Map UI/legacy labels (e.g. 'Base Image', 'TTS') to registry stage keys."""
    s = (stage or "").strip()
    if not s:
        return s
    return _STAGE_ALIASES.get(s.lower(), s.lower() if s.lower() in _STAGE_ALIASES.values() else s)


DEFAULT_NETWORK_SERVICES: List[Dict[str, Any]] = [
    {
        "id": "qwen-image",
        "label": "Qwen Image Studio",
        "enabled": True,
        "health_url": "http://127.0.0.1:8180/docs",
        "health_path": "/docs",
        "base_url": "http://127.0.0.1:8180",
        "base_url_env": "IMAGE_API_URL",
        "container_name": "strix-halo-qwen-image",
        "compose_service": "qwen-image-service",
        "compose_profile": "qwen-image",
        "lifecycle_mode": "compose",
        "start_cmd": "docker compose --profile qwen-image up -d qwen-image-service",
        "stop_cmd": "docker stop strix-halo-qwen-image",
        # Exact only — ernie/ltx must not warm qwen-image via image:*
        "stage_keys": ["image:qwen"],
        "budget_gb": 28,
        "exclusive_group": "uma-heavy",
        "ensure_timeout_s": 180,
    },
    {
        "id": "qwen-tts",
        "label": "Qwen/Kokoro TTS",
        "enabled": True,
        "health_url": "http://127.0.0.1:8010/health",
        "health_path": "/health",
        "base_url": "http://127.0.0.1:8010",
        "base_url_env": "TTS_WORKER_URL",
        "container_name": "strix-halo-qwen-tts",
        "compose_service": "qwen-tts-service",
        "compose_profile": "qwen-tts",
        "lifecycle_mode": "compose",
        "start_cmd": "docker compose --profile qwen-tts up -d qwen-tts-service",
        "stop_cmd": "docker stop strix-halo-qwen-tts",
        "stage_keys": ["tts:qwen-tts", "tts:kokoro", "tts:*"],
        "budget_gb": 10,
        "exclusive_group": "",
        "ensure_timeout_s": 120,
    },
    {
        "id": "heartmula",
        "label": "HeartMuLa music",
        "enabled": True,
        "health_url": "http://127.0.0.1:8011/health",
        "health_path": "/health",
        "base_url": "http://127.0.0.1:8011",
        "base_url_env": "HEARTMULA_URL",
        "container_name": "strix-halo-heartmula",
        "compose_service": "heartmula-service",
        "compose_profile": "heartmula",
        "lifecycle_mode": "compose",
        "start_cmd": "docker compose --profile heartmula up -d heartmula-service",
        "stop_cmd": "docker stop strix-halo-heartmula",
        "stage_keys": ["audio:heartmula", "audio:*"],
        "budget_gb": 14,
        "exclusive_group": "uma-heavy",
        "ensure_timeout_s": 180,
    },
    {
        "id": "comfyui",
        "label": "ComfyUI",
        "enabled": True,
        "health_url": "http://127.0.0.1:8188/system_stats",
        "health_path": "/system_stats",
        "base_url": "http://127.0.0.1:8188",
        "base_url_env": "SLOPFINITY_COMFY_URL",
        "container_name": "strix-halo-comfyui",
        "compose_service": "comfyui-service",
        "compose_profile": "comfyui",
        "lifecycle_mode": "compose",
        "start_cmd": "docker compose --profile comfyui up -d comfyui-service",
        "stop_cmd": "docker stop strix-halo-comfyui",
        # LTX only — wan stays ephemeral docker run (do not warm Comfy for wan)
        "stage_keys": [
            "image:ltx-2.3",
            "video:ltx-2.3",
            "upscale:ltx-spatial",
            "upscale:*",
        ],
        "budget_gb": 12,
        "exclusive_group": "uma-heavy",
        "ensure_timeout_s": 180,
    },
]


def merge_network_services(stored: Any) -> List[Dict[str, Any]]:
    """Merge user config with defaults by id (same pattern as auto_suspend)."""
    if not isinstance(stored, list):
        return [dict(e) for e in DEFAULT_NETWORK_SERVICES]
    by_id = {e.get("id"): e for e in stored if isinstance(e, dict) and e.get("id")}
    out: List[Dict[str, Any]] = []
    seen = set()
    for d in DEFAULT_NETWORK_SERVICES:
        eid = d["id"]
        if eid in by_id:
            merged = dict(d)
            merged.update({k: v for k, v in by_id[eid].items() if v is not None})
            out.append(merged)
        else:
            out.append(dict(d))
        seen.add(eid)
    for e in stored:
        if isinstance(e, dict) and e.get("id") and e["id"] not in seen:
            out.append(e)
    return out


def _entries() -> List[Dict[str, Any]]:
    try:
        from . import config as cfg
        c = cfg.load_config()
        return merge_network_services(c.get("network_services"))
    except Exception:
        return [dict(e) for e in DEFAULT_NETWORK_SERVICES]


def get_service(service_id: str) -> Optional[Dict[str, Any]]:
    for e in _entries():
        if e.get("id") == service_id:
            return e
    return None


def base_url_for(service_id: str) -> str:
    """Resolve public base URL: env wins, then config base_url."""
    e = get_service(service_id) or {}
    env_key = (e.get("base_url_env") or "").strip()
    if env_key:
        v = (os.environ.get(env_key) or "").strip()
        if v:
            for suf in ("/tts", "/music", "/health"):
                if v.rstrip("/").endswith(suf):
                    return v.rstrip("/")[: -len(suf)] or v
            return v.rstrip("/")
    return (e.get("base_url") or "").rstrip("/")


def health_url_for(service_id: str) -> str:
    """Probe URL: explicit health_url, else base + health_path (remote-friendly)."""
    e = get_service(service_id) or {}
    explicit = (e.get("health_url") or "").strip()
    # When env overrides base to a non-loopback host, prefer derived health so
    # operators can set only IMAGE_API_URL / TTS_WORKER_URL / etc.
    base = base_url_for(service_id)
    path = (e.get("health_path") or "/health").strip() or "/health"
    if not path.startswith("/"):
        path = "/" + path
    if base:
        derived = base.rstrip("/") + path
        if explicit:
            # Prefer derived when env moved base off loopback but health still 127.0.0.1
            if "127.0.0.1" in explicit or "localhost" in explicit:
                if "127.0.0.1" not in base and "localhost" not in base:
                    return derived
            return explicit
        return derived
    return explicit


def _match_stage_key(key: str, stage: str, model: str) -> bool:
    if ":" not in key:
        return key == stage
    st, md = key.split(":", 1)
    if st != stage and st != "*":
        return False
    if md == "*" or md == model:
        return True
    return False


def service_for_stage(stage: str, model: str = "") -> Optional[str]:
    """Return service id matching stage:model, preferring exact over wildcard."""
    stage = normalize_stage(stage)
    exact: Optional[str] = None
    wild: Optional[str] = None
    for e in _entries():
        if not e.get("enabled", True):
            continue
        for key in e.get("stage_keys") or []:
            if not _match_stage_key(str(key), stage, model or ""):
                continue
            if str(key).endswith(":*") or str(key) == f"{stage}:*":
                wild = wild or e.get("id")
            else:
                exact = e.get("id")
                break
        if exact:
            break
    return exact or wild


def _docker_env(entry: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Env for lifecycle subprocesses (inherits parent + optional overrides)."""
    env = os.environ.copy()
    e = entry or {}
    host = (e.get("docker_host") or os.environ.get("SLOPFINITY_DOCKER_HOST") or "").strip()
    ctx = (e.get("docker_context") or os.environ.get("SLOPFINITY_DOCKER_CONTEXT") or "").strip()
    if host:
        env["DOCKER_HOST"] = host
    if ctx:
        env["DOCKER_CONTEXT"] = ctx
    return env


def _docker_bin() -> str:
    return (os.environ.get("SLOPFINITY_DOCKER_BIN") or "docker").strip() or "docker"


def _prefix_docker_argv(argv: List[str], entry: Optional[Dict[str, Any]] = None) -> List[str]:
    """Rewrite leading `docker` to use configured bin + optional --context."""
    if not argv:
        return argv
    e = entry or {}
    ctx = (e.get("docker_context") or os.environ.get("SLOPFINITY_DOCKER_CONTEXT") or "").strip()
    dbin = _docker_bin()
    out = list(argv)
    if out[0] == "docker":
        out[0] = dbin
        if ctx:
            out = [out[0], "--context", ctx] + out[1:]
    return out


def resolve_start_cmd(entry: Dict[str, Any]) -> Optional[Cmd]:
    """Synthesize start command from lifecycle_mode + structured fields."""
    mode = (entry.get("lifecycle_mode") or "compose").strip().lower()
    if mode == "none":
        return None
    name = (entry.get("container_name") or "").strip()
    if mode == "container":
        if name:
            return ["docker", "start", name]
        return entry.get("start_cmd") or None
    if mode == "cmd":
        return entry.get("start_cmd") or None
    # compose (default)
    if entry.get("start_cmd"):
        return entry["start_cmd"]
    profile = (entry.get("compose_profile") or "").strip()
    svc = (entry.get("compose_service") or "").strip()
    if profile and svc:
        return f"docker compose --profile {profile} up -d {svc}"
    return None


def resolve_stop_cmd(entry: Dict[str, Any]) -> Optional[Cmd]:
    mode = (entry.get("lifecycle_mode") or "compose").strip().lower()
    if mode == "none":
        return None
    name = (entry.get("container_name") or "").strip()
    if name and mode in ("container", "compose"):
        return ["docker", "stop", name]
    if mode == "cmd":
        return entry.get("stop_cmd") or None
    return entry.get("stop_cmd") or (["docker", "stop", name] if name else None)


def probe(service_id: str, *, timeout: float = 2.0) -> Dict[str, Any]:
    """GET health URL. ok=True if HTTP status < 500 (or connection succeeds)."""
    e = get_service(service_id)
    if not e:
        return {"ok": False, "id": service_id, "error": "unknown service"}
    url = health_url_for(service_id)
    if not url:
        return {"ok": False, "id": service_id, "error": "no health_url"}
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200) or 200
            body = resp.read(256)
        latency = round((time.monotonic() - t0) * 1000, 1)
        ok = int(status) < 500
        return {
            "ok": ok,
            "id": service_id,
            "status": int(status),
            "latency_ms": latency,
            "url": url,
            "detail": body[:80].decode("utf-8", errors="replace") if body else "",
        }
    except urllib.error.HTTPError as ex:
        latency = round((time.monotonic() - t0) * 1000, 1)
        ok = ex.code < 500
        return {
            "ok": ok,
            "id": service_id,
            "status": ex.code,
            "latency_ms": latency,
            "url": url,
            "detail": str(ex.reason or ex),
        }
    except Exception as ex:
        latency = round((time.monotonic() - t0) * 1000, 1)
        return {
            "ok": False,
            "id": service_id,
            "status": 0,
            "latency_ms": latency,
            "url": url,
            "error": str(ex),
        }


def _run_cmd(
    cmd: Cmd,
    *,
    timeout: float = 300.0,
    entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not cmd:
        return {"ok": False, "error": "empty command"}
    if isinstance(cmd, str):
        argv = shlex.split(cmd)
    else:
        argv = list(cmd)
    argv = _prefix_docker_argv(argv, entry)
    env = _docker_env(entry)
    try:
        proc = subprocess.run(
            argv,
            cwd=_compose_cwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=env,
        )
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": (proc.stdout or "")[-400:],
            "stderr": (proc.stderr or "")[-400:],
            "cmd": argv,
        }
    except Exception as ex:
        return {"ok": False, "error": str(ex), "cmd": argv}


def ensure_down(service_id: str) -> Dict[str, Any]:
    """Stop service (idempotent)."""
    e = get_service(service_id)
    if not e:
        return {"ok": False, "id": service_id, "error": "unknown service"}
    if not e.get("enabled", True):
        return {"ok": True, "id": service_id, "skipped": "disabled"}
    p = probe(service_id, timeout=1.0)
    if not p.get("ok"):
        return {"ok": True, "id": service_id, "already_down": True, "probe": p}
    stop = resolve_stop_cmd(e)
    if stop is None:
        return {"ok": True, "id": service_id, "skipped": "lifecycle_mode=none"}
    result = _run_cmd(stop, timeout=120.0, entry=e)
    return {"ok": bool(result.get("ok")), "id": service_id, "action": "stop", **result}


def _exclusive_peers(service_id: str) -> List[str]:
    e = get_service(service_id) or {}
    group = (e.get("exclusive_group") or "").strip()
    if not group:
        return []
    peers = []
    for o in _entries():
        if o.get("id") == service_id:
            continue
        if (o.get("exclusive_group") or "").strip() == group and o.get("enabled", True):
            peers.append(o["id"])
    return peers


def ensure_down_group(group: str) -> Dict[str, Any]:
    """Stop every enabled service in an exclusive_group (e.g. uma-heavy)."""
    group = (group or "").strip()
    results = []
    for o in _entries():
        if not o.get("enabled", True):
            continue
        if (o.get("exclusive_group") or "").strip() != group:
            continue
        results.append(ensure_down(o["id"]))
    return {"ok": all(r.get("ok") for r in results) if results else True, "group": group, "results": results}


def ensure_up(
    service_id: str,
    *,
    timeout_s: Optional[float] = None,
    poll_s: float = 2.0,
    stop_exclusive: bool = True,
) -> Dict[str, Any]:
    """Bring service healthy; start if needed; wait until probe ok.

    When stop_exclusive is True, always park exclusive-group peers first —
    including when the target is already healthy — so sequenced UMA stages
    never leave contending workers warm (e.g. heartmula while qwen-image runs).
    """
    e = get_service(service_id)
    if not e:
        return {"ok": False, "id": service_id, "error": "unknown service"}
    if not e.get("enabled", True):
        return {"ok": False, "id": service_id, "error": "service disabled in config"}

    peer_results = []
    if stop_exclusive:
        for peer in _exclusive_peers(service_id):
            peer_results.append(ensure_down(peer))

    p0 = probe(service_id, timeout=2.0)
    if p0.get("ok"):
        return {
            "ok": True,
            "id": service_id,
            "already_up": True,
            "probe": p0,
            "peers": peer_results,
        }

    start = resolve_start_cmd(e)
    if start is None:
        # lifecycle_mode=none — wait only
        start_result: Dict[str, Any] = {"ok": True, "skipped": "lifecycle_mode=none"}
    else:
        start_result = _run_cmd(
            start,
            timeout=float(timeout_s or e.get("ensure_timeout_s") or 120) + 30,
            entry=e,
        )
        if not start_result.get("ok"):
            return {
                "ok": False,
                "id": service_id,
                "error": "start_cmd failed",
                "start": start_result,
                "peers": peer_results,
            }

    deadline = time.monotonic() + float(timeout_s or e.get("ensure_timeout_s") or 120)
    last = p0
    while time.monotonic() < deadline:
        last = probe(service_id, timeout=2.0)
        if last.get("ok"):
            return {
                "ok": True,
                "id": service_id,
                "started": True,
                "probe": last,
                "start": start_result,
                "peers": peer_results,
            }
        time.sleep(poll_s)

    return {
        "ok": False,
        "id": service_id,
        "error": "timed out waiting for health",
        "probe": last,
        "start": start_result,
        "peers": peer_results,
    }


def ensure_for_stage(stage: str, model: str = "", **kwargs: Any) -> Dict[str, Any]:
    """Resolve stage/model → ensure_up service, or free uma-heavy for one-shots."""
    stage_n = normalize_stage(stage)
    sid = service_for_stage(stage_n, model)
    if sid:
        return ensure_up(sid, **kwargs)
    # Ephemeral GPU jobs (ernie, wan, …): stop resident uma-heavy workers first.
    if stage_n in _ONESHOT_UMA_STAGES and (model or ""):
        cleared = ensure_down_group("uma-heavy")
        return {
            "ok": bool(cleared.get("ok")),
            "skipped": True,
            "reason": f"no service for {stage_n}:{model}; cleared uma-heavy",
            "oneshot": True,
            "clear": cleared,
        }
    return {"ok": True, "skipped": True, "reason": f"no service for {stage_n}:{model}"}


def status_all() -> List[Dict[str, Any]]:
    out = []
    for e in _entries():
        sid = e.get("id")
        if not sid:
            continue
        p = probe(sid, timeout=1.5)
        out.append({
            "id": sid,
            "label": e.get("label"),
            "enabled": e.get("enabled", True),
            "health_url": health_url_for(sid),
            "base_url": base_url_for(sid),
            "container_name": e.get("container_name"),
            "lifecycle_mode": e.get("lifecycle_mode") or "compose",
            "up": bool(p.get("ok")),
            "probe": p,
        })
    return out
