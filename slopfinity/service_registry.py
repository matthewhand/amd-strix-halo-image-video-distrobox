"""Network service registry — probe / ensure_up / ensure_down for toolbox workers.

Lifecycle only (compose/scripts). Generation stays on HTTP URLs.
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

# Repo root for compose commands (override with SLOPFINITY_COMPOSE_DIR).
def _compose_cwd() -> str:
    return os.environ.get("SLOPFINITY_COMPOSE_DIR") or os.getcwd()


Cmd = Union[str, Sequence[str]]


DEFAULT_NETWORK_SERVICES: List[Dict[str, Any]] = [
    {
        "id": "qwen-image",
        "label": "Qwen Image Studio",
        "enabled": True,
        "health_url": "http://127.0.0.1:8180/docs",
        "base_url": "http://127.0.0.1:8180",
        "base_url_env": "IMAGE_API_URL",
        "start_cmd": "docker compose --profile qwen-image up -d qwen-image-service",
        "stop_cmd": "docker stop strix-halo-qwen-image",
        "stage_keys": ["image:qwen", "image:*"],
        "budget_gb": 28,
        "exclusive_group": "uma-heavy",
        "ensure_timeout_s": 180,
    },
    {
        "id": "qwen-tts",
        "label": "Qwen/Kokoro TTS",
        "enabled": True,
        "health_url": "http://127.0.0.1:8010/health",
        "base_url": "http://127.0.0.1:8010",
        "base_url_env": "TTS_WORKER_URL",
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
        "base_url": "http://127.0.0.1:8011",
        "base_url_env": "HEARTMULA_URL",
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
        "base_url": "http://127.0.0.1:8188",
        "base_url_env": "SLOPFINITY_COMFY_URL",
        "start_cmd": "docker compose --profile comfyui up -d comfyui-service",
        "stop_cmd": "docker stop strix-halo-comfyui",
        "stage_keys": [
            "video:ltx-2.3", "video:wan2.2", "video:wan2.5", "video:*",
            "upscale:ltx-spatial", "upscale:*",
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
            # TTS_WORKER_URL often includes /tts — strip known suffixes for base.
            for suf in ("/tts", "/music", "/health"):
                if v.rstrip("/").endswith(suf):
                    return v.rstrip("/")[: -len(suf)] or v
            return v.rstrip("/")
    return (e.get("base_url") or "").rstrip("/")


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


def probe(service_id: str, *, timeout: float = 2.0) -> Dict[str, Any]:
    """GET health_url. ok=True if HTTP status < 500 (or connection succeeds)."""
    e = get_service(service_id)
    if not e:
        return {"ok": False, "id": service_id, "error": "unknown service"}
    url = (e.get("health_url") or "").strip()
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
            "detail": body[:80].decode("utf-8", errors="replace") if body else "",
        }
    except urllib.error.HTTPError as ex:
        latency = round((time.monotonic() - t0) * 1000, 1)
        # 404 on /docs still means the server is up
        ok = ex.code < 500
        return {
            "ok": ok,
            "id": service_id,
            "status": ex.code,
            "latency_ms": latency,
            "detail": str(ex.reason or ex),
        }
    except Exception as ex:
        latency = round((time.monotonic() - t0) * 1000, 1)
        return {
            "ok": False,
            "id": service_id,
            "status": 0,
            "latency_ms": latency,
            "error": str(ex),
        }


def _run_cmd(cmd: Cmd, *, timeout: float = 300.0) -> Dict[str, Any]:
    if not cmd:
        return {"ok": False, "error": "empty command"}
    if isinstance(cmd, str):
        argv = shlex.split(cmd)
        shell = False
    else:
        argv = list(cmd)
        shell = False
    try:
        proc = subprocess.run(
            argv,
            cwd=_compose_cwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
        )
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": (proc.stdout or "")[-400:],
            "stderr": (proc.stderr or "")[-400:],
            "cmd": argv,
        }
    except Exception as ex:
        return {"ok": False, "error": str(ex), "cmd": argv if "argv" in dir() else cmd}


def ensure_down(service_id: str) -> Dict[str, Any]:
    """Stop service (idempotent)."""
    e = get_service(service_id)
    if not e:
        return {"ok": False, "id": service_id, "error": "unknown service"}
    if not e.get("enabled", True):
        return {"ok": True, "id": service_id, "skipped": "disabled"}
    # Already down?
    p = probe(service_id, timeout=1.0)
    if not p.get("ok"):
        return {"ok": True, "id": service_id, "already_down": True, "probe": p}
    stop = e.get("stop_cmd") or ""
    result = _run_cmd(stop, timeout=120.0)
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


def ensure_up(
    service_id: str,
    *,
    timeout_s: Optional[float] = None,
    poll_s: float = 2.0,
    stop_exclusive: bool = True,
) -> Dict[str, Any]:
    """Bring service healthy; start_cmd if needed; wait until probe ok."""
    e = get_service(service_id)
    if not e:
        return {"ok": False, "id": service_id, "error": "unknown service"}
    if not e.get("enabled", True):
        return {"ok": False, "id": service_id, "error": "service disabled in config"}

    p0 = probe(service_id, timeout=2.0)
    if p0.get("ok"):
        return {"ok": True, "id": service_id, "already_up": True, "probe": p0}

    peer_results = []
    if stop_exclusive:
        for peer in _exclusive_peers(service_id):
            peer_results.append(ensure_down(peer))

    start = e.get("start_cmd") or ""
    start_result = _run_cmd(start, timeout=float(timeout_s or e.get("ensure_timeout_s") or 120) + 30)
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
    """Resolve stage/model → service and ensure_up."""
    sid = service_for_stage(stage, model)
    if not sid:
        return {"ok": True, "skipped": True, "reason": f"no service for {stage}:{model}"}
    return ensure_up(sid, **kwargs)


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
            "health_url": e.get("health_url"),
            "base_url": base_url_for(sid),
            "up": bool(p.get("ok")),
            "probe": p,
        })
    return out
