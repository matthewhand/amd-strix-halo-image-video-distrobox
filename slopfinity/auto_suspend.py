"""Auto-suspend dispatcher.

Generalizes PR #40's hardcoded LM Studio SIGSTOP/SIGCONT into a list of
services, each suspended via one of four methods: sigstop, rest_unload,
docker_stop, sigterm. Reads its config list from `config.auto_suspend`
and is fired by the scheduler on every GPU stage entry/exit.

Stdlib only — no new pip deps.

See docs/auto-suspend-design.md for the design rationale and the trade-off
table comparing methods.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def suspend_all(entries: list[dict]) -> list[dict]:
    """Run the configured suspend op for every enabled entry.

    Returns a list of per-entry result dicts shaped like
    `{"id": ..., "method": ..., "ok": True|False, "detail": "...",
       "error": "..." (if ok=False)}`.

    Errors are caught per entry — a failure on one entry does not stop the
    others. The caller (scheduler) emits these results as a websocket event
    so the dashboard can show what happened.
    """
    results: list[dict] = []
    for e in entries or []:
        if not isinstance(e, dict) or not e.get("enabled"):
            continue
        results.append(await _dispatch(e, suspending=True))
    return results


async def resume_all(entries: list[dict]) -> list[dict]:
    """Mirror of `suspend_all` for the resume side.

    `rest_unload` and `sigterm` are no-ops on resume by design — see the
    design doc's "Resume semantics are deliberately asymmetric" section.
    """
    results: list[dict] = []
    for e in entries or []:
        if not isinstance(e, dict) or not e.get("enabled"):
            continue
        results.append(await _dispatch(e, suspending=False))
    return results


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------

async def _dispatch(entry: dict, *, suspending: bool) -> dict:
    eid = entry.get("id") or "?"
    method = entry.get("method") or ""
    base = {"id": eid, "method": method}
    try:
        if method == "sigstop":
            detail = await _sigstop(entry, suspending=suspending)
        elif method == "rest_unload":
            detail = await _rest_unload(entry, suspending=suspending)
        elif method == "docker_stop":
            detail = await _docker_stop(entry, suspending=suspending)
        elif method == "sigterm":
            detail = await _sigterm(entry, suspending=suspending)
        else:
            return {**base, "ok": False, "error": f"unknown method: {method!r}"}
        return {**base, "ok": True, "detail": detail}
    except Exception as ex:
        return {**base, "ok": False, "error": str(ex)}


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

async def _sigstop(entry: dict, *, suspending: bool) -> dict:
    """SIGSTOP/SIGCONT every host PID matching `process_name` (pgrep -af)."""
    name = (entry.get("process_name") or "").strip()
    if not name:
        raise ValueError("sigstop entry missing 'process_name'")
    sig = signal.SIGSTOP if suspending else signal.SIGCONT
    pids = await asyncio.to_thread(_pgrep, name)
    affected: list[int] = []
    errors: list[str] = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            affected.append(pid)
        except (PermissionError, ProcessLookupError) as e:
            errors.append(f"pid {pid}: {e!s}")
    return {"pids": affected, "errors": errors, "match": name}


async def _rest_unload(entry: dict, *, suspending: bool) -> dict:
    """POST a body to `endpoint` on suspend; no-op on resume."""
    if not suspending:
        return {"skipped": "rest_unload has no resume action"}
    endpoint = (entry.get("endpoint") or "").strip()
    if not endpoint:
        raise ValueError("rest_unload entry missing 'endpoint'")
    body = entry.get("body")
    # Default body matches ComfyUI /free shape — covers the most common case.
    if body is None:
        body = {"unload_models": True, "free_memory": True}
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _do() -> int:
        with urllib.request.urlopen(req, timeout=5) as r:
            return int(r.status)

    status = await asyncio.to_thread(_do)
    return {"endpoint": endpoint, "status": status}


async def _docker_stop(entry: dict, *, suspending: bool) -> dict:
    """`docker stop <name>` on suspend; `docker start <name>` on resume."""
    container = (entry.get("container") or "").strip()
    if not container:
        raise ValueError("docker_stop entry missing 'container'")
    cmd = ["docker", "stop" if suspending else "start", container]

    def _run() -> dict:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {
            "rc": cp.returncode,
            "stdout": (cp.stdout or "").strip()[:200],
            "stderr": (cp.stderr or "").strip()[:200],
        }

    res = await asyncio.to_thread(_run)
    if res["rc"] != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> rc={res['rc']} stderr={res['stderr']!r}")
    return {"container": container, "action": cmd[1], **res}


async def _sigterm(entry: dict, *, suspending: bool) -> dict:
    """SIGTERM on suspend; no-op on resume (one-shot graceful shutdown)."""
    if not suspending:
        return {"skipped": "sigterm has no resume action"}
    name = (entry.get("process_name") or "").strip()
    if not name:
        raise ValueError("sigterm entry missing 'process_name'")
    pids = await asyncio.to_thread(_pgrep, name)
    affected: list[int] = []
    errors: list[str] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            affected.append(pid)
        except (PermissionError, ProcessLookupError) as e:
            errors.append(f"pid {pid}: {e!s}")
    return {"pids": affected, "errors": errors, "match": name}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pgrep(pattern: str) -> list[int]:
    """Return PIDs whose full command line matches `pattern` (regex via -af).

    Returns [] on any failure (pgrep missing, no matches, parse error).
    """
    try:
        out = subprocess.run(
            ["pgrep", "-af", pattern],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line.split()[0]))
        except (ValueError, IndexError):
            continue
    return pids


# ---------------------------------------------------------------------------
# Convenience: legacy single-entry helpers used by the /llm/suspend and
# /llm/resume REST endpoints. These keep PR #40's external API working
# even after the lmstudio-specific code path is removed from scheduler.py.
# ---------------------------------------------------------------------------

LEGACY_LMSTUDIO_ENTRY = {
    "id": "lmstudio",
    "label": "LLM (LM Studio)",
    "enabled": True,
    "method": "sigstop",
    "process_name": "LM Studio|lm-studio|ollama serve",
}


async def legacy_suspend_lmstudio() -> dict:
    """Used by POST /llm/suspend — keeps the fleet runner's external API."""
    res = await suspend_all([LEGACY_LMSTUDIO_ENTRY])
    return _legacy_shape(res, key="suspended")


async def legacy_resume_lmstudio() -> dict:
    """Used by POST /llm/resume — keeps the fleet runner's external API."""
    res = await resume_all([LEGACY_LMSTUDIO_ENTRY])
    return _legacy_shape(res, key="resumed")


def _legacy_shape(results: list[dict], *, key: str) -> dict:
    """Reshape a `suspend_all` result list into the PR #40 response shape.

    PR #40's endpoints returned `{"suspended": [pid, ...]}` /
    `{"resumed": [pid, ...]}`. We preserve that for any external caller.
    """
    pids: list[int] = []
    for r in results:
        d = r.get("detail") if isinstance(r, dict) else None
        if isinstance(d, dict):
            pids.extend(d.get("pids") or [])
    return {key: pids, "results": results}
