"""Tiny Slopfinity-style client that wires to the HTTP workers via env URLs.

Instead of calling launch scripts, Slopfinity sets
  HEARTMULA_URL=http://...  or TTS_WORKER_URL=...
and calls these helpers which POST/GET to the configured endpoint.

This module uses only stdlib (urllib) so it has no heavy deps.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return json.loads(body)
            except Exception:
                return {"ok": True, "raw": body, "status": resp.status}
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except Exception:
            err = {"error": e.reason or str(e)}
        err["status"] = e.code
        return err
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return json.loads(body)
            except Exception:
                return {"ok": True, "raw": body}
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except Exception:
            err = {"error": str(e)}
        err["status"] = e.code
        return err
    except Exception as e:
        return {"ok": False, "error": str(e)}


def music_from_env(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to HEARTMULA_URL/music (or /music on the base)."""
    base = os.environ.get("HEARTMULA_URL", "").rstrip("/")
    if not base:
        return {"ok": False, "error": "HEARTMULA_URL not set"}
    url = base if base.endswith("/music") else f"{base}/music"
    return _post_json(url, payload)


def tts_from_env(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to TTS_WORKER_URL/tts."""
    base = os.environ.get("TTS_WORKER_URL", "").rstrip("/")
    if not base:
        return {"ok": False, "error": "TTS_WORKER_URL not set"}
    url = base if base.endswith("/tts") else f"{base}/tts"
    return _post_json(url, payload)


def health_from_env(which: str = "heartmula") -> Dict[str, Any]:
    """GET /health using the appropriate *_URL env."""
    if which == "heartmula":
        base = os.environ.get("HEARTMULA_URL", "").rstrip("/")
        key = "HEARTMULA_URL"
    else:
        base = os.environ.get("TTS_WORKER_URL", "").rstrip("/")
        key = "TTS_WORKER_URL"
    if not base:
        return {"ok": False, "error": f"{key} not set"}
    url = base if base.endswith("/health") else f"{base}/health"
    return _get_json(url)
