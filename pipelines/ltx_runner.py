"""Thin wrapper around scripts.comfyui_api for LTX submission + waiting.

Replaces the inline polling loop currently duplicated in test_chained_wave.
"""
import json
import os
import sys
import time
import urllib.request
import uuid

from . import config

# Make scripts/ importable regardless of caller cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import comfyui_api  # noqa: E402


def submit(workflow, *, server=None, client_id=None):
    """POST a workflow. Returns (prompt_id, client_id) on success, raises on HTTP error."""
    server = server or config.SERVER
    client_id = client_id or str(uuid.uuid4())
    resp = comfyui_api.submit(workflow, server, client_id)
    return resp.get("prompt_id"), client_id


def submit_and_wait(workflow, *, server=None, poll_interval=15):
    """Submit a workflow and block until it completes via /history polling.

    Returns ("success", None) on success, ("error", message) on failure,
    ("error", "submit") if the POST itself failed.
    """
    server = server or config.SERVER
    try:
        pid, _client_id = submit(workflow, server=server)
    except RuntimeError as e:
        return ("error", f"submit: {e}")

    start = time.time()
    while True:
        try:
            with urllib.request.urlopen(
                f"http://{server}/history/{pid}", timeout=5
            ) as r:
                h = json.loads(r.read().decode())
                if h:
                    status = list(h.values())[0].get("status", {})
                    s = status.get("status_str")
                    if s == "success":
                        elapsed = int(time.time() - start)
                        print(f"  DONE in {elapsed}s")
                        return ("success", None)
                    if s == "error":
                        msgs = status.get("messages", [])
                        for m in msgs:
                            if m[0] == "execution_error":
                                err = (
                                    f"{m[1].get('exception_type')} - "
                                    f"{m[1].get('exception_message')}"
                                )
                                print(f"  ERROR: {err}")
                                return ("error", err)
                        return ("error", "unknown")
        except Exception:
            pass
        time.sleep(poll_interval)
