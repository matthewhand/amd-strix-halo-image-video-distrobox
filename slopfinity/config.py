import json
import os
import time

from . import queue_schema

# In-container slopfinity has cwd=/ but the bind-mount lives at /workspace,
# so a relative path like "comfy-outputs/..." silently resolves to a ghost
# file inside the container. Honour SLOPFINITY_STATE_DIR (set to /workspace
# in docker-compose.override.yaml) and fall back to the legacy relative path
# for host-side fleet runners.
_STATE_DIR = os.environ.get("SLOPFINITY_STATE_DIR") or "comfy-outputs/experiments"
CONFIG_FILE = os.path.join(_STATE_DIR, "config.json")
STATE_FILE = os.path.join(_STATE_DIR, "state.json")
QUEUE_FILE = os.path.join(_STATE_DIR, "queue.json")

# System prompt used by run_philosophical_experiments.py when calling the
# local LLM to dream up each fleet video idea. Kept here (not in the gitignored
# runner) so the dashboard can override it via Settings → LLM → Generation.
# Must remain byte-identical to the runner's hardcoded fallback.
DEFAULT_PHILOSOPHICAL_PROMPT = "You are a master cinematic concept artist."

# Defaults for the auto-suggest LLM call (the 🎲 Suggest button on Subjects).
# `suggest_use_subjects` controls whether we feed the LLM the user's current
# Subjects textarea content as style/theme context. `suggest_custom_prompt`
# overrides the built-in suggestion system prompt entirely (empty = default).
DEFAULT_SUGGEST_USE_SUBJECTS = True
DEFAULT_SUGGEST_CUSTOM_PROMPT = ""
# When True, every automatic /subjects/suggest fetch path on the dashboard
# (page-load tryAutoSuggest, hover/scroll/idle prefetch) bails early. The
# manual 🎲 Suggest button stays exempt — explicit user intent always wins.
DEFAULT_SUGGEST_AUTO_DISABLED = False

# Auto-suspend list — see docs/auto-suspend-design.md. Each entry pairs a
# co-resident service with one of four suspension methods. The scheduler
# fires `auto_suspend.suspend_all(...)` on every GPU stage entry and
# `resume_all(...)` on exit.
DEFAULT_AUTO_SUSPEND = [
    {"id": "lmstudio", "label": "LLM (LM Studio)", "enabled": True,
     "method": "sigstop", "process_name": "LM Studio"},
    {"id": "comfyui", "label": "ComfyUI", "enabled": False,
     "method": "rest_unload", "endpoint": "http://localhost:8188/free"},
    {"id": "qwen-tts", "label": "Qwen-TTS worker", "enabled": False,
     "method": "docker_stop", "container": "strix-halo-qwen-tts"},
    {"id": "ollama", "label": "Ollama LLM", "enabled": False,
     "method": "sigstop", "process_name": "ollama"},
]

DEFAULT_CONFIG = {
    "base_model": "ltx-2.3",
    "video_model": "ltx-2.3",
    "upscale": False,
    "frames": 49,
    "size": "1280*720",
    "enhancer_prompt": "You are a master cinematic director. Rewrite the user's prompt into a highly detailed, visually evocative description for an AI video generator. Focus on lighting, texture, and mood. Keep it under 60 words.",
    "infinity_mode": False,
    "infinity_themes": ["existential crisis of lumpy clay robots", "cyberpunk dragons in neon rain"],
    "infinity_index": 0,
    "chains": 10,
    "tier": "auto",
    "consolidation": "overlay",
    "music_gain_db": 0,
    "fade_s": 0.5,
    "chaos_mode": False,
    "chaos_interval_s": 180,
    "when_idle": False,
    "philosophical_prompt": None,
    "suggest_use_subjects": DEFAULT_SUGGEST_USE_SUBJECTS,
    "suggest_custom_prompt": DEFAULT_SUGGEST_CUSTOM_PROMPT,
    "suggest_auto_disabled": DEFAULT_SUGGEST_AUTO_DISABLED,
    "auto_suspend": DEFAULT_AUTO_SUSPEND,
}


def get_philosophical_prompt(config=None):
    """Resolve the fleet system prompt: stored override or built-in default.

    A stored value of None or "" means "use the default", so the runner and
    template renderer can call this without branching on emptiness.
    """
    c = config if isinstance(config, dict) else load_config()
    val = c.get("philosophical_prompt")
    if val is None or val == "":
        return DEFAULT_PHILOSOPHICAL_PROMPT
    return val

def _merge_auto_suspend(stored):
    """Merge canonical DEFAULT_AUTO_SUSPEND entries by id into `stored`.

    User edits to existing entries (enabled/method/process_name/...) win.
    New canonical entries (added in a later release) appear automatically.
    Unknown user-added entries (custom services) are preserved.
    """
    if not isinstance(stored, list):
        return list(DEFAULT_AUTO_SUSPEND)
    by_id = {e.get("id"): e for e in stored if isinstance(e, dict) and e.get("id")}
    out = []
    seen = set()
    # Canonical order first.
    for d in DEFAULT_AUTO_SUSPEND:
        eid = d["id"]
        out.append(by_id.get(eid, dict(d)))
        seen.add(eid)
    # Append any user-only / custom entries the canonical list doesn't know.
    for e in stored:
        if isinstance(e, dict) and e.get("id") and e["id"] not in seen:
            out.append(e)
    return out


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                c = json.load(f)
                for k, v in DEFAULT_CONFIG.items():
                    if k not in c: c[k] = v
                c["auto_suspend"] = _merge_auto_suspend(c.get("auto_suspend"))
                return c
        except: pass
    return dict(DEFAULT_CONFIG, auto_suspend=list(DEFAULT_AUTO_SUSPEND))

SENSITIVE_KEYS = {"api_key"}


def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    # Config may contain API keys; restrict to owner-only.
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except Exception:
        pass


def redact(config):
    """Return a deep copy with sensitive fields blanked for broadcast/transit."""
    import copy
    c = copy.deepcopy(config) if isinstance(config, dict) else config
    if isinstance(c, dict):
        llm = c.get("llm")
        if isinstance(llm, dict) and llm.get("api_key"):
            llm["api_key"] = "***"
        if c.get("api_key"):
            c["api_key"] = "***"
    return c

def get_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: return json.load(f)
        except: pass
    return {"mode": "Idle", "step": "Waiting", "video_index": 0, "total_videos": 0, "chain_index": 0, "total_chains": 0, "current_prompt": "None"}

def set_state(mode="Idle", step="Waiting", video=0, total=0, chain=0, total_chains=0, prompt=""):
    s = {"mode": mode, "step": step, "video_index": video, "total_videos": total, "chain_index": chain, "total_chains": total_chains, "current_prompt": prompt, "ts": time.time()}
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f: json.dump(s, f)

def get_queue():
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, "r") as f:
                items = json.load(f)
            changed = False
            for item in items:
                before = item.get("schema_version")
                queue_schema.migrate_legacy(item)
                if before != queue_schema.SCHEMA_VERSION:
                    changed = True
            if changed:
                save_queue(items)
            return items
        except Exception:
            pass
    return []

def save_queue(q):
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
    with open(QUEUE_FILE, "w") as f: json.dump(q, f)
