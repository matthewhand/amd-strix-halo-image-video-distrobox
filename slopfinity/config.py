import json
import os
import time

CONFIG_FILE = "comfy-outputs/experiments/config.json"
STATE_FILE = "comfy-outputs/experiments/state.json"
QUEUE_FILE = "comfy-outputs/experiments/queue.json"

DEFAULT_CONFIG = {
    "base_model": "ltx-2.3",
    "video_model": "ltx-2.3",
    "upscale": False,
    "frames": 49,
    "size": "1280*720",
    "enhancer_prompt": "You are a master cinematic director. Rewrite the user's prompt into a highly detailed, visually evocative description for an AI video generator. Focus on lighting, texture, and mood. Keep it under 60 words.",
    "infinity_mode": False,
    "infinity_themes": ["existential crisis of lumpy clay robots", "cyberpunk dragons in neon rain"],
    "infinity_index": 0
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                c = json.load(f)
                for k, v in DEFAULT_CONFIG.items():
                    if k not in c: c[k] = v
                return c
        except: pass
    return DEFAULT_CONFIG

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
            with open(QUEUE_FILE, "r") as f: return json.load(f)
        except: pass
    return []

def save_queue(q):
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
    with open(QUEUE_FILE, "w") as f: json.dump(q, f)
