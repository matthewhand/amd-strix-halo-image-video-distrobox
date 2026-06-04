import json
import logging
import os
import time
from datetime import datetime, timezone

from . import queue_schema

_log = logging.getLogger(__name__)

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

# ---------------------------------------------------------------------------
# Defaults for every other hardcoded prompt the pipeline ships with. Surfacing
# these via Settings → Prompts lets users tune output style without editing
# source. A blank/None override means "use built-in default", so the runtime
# loaders can always call the get_*() helper without branching on emptiness.
#
# IMPORTANT: when changing a default below, also update the matching hardcoded
# string in the call site (kept duplicated for byte-identity / hot-path use).
# ---------------------------------------------------------------------------

# `enhancer_prompt` was already a config key (concept-stage rewriter — used by
# slopfinity/workers/concept.py). The new Prompts tab points to the same key
# so Settings can edit it next to the others.

# Multi-stage fan-out system prompt — slopfinity/fanout.py + /enhance distribute.
DEFAULT_FANOUT_SYSTEM_PROMPT = (
    "You are a master multi-stage cinematic director. Produce STRICT JSON "
    "with keys image, video, music, tts. If a stage's seed text is "
    "non-empty, you MUST extend it. Encourage short, memorable aphorisms "
    "when expressing verbal or written statements. Under 40 words per stage. "
    "Return ONLY JSON."
)

# Fleet rewriter user-message template — run_fleet.py queue branch.
# `{seed}` is interpolated with the queued prompt.
DEFAULT_FLEET_USER_PROMPT_TEMPLATE = (
    "Rewrite as a detailed, visually evocative scene description for an AI "
    "image generator. Subject: {seed}. Under 40 words. Include a short, "
    "memorable aphorism as a written or verbal statement in the scene. "
    "Return ONLY the description, no preamble."
)

# Infinity-mode user-message template — run_fleet.py infinity branch.
# `{theme}` is interpolated with the rotating theme.
DEFAULT_INFINITY_USER_PROMPT_TEMPLATE = (
    "Describe a detailed, visually evocative cynical philosophical scene "
    "about: {theme}. Under 40 words. Include a short, memorable aphorism "
    "expressed verbally or written in the scene."
)

# Chaos-mode tangential-suggest system prompt — slopfinity/server.py chaos
# background loop. `{subjects_csv}` is interpolated with the current subjects
# (comma-joined, trimmed to 8).
DEFAULT_CHAOS_SUGGEST_SYSTEM_PROMPT = (
    "You are a concept artist for an AI video fleet. The user is currently "
    "working with these subjects: [{subjects_csv}]. Generate 8 NEW visual "
    "subject ideas that are TANGENTIALLY related. Encourage short, "
    "memorable aphorisms and surreal verbal concepts. 6-14 words each. "
    "Cynical, philosophical, surreal. Output ONLY a JSON array of strings, no prose."
)

# VOID-style fallback when no LLM is reachable — run_fleet.py.
# `{style}` is interpolated with a random art-style pick.
DEFAULT_VOID_FALLBACK_TEMPLATE = "A cynical {style} scene. Text: 'VOID'."

# Defaults for the auto-suggest LLM call (the 🎲 Suggest button on Subjects).
# `suggest_use_subjects` controls whether we feed the LLM the user's current
# Subjects textarea content as style/theme context. `suggest_custom_prompt`
# overrides the built-in suggestion system prompt entirely (empty = default).
DEFAULT_SUGGEST_USE_SUBJECTS = True
DEFAULT_SUGGEST_CUSTOM_PROMPT = ""

# Named suggestion prompts — power the per-mode suggestion-row UX:
#   Endless mode: each marquee row uses a DIFFERENT named prompt; the user
#                 can swap a row's prompt via the in-row chip and the row
#                 re-fetches with the new system prompt.
#   Simple mode:  every row uses the same default ("yes-and") so behaviour
#                 stays predictable.
#   Chat mode:    one batch (no rows), reply-suggestion variant.
#   Raw mode:     suggestions are skipped entirely (LLM-free workflow).
#
# Each entry: {id, title, system, active, builtin}. `builtin` flags the
# 5 starter prompts so a "Restore defaults" button can put them back without
# clobbering user-added entries.
DEFAULT_SUGGEST_PROMPTS = [
    {
        "id": "yes-and", "title": "Yes, and…", "active": True, "builtin": True,
        "system": (
            "You are a story editor. Given the SEED below, output exactly {n} short "
            "next-scene continuations that follow the seed's most natural, supportive "
            "trajectory — 'yes, and…' improv style. Each line ≤ 28 words, plain text, "
            "one per line, no numbering, no bullets, no quotes."
        ),
    },
    {
        "id": "plot-twist", "title": "Plot Twist", "active": True, "builtin": True,
        "system": (
            "You are a story editor. Given the SEED below, output exactly {n} short "
            "continuations that subvert its expected direction — introduce ONE "
            "unexpected element per line that recontextualizes the seed. "
            "Each line ≤ 28 words, plain text, one per line, no numbering, no bullets."
        ),
    },
    {
        "id": "surreal-imagination", "title": "Surreal Imagination", "active": True, "builtin": True,
        "system": (
            "You are a surrealist co-writer. Given the SEED below, output exactly {n} "
            "short continuations that pull the seed into the unreal — dreamlike images, "
            "objects behaving wrong, geometry breaking, time slipping — while keeping "
            "ONE concrete sensory detail per line as an anchor. Each line ≤ 28 words, "
            "plain text, one per line, no numbering, no bullets, no quotes."
        ),
    },
    {
        "id": "questioning-lens", "title": "Questioning Lens", "active": True, "builtin": True,
        "system": (
            "You are an inquisitive narrator. Given the SEED below, output exactly {n} "
            "short continuations that reframe it as a QUESTION the scene is asking — "
            "either a literal question someone in the scene voices, or a tacit question "
            "the camera/subject poses through gesture. Each line ≤ 28 words, plain "
            "text, one per line, no numbering, no bullets, no quotes."
        ),
    },
    {
        "id": "cynic", "title": "Cynic's Take", "active": False, "builtin": True,
        "system": (
            "You are a deadpan cynic. Given the SEED below, output exactly {n} short "
            "continuations that reframe it through a wry, slightly nihilist lens — "
            "without being mean. Each line ≤ 28 words, plain text, one per line, "
            "no numbering, no bullets, no quotes."
        ),
    },
    {
        "id": "wonder", "title": "Childlike Wonder", "active": False, "builtin": True,
        "system": (
            "You are a wide-eyed first-encounter narrator. Given the SEED below, "
            "output exactly {n} short continuations that reframe it as if seen for "
            "the first time, with awe. Each line ≤ 28 words, plain text, one per line, "
            "no numbering, no bullets, no quotes."
        ),
    },
]
# When True, every automatic /subjects/suggest fetch path on the dashboard
# (page-load tryAutoSuggest, hover/scroll/idle prefetch) bails early. The
# manual 🎲 Suggest button stays exempt — explicit user intent always wins.
DEFAULT_SUGGEST_AUTO_DISABLED = False

# When True, simple mode renders the per-row prompt-pill cluster (the
# "spiffy" lead chip showing the named suggest_prompts entry — Yes-and,
# Plot Twist, Concrete Detail, etc.) that endless mode has always had.
# Default OFF so simple mode stays simple; toggling ON also auto-seeds
# the first row using whichever prompt is currently selected in the
# subjects-suggest-prompt-name dropdown.
DEFAULT_SUGGEST_PER_ROW_PROMPTS = False

# Auto-suspend list — see docs/auto-suspend-design.md. Each entry pairs a
# co-resident service with one of four suspension methods. The scheduler
# fires `auto_suspend.suspend_all(...)` on every GPU stage entry and
# `resume_all(...)` on exit.
# Phase 5 — scheduler tuning. `use_planner` flips memory_planner from
# advisory (dashboard-only) to active: acquire_gpu consults the planner's
# resident-set hint before reserving budget so cached models don't reload.
DEFAULT_SCHEDULER = {
    "use_planner": False,
    "memory_safety_gb": 10,
    "llm_cpu_mode": "smart",
    "tts_cpu_mode": "smart",
}


DEFAULT_AUTO_SUSPEND = [
    {"id": "lmstudio", "label": "LLM (LM Studio)", "enabled": True,
     "method": "sigstop", "process_name": "LM Studio", "command": ""},
    {"id": "comfyui", "label": "ComfyUI", "enabled": False,
     "method": "rest_unload", "endpoint": "http://localhost:8188/free", "command": ""},
    {"id": "qwen-tts", "label": "Qwen-TTS worker", "enabled": False,
     "method": "docker_stop", "container": "strix-halo-qwen-tts", "command": ""},
    {"id": "ollama", "label": "Ollama LLM", "enabled": False,
     "method": "sigstop", "process_name": "ollama", "command": ""},
]

DEFAULT_CONFIG = {
    # qwen is the canonical, reliable base image model (standalone LTX-image has
    # no launcher); the rest of the codebase already falls back to 'qwen'.
    "base_model": "qwen",
    "video_model": "ltx-2.3",
    # Pipeline role defaults so fresh configs return "none" (a real skip
    # sentinel) instead of None for these keys.
    "audio_model": "none",
    "tts_model": "none",
    "upscale_model": "none",
    "upscale": False,
    "frames": 49,
    "size": "1280*720",
    "enhancer_prompt": "You are a master cinematic director. Rewrite the user's prompt into a highly detailed, visually evocative description for an AI video generator. Focus on lighting, texture, and mood. Encourage short, memorable aphorisms for any verbal or written elements. Keep it under 60 words.",
    "infinity_mode": False,
    "infinity_themes": ["existential crisis of lumpy clay robots", "cyberpunk dragons in neon rain"],
    "infinity_index": 0,
    "chains": 10,
    # Disk-low generation guard. Blocks /inject (and run_fleet's iter loop)
    # when EITHER threshold trips on the outputs partition:
    #   disk_min_pct  — fail when free %  ≤ this (default 1 = 1% free)
    #   disk_min_gb   — fail when free GB ≤ this (default 5 = 5 GB free)
    # Set either to 0 to disable that check. Both at 0 = guard off.
    "disk_min_pct": 1,
    "disk_min_gb": 5.0,
    # Multi-frame chain handoff — anchors the next chain's first K frames to
    # the previous chain's last K frames via stacked LTXVAddGuide nodes.
    # K=1 reverts to legacy single-last-frame chaining (drifts visibly).
    # K=4 is the consistency-vs-speed sweet spot per research; cap=8.
    "chain_handoff_keyframes": 4,
    # FILM VFI seam smoothing — post-process pass that regenerates the 4
    # frames straddling each chain boundary (last 2 of N-1 + first 2 of N)
    # via FILM frame interpolation (Fannovel16/ComfyUI-Frame-Interpolation).
    # Falls back silently if the node isn't installed.
    "smooth_chain_seams": True,
    "tier": "auto",
    "consolidation": "overlay",
    "music_gain_db": 0,
    "fade_s": 0.5,
    "chaos_mode": False,
    "chaos_interval_s": 180,
    "when_idle": False,
    "philosophical_prompt": None,
    # Prompts tab overrides (None = use built-in default). See get_*() below.
    "fanout_system_prompt": None,
    "fleet_user_prompt_template": None,
    "infinity_user_prompt_template": None,
    "chaos_suggest_system_prompt": None,
    "void_fallback_template": None,
    "suggest_use_subjects": DEFAULT_SUGGEST_USE_SUBJECTS,
    "suggest_custom_prompt": DEFAULT_SUGGEST_CUSTOM_PROMPT,
    "suggest_prompts": DEFAULT_SUGGEST_PROMPTS,
    "suggest_auto_disabled": DEFAULT_SUGGEST_AUTO_DISABLED,
    "suggest_per_row_prompts": DEFAULT_SUGGEST_PER_ROW_PROMPTS,
    "auto_suggest_enabled": True,
    "idle_throttle_pct": 5,
    "creativity_score": 5,
    "quality_score": 5,
    "concurrency_budget_gb": 0.0,
    "auto_suspend": DEFAULT_AUTO_SUSPEND,
    "scheduler": DEFAULT_SCHEDULER,
    # Gate for cloud LLM endpoints in the Settings → LLM provider dropdown.
    # Slopfinity ships local-only by default (False); flipping this on
    # surfaces any cloud providers registered in slopfinity/llm/. The
    # backend `list_providers()` registry currently only contains local
    # entries, so this is a forward-compatible toggle — see TODO in
    # static/app.js (filterProviderDropdown) for the client gate.
    "allow_cloud_endpoints": False,
    # Standalone-mode endpoint URLs. When non-empty, override the env-var
    # / hardcoded defaults so a user running Slopfinity outside the
    # bundled toolbox docker image can point it at their own services.
    # Defaults are the loopback URLs the toolbox compose file exposes;
    # the dashboard works out-of-the-box on the bundled stack and can
    # be re-pointed via Settings → Endpoints (or env vars at startup
    # for headless deployments). Validated through _validate_llm_base_url
    # at save time so an attacker who slips past CSRF (#142) can't
    # repoint TTS / ComfyUI at internal admin panels.
    "tts_worker_url": "http://localhost:8010/tts",
    "comfy_url": "http://localhost:8188",
    # Per-mode suggestion-length budgets (chars, enforced via JSON-schema
    # maxLength). Doubled v323→v324 after user feedback "the short
    # suggestions are too short". Endless beats can now run to short
    # sentences instead of telegraphic fragments; simple chips have room
    # for a real concept; chat replies can be a complete one-liner.
    "suggest_max_len_endless": 80,
    "suggest_max_len_simple": 80,
    "suggest_max_len_chat": 160,
    "show_date_time": False,
    # Badge and progress-bar colour mode.
    # "themed" (default): each stage uses its DaisyUI token (accent/secondary/
    #   warning/info/success) and naturally follows the active theme.
    # "custom": all model badges + progress-bar fills use badge_custom_color.
    "badge_theme": "themed",
    "badge_custom_color": "#7c3aed",
    "pausing_anim_style": "pulse",
}


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


def _resolve_prompt(config, key, default):
    """Generic resolver: stored config[key] override, falling back to default.

    Empty string and None both mean "use built-in default" so callers can
    always blindly call get_xxx() without an emptiness check.
    """
    c = config if isinstance(config, dict) else load_config()
    val = c.get(key)
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    return val


def get_fanout_system_prompt(config=None):
    return _resolve_prompt(config, "fanout_system_prompt", DEFAULT_FANOUT_SYSTEM_PROMPT)


def get_fleet_user_prompt_template(config=None):
    return _resolve_prompt(config, "fleet_user_prompt_template", DEFAULT_FLEET_USER_PROMPT_TEMPLATE)


def get_infinity_user_prompt_template(config=None):
    return _resolve_prompt(config, "infinity_user_prompt_template", DEFAULT_INFINITY_USER_PROMPT_TEMPLATE)


def get_chaos_suggest_system_prompt(config=None):
    return _resolve_prompt(config, "chaos_suggest_system_prompt", DEFAULT_CHAOS_SUGGEST_SYSTEM_PROMPT)


def get_void_fallback_template(config=None):
    return _resolve_prompt(config, "void_fallback_template", DEFAULT_VOID_FALLBACK_TEMPLATE)

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


from .db import engine, init_db as _init_db_raw
from .models import Configuration, QueueItem
from sqlmodel import Session, select

_DB_INITIALIZED = False

def init_db():
    """Idempotent database initialization."""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    _init_db_raw()
    _DB_INITIALIZED = True


def load_config():
    """Load configuration from the database, falling back to JSON if available."""
    # Ensure DB is initialized
    init_db()
    
    with Session(engine) as session:
        # Load all keys from DB
        statement = select(Configuration)
        results = session.exec(statement).all()
        c = {item.key: item.value for item in results}
        
        # Merge defaults for missing keys
        for k, v in DEFAULT_CONFIG.items():
            if k not in c:
                c[k] = v
        
        # Backward compat / First-time migration fallback
        if not results and os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    file_data = json.load(f)
                    for k, v in file_data.items():
                        if k not in c:
                            c[k] = v
                            # Save to DB for future use
                            session.add(Configuration(key=k, value=v))
                    session.commit()
            except Exception:
                pass # Permission issues or malformed JSON

        c["auto_suspend"] = _merge_auto_suspend(c.get("auto_suspend"))
        # Merge scheduler defaults so older configs gain new keys.
        stored_sched = c.get("scheduler") if isinstance(c.get("scheduler"), dict) else {}
        c["scheduler"] = {**DEFAULT_SCHEDULER, **stored_sched}
        return c

def save_config(config):
    """Save configuration to the database."""
    init_db()
    with Session(engine) as session:
        for key, value in config.items():
            existing = session.get(Configuration, key)
            if existing:
                existing.value = value
                existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(existing)
            else:
                session.add(Configuration(key=key, value=value))
        session.commit()
    
    # Optional: Keep JSON in sync as a backup (best effort)
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
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
        except Exception: pass  # torn/corrupt read → fall back to Idle (set_state writes atomically)
    return {"mode": "Idle", "step": "Waiting", "video_index": 0, "total_videos": 0, "chain_index": 0, "total_chains": 0, "current_prompt": "None"}

def set_state(mode="Idle", step="Waiting", video=0, total=0, chain=0, total_chains=0, prompt=""):
    s = {"mode": mode, "step": step, "video_index": video, "total_videos": total, "chain_index": chain, "total_chains": total_chains, "current_prompt": prompt, "ts": time.time()}
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    # Atomic write — get_state() is read on every broadcaster tick; a torn
    # write (reader catching a half-flushed file) would otherwise surface as a
    # transient JSONDecodeError and a spurious "Idle" flash on the dashboard.
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)

# Queue file is the IPC layer between the dashboard handlers, the worker
# fleet, and the legacy run_fleet runner. Multiple writers can race:
# run_fleet's claim/sweep/requeue, the worker base claim/finalize, and the
# FastAPI endpoints (/inject, /queue/cancel, /queue/edit, /queue/requeue,
# /queue/clear-failed, /queue/clear-completed, /queue/toggle-infinity,
# /queue/toggle-polymorphic, /cancel-all, the chat queue tools, and the
# broadcaster's stale-cancelled pruner) all do a get-modify-save dance.
# Because save_queue() does a blind delete-all + reinsert of the WHOLE list,
# two writers in the same window silently overwrite each other: an edit gets
# wiped by a finalizer, a fresh inject vanishes behind a requeue, jobs run
# twice. This was the audit's #1 high-priority correctness bug.
#
# FIX (see docs/queue-concurrency.md): every read-modify-write site now goes
# through the cross-process lock below — either via mutate_queue() (the
# preferred one-shot RMW helper) or an explicit `with queue_lock():` span for
# the few sites with mid-block early-returns / conditional saves. Pure reads
# (get_queue alone, /queue/paginated) intentionally stay lock-free.
#
# Two complementary protections:
#   * _QUEUE_LOCK_FILE  — flock-based filesystem mutex. Survives across
#     processes (host run_fleet vs container slopfinity dashboard) and
#     across worker restarts. Cheap on Linux. Held only for the read +
#     write window; doesn't serialize unrelated callers.
#   * atomic write-rename in save_queue() — a torn write (process killed
#     mid-fwrite) can't leave corrupt JSON on disk. Readers see either the
#     previous valid state or the new one — never half.
#
# `queue_lock()` is a context manager; mutate_queue() wraps the common
# get→modify→save round-trip. Any NEW write path MUST use one of them.
# queue_lock is non-recursive (flock on a fresh fd re-blocks the same
# process) — never nest, and never call get_queue/save_queue/mutate_queue
# from inside a mutator or an open lock span.
import contextlib
import fcntl

_QUEUE_LOCK_FILE = QUEUE_FILE + ".lock"


@contextlib.contextmanager
def queue_lock():
    """Hold an OS-level advisory lock around a get_queue → save_queue
    round-trip. Cooperative across processes (host runner + container
    dashboard + worker subprocesses) since they all open the same
    .lock file. Non-recursive: nesting will deadlock — keep critical
    sections short and avoid calling out to other queue operations
    while holding."""
    os.makedirs(os.path.dirname(_QUEUE_LOCK_FILE), exist_ok=True)
    with open(_QUEUE_LOCK_FILE, "a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def _split_queue_item(item_dict):
    """Split a queue dict into QueueItem(**kwargs) form, funneling any key
    without a dedicated column into the `extra` JSON catch-all.

    Without this, save_queue's `k in model_fields` filter silently drops
    fields like seed_image / stage_prompts / seeds_mode / polymorphic /
    started_ts on the DB roundtrip — breaking the seed-image + FLF2V +
    stage-prompt features whose live consumer (run_fleet) reads them back
    out of the DB. A stable id is stamped if missing.
    """
    item_dict.setdefault("id", queue_schema.make_id())
    known = QueueItem.model_fields
    extra = {k: v for k, v in item_dict.items() if k not in known}
    filtered = {k: v for k, v in item_dict.items() if k in known and k != "extra"}
    # Defensive: if an un-flattened `extra` dict slipped through, merge it
    # (newer top-level unknowns win over stale nested copies).
    prev = item_dict.get("extra")
    if isinstance(prev, dict):
        extra = {**prev, **extra}
    filtered["extra"] = extra
    return filtered


def _flatten_queue_item(m):
    """model_dump() a QueueItem and lift its `extra` catch-all back to the top
    level so callers see one flat dict (exactly as before the column existed).
    setdefault so a real column is never clobbered by a stale extra entry."""
    d = m.model_dump()
    extra = d.pop("extra", None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            d.setdefault(k, v)
    return d


def get_queue():
    """Retrieve the queue from the database, falling back to JSON for migration."""
    init_db()
    with Session(engine) as session:
        statement = select(QueueItem).order_by(QueueItem.ts)
        items = session.exec(statement).all()
        if not items and os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, "r") as f:
                    file_data = json.load(f)
            except Exception as e:
                file_data = []
                _log.warning("get_queue: could not read legacy %s: %s", QUEUE_FILE, e)
            # Per-row so one corrupt legacy item is skipped (and logged) rather
            # than silently aborting the entire migration.
            for item in file_data:
                try:
                    queue_schema.migrate_legacy(item)
                    q_item = QueueItem(**_split_queue_item(item))
                    session.add(q_item)
                except Exception as e:
                    _log.warning("get_queue: skipping un-migratable legacy item: %s", e)
            try:
                session.commit()
                items = session.exec(statement).all()
            except Exception as e:
                _log.warning("get_queue: legacy migration commit failed: %s", e)

        return [_flatten_queue_item(m) for m in items]

def save_queue(q):
    """Save the queue to the database and sync to JSON."""
    init_db()
    with Session(engine) as session:
        # Delete all existing and replace (simulates the current list behavior)
        # In the future, we should do delta updates for performance.
        session.exec(select(QueueItem)).all() # load for session
        for item in session.exec(select(QueueItem)).all():
            session.delete(item)
        
        for item_dict in q:
            # _split_queue_item stamps a stable id (run_fleet appends iter rows
            # without one — otherwise the id default_factory churns a fresh uuid
            # every save and status tracking can't follow an item across saves)
            # and funnels non-column keys into `extra` so they aren't dropped.
            session.add(QueueItem(**_split_queue_item(item_dict)))
        session.commit()

    # Sync to JSON as backup
    try:
        os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
        tmp = QUEUE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(q, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, QUEUE_FILE)
    except Exception:
        pass


def mutate_queue(mutator):
    """Atomic read→modify→write of the queue under the cross-process lock.

    save_queue does a blind delete-all + reinsert of the whole list, so two
    processes doing get_queue→modify→save_queue concurrently (host run_fleet +
    container dashboard + worker subprocesses) lose each other's edits. This
    runs the full cycle inside queue_lock() (an flock shared via the same
    SLOPFINITY_STATE_DIR), serialising all writers.

    `mutator(queue_list)` mutates the list in place and/or returns a new list;
    the resulting list is persisted and returned.

    CONTRACT: the mutator must NOT call get_queue/save_queue/mutate_queue
    (queue_lock is non-recursive → re-entry deadlocks) and must not block on
    network/subprocess/GPU work while the lock is held — keep it pure list work.
    """
    with queue_lock():
        q = get_queue()
        result = mutator(q)
        q = result if isinstance(result, list) else q
        save_queue(q)
        return q


_CONFIG_LOCK_FILE = CONFIG_FILE + ".lock"


@contextlib.contextmanager
def config_lock():
    """Cross-process advisory lock for a load_config → save_config round-trip.
    Separate flock from queue_lock; non-recursive (don't nest)."""
    os.makedirs(os.path.dirname(_CONFIG_LOCK_FILE), exist_ok=True)
    with open(_CONFIG_LOCK_FILE, "a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def mutate_config(mutator):
    """Locked read→modify→write of the config.

    save_config upserts per key, but callers pass the WHOLE config dict, so two
    unsynchronised loops (run_fleet's infinity-index increment and the
    broadcaster's chaos_rotator theme refresh) can revert each other's keys —
    e.g. run_fleet saving a stale infinity_themes back over a fresh rotation.
    Routing both through this serialises them.

    NOTE: dashboard settings endpoints still call save_config directly and are
    not yet serialised against this — opt-in callers only. CONTRACT: the mutator
    must not call load_config/save_config/mutate_config (non-recursive lock) or
    block on I/O while held.
    """
    with config_lock():
        c = load_config()
        result = mutator(c)
        c = result if isinstance(result, dict) else c
        save_config(c)
        return c
