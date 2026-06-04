# Config, Settings & Locking

## Overview

Slopfinity manages three critical persistence layers with overlapping semantics and failure modes:

1. **Configuration** (`config.json` + SQLite) — pipeline models, prompts, URLs, thresholds
2. **Queue** (`queue.json` + SQLite) — jobs for fleet execution (run_fleet and workers)
3. **Settings endpoints** (FastAPI routes) — user-facing GET/POST that read/write config

The core problem fixed in commits **68e5e33** and **ddab557** is **lost updates**: two independent background loops (run_fleet's infinity-mode index rotation + broadcaster's chaos_rotator theme refresh) can clobber each other's config keys when they do unsynchronised full-config load→modify→save cycles. The settings endpoints also had a subtle roundtrip bug where displayed fields (disk_min_pct/disk_min_gb) could not be persisted via POST.

## Architecture: Config Load/Save

### Persistence Backend (SQLite + JSON Backup)

**File:** `slopfinity/config.py:415–472`

Config is primarily stored in SQLite (`Configuration` table), with optional JSON backup synced at save time for disaster recovery:

```python
def load_config():
    """Load configuration from the database, falling back to JSON if available."""
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
                            session.add(Configuration(key=k, value=v))
                    session.commit()
            except Exception:
                pass
        
        c["auto_suspend"] = _merge_auto_suspend(c.get("auto_suspend"))
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
                existing.updated_at = datetime.utcnow()
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
```

**Key invariants:**

- **Per-key upsert:** `save_config()` upserts every key in the dict individually to SQLite. This is semantically sound for a single writer but breaks under concurrent updates (see "Lost-Update Race" below).
- **Default merging:** `load_config()` merges all missing keys from `DEFAULT_CONFIG` so callers always get a complete dict.
- **Scheduler defaults:** Stored scheduler subdicts are merged with `DEFAULT_SCHEDULER`, auto-adding new keys (e.g., `use_planner`) in releases.
- **JSON backup:** Optional; survives DB corruption or filesystem issues. Kept 0o600-readable.

### Default Configuration

**File:** `slopfinity/config.py:179–293`

The canonical defaults define every model role, prompt, threshold, and feature flag:

```python
DEFAULT_SCHEDULER = {
    "use_planner": False,
    "memory_safety_gb": 10,
    "llm_cpu_mode": "smart",
    "tts_cpu_mode": "smart",
}

DEFAULT_CONFIG = {
    "base_model": "qwen",
    "video_model": "ltx-2.3",
    "audio_model": "none",
    "tts_model": "none",
    # ... disk guard thresholds
    "disk_min_pct": 1,
    "disk_min_gb": 5.0,
    # ... server-fetched endpoint URLs
    "tts_worker_url": "http://localhost:8010/tts",
    "comfy_url": "http://localhost:8188",
    # ... scheduler and other keys
}
```

Notable entries:

- **`disk_min_pct`, `disk_min_gb`:** Two guards on the outputs partition. A generation is blocked if EITHER trips (free% ≤ pct OR free GB ≤ gb). Both 0 = disabled. Set on General tab → Disk Space thresholds.
- **`tts_worker_url`, `comfy_url`:** Server-fetched URLs, now validated by SSRF guard (see "SSRF Protection" below).
- **`scheduler`:** Subdict with CPU-offload mode and memory planner config (see "llm_cpu_mode Namespace Bug" below).

## The Lost-Update Race (Commit 68e5e33)

### The Failure Mode

Two background loops, running concurrently and unsynchronised, both do:

```
config = load_config()
# modify config["some_key"]
save_config(config)
```

**Loop 1 (run_fleet, `run_fleet.py:403–420`):** Increments infinity_index and picks a theme atomically.

```python
config = cfg.load_config()
if config.get("infinity_mode"):
    def _advance(c):
        th = c.get("infinity_themes") or []
        if not th:
            return c
        i = c.get("infinity_index", 0) % len(th)
        _picked["theme"] = th[i]
        c["infinity_index"] = i + 1
        return c
    # Before fix: config = cfg.load_config(); _advance(config); cfg.save_config(config)
    # After fix: uses config_lock
    config = cfg.mutate_config(_advance)
```

**Loop 2 (broadcaster chaos_rotator, `slopfinity/broadcaster.py:262–310`):** Fetches new themes from LLM and resets the index.

```python
async def chaos_rotator():
    # ... every ~10s when chaos_mode is on:
    if arr:
        def _set_themes(c):
            c["infinity_themes"] = arr
            c["infinity_index"] = 0  # Reset index
            return c
        # Before fix: config = load_config(); _set_themes(config); save_config(config)
        # After fix: uses mutate_config
        config = await asyncio.to_thread(cfg.mutate_config, _set_themes)
```

**Scenario:** Run_fleet loads config T0 with `infinity_themes=["theme_A"]` and `infinity_index=0`. Broadcaster loads same config at T0, then LLM-fetches new themes and saves at T1 with `infinity_themes=["theme_B", "theme_C"]` and `infinity_index=0`. Run_fleet then saves at T2 with its stale-loaded `infinity_themes=["theme_A"]` (overwriting the fresh rotation) but `infinity_index=1`. Result: the user's fresh theme list is clobbered, and the index is out of sync.

### The Fix: config_lock + mutate_config

**File:** `slopfinity/config.py:674–709`

Two new primitives serialise config RMW:

```python
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
    broadcaster's chaos_rotator theme refresh) can revert each other's keys.
    Routing both through this serialises them.
    
    NOTE: dashboard settings endpoints still call save_config directly — opt-in.
    CONTRACT: mutator must not call load_config/save_config/mutate_config
    (non-recursive lock) or block on I/O while held."""
    with config_lock():
        c = load_config()
        result = mutator(c)
        c = result if isinstance(result, dict) else c
        save_config(c)
        return c
```

**Why separate from queue_lock?** The queue lock serialises job-list RMW; config is unrelated, and a single serialization point would bottleneck both systems. The tests verify they are independent:

**File:** `tests/test_config_extras.py:85–98`

```python
def test_config_lock_is_exclusive():
    import fcntl
    import pytest
    from slopfinity import config as cfg
    with cfg.config_lock():
        with open(cfg._CONFIG_LOCK_FILE, "a+") as fd:
            # Exclusive lock blocks second acquire
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    
    # config_lock and queue_lock are independent flocks
    with cfg.queue_lock():
        with open(cfg._CONFIG_LOCK_FILE, "a+") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # succeeds
            fcntl.flock(fd, fcntl.LOCK_UN)
```

**Which callers use mutate_config?**

- `run_fleet.py:420` — infinity-mode theme selection
- `slopfinity/broadcaster.py:307` — chaos_rotator theme refresh
- **NOT** settings endpoints (POST /settings, POST /config) — they still call `save_config()` directly (opt-in pattern; a future refactor can adopt `mutate_config` for those as well)

### Verification

**File:** `tests/test_config_extras.py:101–106`

```python
def test_mutate_config_reads_modifies_saves():
    from slopfinity import config as cfg
    cfg.save_config({"infinity_index": 5, "infinity_themes": ["x"]})
    out = cfg.mutate_config(lambda c: {**c, "infinity_index": c.get("infinity_index", 0) + 1})
    assert out["infinity_index"] == 6
    assert cfg.load_config()["infinity_index"] == 6
```

## The llm_cpu_mode Namespace Bug (Commit ddab557)

### The Failure Mode

Settings → LLM → Generation offers three CPU-offload modes: `"gpu"` (always GPU), `"smart"` (GPU if idle else CPU), `"cpu"` (always CPU). The user's choice is persisted at `config["scheduler"]["llm_cpu_mode"]`, as shown in the Settings GET:

**File:** `slopfinity/routers/config.py:159–162`

```python
"llm_cpu_mode": _coerce_cpu_mode(
    (c.get("scheduler") or {}).get("llm_cpu_mode")
    or (c.get("scheduler") or {}).get("llm_cpu_only")
),
```

But the LLM layer read it from the wrong bucket. The pre-fix code was:

```python
# WRONG: read from c.get("llm", {}).get("scheduler")  [which never exists]
scheduler_obj = c.get("llm", {})
cpu_mode = scheduler_obj.get("scheduler")  # Always None!
# Falls back to default "smart" every time
return cpu_mode or "smart"
```

The user could toggle CPU mode a hundred times in Settings, but the LLM layer would always compute as if it were "smart" (GPU when idle, CPU when busy). The mode was being **persisted correctly** but **read from the wrong namespace**, a classic schema-drift bug.

### The Fix: _llm_cpu_mode()

**File:** `slopfinity/llm/__init__.py:41–48`

```python
def _llm_cpu_mode() -> str:
    """LLM CPU-offload mode, read from config['scheduler']['llm_cpu_mode'] —
    the bucket the settings endpoint actually persists it under. The previous
    read used the llm sub-config's 'scheduler' key, which never exists, so the
    user's choice silently fell back to 'smart' every time."""
    from .. import config as cfg
    c = cfg.load_config()
    return (c.get("scheduler") or {}).get("llm_cpu_mode") or "smart"
```

Now used in two places:

1. **lmstudio_call()** (line 80): Decides whether to prefer CPU or GPU endpoints.
2. **lmstudio_chat_raw()** (line 161): Same decision for multi-turn chat.

Both now read the correct bucket:

```python
cpu_mode = _llm_cpu_mode()  # Reads from scheduler dict
gpu = scheduler.get_gpu()
is_gpu_busy = gpu.resident_gb > 0 or bool(gpu.in_flight)

prefer_cpu = (cpu_mode == "smart" and is_gpu_busy) or cpu_mode == "cpu"
```

### Verification

**File:** `tests/test_config_extras.py:73–82`

```python
def test_llm_cpu_mode_reads_scheduler_bucket():
    from slopfinity import config as cfg
    from slopfinity.llm import _llm_cpu_mode
    cfg.save_config({"scheduler": {"llm_cpu_mode": "cpu"}})
    assert _llm_cpu_mode() == "cpu"
    cfg.save_config({"scheduler": {"llm_cpu_mode": "smart"}})
    assert _llm_cpu_mode() == "smart"
```

## Disk Thresholds: GET-but-not-POST Bug (Commit ddab557)

### The Failure Mode

Settings → General → Disk Space Thresholds shows two fields: `disk_min_pct` (minimum free %) and `disk_min_gb` (minimum free GB). The GET endpoint returns them:

**File:** `slopfinity/routers/config.py:128–133`

```python
"disk_min_pct": float(
    c.get("disk_min_pct") if c.get("disk_min_pct") is not None else 1
),
"disk_min_gb": float(
    c.get("disk_min_gb") if c.get("disk_min_gb") is not None else 5
),
```

But the POST endpoint silently dropped them:

```python
# OLD CODE (before fix):
# Settings POST had no branch for disk_min_pct or disk_min_gb,
# so "if disk_min_pct in data: ..." was never executed
# → user changes to these fields were accepted, displayed, but not saved
```

Result: a user sets disk_min_pct to 10 via the UI, sees it reflected on reload (because GET returns the old default 1 from a read of the fresh database with no user change), saves, reloads, and finds it reset to 1. The fields were **displayed but unsaveable**.

### The Fix: Persist Disk Thresholds in POST

**File:** `slopfinity/routers/config.py:210–217`

```python
# Disk-guard thresholds (General tab) — settings_get surfaces these, so the
# POST must persist them too (they were silently dropped before).
for _dk in ("disk_min_pct", "disk_min_gb"):
    if _dk in data:
        try:
            c[_dk] = max(0.0, float(data.get(_dk) or 0))
        except (TypeError, ValueError):
            pass
```

Now both GET and POST handle these keys symmetrically.

### Verification

**File:** `tests/test_server_config.py:85–93`

```python
async def test_disk_thresholds_persist(self, client, default_config):
    saved = {}
    cfg_copy = dict(default_config)
    with mock.patch("slopfinity.server.cfg.load_config", return_value=cfg_copy), \
         mock.patch("slopfinity.server.cfg.save_config", side_effect=lambda c: saved.update(c)):
        resp = await client.post("/settings", json={"disk_min_pct": 7, "disk_min_gb": 12})
    assert resp.status_code == 200
    assert saved.get("disk_min_pct") == 7.0
    assert saved.get("disk_min_gb") == 12.0
```

## SSRF Protection: validate_llm_base_url Applied to tts_worker_url & comfy_url (Commit ddab557)

### The Vulnerability

Settings → Endpoints allows users to re-point TTS and ComfyUI to custom servers:

```json
{
  "tts_worker_url": "http://user-supplied.example.com/tts",
  "comfy_url": "http://user-supplied.example.com:8188"
}
```

The slopfinity server **fetches** these URLs server-side to call TTS and ComfyUI (not the browser). A CSRF-bypassing attacker or social-engineering could inject a malicious URL pointing at internal admin panels:

```
tts_worker_url: "http://169.254.169.254/latest/meta-data/"  (AWS metadata)
comfy_url: "http://127.0.0.1:9000/admin"  (internal service)
```

**Before the fix:** POST /settings accepted these URLs unchecked.

### The Fix: Apply SSRF Guard

**File:** `slopfinity/routers/config.py:194–208`

```python
# Endpoints tab — direct top-level keys. The server fetches these URLs, so
# run them through the same SSRF guard as the LLM base_url (blocks the
# cloud-metadata IP, multicast/reserved ranges, non-http schemes).
import slopfinity.net_guard as _net_guard
for _uk in ("tts_worker_url", "comfy_url"):
    if _uk in data:
        _u = str(data[_uk]).strip()
        if _u:
            try:
                _net_guard.validate_llm_base_url(_u)
            except Exception as _e:
                return JSONResponse(
                    {"ok": False, "error": f"{_uk}: {_e}"}, status_code=400
                )
        c[_uk] = _u
```

### SSRF Guard Details

**File:** `slopfinity/net_guard.py:28–52`

```python
def validate_llm_base_url(url: str) -> str:
    """Return `url` if it's safe to fetch server-side; raise ValueError otherwise.
    
    Allows http/https to loopback, private (LAN) and public hosts; blocks other
    schemes and cloud-metadata / reserved / multicast addresses."""
    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https"):
        raise ValueError("base_url scheme must be http or https")
    host = u.hostname
    if not host:
        raise ValueError("base_url is missing a host")
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ValueError(f"base_url DNS resolution failed: {exc}")
    if not infos:
        raise ValueError("base_url did not resolve")
    for info in infos:
        sockaddr = info[4]
        ip = ipaddress.ip_address(sockaddr[0])
        if _blocked_ip(ip):
            raise ValueError("base_url resolves to a blocked address (metadata/reserved)")
    return url

def _blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_multicast or ip.is_reserved or ip.is_unspecified
        or (ip.version == 4 and ip in _METADATA_V4)
        or (ip.version == 6 and ip in _METADATA_V6)
    )
```

**Blocked categories:**

- Non-HTTP(S) schemes (file://, gopher://, etc.)
- Multicast (224.0.0.0/4)
- Reserved (100.64.0.0/10, etc.)
- Unspecified (0.0.0.0, ::)
- AWS metadata (169.254.0.0/16, including 169.254.169.254)
- AWS IMDSv6 (fd00:ec2::/64)

**Allowed:**

- Loopback (127.0.0.1, ::1) — localhost services
- Private/LAN (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) — LAN deployments
- Public IPs — cloud providers (OpenAI, etc.)

### Verification

**File:** `tests/test_server_config.py:95–103`

```python
async def test_ssrf_metadata_url_rejected(self, client, default_config):
    cfg_copy = dict(default_config)
    with mock.patch("slopfinity.server.cfg.load_config", return_value=cfg_copy), \
         mock.patch("slopfinity.server.cfg.save_config"):
        resp = await client.post(
            "/settings",
            json={"tts_worker_url": "http://169.254.169.254/latest/meta-data/"},
        )
    assert resp.status_code == 400
```

## API Key Redaction

### load_config() and Broadcast Redaction

**File:** `slopfinity/config.py:475–485`

```python
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
```

Masks `config["llm"]["api_key"]` and any top-level `api_key` before broadcasting to WebSocket clients or returning in HTTP responses.

### Settings GET Redaction

**File:** `slopfinity/routers/config.py:72–80`

```python
@router.get("/settings")
async def settings_get():
    """Return current settings. `api_key` is masked in transit."""
    c = cfg.load_config()
    llm = dict(DEFAULT_LLM_CONFIG)
    llm.update(c.get("llm") or {})
    has_key = bool(llm.get("api_key"))
    llm_safe = dict(llm)
    llm_safe["api_key"] = "***" if has_key else ""
```

### Settings POST: Mask Token Pass-Through

**File:** `slopfinity/routers/config.py:219–228`

```python
llm_in = data.get("llm") or {}
if isinstance(llm_in, dict):
    current_llm = dict(DEFAULT_LLM_CONFIG)
    current_llm.update(c.get("llm") or {})
    for k, v in llm_in.items():
        if k == "api_key":
            if v in ("", "***", None):
                continue  # Don't overwrite with mask
            current_llm[k] = v
        else:
            current_llm[k] = v
```

If the client echoes back `"***"` (the mask token), the POST treats it as "no change" and skips the update. This allows the UI to submit the full LLM dict without leaking the secret.

## Settings Endpoints Architecture

### GET /settings

Returns a flat dict of displayable settings:

**File:** `slopfinity/routers/config.py:72–181`

Includes:

- LLM config (provider, base_url, model_id, temperature, api_key masked, etc.)
- Disk thresholds (disk_min_pct, disk_min_gb)
- Endpoint URLs (tts_worker_url, comfy_url)
- Prompts (philosophical_prompt, fanout_system_prompt, etc.)
- Scheduler (memory_safety_gb, use_planner, llm_cpu_mode, tts_cpu_mode)
- Model loading preferences (sticky, eager_unload)
- Auto-suspend list
- Provider list (filtered by allow_cloud_endpoints toggle)

### POST /settings

Partial update (only updated keys in `data`):

**File:** `slopfinity/routers/config.py:184–337`

- **Endpoint URLs** (tts_worker_url, comfy_url): Validated by `net_guard.validate_llm_base_url()`, then persisted.
- **Disk thresholds** (disk_min_pct, disk_min_gb): Coerced to float ≥ 0, persisted.
- **LLM config:** Merges with defaults, coerces numerics (temperature, max_retries, timeout_s).
- **Scheduler, model_loading:** Top-level merge (existing keys updated, new keys preserved).
- **Prompt overrides:** Empty string / None both mean "use built-in default"; persisted as-is.
- **Cloud-endpoints toggle:** Persists allow_cloud_endpoints bool.
- **Auto-suspend list:** Validated per entry (id, label, enabled, method, method-specific fields), reconstructed.

All updates call `cfg.save_config(c)` directly (not `mutate_config()`). This is a known opt-in limitation; the endpoints are not yet serialised against background loops.

### GET /config

Not discussed in this scope (model-selection endpoint).

### TTS Speed Validation (Commit 68e5e33)

**File:** `slopfinity/routers/runner.py:302–313`

```python
speed_raw = data.get("speed")
try:
    speed = float(speed_raw) if speed_raw is not None else None
except (TypeError, ValueError):
    speed = None
# Clamp to the supported TTS range instead of forwarding e.g. speed=999 to
# the worker (the chat tool already validates 0.5–2.0; match it here).
if speed is not None and not (0.5 <= speed <= 2.0):
    return JSONResponse(
        {"ok": False, "error": "speed must be between 0.5 and 2.0"},
        status_code=400,
    )
```

Rejects out-of-range speed values before forwarding to the TTS worker, mirroring chat-tool validation.

## Summary of Key Mechanisms

| Mechanism | File | Purpose | Key Calls |
|-----------|------|---------|-----------|
| load_config() | config.py:415 | Load all config from DB + defaults | Used by every settings read |
| save_config(config) | config.py:451 | Upsert config keys to DB | Used by settings POST + runs endpoints |
| config_lock() | config.py:677 | Exclusive flock for RMW | Used by mutate_config only |
| mutate_config(fn) | config.py:690 | Locked RMW of entire config | run_fleet:420, broadcaster:307 |
| mutate_queue(fn) | config.py:650 | Locked RMW of queue (separate lock) | run_fleet, dashboard endpoints |
| redact(config) | config.py:475 | Mask api_key for broadcast | ws_manager, http responses |
| validate_llm_base_url(url) | net_guard.py:28 | SSRF guard for server-fetched URLs | settings POST for tts_worker_url, comfy_url |
| _llm_cpu_mode() | llm/__init__.py:41 | Read CPU-offload mode from correct bucket | lmstudio_call:80, lmstudio_chat_raw:161 |

## Residuals & Future Work

1. **Settings endpoints not yet serialised:** POST /settings and POST /config call `save_config()` directly, not `mutate_config()`. If a background loop calls `mutate_config()` concurrently, the settings write could be clobbered. Resolve by wrapping endpoints in `mutate_config()` or using a single serialization point.

2. **Queue lock separate from config lock:** Semantically correct but adds state-management overhead. Both use the same `SLOPFINITY_STATE_DIR`, so a single flock might suffice in future. Verify independence is not a requirement before consolidating.

3. **JSON backup best-effort:** `save_config()` syncs JSON backup with `try/except: pass`, so a failing JSON write is silent. For headless deployments relying on JSON, add logging or fail-fast semantics.

4. **Backward compat for cpu_mode:** The code still supports old boolean `llm_cpu_only` fields (coerced to "cpu"/"gpu"/"smart" strings). Can be removed once all configs are migrated.

## Failure Modes & Edge Cases

| Failure Mode | Cause | Mitigation |
|------|-------|-----------|
| Infinity-theme clobber | run_fleet and broadcaster both do unsynchronized load→modify→save | config_lock + mutate_config; routes both through serialization |
| CPU-mode silently ignored | LLM layer read from wrong namespace (llm.scheduler instead of scheduler.llm_cpu_mode) | _llm_cpu_mode() reads correct bucket; LLM endpoints check it |
| Disk thresholds unsaveable | POST /settings had no handler; GET showed them but POST dropped them | Disk-threshold persisting branch in settings_post; round-trip verified |
| SSRF via URL injection | POST /settings accepted arbitrary tts_worker_url / comfy_url unchecked | validate_llm_base_url() applied; blocks metadata, reserved, non-http schemes |
| API key exposure | api_key returned in GET /settings or broadcast frames | redact() blanks api_key to "***"; POST skips update if echoed as mask token |
| Mask-token bypass | If client sends real api_key, it's stored; if "***", it's ignored | POST branch checks `if v in ("", "***", None): continue` |

## Verification

Full test suite passes (239+ tests):

```bash
pytest tests/test_config_extras.py -v           # Config locking, mutate_config RMW
pytest tests/test_server_config.py -v           # Settings endpoint round-trip
```

Key test scenarios:

- **test_mutate_config_reads_modifies_saves:** Atomicity of config RMW.
- **test_config_lock_is_exclusive:** flock blocks concurrent acquire.
- **test_llm_cpu_mode_reads_scheduler_bucket:** Correct namespace read.
- **test_disk_thresholds_persist:** GET ↔ POST round-trip.
- **test_ssrf_metadata_url_rejected:** Metadata endpoint blocked.

---

**Relevant commits:**
- **68e5e33:** config_lock + mutate_config, broadcaster exception logging, TTS speed validation, frame leak cleanup
- **ddab557:** llm_cpu_mode namespace fix, disk-threshold persistence, SSRF guard on tts_worker_url/comfy_url, requeue stage-dict reset
