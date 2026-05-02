import os
import json
import random
from fastapi import APIRouter, Request, Body
from fastapi.responses import JSONResponse, HTMLResponse
import slopfinity.config as cfg
from slopfinity.llm.probe import discover as llm_discover, ping as llm_ping
from slopfinity.llm import DEFAULT_LLM_CONFIG, list_providers
from fastapi.templating import Jinja2Templates
from slopfinity.paths import TEMPLATES_DIR
import slopfinity.branding as _branding

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Real-model candidate pools per role. `__random__` picks uniformly from
# the role's pool when /config arrives.
_RANDOM_CANDIDATES = {
    "base_model": ["qwen", "ernie"],
    "audio_model": ["heartmula"],
    "tts_model": ["qwen-tts", "kokoro"],
}


def _coerce_cpu_mode(raw) -> str:
    """Normalize CPU mode input to 'gpu', 'smart', or 'cpu'."""
    if raw is None:
        return "smart"
    if isinstance(raw, bool):
        return "cpu" if raw else "gpu"
    s = str(raw).lower().strip()
    if s in ("cpu", "smart", "gpu"):
        return s
    return "smart"


def _cpu_mode_to_bool(raw) -> bool | None:
    """Convert cpu mode string back to boolean for backward compat."""
    mode = _coerce_cpu_mode(raw)
    if mode == "cpu":
        return True
    if mode == "gpu":
        return False
    return None


router = APIRouter()


@router.post("/config")
async def update_config(data: dict = Body(...)):
    config = cfg.load_config()
    # Resolve `__random__` placeholders to a concrete model from the role's
    # candidate pool. We persist the chosen model (not the sentinel) so the
    # rest of the pipeline never has to know about pseudo-models. `__slopped__`
    # without a concrete file selection is treated the same as `__random__`
    # (fall back to a real model) — the UI is responsible for sending
    # `slopped:<filename>` once the user picks one.
    for role, pool in _RANDOM_CANDIDATES.items():
        v = data.get(role)
        if v == "__random__" or v == "__slopped__":
            data[role] = random.choice(pool) if pool else "none"
    config.update(data)
    cfg.save_config(config)
    return {"status": "ok"}


@router.get("/settings")
async def settings_get():
    """Return current settings. `api_key` is masked in transit."""
    c = cfg.load_config()
    llm = dict(DEFAULT_LLM_CONFIG)
    llm.update(c.get("llm") or {})
    has_key = bool(llm.get("api_key"))
    llm_safe = dict(llm)
    llm_safe["api_key"] = "***" if has_key else ""
    branding_cfg = c.get("branding") or {}
    # Cloud-endpoints gate. The provider registry today is local-only
    # (see slopfinity/llm/providers.py) so the list returned here is the
    # same regardless of the toggle — but we still surface the bool so
    # the client can pre-toggle the UI checkbox and so a future cloud
    # provider entry can be filtered server-side without a UI change.
    allow_cloud = bool(c.get("allow_cloud_endpoints", False))
    _LOCAL_PROVIDERS = {"lmstudio", "ollama", "vllm", "llamacpp", "custom"}
    all_providers = list_providers()
    if allow_cloud:
        providers_filtered = list(all_providers)
    else:
        providers_filtered = [p for p in all_providers if p in _LOCAL_PROVIDERS]
    return {
        "llm": llm_safe,
        "llm_has_api_key": has_key,
        "providers": providers_filtered,
        "allow_cloud_endpoints": allow_cloud,
        "branding": {
            "active": branding_cfg.get("active") or "slopfinity",
            "profiles": _branding.list_profiles(),
        },
        "philosophical_prompt": c.get("philosophical_prompt") or "",
        "philosophical_prompt_default": cfg.DEFAULT_PHILOSOPHICAL_PROMPT,
        # Prompts tab — overrides + their built-in defaults (so the UI can
        # pre-fill placeholders and offer "Reset to default" without a second
        # round-trip). Empty string = "use default" semantics, same as the
        # philosophical_prompt pattern above.
        "enhancer_prompt": c.get("enhancer_prompt") or "",
        "enhancer_prompt_default": cfg.DEFAULT_CONFIG["enhancer_prompt"],
        "fanout_system_prompt": c.get("fanout_system_prompt") or "",
        "fanout_system_prompt_default": cfg.DEFAULT_FANOUT_SYSTEM_PROMPT,
        "fleet_user_prompt_template": c.get("fleet_user_prompt_template") or "",
        "fleet_user_prompt_template_default": cfg.DEFAULT_FLEET_USER_PROMPT_TEMPLATE,
        "infinity_user_prompt_template": c.get("infinity_user_prompt_template") or "",
        "infinity_user_prompt_template_default": cfg.DEFAULT_INFINITY_USER_PROMPT_TEMPLATE,
        "chaos_suggest_system_prompt": c.get("chaos_suggest_system_prompt") or "",
        "chaos_suggest_system_prompt_default": cfg.DEFAULT_CHAOS_SUGGEST_SYSTEM_PROMPT,
        "void_fallback_template": c.get("void_fallback_template") or "",
        "void_fallback_template_default": cfg.DEFAULT_VOID_FALLBACK_TEMPLATE,
        "suggest_use_subjects": bool(
            c.get("suggest_use_subjects", cfg.DEFAULT_SUGGEST_USE_SUBJECTS)
        ),
        "suggest_custom_prompt": c.get("suggest_custom_prompt") or "",
        "suggest_auto_disabled": bool(
            c.get("suggest_auto_disabled", cfg.DEFAULT_SUGGEST_AUTO_DISABLED)
        ),
        "disk_min_pct": float(
            c.get("disk_min_pct") if c.get("disk_min_pct") is not None else 1
        ),
        "disk_min_gb": float(
            c.get("disk_min_gb") if c.get("disk_min_gb") is not None else 5
        ),
        # Named suggestion prompts. Active entries appear in the unified
        # Suggestions badge dropdown + drive Endless per-row selectors.
        "suggest_prompts": list(
            c.get("suggest_prompts") or cfg.DEFAULT_SUGGEST_PROMPTS
        ),
        "auto_suspend": c.get("auto_suspend") or list(cfg.DEFAULT_AUTO_SUSPEND),
        # Per-model loading prefs (Settings → Scheduler → "Per-model loading
        # preferences"). Both lists default to empty; the hydrator on the
        # client toggles checkboxes based on membership.
        "model_loading": {
            "sticky": list((c.get("model_loading") or {}).get("sticky") or []),
            "eager_unload": list(
                (c.get("model_loading") or {}).get("eager_unload") or []
            ),
        },
        "scheduler": {
            "memory_safety_gb": (c.get("scheduler") or {}).get("memory_safety_gb", 10),
            "use_planner": bool((c.get("scheduler") or {}).get("use_planner", False)),
            # CPU offload mode for LLM and TTS stages.
            # Three-way string: "gpu" | "smart" | "cpu"
            #   "gpu"   — always use GPU (never offload)
            #   "smart" — use GPU when GPU utilisation is 0%, else CPU
            #   "cpu"   — always use CPU (original default behaviour)
            # Backward compat: if the stored value is a boolean (old schema),
            # True → "cpu" and False → "gpu". Default is "smart".
            "llm_cpu_mode": _coerce_cpu_mode(
                (c.get("scheduler") or {}).get("llm_cpu_mode")
                or (c.get("scheduler") or {}).get("llm_cpu_only")
            ),
            "tts_cpu_mode": _coerce_cpu_mode(
                (c.get("scheduler") or {}).get("tts_cpu_mode")
                or (c.get("scheduler") or {}).get("tts_cpu_only")
            ),
            # Derived booleans kept for backward compat with any external
            # code that still reads them. "smart" resolves to None here —
            # callers that need the live decision should use _resolve_cpu_mode().
            "llm_cpu_only": _cpu_mode_to_bool(
                (c.get("scheduler") or {}).get("llm_cpu_mode")
                or (c.get("scheduler") or {}).get("llm_cpu_only")
            ),
            "tts_cpu_only": _cpu_mode_to_bool(
                (c.get("scheduler") or {}).get("tts_cpu_mode")
                or (c.get("scheduler") or {}).get("tts_cpu_only")
            ),
        },
    }


@router.post("/settings")
async def settings_post(data: dict = Body(...)):
    """Partial update of settings. Writes to `config.json` under `llm.*`.

    - `api_key == ""` is treated as "no change" (mask token) — strip it.
    - `api_key == "***"` means the client echoed back the mask — also strip.
    - Any explicit non-empty value is persisted.
    """
    c = cfg.load_config()
    llm_in = data.get("llm") or {}
    if isinstance(llm_in, dict):
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


@router.get("/settings/models")
async def settings_models(
    base_url: str = "", provider: str = "lmstudio", api_key: str = ""
):
    """Proxy list_models to the chosen local provider (never call from browser)."""
    from slopfinity.llm.providers import get_provider

    if not base_url:
        return JSONResponse({"ok": False, "error": "missing base_url"}, status_code=400)
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


@router.post("/settings/test")
async def settings_test(data: dict = Body(...)):
    base_url = (data.get("base_url") or "").strip()
    provider = (data.get("provider") or "lmstudio").strip()
    model_id = (data.get("model_id") or "").strip()
    api_key = data.get("api_key") or ""
    if api_key in ("***",):
        api_key = (cfg.load_config().get("llm") or {}).get("api_key") or ""
    if not base_url or not model_id:
        return {"ok": False, "error": "base_url and model_id required", "latency_ms": 0}
    # Also count models to enrich the ✓ badge
    from slopfinity.llm.providers import get_provider

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


@router.get("/settings/probe")
async def settings_probe():
    """Async scan of well-known local LLM ports."""
    found = await llm_discover(timeout=1.0)
    return {"ok": True, "endpoints": found}
