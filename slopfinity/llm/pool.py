import os
from dotenv import load_dotenv
import asyncio

from .providers import get_provider

load_dotenv()

# Operator default: Ollama on a dedicated NVIDIA box for CV/prompt LLM work.
# LM Studio (:1234) remains optional via env / config.json / failover.
_DEFAULT_PRIMARY_URL = "http://localhost:11434/v1"
_DEFAULT_PRIMARY_MODEL = "gemma4:12b"
_DEFAULT_FAILOVER_URL = "http://localhost:1234/v1"


def _config_llm_fallback():
    """Primary (url, model) from config.json's `llm` block.

    The env pool (SLOPFINITY_LLM_*) is the deployment override, but when no
    primary URL is set in the environment we honor the operator's saved
    config.json instead of silently defaulting. This keeps the documented
    config.json → LLM contract working (and lets the mock integration tests,
    which seed config.json with a dynamic port, resolve the right endpoint).
    Imported lazily to avoid an import cycle.
    """
    try:
        from .. import config as _cfg
        llm = (_cfg.load_config() or {}).get("llm") or {}
        return llm.get("base_url") or "", llm.get("model_id") or ""
    except Exception:
        return "", ""


def get_env_pool_config():
    """Reads the LLM endpoint pool from the environment (SLOPFINITY_LLM_*).

    Primary defaults to Ollama :11434 / gemma4:12b (dedicated NVIDIA LLM).
    LM Studio :1234 is the default failover when no FAILOVER_URLS env is set.
    """
    cfg_url, cfg_model = _config_llm_fallback()
    primary_url = (
        os.environ.get("SLOPFINITY_LLM_PRIMARY_URL")
        or cfg_url
        or _DEFAULT_PRIMARY_URL
    )
    primary_model = (
        os.environ.get("SLOPFINITY_LLM_PRIMARY_MODEL")
        or cfg_model
        or _DEFAULT_PRIMARY_MODEL
    )

    # Secondary slot historically named "cpu" — keep for API shape; default empty
    # so we don't double-probe the same Ollama host as primary.
    cpu_url = os.environ.get("SLOPFINITY_LLM_CPU_URL", "")
    cpu_model = os.environ.get("SLOPFINITY_LLM_CPU_MODEL", "")

    failover_urls_str = os.environ.get("SLOPFINITY_LLM_FAILOVER_URLS", "")
    failover_models_str = os.environ.get("SLOPFINITY_LLM_FAILOVER_MODELS", "")

    if failover_urls_str.strip():
        failover_urls = [u.strip() for u in failover_urls_str.split(",") if u.strip()]
        failover_models = (
            [m.strip() for m in failover_models_str.split(",")]
            if failover_models_str
            else []
        )
    else:
        # Default failover: LM Studio for heavier / alternate models when set.
        failover_urls = [_DEFAULT_FAILOVER_URL]
        failover_models = [""]

    while len(failover_models) < len(failover_urls):
        failover_models.append("")

    failovers = [{"url": u, "model": m} for u, m in zip(failover_urls, failover_models)]

    return {
        "primary": {"url": primary_url, "model": primary_model},
        "cpu": {"url": cpu_url, "model": cpu_model},
        "failovers": failovers,
    }


def _normalize_url(url):
    """Canonical form for URL identity comparison.

    Case-insensitive and trailing-slash-insensitive so that, e.g.,
    "http://Host:1234/v1/" and "http://host:1234/v1" are treated as the
    same endpoint and not probed/tried twice.
    """
    return (url or "").strip().rstrip("/").lower()


def _dedup_endpoints(endpoints):
    """De-duplicate a priority-ordered list of endpoint dicts.

    Order/priority is preserved (the caller supplies the chain in the
    desired order, e.g. primary, then cpu, then failovers). Duplicates
    are dropped on (normalized-url, model) identity, so the same endpoint
    is never probed/tried twice. An endpoint serving a *different* model
    on the same URL is kept (it is a distinct (url, model) pair).
    Endpoints with a blank URL are skipped.
    """
    seen = set()
    deduped = []
    for ep in endpoints:
        url = ep.get("url") if isinstance(ep, dict) else None
        if not url or not url.strip():
            continue
        key = (_normalize_url(url), (ep.get("model") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ep)
    return deduped

async def probe_endpoint(url, default_model, provider_name="lmstudio", timeout=5):
    """Probes an endpoint to see if it's alive and what models it has."""
    provider = get_provider(provider_name)
    try:
        models = await asyncio.to_thread(provider.list_models, url, None, timeout)
        ok = isinstance(models, list)
        
        available_models = [m["id"] for m in models] if ok else []
        
        # Pick model
        selected_model = default_model
        if not selected_model and available_models:
            non_embed = [m for m in available_models if "embed" not in m.lower()]
            selected_model = non_embed[0] if non_embed else available_models[0]
            
        return {
            "url": url,
            "ok": ok,
            "selected_model": selected_model,
            "available_models": available_models,
            "error": None
        }
    except Exception as e:
        return {
            "url": url,
            "ok": False,
            "selected_model": default_model,
            "available_models": [],
            "error": str(e)
        }

async def get_pool_status():
    """Probes all endpoints in the pool and returns their status.

    The pool is de-duplicated by (normalized-url, model) identity before
    probing so the same endpoint is never probed twice. Priority order is
    preserved (primary, then cpu, then failovers). The primary and cpu
    slots are always returned (even if a failover duplicates them); only
    redundant *failovers* are dropped from the probed set.
    """
    cfg = get_env_pool_config()

    def _provider_for(url: str, default: str = "ollama") -> str:
        u = (url or "").lower()
        if ":1234" in u:
            return "lmstudio"
        if ":11434" in u or "ollama" in u:
            return "ollama"
        return default

    primary_task = probe_endpoint(
        cfg["primary"]["url"],
        cfg["primary"]["model"],
        provider_name=_provider_for(cfg["primary"]["url"]),
    )
    cpu_task = probe_endpoint(
        cfg["cpu"]["url"],
        cfg["cpu"]["model"],
        provider_name=_provider_for(cfg["cpu"]["url"], "ollama"),
    )

    # Dedup failovers against the primary/cpu slots and against each other.
    seen = {
        (_normalize_url(cfg["primary"]["url"]), (cfg["primary"]["model"] or "").strip()),
        (_normalize_url(cfg["cpu"]["url"]), (cfg["cpu"]["model"] or "").strip()),
    }
    unique_failovers = []
    for f in cfg["failovers"]:
        if not f.get("url") or not f["url"].strip():
            continue
        key = (_normalize_url(f["url"]), (f.get("model") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        unique_failovers.append(f)

    failover_tasks = [
        probe_endpoint(f["url"], f["model"], provider_name="ollama")
        for f in unique_failovers
    ]

    results = await asyncio.gather(primary_task, cpu_task, *failover_tasks)

    return {
        "primary": results[0],
        "cpu": results[1],
        "failovers": list(results[2:])
    }
