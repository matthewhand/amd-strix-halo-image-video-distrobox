import os
from dotenv import load_dotenv
import asyncio

from .providers import get_provider

load_dotenv()

def get_env_pool_config():
    """Reads the LLM endpoint pool from .env."""
    primary_url = os.environ.get("SLOPFINITY_LLM_PRIMARY_URL", "http://localhost:1234/v1")
    primary_model = os.environ.get("SLOPFINITY_LLM_PRIMARY_MODEL", "")
    
    cpu_url = os.environ.get("SLOPFINITY_LLM_CPU_URL", "http://localhost:11434/v1")
    cpu_model = os.environ.get("SLOPFINITY_LLM_CPU_MODEL", "")
    
    failover_urls_str = os.environ.get("SLOPFINITY_LLM_FAILOVER_URLS", "")
    failover_models_str = os.environ.get("SLOPFINITY_LLM_FAILOVER_MODELS", "")
    
    failover_urls = [u.strip() for u in failover_urls_str.split(",") if u.strip()]
    failover_models = [m.strip() for m in failover_models_str.split(",")] if failover_models_str else []
    
    # Pad models list if shorter than urls
    while len(failover_models) < len(failover_urls):
        failover_models.append("")
        
    failovers = [{"url": u, "model": m} for u, m in zip(failover_urls, failover_models)]
    
    return {
        "primary": {"url": primary_url, "model": primary_model},
        "cpu": {"url": cpu_url, "model": cpu_model},
        "failovers": failovers
    }

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
    """Probes all endpoints in the pool and returns their status."""
    cfg = get_env_pool_config()
    
    primary_task = probe_endpoint(cfg["primary"]["url"], cfg["primary"]["model"])
    cpu_task = probe_endpoint(cfg["cpu"]["url"], cfg["cpu"]["model"], provider_name="ollama")
    
    failover_tasks = [
        probe_endpoint(f["url"], f["model"], provider_name="ollama")
        for f in cfg["failovers"]
    ]
    
    results = await asyncio.gather(primary_task, cpu_task, *failover_tasks)
    
    return {
        "primary": results[0],
        "cpu": results[1],
        "failovers": results[2:]
    }
