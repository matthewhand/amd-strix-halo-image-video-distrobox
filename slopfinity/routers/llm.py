import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import asyncio
import slopfinity.scheduler as sched
import slopfinity.config as cfg


router = APIRouter()

@router.get("/llm/health")
async def llm_health_endpoint():
    """Cheap reachability probe for the configured LLM provider.

    HTTP-only: hits the provider's `/v1/models` (or native equivalent) and
    treats a successful response as "alive". Deliberately does NOT run an
    inference — the dashboard polls this every 60 s, and burning a token
    each time would force the model to stay resident even while the GPU
    is busy with diffusion stages (the auto-suspend dance would constantly
    fight a synthetic ping). A reachable HTTP server is all the UI needs
    to decide whether LLM-dependent modes (Endless / Simple / Chat) are
    available; Raw mode is the fallback when this returns ok=false.
    """
    from slopfinity.llm.providers import get_provider
    config = cfg.load_config()
    llm = config.get("llm") or {}
    provider = llm.get("provider") or "lmstudio"
    base_url = llm.get("base_url") or "http://localhost:1234/v1"
    model_id = llm.get("model_id") or ""
    api_key = llm.get("api_key") or None
    try:
        models = await asyncio.to_thread(
            get_provider(provider).list_models, base_url, api_key, 5
        )
        ok = isinstance(models, list)
        return {"ok": ok, "provider": provider, "model_id": model_id,
                "model_count": len(models) if ok else 0,
                "error": None if ok else "no models endpoint"}
    except Exception as e:
        return {"ok": False, "provider": provider, "model_id": model_id,
                "error": str(e)}

@router.post("/llm/suspend")
async def llm_suspend_endpoint():
    """Manually SIGSTOP any running local LLM (LM Studio / Ollama).

    Independent of the `llm.auto_suspend` toggle — gives the user a one-shot
    pause for ad-hoc memory triage. Resume via POST /llm/resume.
    """
    result = await sched.suspend_llm_async()
    return {"ok": True, **result}

@router.post("/llm/resume")
async def llm_resume_endpoint():
    """Manually SIGCONT any suspended local LLM process."""
    result = await sched.resume_llm_async()
    return {"ok": True, **result}
