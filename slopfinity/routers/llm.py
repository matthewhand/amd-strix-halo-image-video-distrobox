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
    Now queries the primary from the pool.
    """
    from slopfinity.llm.pool import get_pool_status
    pool_status = await get_pool_status()
    
    primary = pool_status["primary"]
    return {
        "ok": primary["ok"], 
        "provider": "lmstudio", 
        "model_id": primary["selected_model"],
        "model_count": len(primary["available_models"]),
        "error": primary["error"]
    }

@router.get("/llm/pool")
async def llm_pool_endpoint():
    """Returns the full status of the distributed LLM failover pool."""
    from slopfinity.llm.pool import get_pool_status
    return await get_pool_status()

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
