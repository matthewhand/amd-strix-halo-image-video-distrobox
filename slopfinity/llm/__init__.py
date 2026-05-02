"""Local-only LLM abstraction package.

Back-compat surface: `lmstudio_call(sys_p, user_p)` is still exported,
but it now respects `config.json -> llm.*` so any configured local
provider (LM Studio, Ollama, vLLM, llama.cpp, custom OpenAI-compat) is
used. No cloud providers are supported.
"""
from __future__ import annotations

from .providers import get_provider, list_providers, Provider
from .probe import discover, ping


DEFAULT_LLM_CONFIG = {
    "provider": "lmstudio",
    "base_url": "http://localhost:1234/v1",
    "model_id": "",
    "api_key": "",
    "temperature": 0.7,
    "max_retries": 2,
    "timeout_s": 60,
    "extra_headers": {},
    # When true, the scheduler SIGSTOPs local LLM processes before each
    # heavy GPU stage and SIGCONTs after, freeing ~8 GB of unified RAM.
    # Default off — opt-in via Settings → LLM → Generation.
    "auto_suspend": False,
}


def _load_llm_cfg() -> dict:
    from .. import config as cfg
    c = cfg.load_config()
    llm = dict(DEFAULT_LLM_CONFIG)
    llm.update(c.get("llm") or {})
    return llm


def _auto_pick_model(provider, base_url, api_key, timeout) -> str | None:
    try:
        models = provider.list_models(base_url, api_key=api_key or None, timeout=timeout)
    except Exception:
        return None
    non_embed = [m["id"] for m in models if "embed" not in (m["id"] or "").lower()]
    if non_embed:
        return non_embed[0]
    return models[0]["id"] if models else None


def lmstudio_call(sys_p: str, user_p: str, response_format: dict | None = None) -> str:
    """Back-compat entry point used by /enhance + /subjects/suggest.

    Name retained for history; now dispatches to the configured local
    provider. Optional `response_format` passes an OpenAI-style
    structured-output spec through to the provider — used by
    /subjects/suggest to constrain LLM output to a strict
    {"suggestions": [...]} JSON document so chips can never contain
    markdown / scaffolding leaks.
    """
    from .pool import get_env_pool_config
    from .. import scheduler

    llm = _load_llm_cfg()
    timeout = int(llm.get("timeout_s") or 60)
    temperature = float(llm.get("temperature") or 0.7)
    extra_headers = llm.get("extra_headers") or None
    
    cpu_mode = (llm.get("scheduler") or {}).get("llm_cpu_mode") or "smart"
    is_gpu_busy = scheduler.GPU.resident_gb > 0 or bool(scheduler.GPU.in_flight)
    
    pool_cfg = get_env_pool_config()
    prefer_cpu = (cpu_mode == "smart" and is_gpu_busy) or cpu_mode == "cpu"
    
    endpoints = []
    if prefer_cpu and pool_cfg["cpu"]["url"]:
        endpoints.append(pool_cfg["cpu"])
    if pool_cfg["primary"]["url"]:
        endpoints.append(pool_cfg["primary"])
    for f in pool_cfg["failovers"]:
        if f["url"]:
            endpoints.append(f)
            
    seen = set()
    unique_endpoints = []
    for ep in endpoints:
        if ep["url"] not in seen:
            seen.add(ep["url"])
            unique_endpoints.append(ep)

    last_err = None
    for ep in unique_endpoints:
        base_url = ep["url"].rstrip("/")
        model_id = ep["model"]
        
        provider_name = "ollama" if "11434" in base_url else "lmstudio"
        provider = get_provider(provider_name)
        
        if not model_id:
            model_id = _auto_pick_model(provider, base_url, None, timeout=5) or ""
            if not model_id:
                last_err = f"No model available on {base_url}"
                continue
                
        try:
            return provider.chat(
                base_url=base_url,
                model_id=model_id,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p},
                ],
                temperature=temperature,
                api_key=None,
                timeout=timeout,
                extra_headers=extra_headers,
                response_format=response_format,
            )
        except Exception as e:
            last_err = e
            continue
            
    return f"Error: All endpoints failed. Last error: {last_err}"


def lmstudio_chat_raw(messages: list, tools: list | None = None,
                       temperature: float | None = None,
                       timeout_override: int | None = None) -> dict:
    """Multi-turn chat with optional tool-calling. Returns the full
    ``choices[0].message`` dict (content + optional tool_calls) so the
    caller can detect tool-call requests and feed the results back.

    Wraps the configured provider's /v1/chat/completions endpoint with
    OpenAI-compat tool plumbing. Used by the dashboard's Chat mode.
    """
    import urllib.request
    import urllib.error
    import json
    from .providers import _http_json, _auth_headers

    from .pool import get_env_pool_config
    from .. import scheduler

    llm = _load_llm_cfg()
    timeout = int(timeout_override or llm.get("timeout_s") or 120)
    temp = float(temperature if temperature is not None else (llm.get("temperature") or 0.7))
    extra_headers = llm.get("extra_headers") or None
    
    cpu_mode = (llm.get("scheduler") or {}).get("llm_cpu_mode") or "smart"
    is_gpu_busy = scheduler.GPU.resident_gb > 0 or bool(scheduler.GPU.in_flight)
    
    pool_cfg = get_env_pool_config()
    prefer_cpu = (cpu_mode == "smart" and is_gpu_busy) or cpu_mode == "cpu"
    
    endpoints = []
    if prefer_cpu and pool_cfg["cpu"]["url"]:
        endpoints.append(pool_cfg["cpu"])
    if pool_cfg["primary"]["url"]:
        endpoints.append(pool_cfg["primary"])
    for f in pool_cfg["failovers"]:
        if f["url"]:
            endpoints.append(f)
            
    seen = set()
    unique_endpoints = []
    for ep in endpoints:
        if ep["url"] not in seen:
            seen.add(ep["url"])
            unique_endpoints.append(ep)

    last_err = None
    for ep in unique_endpoints:
        base_url = ep["url"].rstrip("/")
        model_id = ep["model"]
        
        provider_name = "ollama" if "11434" in base_url else "lmstudio"
        provider = get_provider(provider_name)
        
        if not model_id:
            model_id = _auto_pick_model(provider, base_url, None, timeout=5) or ""
            if not model_id:
                last_err = f"No model available on {base_url}"
                continue

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temp,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = base_url + "/chat/completions"
        try:
            data = _http_json("POST", url, body=payload,
                              headers=_auth_headers(None, extra_headers),
                              timeout=timeout)
            msg = data.get("choices", [{}])[0].get("message") or {}
            if msg.get("content") is None:
                msg["content"] = ""
            return msg
        except Exception as e:
            last_err = e
            continue

    return {"role": "assistant", "content": f"Error: All endpoints failed. Last error: {last_err}"}


__all__ = [
    "lmstudio_call",
    "lmstudio_chat_raw",
    "get_provider",
    "list_providers",
    "Provider",
    "discover",
    "ping",
    "DEFAULT_LLM_CONFIG",
]

import asyncio
_LLM_LOCK = asyncio.Lock()
