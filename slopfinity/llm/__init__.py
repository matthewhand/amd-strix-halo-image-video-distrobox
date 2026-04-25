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


def lmstudio_call(sys_p: str, user_p: str) -> str:
    """Back-compat entry point used by /enhance.

    Name retained for history; now dispatches to the configured local provider.
    """
    llm = _load_llm_cfg()
    provider = get_provider(llm.get("provider") or "lmstudio")
    base_url = llm.get("base_url") or "http://localhost:1234/v1"
    api_key = llm.get("api_key") or None
    timeout = int(llm.get("timeout_s") or 60)
    temperature = float(llm.get("temperature") or 0.7)
    model_id = llm.get("model_id") or ""
    extra_headers = llm.get("extra_headers") or None

    if not model_id:
        model_id = _auto_pick_model(provider, base_url, api_key, timeout=5) or ""
        if not model_id:
            return "Error: no model available from configured provider"

    max_retries = max(0, int(llm.get("max_retries") or 0))
    last_err = None
    for _ in range(max_retries + 1):
        try:
            return provider.chat(
                base_url=base_url,
                model_id=model_id,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p},
                ],
                temperature=temperature,
                api_key=api_key,
                timeout=timeout,
                extra_headers=extra_headers,
            )
        except Exception as e:
            last_err = e
    return f"Error: {last_err}"


__all__ = [
    "lmstudio_call",
    "get_provider",
    "list_providers",
    "Provider",
    "discover",
    "ping",
    "DEFAULT_LLM_CONFIG",
]
