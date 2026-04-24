"""Discover local LLM endpoints by scanning well-known ports."""
from __future__ import annotations

import asyncio
import time
import urllib.request
import urllib.error
import json


# (port, default_provider, base_url_template)
CANDIDATES = [
    (1234, "lmstudio", "http://localhost:{port}/v1"),
    (11434, "ollama", "http://localhost:{port}"),
    (8080, "llamacpp", "http://localhost:{port}/v1"),
    (8000, "vllm", "http://localhost:{port}/v1"),
]


def _probe_one(port: int, provider: str, base_url: str, timeout: float = 1.0) -> dict | None:
    urls_to_try = []
    if provider == "ollama":
        urls_to_try.append((base_url + "/v1/models", "ollama-compat"))
        urls_to_try.append((base_url + "/api/tags", "ollama-native"))
    else:
        urls_to_try.append((base_url.rstrip("/") + "/models", provider))

    for url, flavor in urls_to_try:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
                if flavor == "ollama-native":
                    return {
                        "port": port,
                        "provider": "ollama",
                        "base_url": base_url,
                        "model_count": len(data.get("models") or []),
                        "flavor": "native",
                    }
                if flavor == "ollama-compat":
                    return {
                        "port": port,
                        "provider": "ollama",
                        "base_url": base_url + "/v1",
                        "model_count": len(data.get("data") or data.get("models") or []),
                        "flavor": "openai-compat",
                    }
                return {
                    "port": port,
                    "provider": provider,
                    "base_url": base_url,
                    "model_count": len(data.get("data") or data.get("models") or []),
                    "flavor": "openai-compat",
                }
        except Exception:
            continue
    return None


async def discover(timeout: float = 1.0) -> list[dict]:
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _probe_one, port, provider, tmpl.format(port=port), timeout)
        for (port, provider, tmpl) in CANDIDATES
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


def ping(base_url: str, provider: str, model_id: str, api_key: str | None = None, timeout: int = 10) -> dict:
    """Send a 1-token ping and return {ok, latency_ms, error}."""
    from .providers import get_provider
    p = get_provider(provider)
    t0 = time.time()
    try:
        msg = p.chat(
            base_url=base_url,
            model_id=model_id,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            api_key=api_key or None,
            timeout=timeout,
            max_tokens=1,
        )
        return {"ok": True, "latency_ms": int((time.time() - t0) * 1000), "reply": (msg or "")[:80]}
    except Exception as e:
        return {"ok": False, "latency_ms": int((time.time() - t0) * 1000), "error": str(e)}
