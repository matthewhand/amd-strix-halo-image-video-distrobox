"""Local-only LLM provider implementations.

Slopfinity is a local-only stack. NO cloud providers (OpenAI, Anthropic,
Groq, Together, Fireworks, etc.) are supported or will be added here.
All providers listed below expect an endpoint bound to localhost or a
LAN address inside the user's own network.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Protocol, Optional


DEFAULT_TIMEOUT = 60
DEFAULT_LIST_TIMEOUT = 5


def _http_json(method: str, url: str, *, body=None, headers=None, timeout: int = DEFAULT_TIMEOUT):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if not raw:
            return {}
        return json.loads(raw)


class Provider(Protocol):
    name: str

    def list_models(self, base_url: str, api_key: Optional[str] = None, timeout: int = DEFAULT_LIST_TIMEOUT) -> list[dict]: ...

    def chat(
        self,
        base_url: str,
        model_id: str,
        messages: list[dict],
        temperature: float,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        extra_headers: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str: ...


def _auth_headers(api_key: Optional[str], extra_headers: Optional[dict] = None) -> dict:
    h = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        h.update(extra_headers)
    return h


class _OpenAICompatBase:
    """Shared implementation for any OpenAI-compatible local server."""
    name = "openai-compat"

    def list_models(self, base_url, api_key=None, timeout=DEFAULT_LIST_TIMEOUT):
        url = base_url.rstrip("/") + "/models"
        data = _http_json("GET", url, headers=_auth_headers(api_key), timeout=timeout)
        items = data.get("data") or data.get("models") or []
        out = []
        for m in items:
            if isinstance(m, dict):
                out.append({"id": m.get("id") or m.get("name") or "", "raw": m})
            elif isinstance(m, str):
                out.append({"id": m, "raw": {}})
        return [m for m in out if m["id"]]

    def chat(self, base_url, model_id, messages, temperature, api_key=None,
             timeout=DEFAULT_TIMEOUT, extra_headers=None, max_tokens=None,
             response_format=None):
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # OpenAI / LM Studio / llama.cpp structured-output support.
        # Pass-through dict like
        #   {"type": "json_schema", "json_schema": {"name": "...",
        #    "schema": {...}, "strict": true}}
        # constrains the LLM to emit a JSON document matching the
        # schema. Used by suggestion fetches so the model can't leak
        # markdown headers / scaffolding text into chips.
        if response_format is not None:
            payload["response_format"] = response_format
        data = _http_json(
            "POST", url, body=payload,
            headers=_auth_headers(api_key, extra_headers),
            timeout=timeout,
        )
        try:
            msg = data["choices"][0]["message"]
            # Reasoning models (qwen3.5-claude-distill, deepseek-r1, etc.)
            # split output into `content` (final answer) + `reasoning_content`
            # (chain-of-thought). When the response is truncated mid-thought
            # `content` can be empty while `reasoning_content` carries the
            # actual text. Fall through to reasoning_content so callers
            # don't get an empty string back.
            content = (msg.get("content") or "").strip()
            if not content:
                content = (msg.get("reasoning_content") or "").strip()
            return content
        except Exception as e:
            raise RuntimeError(f"Malformed chat response: {e}: {data}")


class LMStudioProvider(_OpenAICompatBase):
    name = "lmstudio"


class VLLMProvider(_OpenAICompatBase):
    name = "vllm"


class LlamaCppProvider(_OpenAICompatBase):
    name = "llamacpp"


class OpenAICompatProvider(_OpenAICompatBase):
    """Catch-all for any user-specified OpenAI-compatible local server."""
    name = "custom"


class OllamaProvider:
    """Ollama native API with OpenAI-compat fallback preference.

    If `{base}/v1/models` responds, we treat it as OpenAI-compat (preferred).
    Otherwise fall back to the native `/api/chat` + `/api/tags` endpoints.
    """
    name = "ollama"

    def _compat_base(self, base_url: str) -> str:
        b = base_url.rstrip("/")
        if b.endswith("/v1"):
            return b
        return b + "/v1"

    def _native_base(self, base_url: str) -> str:
        b = base_url.rstrip("/")
        if b.endswith("/v1"):
            b = b[:-3]
        return b

    def _openai_ok(self, base_url, api_key, timeout) -> bool:
        try:
            url = self._compat_base(base_url) + "/models"
            _http_json("GET", url, headers=_auth_headers(api_key), timeout=timeout)
            return True
        except Exception:
            return False

    def list_models(self, base_url, api_key=None, timeout=DEFAULT_LIST_TIMEOUT):
        if self._openai_ok(base_url, api_key, timeout):
            return _OpenAICompatBase().list_models(self._compat_base(base_url), api_key, timeout)
        url = self._native_base(base_url) + "/api/tags"
        data = _http_json("GET", url, timeout=timeout)
        return [{"id": m.get("name", ""), "raw": m} for m in (data.get("models") or []) if m.get("name")]

    def chat(self, base_url, model_id, messages, temperature, api_key=None,
             timeout=DEFAULT_TIMEOUT, extra_headers=None, max_tokens=None,
             response_format=None):
        if self._openai_ok(base_url, api_key, timeout):
            return _OpenAICompatBase().chat(
                self._compat_base(base_url), model_id, messages, temperature,
                api_key, timeout, extra_headers, max_tokens, response_format,
            )
        url = self._native_base(base_url) + "/api/chat"
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        # Ollama native API supports `format: "json"` for free-form JSON
        # constraint (no schema). Best-effort: if response_format asks
        # for JSON, request JSON mode; the schema details are dropped.
        if response_format and response_format.get("type", "").startswith("json"):
            payload["format"] = "json"
        data = _http_json("POST", url, body=payload,
                          headers=_auth_headers(None, extra_headers), timeout=timeout)
        msg = (data.get("message") or {}).get("content", "")
        return (msg or "").strip()


_REGISTRY = {
    "lmstudio": LMStudioProvider(),
    "ollama": OllamaProvider(),
    "vllm": VLLMProvider(),
    "llamacpp": LlamaCppProvider(),
    "custom": OpenAICompatProvider(),
}


def get_provider(name: str) -> Provider:
    key = (name or "").lower().strip()
    return _REGISTRY.get(key, _REGISTRY["lmstudio"])


def list_providers() -> list[str]:
    return list(_REGISTRY.keys())
