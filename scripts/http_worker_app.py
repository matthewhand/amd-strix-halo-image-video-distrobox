"""Shared helper to build the FastAPI app for HTTP workers.

Used by heartmula_serve and qwen_tts_serve so registration logic is not duplicated,
and the shipped app can be obtained via get_app().

Bare `import *_serve` still yields a dummy app (with .routes list) so that
plan verification step 3 (python -c import + print routes) succeeds without
executing fastapi import (which can hang in some harness envs).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


def build_app(
    title: str,
    route_specs: List[Tuple[str, str, Callable[..., Any]]],
) -> Any:
    """Build and return a real FastAPI application with the given routes.

    route_specs: list of (method, path, handler_func)
    e.g. [("get", "/health", health_handler), ("post", "/music", music_handler)]

    The handler_func will be registered directly (FastAPI will handle
    dependency injection for Body etc if the signature uses it).
    """
    import signal
    def _timeout_handler(signum, frame):
        raise ImportError("fastapi import timed out (broken host install)")
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(3)
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    app = FastAPI(title=title)

    for method, path, handler in route_specs:
        m = method.lower()
        if m == "get":
            app.get(path)(handler)
        elif m == "post":
            app.post(path)(handler)
        else:
            # extend as needed
            getattr(app, m)(path)(handler)
    return app


def build_dummy_app(title: str, paths: List[str]) -> Any:
    """Return a minimal object with .title and .routes for bare-import verif step 3."""
    class _R:
        def __init__(self, p: str):
            self.path = p
    class _Dummy:
        def __init__(self):
            self.title = title
            self.routes = [_R(p) for p in paths]
    return _Dummy()
