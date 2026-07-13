"""Hermetic smoke test for the router endpoints that 500'd after the Phase-1
server.py -> routers split dropped a batch of module-level imports / constants
/ helpers, leaving undefined names that raised NameError at request time.

These assertions guard against regression of those specific 500s:
  - GET /coordinator/status   (needed _coordinator + _coord_imp_err_repr)
  - GET /pipeline/plan        (needed the corrected memory_planner import)
  - GET /pipeline/slopped     (needed _SLOPPED_EXTS)
  - GET /disk/guard           (sanity: shares the runner router)
  - GET /healthz              (app still boots clean)

GET-only on purpose: the CSRF/Origin middleware only guards mutating methods,
so a plain TestClient GET sails through without origin spoofing.

EXP_DIR is read from SLOPFINITY_EXP_DIR at import time (slopfinity.paths), so we
point it at a throwaway tmp dir *before* importing the app.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Must be set before any slopfinity import resolves EXP_DIR.
_TMP = tempfile.mkdtemp(prefix="slop_router_smoke_")
os.environ.setdefault("SLOPFINITY_EXP_DIR", _TMP)

from slopfinity.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize(
    "path",
    [
        "/healthz",
        "/coordinator/status",
        "/pipeline/plan",
        "/pipeline/slopped?role=image",
        "/disk/guard",
        "/services",
    ],
)
def test_router_endpoints_non_5xx(client, path):
    """None of these may 5xx — that was the symptom of the dropped imports."""
    resp = client.get(path)
    assert resp.status_code < 500, (
        f"GET {path} returned {resp.status_code} "
        f"(regression of dropped-import 500): {resp.text[:300]}"
    )
