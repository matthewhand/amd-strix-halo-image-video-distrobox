"""Tests for slopfinity.workers.base.StageWorker.

Phase 1 (`slopfinity.queue_schema`) has not landed yet, so these tests
inject a minimal in-process schema stub into `base.qs`. Once Phase 1
merges the same tests will run against the real module without changes
to test logic — the stub mirrors the public contract documented in
docs/queueing-refactor-design.md.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Optional

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import config as _config  # noqa: E402
from slopfinity.workers import base as worker_base  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal queue_schema stub matching the Phase 1 contract.
# ---------------------------------------------------------------------------
STAGE_ORDER = ["concept", "image", "video", "audio", "tts", "post", "merge"]
PREREQS = {
    "concept": [],
    "image": ["concept"],
    "video": ["image"],
    "audio": ["concept"],
    "tts": ["concept"],
    "post": ["video"],
    "merge": ["post", "audio", "tts"],
}
ROLE_STAGE = {
    "llm": ["concept"],
    "image": ["image"],
    "video": ["video", "post"],
    "audio": ["audio"],
    "tts": ["tts"],
    "post": ["post"],
    "ffmpeg": ["merge"],
}


class _SchemaStub:
    STAGE_ORDER = STAGE_ORDER
    PREREQS = PREREQS
    ROLE_STAGE = ROLE_STAGE

    @staticmethod
    def migrate_legacy(item: dict) -> dict:
        if item.get("schema_version") == 2:
            return item
        item = dict(item)
        item["schema_version"] = 2
        item.setdefault("id", f"q-{id(item)}")
        item["stages"] = {s: {"status": "needs"} for s in STAGE_ORDER}
        return item

    @staticmethod
    def stage_status(item: dict, stage: str) -> str:
        return ((item.get("stages") or {}).get(stage) or {}).get("status", "needs")

    @staticmethod
    def set_stage_status(item: dict, stage: str, status: str, **fields) -> None:
        stages = item.setdefault("stages", {})
        s = stages.setdefault(stage, {})
        s["status"] = status
        for k, v in fields.items():
            s[k] = v

    @staticmethod
    def prerequisites_met(item: dict, stage: str) -> bool:
        for prereq in PREREQS.get(stage, []):
            st = _SchemaStub.stage_status(item, prereq)
            if st not in ("done", "skipped"):
                return False
        return True

    @staticmethod
    def next_stage_for_role(item: dict, role: str) -> Optional[str]:
        for stage in ROLE_STAGE.get(role, []):
            if _SchemaStub.stage_status(item, stage) == "needs":
                return stage
        return None


@pytest.fixture
def schema(monkeypatch):
    monkeypatch.setattr(worker_base, "qs", _SchemaStub)
    return _SchemaStub


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    """Point config's QUEUE_FILE at a temp file so tests don't touch real state."""
    qfile = str(tmp_path / "queue.json")
    monkeypatch.setattr(_config, "QUEUE_FILE", qfile)
    return qfile


def _mk_item(item_id: str, stages: dict) -> dict:
    return {
        "id": item_id,
        "schema_version": 2,
        "prompt": f"prompt-{item_id}",
        "stages": {s: dict(v) for s, v in stages.items()},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_qs_guard_raises_when_unavailable(monkeypatch, isolated_state):
    """If queue_schema is missing, methods raise RuntimeError."""
    monkeypatch.setattr(worker_base, "qs", None)
    w = worker_base.StageWorker("w0")
    with pytest.raises(RuntimeError, match="queue_schema not available"):
        asyncio.run(w.claim_next())


def test_idle_when_no_items(schema, isolated_state):
    _config.save_queue([])
    w = worker_base.StageWorker("w0")
    w.role = "llm"
    assert asyncio.run(w.claim_next()) is None
    assert asyncio.run(w.run_once()) is False


def test_claim_picks_eligible_needs(schema, isolated_state):
    item = _mk_item("a", {s: {"status": "needs"} for s in STAGE_ORDER})
    _config.save_queue([item])

    w = worker_base.StageWorker("llm-w0")
    w.role = "llm"
    claimed = asyncio.run(w.claim_next())
    assert claimed is not None
    got_item, got_stage = claimed
    assert got_stage == "concept"
    assert got_item["id"] == "a"

    persisted = _config.get_queue()
    assert persisted[0]["stages"]["concept"]["status"] == "working"
    assert persisted[0]["stages"]["concept"]["worker"] == "llm-w0"
    assert "started_ts" in persisted[0]["stages"]["concept"]


def test_claim_respects_prereqs(schema, isolated_state):
    """ImageWorker shouldn't claim when concept isn't done yet."""
    item = _mk_item(
        "a",
        {
            "concept": {"status": "needs"},
            "image": {"status": "needs"},
            "video": {"status": "needs"},
            "audio": {"status": "needs"},
            "tts": {"status": "needs"},
            "post": {"status": "needs"},
            "merge": {"status": "needs"},
        },
    )
    _config.save_queue([item])

    w = worker_base.StageWorker("img-w0")
    w.role = "image"
    assert asyncio.run(w.claim_next()) is None  # prereq concept not done


def test_claim_after_prereq_done(schema, isolated_state):
    item = _mk_item(
        "a",
        {
            "concept": {"status": "done"},
            "image": {"status": "needs"},
            "video": {"status": "needs"},
            "audio": {"status": "needs"},
            "tts": {"status": "needs"},
            "post": {"status": "needs"},
            "merge": {"status": "needs"},
        },
    )
    _config.save_queue([item])

    w = worker_base.StageWorker("img-w0")
    w.role = "image"
    claimed = asyncio.run(w.claim_next())
    assert claimed is not None
    _, stage = claimed
    assert stage == "image"


def test_atomic_claim_race(schema, isolated_state):
    """Two workers see the same item; only one claims it."""
    item = _mk_item("a", {s: {"status": "needs"} for s in STAGE_ORDER})
    _config.save_queue([item])

    w1 = worker_base.StageWorker("llm-w0")
    w1.role = "llm"
    w2 = worker_base.StageWorker("llm-w1")
    w2.role = "llm"

    # Sequential calls — second worker sees the working state and returns None.
    first = asyncio.run(w1.claim_next())
    second = asyncio.run(w2.claim_next())

    assert first is not None
    assert second is None
    persisted = _config.get_queue()
    assert persisted[0]["stages"]["concept"]["worker"] == "llm-w0"


def test_claim_skips_already_working(schema, isolated_state):
    item = _mk_item(
        "a",
        {
            "concept": {"status": "working", "worker": "other"},
            "image": {"status": "needs"},
            "video": {"status": "needs"},
            "audio": {"status": "needs"},
            "tts": {"status": "needs"},
            "post": {"status": "needs"},
            "merge": {"status": "needs"},
        },
    )
    _config.save_queue([item])

    w = worker_base.StageWorker("llm-w0")
    w.role = "llm"
    assert asyncio.run(w.claim_next()) is None


def test_run_once_success_writes_done(schema, isolated_state):
    item = _mk_item("a", {s: {"status": "needs"} for s in STAGE_ORDER})
    _config.save_queue([item])

    class W(worker_base.StageWorker):
        role = "llm"

        async def run_stage(self, item, stage):
            return {"ok": True, "output": "rewritten"}

    w = W("llm-w0")
    assert asyncio.run(w.run_once()) is True

    persisted = _config.get_queue()
    cs = persisted[0]["stages"]["concept"]
    assert cs["status"] == "done"
    assert cs["output"] == "rewritten"
    assert "completed_ts" in cs


def test_run_once_failure_writes_failed(schema, isolated_state):
    item = _mk_item("a", {s: {"status": "needs"} for s in STAGE_ORDER})
    _config.save_queue([item])

    class W(worker_base.StageWorker):
        role = "llm"

        async def run_stage(self, item, stage):
            raise ValueError("boom")

    w = W("llm-w0")
    assert asyncio.run(w.run_once()) is True

    persisted = _config.get_queue()
    cs = persisted[0]["stages"]["concept"]
    assert cs["status"] == "failed"
    assert "boom" in cs["error"]


def test_claim_migrates_legacy_v1_items(schema, isolated_state):
    """v1 items (no `stages` dict) get migrated on read."""
    legacy = {"prompt": "hi", "status": "pending"}
    _config.save_queue([legacy])

    w = worker_base.StageWorker("llm-w0")
    w.role = "llm"
    claimed = asyncio.run(w.claim_next())
    assert claimed is not None
    _, stage = claimed
    assert stage == "concept"

    persisted = _config.get_queue()
    assert persisted[0]["schema_version"] == 2
    assert "stages" in persisted[0]


def test_claim_picks_first_eligible_in_order(schema, isolated_state):
    """Items are scanned in queue order; first eligible wins."""
    items = [
        _mk_item(
            "a",
            {
                "concept": {"status": "done"},
                "image": {"status": "done"},
                "video": {"status": "needs"},
                "audio": {"status": "needs"},
                "tts": {"status": "needs"},
                "post": {"status": "needs"},
                "merge": {"status": "needs"},
            },
        ),
        _mk_item("b", {s: {"status": "needs"} for s in STAGE_ORDER}),
    ]
    _config.save_queue(items)

    w = worker_base.StageWorker("llm-w0")
    w.role = "llm"
    claimed = asyncio.run(w.claim_next())
    assert claimed is not None
    got_item, _ = claimed
    # First eligible llm stage is item b's concept (item a's concept already done).
    assert got_item["id"] == "b"
