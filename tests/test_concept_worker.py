"""Tests for slopfinity.workers.concept.ConceptWorker.

Mocks `lmstudio_call` so we exercise the worker without hitting a real
local LLM. Like test_worker_base, injects a Phase 1 queue_schema stub
into base.qs.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import config as _config  # noqa: E402
from slopfinity.workers import base as worker_base  # noqa: E402
from slopfinity.workers import concept as concept_mod  # noqa: E402


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
    def migrate_legacy(item):
        if item.get("schema_version") == 2:
            return item
        item = dict(item)
        item["schema_version"] = 2
        item.setdefault("id", f"q-{id(item)}")
        item["stages"] = {s: {"status": "needs"} for s in STAGE_ORDER}
        return item

    @staticmethod
    def stage_status(item, stage):
        return ((item.get("stages") or {}).get(stage) or {}).get("status", "needs")

    @staticmethod
    def set_stage_status(item, stage, status, **fields):
        s = item.setdefault("stages", {}).setdefault(stage, {})
        s["status"] = status
        for k, v in fields.items():
            s[k] = v

    @staticmethod
    def prerequisites_met(item, stage):
        for prereq in PREREQS.get(stage, []):
            if _SchemaStub.stage_status(item, prereq) not in ("done", "skipped"):
                return False
        return True

    @staticmethod
    def next_stage_for_role(item, role) -> Optional[str]:
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
    monkeypatch.setattr(_config, "QUEUE_FILE", str(tmp_path / "queue.json"))


def _mk_item(item_id, prompt, stages=None, snapshot=None):
    item = {
        "id": item_id,
        "schema_version": 2,
        "prompt": prompt,
        "stages": stages or {s: {"status": "needs"} for s in STAGE_ORDER},
    }
    if snapshot is not None:
        item["config_snapshot"] = snapshot
    return item


def test_concept_worker_role():
    assert concept_mod.ConceptWorker.role == "llm"


def test_concept_worker_happy_path(schema, isolated_state, monkeypatch):
    item = _mk_item("a", "a cat in a hat", snapshot={"enhancer_prompt": "Rewrite cinematically."})
    _config.save_queue([item])

    fake = mock.Mock(return_value="A regal feline poised under dramatic chiaroscuro lighting...")
    monkeypatch.setattr(concept_mod, "lmstudio_call", fake)

    w = concept_mod.ConceptWorker("llm-w0")
    assert asyncio.run(w.run_once()) is True

    fake.assert_called_once_with("Rewrite cinematically.", "a cat in a hat")

    persisted = _config.get_queue()
    cs = persisted[0]["stages"]["concept"]
    assert cs["status"] == "done"
    assert cs["output"].startswith("A regal feline")
    # Image stage's prereqs are now met.
    assert _SchemaStub.prerequisites_met(persisted[0], "image") is True


def test_concept_worker_falls_back_to_live_config(schema, isolated_state, monkeypatch):
    """No config_snapshot → load_config()['enhancer_prompt'] is used."""
    item = _mk_item("a", "hello world")
    _config.save_queue([item])

    monkeypatch.setattr(
        concept_mod._config,
        "load_config",
        lambda: {"enhancer_prompt": "BE TERSE"},
    )
    fake = mock.Mock(return_value="hi.")
    monkeypatch.setattr(concept_mod, "lmstudio_call", fake)

    w = concept_mod.ConceptWorker("llm-w0")
    asyncio.run(w.run_once())
    fake.assert_called_once_with("BE TERSE", "hello world")


def test_concept_worker_empty_prompt_fails(schema, isolated_state, monkeypatch):
    item = _mk_item("a", "   ", snapshot={"enhancer_prompt": "x"})
    _config.save_queue([item])
    fake = mock.Mock()
    monkeypatch.setattr(concept_mod, "lmstudio_call", fake)

    w = concept_mod.ConceptWorker("llm-w0")
    asyncio.run(w.run_once())

    fake.assert_not_called()
    cs = _config.get_queue()[0]["stages"]["concept"]
    assert cs["status"] == "failed"
    assert "empty prompt" in cs["error"]


def test_concept_worker_llm_error_string_is_failure(schema, isolated_state, monkeypatch):
    """lmstudio_call returns 'Error: ...' on connection failure → failed."""
    item = _mk_item("a", "x", snapshot={"enhancer_prompt": "y"})
    _config.save_queue([item])

    monkeypatch.setattr(
        concept_mod, "lmstudio_call", mock.Mock(return_value="Error: connection refused")
    )

    w = concept_mod.ConceptWorker("llm-w0")
    asyncio.run(w.run_once())

    cs = _config.get_queue()[0]["stages"]["concept"]
    assert cs["status"] == "failed"
    assert "connection refused" in cs["error"]


def test_concept_worker_exception_is_failure(schema, isolated_state, monkeypatch):
    item = _mk_item("a", "x", snapshot={"enhancer_prompt": "y"})
    _config.save_queue([item])

    monkeypatch.setattr(
        concept_mod, "lmstudio_call", mock.Mock(side_effect=RuntimeError("boom"))
    )

    w = concept_mod.ConceptWorker("llm-w0")
    asyncio.run(w.run_once())

    cs = _config.get_queue()[0]["stages"]["concept"]
    assert cs["status"] == "failed"
    assert "boom" in cs["error"]


def test_concept_worker_unblocks_image_stage(schema, isolated_state, monkeypatch):
    """After concept done, image stage prereqs are met (key invariant)."""
    item = _mk_item("a", "x", snapshot={"enhancer_prompt": "y"})
    _config.save_queue([item])

    monkeypatch.setattr(concept_mod, "lmstudio_call", mock.Mock(return_value="ok"))

    pre = _config.get_queue()[0]
    assert _SchemaStub.prerequisites_met(pre, "image") is False

    w = concept_mod.ConceptWorker("llm-w0")
    asyncio.run(w.run_once())

    post = _config.get_queue()[0]
    assert _SchemaStub.prerequisites_met(post, "image") is True
    assert _SchemaStub.prerequisites_met(post, "audio") is True
    assert _SchemaStub.prerequisites_met(post, "tts") is True
