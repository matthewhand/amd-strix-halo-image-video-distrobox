"""Compat shim for Phase 1+2 modules that may not yet be on `main`.

Phase 3 worker subclasses depend on:
  - slopfinity.queue_schema     (Phase 1)
  - slopfinity.workers.base     (Phase 2 — StageWorker)

Either may not exist when this PR lands, so we provide minimal placeholder
types. Once Phases 1+2 merge, this module is unused and can be deleted.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------- Phase 2: StageWorker ----------------------------------------------

try:  # pragma: no cover — exercised only after Phase 2 lands
    from .base import StageWorker as _RealStageWorker  # type: ignore[attr-defined]
    StageWorker = _RealStageWorker
    HAS_REAL_STAGE_WORKER = True
except Exception:  # ImportError or anything else
    HAS_REAL_STAGE_WORKER = False

    class StageWorker:  # type: ignore[no-redef]
        """Minimal placeholder — mirrors the Phase 2 contract.

        Subclasses set `role` (str) and override `run_stage(item)` to return
        a dict like `{"ok": bool, "output": Any, "asset": Optional[str]}`.

        The real Phase 2 base provides `claim()`, queue.json mutation, event
        emission, etc. We expose just enough surface so subclasses don't have
        to branch on whether Phase 2 is present.
        """

        role: str = ""

        def __init__(self, role: Optional[str] = None) -> None:
            if role is not None:
                self.role = role

        def can_claim(self, item: Any) -> bool:
            """True if this worker's role matches the item's next pending stage."""
            try:
                stages = getattr(item, "stages", None) or item.get("stages", {})
                stage = stages.get(self.role) if isinstance(stages, dict) else getattr(stages, self.role, None)
                if stage is None:
                    return False
                status = stage.get("status") if isinstance(stage, dict) else getattr(stage, "status", None)
                return status in (None, "pending", "queued")
            except Exception:
                return False

        async def run_stage(self, item: Any) -> dict:  # pragma: no cover — abstract
            raise NotImplementedError


# ---------- Phase 1: queue_schema --------------------------------------------

try:  # pragma: no cover — exercised only after Phase 1 lands
    from .. import queue_schema as _qs  # type: ignore[attr-defined]
    HAS_QUEUE_SCHEMA = True
    QueueItem = getattr(_qs, "QueueItem", dict)
except Exception:
    HAS_QUEUE_SCHEMA = False
    QueueItem = dict  # type: ignore[misc,assignment]


def stage_get(item: Any, role: str, key: str, default: Any = None) -> Any:
    """Read `item.stages[role][key]` tolerating dict-or-object shapes."""
    try:
        stages = getattr(item, "stages", None)
        if stages is None and isinstance(item, dict):
            stages = item.get("stages", {})
        stage = stages.get(role) if isinstance(stages, dict) else getattr(stages, role, None)
        if stage is None:
            return default
        if isinstance(stage, dict):
            return stage.get(key, default)
        return getattr(stage, key, default)
    except Exception:
        return default


def config_snapshot_get(item: Any, key: str, default: Any = None) -> Any:
    """Read `item.config_snapshot[key]` tolerating dict-or-object shapes."""
    try:
        snap = getattr(item, "config_snapshot", None)
        if snap is None and isinstance(item, dict):
            snap = item.get("config_snapshot", {})
        if isinstance(snap, dict):
            return snap.get(key, default)
        return getattr(snap, key, default)
    except Exception:
        return default


def item_v_idx(item: Any) -> int:
    """Read `item.v_idx` (or item['v_idx']); default 0."""
    try:
        v = getattr(item, "v_idx", None)
        if v is None and isinstance(item, dict):
            v = item.get("v_idx", 0)
        return int(v or 0)
    except Exception:
        return 0
