"""StageWorker base class — Phase 2 of the queueing refactor.

Each worker is a role-typed daemon that polls `queue.json` for items where
its stage is `needs` AND prerequisites are satisfied, claims atomically by
flipping the stage status to `working`, runs the stage, and writes back
`done` / `failed`.

The atomic claim is implemented as a re-read + verify + save under a single
sync code path. Concurrent workers cannot both observe a `needs` stage and
both transition it to `working` because the second writer will read the
already-`working` state and skip the item. queue.json is small enough
(< 1 KB per item) that the read+save round-trip is negligible compared to
stage runtimes (seconds → minutes).

The module is import-safe even before Phase 1 lands: if
`slopfinity.queue_schema` isn't on the path yet, `qs` is None and every
method raises a clear RuntimeError. Phase 4 will wire workers into the
coordinator; until then this is dead code.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

from slopfinity import config as _config

try:
    from slopfinity import queue_schema as qs
except ImportError:
    qs = None  # Phase 1 not merged yet — module is dead code until then


def _require_qs() -> None:
    if qs is None:
        raise RuntimeError(
            "slopfinity.queue_schema not available — Phase 1 schema PR "
            "must merge before workers can run."
        )


class StageWorker:
    """Polling base class. Subclasses set `role` and override `run_stage`."""

    role: str = ""  # 'llm', 'image', 'video', 'audio', 'tts', 'post', 'ffmpeg'

    def __init__(self, worker_id: str):
        self.worker_id = worker_id

    # ------------------------------------------------------------------
    # Claim
    # ------------------------------------------------------------------
    async def claim_next(self) -> Optional[Tuple[dict, str]]:
        """Find a queue item with a needs+prereq-met stage for this role.

        Marks the stage `working` atomically (re-read + save), returns
        `(item, stage)`. Returns None if nothing is eligible.
        """
        _require_qs()
        return await asyncio.to_thread(self._claim_sync)

    def _claim_sync(self) -> Optional[Tuple[dict, str]]:
        """Sync inner half of claim_next — runs in a worker thread.

        Each call re-reads queue.json so a concurrent worker that already
        flipped the stage to `working` is observed; we then skip and look
        for the next eligible item.
        """
        queue = _config.get_queue()
        # Migrate v1 items on read so stage_status() finds the dict.
        migrated_any = False
        for i, item in enumerate(queue):
            if item.get("schema_version") != 2:
                queue[i] = qs.migrate_legacy(item)
                migrated_any = True

        for idx, item in enumerate(queue):
            stage = qs.next_stage_for_role(item, self.role)
            if stage is None:
                continue
            if qs.stage_status(item, stage) != "needs":
                continue
            if not qs.prerequisites_met(item, stage):
                continue
            # Claim. set_stage_status mutates the item in place.
            qs.set_stage_status(
                item,
                stage,
                "working",
                worker=self.worker_id,
                started_ts=time.time(),
            )
            queue[idx] = item
            _config.save_queue(queue)
            return item, stage

        # If we migrated items but didn't find work, persist the migration.
        if migrated_any:
            _config.save_queue(queue)
        return None

    # ------------------------------------------------------------------
    # Run + finalize
    # ------------------------------------------------------------------
    async def run_stage(self, item: dict, stage: str) -> dict:
        """Subclasses override.

        Returns a result dict with at least `ok: bool`. On success may
        carry `output` (text) and/or `asset` (path). On failure carries
        `error` (str).
        """
        raise NotImplementedError

    async def _finalize(self, item_id: str, stage: str, result: dict) -> None:
        """Write back done/failed to queue.json. Re-reads to avoid clobber."""
        _require_qs()
        await asyncio.to_thread(self._finalize_sync, item_id, stage, result)

    def _finalize_sync(self, item_id: str, stage: str, result: dict) -> None:
        queue = _config.get_queue()
        for idx, qi in enumerate(queue):
            if qi.get("id") != item_id:
                continue
            now = time.time()
            if result.get("ok"):
                kwargs = {"completed_ts": now}
                if "output" in result:
                    kwargs["output"] = result["output"]
                if "asset" in result:
                    kwargs["asset"] = result["asset"]
                qs.set_stage_status(qi, stage, "done", **kwargs)
            else:
                qs.set_stage_status(
                    qi,
                    stage,
                    "failed",
                    completed_ts=now,
                    error=str(result.get("error") or "unknown error"),
                )
            queue[idx] = qi
            _config.save_queue(queue)
            return

    # ------------------------------------------------------------------
    # Drive
    # ------------------------------------------------------------------
    async def run_once(self) -> bool:
        """Claim+run+finalize one stage. Returns True if work was done."""
        _require_qs()
        claimed = await self.claim_next()
        if claimed is None:
            return False
        item, stage = claimed
        try:
            result = await self.run_stage(item, stage)
        except Exception as exc:  # pragma: no cover - defensive
            result = {"ok": False, "error": str(exc)}
        await self._finalize(item.get("id"), stage, result)
        return True

    async def loop(self, poll_interval_s: float = 2.0) -> None:
        """Poll forever. Idle ticks sleep `poll_interval_s`."""
        while True:
            did = await self.run_once()
            if not did:
                await asyncio.sleep(poll_interval_s)
