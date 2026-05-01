"""Slopfinity Coordinator — Phase 4 of the queueing refactor.

Replaces the linear stage-by-stage logic of `run_philosophical_experiments.py`
with a fan of concurrent `StageWorker` loops. Each worker pulls items from
the per-stage queue (Phase 1 schema) when its stage is `pending`, runs the
stage via the existing `slopfinity.workers` shims, and writes the result
back into `item.stages.<stage>.status`.

The actual GPU/budget serialization still lives in
`slopfinity.scheduler.acquire_gpu` — workers can run their *queue polling*
concurrently, but the GPU-resident sections are still gated by the
budget-aware lock. This means N stages may be `running` from the queue's
perspective while only M (budget-fitting) actually hold the GPU.

Defensive imports: Phases 1-3 (queue_schema + StageWorker base + per-stage
workers) may not be merged yet. We wrap the imports so this module can
land first and degrade gracefully (the `Coordinator` raises a clear error
on `run()` if its prerequisites are missing).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defensive imports — Phases 1-3 may land after this PR.
# ---------------------------------------------------------------------------
_IMPORT_ERROR: Optional[Exception] = None

try:
    # Phases 1-3 of the queueing refactor have shipped — each worker
    # class lives in its own module under slopfinity.workers.* (not the
    # `stage_workers` aggregate the original PR speculatively imported).
    # Without this fix the defensive `except` below permanently set
    # `_WORKERS_AVAILABLE = False` and `Coordinator.run()` raised
    # "Land Phases 1-3 first" forever — i.e. the entire Phase-4
    # concurrent-mode codepath was dead.
    from .workers.concept import ConceptWorker
    from .workers.image import ImageWorker
    from .workers.video import VideoWorker
    from .workers.audio import AudioWorker
    from .workers.tts import TTSWorker
    from .workers.post import PostWorker
    from .workers.merge import MergeWorker
    _WORKERS_AVAILABLE = True
except Exception as e:  # ImportError or AttributeError if a worker disappears
    _IMPORT_ERROR = e
    _WORKERS_AVAILABLE = False

    # Stub placeholders so type-hinting / instantiation failures are clear.
    class _MissingWorker:  # noqa: N801
        def __init__(self, name: str):
            self.name = name

        async def loop(self, poll_interval_s: float = 2.0) -> None:
            raise RuntimeError(
                "StageWorker classes are unavailable — Phase 1-3 of the "
                "queueing refactor must land before the coordinator can run. "
                f"Underlying import error: {_IMPORT_ERROR!r}"
            )

    class ConceptWorker(_MissingWorker):  # type: ignore[no-redef]
        pass

    class ImageWorker(_MissingWorker):  # type: ignore[no-redef]
        pass

    class VideoWorker(_MissingWorker):  # type: ignore[no-redef]
        pass

    class AudioWorker(_MissingWorker):  # type: ignore[no-redef]
        pass

    class TTSWorker(_MissingWorker):  # type: ignore[no-redef]
        pass

    class PostWorker(_MissingWorker):  # type: ignore[no-redef]
        pass

    class MergeWorker(_MissingWorker):  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Persistent running-state file. The dashboard's start/stop endpoints
# (Phase 4 / item 2) flip this so a slopfinity restart can resume the
# coordinator if it was running before the restart.
# ---------------------------------------------------------------------------
def _state_dir() -> str:
    return os.environ.get("SLOPFINITY_STATE_DIR") or "comfy-outputs/experiments"


COORDINATOR_STATE_FILE = os.path.join(_state_dir(), "coordinator.state.json")


def _read_state() -> dict:
    import json
    try:
        with open(COORDINATOR_STATE_FILE, "r") as f:
            v = json.load(f)
            if isinstance(v, dict):
                return v
    except Exception:
        pass
    return {"running": False, "ts": 0}


def _write_state(state: dict) -> None:
    import json
    os.makedirs(os.path.dirname(COORDINATOR_STATE_FILE), exist_ok=True)
    with open(COORDINATOR_STATE_FILE, "w") as f:
        json.dump(state, f)


def is_running() -> bool:
    """Best-effort 'is the coordinator supposed to be running' flag.

    Reads the persisted state file. The actual liveness of the asyncio
    tasks is tracked in `Coordinator._task_set` for the in-process case;
    this helper exists for cross-process callers (the dashboard restart
    path).
    """
    return bool(_read_state().get("running"))


def mark_running(running: bool) -> None:
    _write_state({"running": bool(running), "ts": time.time()})


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
class Coordinator:
    """Run all stage workers concurrently against the shared queue.

    Each worker is an independent `loop()` coroutine. Concurrency among
    GPU-heavy stages is still bounded by `scheduler.acquire_gpu`; the
    coordinator just removes the *single-threaded job ordering* constraint
    that the legacy fleet runner imposed.

    Usage:

        co = Coordinator()
        await co.run()                  # runs forever; cancel to stop

    Or, from another async context:

        co = Coordinator()
        await co.start()
        ...
        await co.stop()
    """

    def __init__(self, poll_interval_s: float = 2.0) -> None:
        self.poll_interval_s = float(poll_interval_s)
        # Instantiate workers up front so ID / labels are predictable
        # in event streams and logs.
        self.workers = [
            ConceptWorker("llm-w0"),
            ImageWorker("img-w0"),
            VideoWorker("vid-w0"),
            AudioWorker("aud-w0"),
            TTSWorker("tts-w0"),
            PostWorker("post-w0"),
            MergeWorker("merge-w0"),
        ]
        self._task_set: List[asyncio.Task] = []
        self._stopping = False

    # --- lifecycle -------------------------------------------------------
    async def run(self) -> None:
        """Spawn all worker loops and `gather` until cancelled.

        Cancellation propagates: cancelling the outer task cancels each
        per-worker task, and `loop()` is expected to honour
        `asyncio.CancelledError` cleanly (released GPU lock, no half-written
        queue rows).
        """
        if not _WORKERS_AVAILABLE:
            raise RuntimeError(
                "Coordinator cannot run: stage workers are not importable. "
                f"Underlying error: {_IMPORT_ERROR!r}. "
                "Land Phases 1-3 of the queueing refactor first."
            )
        await self.start()
        try:
            await asyncio.gather(*self._task_set)
        except asyncio.CancelledError:
            await self.stop()
            raise

    async def start(self) -> None:
        """Spawn worker tasks (idempotent)."""
        if self._task_set:
            return
        if not _WORKERS_AVAILABLE:
            raise RuntimeError(
                "Coordinator.start: stage workers are not importable. "
                f"Underlying error: {_IMPORT_ERROR!r}"
            )
        self._stopping = False
        self._task_set = [
            asyncio.create_task(
                w.loop(poll_interval_s=self.poll_interval_s),
                name=f"stage-worker:{getattr(w, 'name', w.__class__.__name__)}",
            )
            for w in self.workers
        ]
        mark_running(True)
        log.info("Coordinator started %d stage workers", len(self._task_set))

    async def stop(self, timeout_s: float = 5.0) -> None:
        """Cancel all worker tasks and wait for them to settle."""
        if self._stopping:
            return
        self._stopping = True
        try:
            for t in self._task_set:
                if not t.done():
                    t.cancel()
            if self._task_set:
                # Give workers a chance to exit their `acquire_gpu` blocks.
                done, pending = await asyncio.wait(
                    self._task_set, timeout=timeout_s
                )
                for t in pending:
                    log.warning(
                        "Coordinator: worker %s did not exit within %.1fs",
                        t.get_name(), timeout_s,
                    )
        finally:
            self._task_set = []
            mark_running(False)
            log.info("Coordinator stopped")

    # --- introspection ---------------------------------------------------
    def status(self) -> dict:
        return {
            "running": bool(self._task_set) and not self._stopping,
            "workers": [
                {
                    "name": getattr(w, "name", w.__class__.__name__),
                    "stage": w.__class__.__name__.replace("Worker", "").lower(),
                }
                for w in self.workers
            ],
            "import_ok": _WORKERS_AVAILABLE,
            "import_error": repr(_IMPORT_ERROR) if _IMPORT_ERROR else None,
        }


# Module-level singleton used by the FastAPI endpoints. Lazy so importing
# this module is cheap (no event loop, no worker construction yet).
_singleton: Optional[Coordinator] = None


def get_coordinator() -> Coordinator:
    global _singleton
    if _singleton is None:
        _singleton = Coordinator()
    return _singleton


__all__ = [
    "Coordinator",
    "get_coordinator",
    "is_running",
    "mark_running",
    "COORDINATOR_STATE_FILE",
]


# ---------------------------------------------------------------------------
# CLI: `python -m slopfinity.coordinator`
#
# Replaces the legacy `run_philosophical_experiments.py` fleet runner for
# users who want the new architecture. The legacy runner is still shipped
# (and still works) — see docs/concurrent-mode-design.md for the migration
# note.
# ---------------------------------------------------------------------------
def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m slopfinity.coordinator",
        description=(
            "Run the Slopfinity Phase-4 Coordinator standalone. "
            "Spawns one StageWorker per pipeline stage; each worker pulls "
            "from the per-stage queue concurrently. GPU/budget serialization "
            "is enforced by scheduler.acquire_gpu."
        ),
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds each StageWorker sleeps between queue polls (default: 2.0).",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("SLOPFINITY_LOG_LEVEL", "INFO"),
        help="Python logging level (default: INFO; or $SLOPFINITY_LOG_LEVEL).",
    )
    return p


def _cli_main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    co = Coordinator(poll_interval_s=args.poll_interval)
    log.info("Starting coordinator (poll_interval=%.1fs)", args.poll_interval)
    try:
        asyncio.run(co.run())
    except KeyboardInterrupt:
        log.info("Coordinator interrupted by SIGINT — stopping cleanly")
        return 130
    except RuntimeError as e:
        log.error("Coordinator failed to start: %s", e)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
