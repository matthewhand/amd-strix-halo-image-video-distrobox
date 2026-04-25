"""ConceptWorker — first concrete StageWorker.

Runs the `concept` stage: feeds the user's queued prompt through the
configured local LLM with the `enhancer_prompt` system message and
records the rewritten text on the item's stage output.

This is the reference implementation for Phase 2; Phase 3 adds the
heavyweight Image / Video / Audio / TTS / Post / Merge workers using
the same harness.
"""
from __future__ import annotations

import asyncio

from slopfinity import config as _config
from slopfinity.llm import lmstudio_call
from slopfinity.workers.base import StageWorker


class ConceptWorker(StageWorker):
    role = "llm"

    async def run_stage(self, item: dict, stage: str) -> dict:
        prompt = item.get("prompt") or ""
        if not prompt.strip():
            return {"ok": False, "error": "item has empty prompt"}

        # Resolve enhancer prompt from the item snapshot if present
        # (so retroactive config changes don't mutate in-flight items),
        # else from live config.
        snap = item.get("config_snapshot") or {}
        sys_prompt = snap.get("enhancer_prompt") or _config.load_config().get(
            "enhancer_prompt"
        )
        if not sys_prompt:
            return {"ok": False, "error": "no enhancer_prompt configured"}

        try:
            rewritten = await asyncio.to_thread(lmstudio_call, sys_prompt, prompt)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        if not rewritten or rewritten.startswith("Error:"):
            return {"ok": False, "error": rewritten or "empty LLM response"}

        return {"ok": True, "output": rewritten}
