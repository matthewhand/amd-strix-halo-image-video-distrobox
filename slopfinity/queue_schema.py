"""Queue schema v2 — typed per-stage status for queue items.

See docs/queueing-refactor-design.md. Phase 1 ships migration + helpers;
the runner still consults legacy fields. Phase 2+ workers will read the
``stages`` map exclusively.
"""
import time
import uuid
from typing import Optional

SCHEMA_VERSION = 2
STAGE_ORDER = ["concept", "image", "video", "audio", "tts", "post", "merge"]
PREREQS = {
    "concept": [], "image": ["concept"], "video": ["image"],
    "audio": ["concept"], "tts": ["concept"], "post": ["video"],
    "merge": ["post", "audio", "tts"],
}
ROLE_STAGE = {
    "concept": "llm", "image": "image", "video": "video",
    "audio": "audio", "tts": "tts", "post": "post", "merge": "ffmpeg",
}


def make_id() -> str:
    return f"q-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"


def migrate_legacy(item: dict) -> dict:
    if item.get("schema_version") == SCHEMA_VERSION and "stages" in item:
        return item
    item.setdefault("id", make_id())
    item["schema_version"] = SCHEMA_VERSION
    legacy_status = item.get("status", "pending")
    succeeded = item.get("succeeded")
    image_only = bool(item.get("image_only", False))
    if legacy_status == "done" and succeeded is True:
        bulk = "done"
    elif legacy_status == "done" and succeeded is False:
        bulk = "failed"
    elif legacy_status == "cancelled":
        bulk = "skipped"
    else:
        bulk = "needs"
    config = item.get("config_snapshot") or {}
    stages = {}
    for s in STAGE_ORDER:
        if image_only and s in ("video", "audio", "tts", "post", "merge"):
            stages[s] = {"status": "skipped"}
        else:
            entry = {"status": bulk}
            if s == "image":
                entry["model"] = config.get("base_model", "qwen")
            elif s == "video":
                entry["model"] = config.get("video_model", "ltx-2.3")
            elif s == "audio":
                entry["model"] = config.get("audio_model", "heartmula")
            elif s == "tts":
                entry["model"] = config.get("tts_model", "qwen-tts")
            elif s == "post":
                entry["model"] = config.get("upscale_model", "ltx-spatial")
            stages[s] = entry
    item["stages"] = stages
    return item


def stage_status(item: dict, stage: str) -> str:
    return ((item.get("stages") or {}).get(stage) or {}).get("status", "needs")


def set_stage_status(item: dict, stage: str, status: str, **fields) -> None:
    item.setdefault("stages", {}).setdefault(stage, {})
    item["stages"][stage]["status"] = status
    item["stages"][stage].update(fields)
    if status in ("done", "failed", "skipped"):
        item["stages"][stage].setdefault("completed_ts", time.time())
    if status == "working":
        item["stages"][stage].setdefault("started_ts", time.time())


def prerequisites_met(item: dict, stage: str) -> bool:
    for p in PREREQS.get(stage, []):
        if stage_status(item, p) not in ("done", "skipped"):
            return False
    return True


def next_stage_for_role(items: list, role: str) -> Optional[tuple]:
    role_stages = [s for s, r in ROLE_STAGE.items() if r == role]
    for item in items:
        for stage in role_stages:
            if stage_status(item, stage) == "needs" and prerequisites_met(item, stage):
                return (item, stage)
    return None


def overall_status(item: dict) -> str:
    stages = item.get("stages") or {}
    if not stages:
        return item.get("status", "pending")
    statuses = [s.get("status", "needs") for s in stages.values()]
    if all(s in ("done", "skipped") for s in statuses):
        return "done" if any(s == "done" for s in statuses) else "skipped"
    if any(s == "failed" for s in statuses):
        return "failed"
    if any(s == "working" for s in statuses):
        return "running"
    return "pending"
