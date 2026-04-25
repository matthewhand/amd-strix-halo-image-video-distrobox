"""Phase 1 queue schema tests — migration, prerequisites, role dispatch."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slopfinity import queue_schema as qs  # noqa: E402


def test_migrate_pending_legacy_item_has_all_stages_needs():
    item = {"status": "pending", "config_snapshot": {"base_model": "qwen"}}
    qs.migrate_legacy(item)
    assert item["schema_version"] == qs.SCHEMA_VERSION
    assert set(item["stages"].keys()) == set(qs.STAGE_ORDER)
    for stage in qs.STAGE_ORDER:
        assert item["stages"][stage]["status"] == "needs"
    assert item["stages"]["image"]["model"] == "qwen"


def test_migrate_done_succeeded_legacy_item_has_all_stages_done():
    item = {"status": "done", "succeeded": True}
    qs.migrate_legacy(item)
    for stage in qs.STAGE_ORDER:
        assert item["stages"][stage]["status"] == "done"


def test_migrate_done_failed_legacy_item_has_all_stages_failed():
    item = {"status": "done", "succeeded": False}
    qs.migrate_legacy(item)
    for stage in qs.STAGE_ORDER:
        assert item["stages"][stage]["status"] == "failed"


def test_migrate_cancelled_legacy_item_has_all_stages_skipped():
    item = {"status": "cancelled"}
    qs.migrate_legacy(item)
    for stage in qs.STAGE_ORDER:
        assert item["stages"][stage]["status"] == "skipped"


def test_migrate_image_only_skips_post_image_stages():
    item = {"status": "pending", "image_only": True}
    qs.migrate_legacy(item)
    assert item["stages"]["concept"]["status"] == "needs"
    assert item["stages"]["image"]["status"] == "needs"
    for s in ("video", "audio", "tts", "post", "merge"):
        assert item["stages"][s]["status"] == "skipped"


def test_migrate_is_idempotent():
    item = {"status": "pending"}
    qs.migrate_legacy(item)
    snapshot = {k: dict(v) for k, v in item["stages"].items()}
    item_id = item["id"]
    qs.migrate_legacy(item)
    qs.migrate_legacy(item)
    assert item["id"] == item_id
    assert item["schema_version"] == qs.SCHEMA_VERSION
    for stage, before in snapshot.items():
        assert item["stages"][stage] == before


def test_prerequisites_simple_chain():
    item = {"status": "pending"}
    qs.migrate_legacy(item)
    # concept has no prereqs.
    assert qs.prerequisites_met(item, "concept") is True
    # image needs concept done.
    assert qs.prerequisites_met(item, "image") is False
    qs.set_stage_status(item, "concept", "done")
    assert qs.prerequisites_met(item, "image") is True
    # video needs image done.
    assert qs.prerequisites_met(item, "video") is False
    qs.set_stage_status(item, "image", "done")
    assert qs.prerequisites_met(item, "video") is True


def test_prerequisites_merge_needs_post_audio_tts():
    item = {"status": "pending"}
    qs.migrate_legacy(item)
    qs.set_stage_status(item, "concept", "done")
    qs.set_stage_status(item, "image", "done")
    qs.set_stage_status(item, "video", "done")
    qs.set_stage_status(item, "post", "done")
    # missing audio + tts.
    assert qs.prerequisites_met(item, "merge") is False
    qs.set_stage_status(item, "audio", "done")
    assert qs.prerequisites_met(item, "merge") is False
    qs.set_stage_status(item, "tts", "done")
    assert qs.prerequisites_met(item, "merge") is True
    # skipped also satisfies prereq.
    item2 = {"status": "pending"}
    qs.migrate_legacy(item2)
    for s in ("concept", "image", "video", "post", "audio", "tts"):
        qs.set_stage_status(item2, s, "skipped")
    assert qs.prerequisites_met(item2, "merge") is True


def test_next_stage_for_role_picks_first_ready():
    a = {"status": "pending"}
    b = {"status": "pending"}
    qs.migrate_legacy(a)
    qs.migrate_legacy(b)
    # llm role = concept stage. Both items need it; first wins.
    pick = qs.next_stage_for_role([a, b], "llm")
    assert pick is not None
    item, stage = pick
    assert item is a
    assert stage == "concept"
    # mark a done; next pick should be b's concept.
    qs.set_stage_status(a, "concept", "done")
    pick = qs.next_stage_for_role([a, b], "llm")
    assert pick is not None and pick[0] is b and pick[1] == "concept"
    # video role: needs image done. Neither item has image done -> None.
    assert qs.next_stage_for_role([a, b], "video") is None


def test_next_stage_for_role_skips_skipped_and_done():
    a = {"status": "pending", "image_only": True}
    qs.migrate_legacy(a)
    qs.set_stage_status(a, "concept", "done")
    qs.set_stage_status(a, "image", "done")
    # image-only item: video/audio/tts/post/merge are skipped.
    assert qs.next_stage_for_role([a], "video") is None
    assert qs.next_stage_for_role([a], "ffmpeg") is None
    assert qs.next_stage_for_role([a], "image") is None  # already done


def test_overall_status_derivation():
    item = {"status": "pending"}
    qs.migrate_legacy(item)
    assert qs.overall_status(item) == "pending"
    qs.set_stage_status(item, "concept", "working")
    assert qs.overall_status(item) == "running"
    # mark all stages done -> done.
    item2 = {"status": "pending"}
    qs.migrate_legacy(item2)
    for s in qs.STAGE_ORDER:
        qs.set_stage_status(item2, s, "done")
    assert qs.overall_status(item2) == "done"
    # any failed -> failed.
    item3 = {"status": "pending"}
    qs.migrate_legacy(item3)
    qs.set_stage_status(item3, "concept", "done")
    qs.set_stage_status(item3, "image", "failed")
    assert qs.overall_status(item3) == "failed"
    # all skipped -> skipped.
    item4 = {"status": "cancelled"}
    qs.migrate_legacy(item4)
    assert qs.overall_status(item4) == "skipped"


def test_overall_status_empty_stages_falls_back_to_legacy():
    item = {"status": "pending", "stages": {}}
    assert qs.overall_status(item) == "pending"
    item2 = {"status": "done"}
    assert qs.overall_status(item2) == "done"
