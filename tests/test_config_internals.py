"""Unit tests for slopfinity.config internals.

Covers under-tested logic in slopfinity/config.py:
  * prompt-resolution helpers (philosophical / fanout / fleet / infinity /
    chaos / void) and the generic _resolve_prompt fallback semantics,
  * _merge_auto_suspend merge semantics (canonical auto-add, user-edit win,
    custom-entry preservation),
  * queue_lock() context-manager behaviour and atomic write-rename in
    save_queue / get_queue round-trips.

NOTE: the cpu-mode bool<->string coercion referenced by the task brief does
NOT live in slopfinity/config.py — it is implemented in
slopfinity/routers/config.py (_coerce_cpu_mode / _cpu_mode_to_bool).
Importing that module has an unavoidable filesystem side-effect (it imports
slopfinity.paths, which os.makedirs() a /workspace/* dir at import time),
so it can't be exercised hermetically here without modifying source. It is
therefore intentionally left uncovered by this file. See the agent report.

All file-touching tests are hermetic: the config module computes its file
paths at import time from SLOPFINITY_STATE_DIR, so we monkeypatch the
module-level CONFIG_FILE / QUEUE_FILE / _QUEUE_LOCK_FILE attributes onto a
tmp dir rather than mutating any real state directory.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import config as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Hermetic state-dir fixture: redirect every config file path to a tmp dir.
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    """Point all config module-level file paths at an isolated tmp dir."""
    config_file = tmp_path / "config.json"
    queue_file = tmp_path / "queue.json"
    state_file = tmp_path / "state.json"
    lock_file = tmp_path / "queue.json.lock"
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(config_file))
    monkeypatch.setattr(cfg, "QUEUE_FILE", str(queue_file))
    monkeypatch.setattr(cfg, "STATE_FILE", str(state_file))
    monkeypatch.setattr(cfg, "_QUEUE_LOCK_FILE", str(lock_file))
    return tmp_path


# ---------------------------------------------------------------------------
# _resolve_prompt + get_* prompt resolvers
# ---------------------------------------------------------------------------

class TestPromptResolvers:
    def test_philosophical_default_when_none(self):
        assert cfg.get_philosophical_prompt({"philosophical_prompt": None}) == \
            cfg.DEFAULT_PHILOSOPHICAL_PROMPT

    def test_philosophical_default_when_empty_string(self):
        assert cfg.get_philosophical_prompt({"philosophical_prompt": ""}) == \
            cfg.DEFAULT_PHILOSOPHICAL_PROMPT

    def test_philosophical_override_wins(self):
        assert cfg.get_philosophical_prompt(
            {"philosophical_prompt": "custom system"}
        ) == "custom system"

    def test_philosophical_missing_key_uses_default(self):
        assert cfg.get_philosophical_prompt({}) == cfg.DEFAULT_PHILOSOPHICAL_PROMPT

    def test_resolve_prompt_none_returns_default(self):
        assert cfg._resolve_prompt({"k": None}, "k", "DEF") == "DEF"

    def test_resolve_prompt_empty_returns_default(self):
        assert cfg._resolve_prompt({"k": ""}, "k", "DEF") == "DEF"

    def test_resolve_prompt_whitespace_only_returns_default(self):
        # Only the generic _resolve_prompt strips whitespace (philosophical
        # uses an exact == "" check), so verify the strip path explicitly.
        assert cfg._resolve_prompt({"k": "   \n\t "}, "k", "DEF") == "DEF"

    def test_resolve_prompt_override_wins(self):
        assert cfg._resolve_prompt({"k": "value"}, "k", "DEF") == "value"

    def test_resolve_prompt_non_string_truthy_passes_through(self):
        # A non-string truthy value isn't stripped/blanked — it's returned.
        assert cfg._resolve_prompt({"k": 123}, "k", "DEF") == 123

    def test_fanout_default(self):
        assert cfg.get_fanout_system_prompt({}) == cfg.DEFAULT_FANOUT_SYSTEM_PROMPT

    def test_fanout_override(self):
        assert cfg.get_fanout_system_prompt(
            {"fanout_system_prompt": "X"}
        ) == "X"

    def test_fleet_default(self):
        assert cfg.get_fleet_user_prompt_template({}) == \
            cfg.DEFAULT_FLEET_USER_PROMPT_TEMPLATE

    def test_fleet_override(self):
        assert cfg.get_fleet_user_prompt_template(
            {"fleet_user_prompt_template": "{seed} go"}
        ) == "{seed} go"

    def test_infinity_default(self):
        assert cfg.get_infinity_user_prompt_template({}) == \
            cfg.DEFAULT_INFINITY_USER_PROMPT_TEMPLATE

    def test_infinity_override(self):
        assert cfg.get_infinity_user_prompt_template(
            {"infinity_user_prompt_template": "{theme}!"}
        ) == "{theme}!"

    def test_chaos_default(self):
        assert cfg.get_chaos_suggest_system_prompt({}) == \
            cfg.DEFAULT_CHAOS_SUGGEST_SYSTEM_PROMPT

    def test_chaos_override(self):
        assert cfg.get_chaos_suggest_system_prompt(
            {"chaos_suggest_system_prompt": "chaos"}
        ) == "chaos"

    def test_void_default(self):
        assert cfg.get_void_fallback_template({}) == \
            cfg.DEFAULT_VOID_FALLBACK_TEMPLATE

    def test_void_override(self):
        assert cfg.get_void_fallback_template(
            {"void_fallback_template": "void {style}"}
        ) == "void {style}"

    def test_resolvers_load_config_when_no_dict(self, tmp_state, monkeypatch):
        # When passed a non-dict, the resolver falls back to load_config().
        # With a hermetic empty state dir, load_config() returns defaults,
        # so the resolver returns the built-in default prompt.
        assert cfg.get_fanout_system_prompt(None) == \
            cfg.DEFAULT_FANOUT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _merge_auto_suspend semantics
# ---------------------------------------------------------------------------

class TestMergeAutoSuspend:
    def test_non_list_returns_canonical_copy(self):
        out = cfg._merge_auto_suspend(None)
        ids = [e["id"] for e in out]
        assert ids == [e["id"] for e in cfg.DEFAULT_AUTO_SUSPEND]
        # Must be a fresh list, not the module constant.
        assert out is not cfg.DEFAULT_AUTO_SUSPEND

    def test_empty_list_gets_all_canonical(self):
        out = cfg._merge_auto_suspend([])
        ids = [e["id"] for e in out]
        assert ids == [e["id"] for e in cfg.DEFAULT_AUTO_SUSPEND]

    def test_user_edit_to_existing_entry_wins(self):
        stored = [{"id": "lmstudio", "enabled": False, "method": "sigterm",
                   "process_name": "LM Studio"}]
        out = cfg._merge_auto_suspend(stored)
        entry = next(e for e in out if e["id"] == "lmstudio")
        # User's edits preserved verbatim, not overwritten by canonical default.
        assert entry["enabled"] is False
        assert entry["method"] == "sigterm"

    def test_missing_canonical_entries_auto_added(self):
        # User only stored one canonical entry; the other 3 should appear.
        stored = [{"id": "lmstudio", "enabled": True, "method": "sigstop"}]
        out = cfg._merge_auto_suspend(stored)
        ids = {e["id"] for e in out}
        for d in cfg.DEFAULT_AUTO_SUSPEND:
            assert d["id"] in ids

    def test_custom_user_entry_preserved_and_appended(self):
        custom = {"id": "my-service", "enabled": True, "method": "docker_stop",
                  "container": "mine"}
        out = cfg._merge_auto_suspend([custom])
        assert custom in out
        # Custom entries come after all canonical entries.
        canonical_ids = [e["id"] for e in cfg.DEFAULT_AUTO_SUSPEND]
        assert out[len(canonical_ids)]["id"] == "my-service"

    def test_canonical_order_preserved(self):
        # Even when stored in a different order, output follows canonical order.
        stored = list(reversed([dict(d) for d in cfg.DEFAULT_AUTO_SUSPEND]))
        out = cfg._merge_auto_suspend(stored)
        ids = [e["id"] for e in out]
        assert ids == [d["id"] for d in cfg.DEFAULT_AUTO_SUSPEND]

    def test_entries_without_id_are_dropped(self):
        stored = [{"enabled": True, "method": "sigstop"}]  # no id
        out = cfg._merge_auto_suspend(stored)
        # Only canonical entries survive; the id-less custom one is skipped.
        assert all(e.get("id") for e in out)
        assert len(out) == len(cfg.DEFAULT_AUTO_SUSPEND)


# ---------------------------------------------------------------------------
# load_config merge of auto_suspend + scheduler
# ---------------------------------------------------------------------------

class TestLoadConfigMerge:
    def test_no_file_returns_fresh_defaults(self, tmp_state):
        c = cfg.load_config()
        assert c["base_model"] == cfg.DEFAULT_CONFIG["base_model"]
        # auto_suspend / scheduler are fresh copies, not the module constants.
        assert c["auto_suspend"] is not cfg.DEFAULT_AUTO_SUSPEND
        assert c["scheduler"] is not cfg.DEFAULT_SCHEDULER

    def test_stored_config_gains_new_default_keys(self, tmp_state):
        (tmp_state / "config.json").write_text(json.dumps({"chains": 99}))
        c = cfg.load_config()
        assert c["chains"] == 99  # user value preserved
        assert "base_model" in c  # default key backfilled
        assert c["base_model"] == cfg.DEFAULT_CONFIG["base_model"]

    def test_stored_scheduler_merges_with_defaults(self, tmp_state):
        (tmp_state / "config.json").write_text(
            json.dumps({"scheduler": {"memory_safety_gb": 42}})
        )
        c = cfg.load_config()
        assert c["scheduler"]["memory_safety_gb"] == 42  # user value
        assert c["scheduler"]["llm_cpu_mode"] == "smart"  # default backfilled

    def test_stored_auto_suspend_is_merged(self, tmp_state):
        (tmp_state / "config.json").write_text(
            json.dumps({"auto_suspend": [
                {"id": "custom-x", "enabled": True, "method": "sigstop"}
            ]})
        )
        c = cfg.load_config()
        ids = {e["id"] for e in c["auto_suspend"]}
        assert "custom-x" in ids
        # canonical entries auto-added
        assert "lmstudio" in ids

    def test_corrupt_config_falls_back_to_defaults(self, tmp_state):
        (tmp_state / "config.json").write_text("{ not valid json")
        c = cfg.load_config()
        assert c["base_model"] == cfg.DEFAULT_CONFIG["base_model"]


# ---------------------------------------------------------------------------
# queue_lock + atomic save_queue / get_queue
# ---------------------------------------------------------------------------

class TestQueueLock:
    def test_queue_lock_is_context_manager(self, tmp_state):
        # Entering and exiting should not raise and should create the lockfile.
        with cfg.queue_lock():
            assert os.path.exists(cfg._QUEUE_LOCK_FILE)

    def test_queue_lock_round_trip_under_lock(self, tmp_state):
        with cfg.queue_lock():
            cfg.save_queue([{"id": "a", "schema_version": 999}])
        assert os.path.exists(cfg.QUEUE_FILE)

    def test_queue_lock_reentrant_via_separate_threads_serializes(self, tmp_state):
        # flock(LOCK_EX) on the same file from two threads (each opening its
        # own fd) must serialize. We assert the second thread cannot enter
        # while the first holds the lock.
        order = []
        first_holding = threading.Event()
        release_first = threading.Event()

        def first():
            with cfg.queue_lock():
                order.append("first-enter")
                first_holding.set()
                release_first.wait(timeout=5)
                order.append("first-exit")

        def second():
            first_holding.wait(timeout=5)
            with cfg.queue_lock():
                order.append("second-enter")

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        # While first holds the lock, second must NOT have entered.
        first_holding.wait(timeout=5)
        assert "second-enter" not in order
        release_first.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert order == ["first-enter", "first-exit", "second-enter"]


class TestAtomicQueueWrite:
    def test_save_queue_writes_valid_json(self, tmp_state):
        data = [{"id": "x", "schema_version": 999}]
        cfg.save_queue(data)
        with open(cfg.QUEUE_FILE) as f:
            assert json.load(f) == data

    def test_save_queue_uses_temp_then_rename(self, tmp_state, monkeypatch):
        # Verify the atomic-write contract: write goes to a .tmp sibling and
        # is then os.replace()'d onto the final path.
        seen = {}
        real_replace = os.replace

        def spy_replace(src, dst):
            seen["src"] = src
            seen["dst"] = dst
            return real_replace(src, dst)

        monkeypatch.setattr(cfg.os, "replace", spy_replace)
        cfg.save_queue([{"id": "y", "schema_version": 999}])
        assert seen["src"] == cfg.QUEUE_FILE + ".tmp"
        assert seen["dst"] == cfg.QUEUE_FILE
        # The temp file is gone after the rename.
        assert not os.path.exists(cfg.QUEUE_FILE + ".tmp")

    def test_save_queue_no_stale_tmp_on_success(self, tmp_state):
        cfg.save_queue([{"id": "z", "schema_version": 999}])
        assert not os.path.exists(cfg.QUEUE_FILE + ".tmp")

    def test_get_queue_missing_file_returns_empty(self, tmp_state):
        assert cfg.get_queue() == []

    def test_get_queue_reads_back_saved(self, tmp_state):
        # Save with the current schema version so get_queue doesn't rewrite.
        from slopfinity import queue_schema
        item = {"id": "q1", "schema_version": queue_schema.SCHEMA_VERSION}
        cfg.save_queue([item])
        out = cfg.get_queue()
        assert isinstance(out, list)
        assert out and out[0]["id"] == "q1"

    def test_get_queue_corrupt_returns_empty(self, tmp_state):
        with open(cfg.QUEUE_FILE, "w") as f:
            f.write("{ broken")
        assert cfg.get_queue() == []
