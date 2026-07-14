"""Port of d4e3d4e _prune_done_archive safety contract tests (JSON queue on land)."""


def test_prune_done_archive_never_drops_active_rows():
    # SAFETY: the done/cancelled archive prune guards the LIVE queue, so it may
    # only ever drop rows whose status is exactly done/cancelled. Even when those
    # vastly exceed the cap, every pending/working/error/failed/blocked/None row
    # must survive untouched and in order.
    from slopfinity import config as cfg
    active = [
        {"prompt": "p", "ts": 1.0, "status": "pending"},
        {"prompt": "w", "ts": 2.0, "status": "working"},
        {"prompt": "e", "ts": 3.0, "status": "error"},
        {"prompt": "f", "ts": 4.0, "status": "failed"},
        {"prompt": "b", "ts": 5.0, "status": "blocked"},
        {"prompt": "n", "ts": 6.0, "status": None},
    ]
    done = [{"prompt": f"d{i}", "ts": 100.0 + i, "status": "done"} for i in range(50)]
    out = cfg._prune_done_archive(active + done, cap=10)
    assert [x["prompt"] for x in out if x["status"] not in ("done", "cancelled")] == \
        ["p", "w", "e", "f", "b", "n"]
    kept_done = [x for x in out if x["status"] == "done"]
    assert len(kept_done) == 10  # capped to the 10 most-recent by ts
    assert {x["prompt"] for x in kept_done} == {f"d{i}" for i in range(40, 50)}


def test_prune_done_archive_noop_returns_same_object():
    # Common case: at/under the cap the input list is returned UNCHANGED (the
    # same object) — no allocation, no reordering, no surprise on every save.
    from slopfinity import config as cfg
    q = [{"prompt": f"d{i}", "ts": float(i), "status": "done"} for i in range(5)]
    q.append({"prompt": "live", "ts": 99.0, "status": "pending"})
    assert cfg._prune_done_archive(q, cap=10) is q


def test_prune_done_archive_preserves_positions():
    # Kept rows stay in their original positions (interleaved active+done): only
    # the oldest surplus terminal rows are removed, nothing is reordered.
    from slopfinity import config as cfg
    q = [
        {"prompt": "d_old", "ts": 1.0, "status": "done"},
        {"prompt": "live1", "ts": 2.0, "status": "working"},
        {"prompt": "d_mid", "ts": 3.0, "status": "cancelled"},
        {"prompt": "live2", "ts": 4.0, "status": "pending"},
        {"prompt": "d_new", "ts": 5.0, "status": "done"},
    ]
    out = cfg._prune_done_archive(q, cap=1)  # keep only the single newest terminal
    assert [x["prompt"] for x in out] == ["live1", "live2", "d_new"]


def test_save_queue_bounds_done_archive_end_to_end():
    # Integration: save_queue prunes before persisting, so get_queue never
    # returns more than the cap of done/cancelled rows — yet keeps every active
    # row. This caps queue-LIST history only; generated files on disk are
    # untouched (they live in EXP_DIR, not the queue table).
    from slopfinity import config as cfg
    cap = cfg._DONE_ARCHIVE_CAP
    try:
        active = [
            {"prompt": "pending1", "ts": 5.0, "status": "pending"},
            {"prompt": "working1", "ts": 6.0, "status": "working"},
            {"prompt": "failed1", "ts": 7.0, "status": "failed"},
        ]
        done = [{"prompt": f"done{i}", "ts": 1000.0 + i, "status": "done"}
                for i in range(cap + 100)]
        cfg.save_queue(active + done)
        got = cfg.get_queue()
        kept_done = [x for x in got if x["status"] == "done"]
        kept_active = [x for x in got if x["status"] in ("pending", "working", "failed")]
        assert len(kept_done) == cap, f"done not capped: {len(kept_done)}"
        assert len(kept_active) == 3, "active rows dropped"
        assert {x["prompt"] for x in kept_done} == {f"done{i}" for i in range(100, cap + 100)}
    finally:
        cfg.save_queue([])
