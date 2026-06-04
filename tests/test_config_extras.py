"""DEFAULT_CONFIG role keys + save_queue stable-id stamping."""


def test_default_config_role_keys_and_qwen_base():
    from slopfinity.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["base_model"] == "qwen"
    assert DEFAULT_CONFIG["audio_model"] == "none"
    assert DEFAULT_CONFIG["tts_model"] == "none"
    assert DEFAULT_CONFIG["upscale_model"] == "none"


def test_save_queue_stamps_stable_id():
    # run_fleet appends rows without 'id'; save_queue must stamp a stable id so
    # the id doesn't churn (fresh uuid) on every save.
    from slopfinity import config as cfg
    try:
        item = {"prompt": "idtest", "ts": 123.0, "status": "pending"}
        cfg.save_queue([item])
        assert item.get("id")
        first = item["id"]
        cfg.save_queue([item])
        assert item["id"] == first
    finally:
        cfg.save_queue([])


def test_queue_item_extra_fields_survive_db_roundtrip():
    """Fields without a dedicated QueueItem column (seed_image, stage_prompts,
    seeds_mode, polymorphic, started_ts, …) must survive the DB roundtrip via
    the `extra` catch-all — they used to be silently dropped, breaking the
    seed-image / FLF2V / stage-prompt features in run_fleet."""
    from slopfinity import config as cfg
    payload = {
        "prompt": "p", "ts": 7.0, "status": "pending",
        "seed_image": "seed_x.png", "seeds_mode": "per-task",
        "seed_images": ["seed_a.png", "seed_b.png"], "seed_prompt": "edited",
        "stage_prompts": {"video": "vp"}, "stage_prompts_raw": "raw",
        "polymorphic": True, "started_ts": 11.0, "requeued_from_ts": 3.0,
    }
    extras = [k for k in payload if k not in ("prompt", "ts", "status")]
    try:
        cfg.save_queue([dict(payload)])
        got = cfg.get_queue()
        assert len(got) == 1
        g = got[0]
        assert "extra" not in g, "extra wrapper must be flattened, not exposed"
        for k in extras:
            assert g.get(k) == payload[k], f"dropped/changed on roundtrip: {k}"
        # second roundtrip (run_fleet claim->mutate->save) must also preserve
        cfg.save_queue(got)
        g2 = cfg.get_queue()[0]
        for k in extras:
            assert g2.get(k) == payload[k], f"dropped on 2nd roundtrip: {k}"
    finally:
        cfg.save_queue([])


def test_queue_lock_is_exclusive():
    # queue_lock holds an exclusive flock — a second non-blocking acquire fails.
    import fcntl
    import pytest
    from slopfinity import config as cfg
    with cfg.queue_lock():
        with open(cfg._QUEUE_LOCK_FILE, "a+") as fd:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # after release it acquires fine
    with open(cfg._QUEUE_LOCK_FILE, "a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)


def test_config_lock_is_exclusive():
    import fcntl
    import pytest
    from slopfinity import config as cfg
    with cfg.config_lock():
        with open(cfg._CONFIG_LOCK_FILE, "a+") as fd:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # config_lock and queue_lock are independent flocks — holding one must not
    # block the other.
    with cfg.queue_lock():
        with open(cfg._CONFIG_LOCK_FILE, "a+") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)


def test_mutate_config_reads_modifies_saves():
    from slopfinity import config as cfg
    cfg.save_config({"infinity_index": 5, "infinity_themes": ["x"]})
    out = cfg.mutate_config(lambda c: {**c, "infinity_index": c.get("infinity_index", 0) + 1})
    assert out["infinity_index"] == 6
    assert cfg.load_config()["infinity_index"] == 6


def test_mutate_queue_reads_modifies_saves():
    from slopfinity import config as cfg
    try:
        cfg.save_queue([])
        out = cfg.mutate_queue(
            lambda q: q + [{"prompt": "m", "ts": 1.0, "status": "pending"}]
        )
        assert any(i["prompt"] == "m" for i in out)
        assert any(i["prompt"] == "m" for i in cfg.get_queue())
    finally:
        cfg.save_queue([])


# The worker each subprocess runs: append M rows via mutate_queue. With the
# cross-process lock every append survives; with a bare get→modify→save the
# blind delete-all+reinsert in save_queue would drop overlapping writes.
_CONCURRENT_WORKER = (
    "import sys\n"
    "from slopfinity import config as cfg\n"
    "wid, m = sys.argv[1], int(sys.argv[2])\n"
    "for i in range(m):\n"
    "    rid = '%s-%d' % (wid, i)\n"
    "    cfg.mutate_queue(\n"
    "        lambda q, rid=rid, i=i: q + [{'prompt': rid, 'id': rid,\n"
    "            'ts': float(int(wid) * 10000 + i), 'status': 'pending'}]\n"
    "    )\n"
)


def test_mutate_queue_no_lost_updates_across_processes(tmp_path):
    """End-to-end proof the lost-update race is closed: N real subprocesses
    each append M rows concurrently; all N*M must land (no clobber)."""
    import os
    import subprocess
    import sys

    state_dir = tmp_path / "sf_conc"
    env = dict(os.environ)
    env["SLOPFINITY_STATE_DIR"] = str(state_dir)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    N, M = 5, 20
    # Seed an empty queue in the shared state dir before fanning out.
    subprocess.run(
        [sys.executable, "-c",
         "from slopfinity import config as cfg; cfg.save_queue([])"],
        env=env, check=True, capture_output=True,
    )

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CONCURRENT_WORKER, str(w), str(M)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        for w in range(N)
    ]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()

    result = subprocess.run(
        [sys.executable, "-c",
         "from slopfinity import config as cfg; "
         "q = cfg.get_queue(); "
         "print(len(q)); "
         "print(len({i['id'] for i in q}))"],
        env=env, check=True, capture_output=True, text=True,
    )
    total, distinct = (int(x) for x in result.stdout.split())
    assert total == N * M, f"lost updates: got {total} of {N * M}"
    assert distinct == N * M, f"id collisions / drops: {distinct} distinct"
