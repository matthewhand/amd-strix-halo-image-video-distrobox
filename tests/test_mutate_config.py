"""Land unit coverage for config_lock + mutate_config (d4bfdee slice)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest


@pytest.fixture
def conf_env(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("SLOPFINITY_STATE_DIR", str(state))
    import importlib
    import slopfinity.config as cfg
    importlib.reload(cfg)
    cfg.save_config({"k": 0, "llm": {"api_key": ""}})
    yield cfg
    importlib.reload(cfg)


def test_mutate_config_serialises_increments(conf_env):
    cfg = conf_env
    n = 20
    errors = []

    def bump(_i):
        try:
            def mut(c):
                c = dict(c)
                c["k"] = int(c.get("k") or 0) + 1
                return c
            cfg.mutate_config(mut)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=bump, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    final = cfg.load_config()
    assert final["k"] == n


def test_save_config_atomic_rename(conf_env):
    cfg = conf_env
    cfg.save_config({"hello": "world"})
    path = Path(os.environ["SLOPFINITY_STATE_DIR"]) / "config.json"
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["hello"] == "world"
