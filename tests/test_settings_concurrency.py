"""Lost-update concurrency proof for POST /config-style RMW (land-adapted).

Flam's suite imports exported _apply_settings; land keeps settings merge as a
nested _apply inside settings_post. This port proves the same lock contract for
the full-dict config path that both /config and mutate_config use, without
extracting settings helpers.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_CONFIG_WORKER = (
    "import sys\n"
    "from slopfinity import config as cfg\n"
    "wid = int(sys.argv[1])\n"
    "data = {'cfgkey_%d' % wid: wid}\n"
    "cfg.mutate_config(lambda c: {**c, **data})\n"
)

_SCHEDULER_WORKER = (
    "import sys\n"
    "from slopfinity import config as cfg\n"
    "wid = int(sys.argv[1])\n"
    "key = 'k%d' % wid\n"
    "def mut(c):\n"
    "    c = dict(c)\n"
    "    sched = dict(c.get('scheduler') or {})\n"
    "    sched[key] = wid\n"
    "    c['scheduler'] = sched\n"
    "    return c\n"
    "cfg.mutate_config(mut)\n"
)


def _fanout(tmp_path, worker_src, n):
    state_dir = tmp_path / "sf_settings_conc"
    env = dict(os.environ)
    env["SLOPFINITY_STATE_DIR"] = str(state_dir)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from slopfinity import config as cfg; "
            "cfg.save_config({'scheduler': {'seed': True}})",
        ],
        env=env,
        check=True,
        capture_output=True,
    )

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", worker_src, str(i)],
            env=env,
            cwd=os.getcwd(),
        )
        for i in range(n)
    ]
    for p in procs:
        assert p.wait(timeout=30) == 0

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from slopfinity import config as cfg; "
            "print(json.dumps(cfg.load_config()))",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    return json.loads(out.stdout)


def test_config_no_lost_updates_across_processes(tmp_path):
    n = 5
    final = _fanout(tmp_path, _CONFIG_WORKER, n)
    for i in range(n):
        assert final.get(f"cfgkey_{i}") == i, final


def test_scheduler_bucket_no_lost_updates_across_processes(tmp_path):
    n = 5
    final = _fanout(tmp_path, _SCHEDULER_WORKER, n)
    sched = final.get("scheduler") or {}
    assert sched.get("seed") is True
    for i in range(n):
        assert sched.get(f"k{i}") == i, sched
