#!/usr/bin/env python3
"""11-hour variety inject stress + memory watchdog orchestrator.

Single process:
  - samples /proc/meminfo every SAMPLE_S seconds → CSV timeseries
  - if MemAvailable < threshold → pause queue + block inject (+ park heavies if critical)
  - when queue has no active jobs and memory OK → inject next escalating unique prompt
  - infinity=0, short frames via live config snapshot

Usage:
  python3 scripts/stress_11h_variety_orchestrator.py \\
    --scratch /tmp/grok-goal-.../implementer \\
    --hours 11 --sample-s 60 --min-available-gb 12
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from slopfinity import stress_memory as sm  # noqa: E402


def log(path: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def http_json(method: str, url: str, *, data=None, timeout: float = 60.0):
    headers = {}
    body = None
    if data is not None:
        if isinstance(data, dict):
            body = urllib.parse.urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = data
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}


def queue_has_active(exp_dir: Path) -> bool:
    qpath = exp_dir / "queue.json"
    if not qpath.is_file():
        return False
    try:
        q = json.loads(qpath.read_text())
    except Exception:
        return False
    for it in q if isinstance(q, list) else []:
        if isinstance(it, dict) and it.get("status") in (
            "pending",
            "working",
            "queued",
            "running",
        ):
            return True
    return False


def ensure_config_short(exp_dir: Path) -> None:
    cfg = exp_dir / "config.json"
    if not cfg.is_file():
        return
    try:
        c = json.loads(cfg.read_text())
        c["frames"] = 17
        c["infinity_mode"] = False
        c["when_idle"] = False
        cfg.write_text(json.dumps(c, indent=2))
    except Exception:
        pass


def apply_actions(
    actions: list,
    *,
    slop: str,
    log_path: Path,
) -> None:
    if "pause_queue" in actions:
        try:
            http_json("POST", f"{slop}/queue/pause")
            log(log_path, "watchdog: pause_queue")
        except Exception as e:
            log(log_path, f"watchdog: pause_queue failed {e}")
    if "park_heavies" in actions:
        for name in (
            "strix-halo-comfyui",
            "strix-halo-mage-image",
            "strix-halo-heartmula",
        ):
            try:
                subprocess.run(
                    ["docker", "stop", name],
                    capture_output=True,
                    timeout=120,
                )
                log(log_path, f"watchdog: park {name}")
            except Exception as e:
                log(log_path, f"watchdog: park {name} failed {e}")


def resume_if_ok(slop: str, log_path: Path) -> None:
    try:
        st = http_json("GET", f"{slop}/queue/pause-state")
        if st.get("paused"):
            http_json("POST", f"{slop}/queue/resume")
            log(log_path, "watchdog: resume_queue (memory recovered)")
    except Exception as e:
        log(log_path, f"watchdog: resume check failed {e}")


def inject_prompt(slop: str, prompt: str, rank: int) -> dict:
    return http_json(
        "POST",
        f"{slop}/inject",
        data={
            "prompt": prompt,
            "priority": str(max(1, 100 - rank)),
            "infinity": "0",
            "concurrent": "1",
        },
        timeout=60.0,
    )


def write_report(scratch: Path) -> Path:
    ts_path = scratch / "memory_timeseries.csv"
    inj_path = scratch / "variety_inject_log.txt"
    samples = (
        sm.parse_timeseries_csv(ts_path.read_text())
        if ts_path.is_file()
        else []
    )
    rep = sm.series_report(samples)
    # variety
    injects = 0
    unique = set()
    if inj_path.is_file():
        for line in inj_path.read_text().splitlines():
            if "INJECT rank=" in line:
                injects += 1
            if "prompt=" in line:
                unique.add(line.split("prompt=", 1)[-1][:200])
    # queue summary if present
    exp = Path(
        os.environ.get(
            "SLOPFINITY_EXP_DIR",
            str(_ROOT / "comfy-outputs" / "experiments"),
        )
    )
    q_stats = {}
    if (exp / "queue.json").is_file():
        try:
            q = json.loads((exp / "queue.json").read_text())
            from collections import Counter

            q_stats = dict(
                Counter(
                    it.get("status")
                    for it in q
                    if isinstance(it, dict)
                )
            )
        except Exception:
            pass

    avail = rep["mem_available_gb"]
    used = rep["mem_used_gb"]
    lines = [
        "# 11h stress memory report",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"Timeseries: `{ts_path}`",
        "",
        "## Memory (from real samples)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| sample_count | {rep['sample_count']} |",
        f"| duration_hours | {rep['duration_hours']:.4f} |",
        f"| mem_total_gb (last) | {rep.get('mem_total_gb', 0):.2f} |",
        f"| MemAvailable_GB min | {avail['min']:.4f} |",
        f"| MemAvailable_GB avg | {avail['avg']:.4f} |",
        f"| MemAvailable_GB max | {avail['max']:.4f} |",
        f"| MemUsed_GB min | {used['min']:.4f} |",
        f"| MemUsed_GB avg | {used['avg']:.4f} |",
        f"| MemUsed_GB max | {used['max']:.4f} |",
        "",
        "## Variety inject summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| inject_attempts_logged | {injects} |",
        f"| unique_prompt_lines | {len(unique)} |",
        f"| queue_status_counts | {q_stats} |",
        "",
    ]
    out = scratch / "memory_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--sample-s", type=float, default=60.0)
    ap.add_argument("--min-available-gb", type=float, default=12.0)
    ap.add_argument("--slop-url", default=os.environ.get("SLOPFINITY_URL", "http://127.0.0.1:9099"))
    ap.add_argument(
        "--exp-dir",
        default=os.environ.get(
            "SLOPFINITY_EXP_DIR",
            str(_ROOT / "comfy-outputs" / "experiments"),
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        p = write_report(scratch)
        print(p.read_text())
        return 0

    ts_path = scratch / "memory_timeseries.csv"
    log_path = scratch / "orchestrator.log"
    inj_path = scratch / "variety_inject_log.txt"
    state_path = scratch / "orchestrator_state.json"

    if not ts_path.is_file():
        ts_path.write_text(sm.CSV_HEADER + "\n", encoding="utf-8")

    exp_dir = Path(args.exp_dir)
    ensure_config_short(exp_dir)
    slop = args.slop_url.rstrip("/")

    deadline = time.time() + float(args.hours) * 3600.0
    rank = 0
    if state_path.is_file():
        try:
            st = json.loads(state_path.read_text())
            rank = int(st.get("rank", 0))
        except Exception:
            pass

    log(
        log_path,
        f"START hours={args.hours} sample_s={args.sample_s} "
        f"min_avail_gb={args.min_available_gb} rank={rank}",
    )
    # env snapshot
    snap = scratch / "stress_env_snapshot.txt"
    with snap.open("a", encoding="utf-8") as f:
        f.write(f"\n=== start {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        try:
            s = sm.read_meminfo()
            f.write(sm.sample_to_csv_row(s) + "\n")
        except Exception as e:
            f.write(f"mem err {e}\n")
        f.write(f"exp={exp_dir} slop={slop}\n")

    sample_i = 0
    while time.time() < deadline:
        t_loop = time.time()
        try:
            sample = sm.read_meminfo()
        except Exception as e:
            log(log_path, f"meminfo read failed: {e}")
            sample = sm.MemSample(0, 0, 0, 0, 0)

        with ts_path.open("a", encoding="utf-8") as f:
            f.write(sm.sample_to_csv_row(sample) + "\n")
        sample_i += 1

        actions = sm.decide_watchdog_action(
            sample, min_available_gb=args.min_available_gb
        )
        if actions:
            log(
                log_path,
                f"WATCHDOG avail_gb={sample.mem_available_gb:.2f} actions={actions}",
            )
            apply_actions(actions, slop=slop, log_path=log_path)
        else:
            # Memory OK — resume if we had paused
            resume_if_ok(slop, log_path)
            # Inject when idle
            if not queue_has_active(exp_dir):
                prompt = sm.escalate_prompt(rank, seed=args.seed)
                try:
                    # force infinity off on disk before inject snapshot
                    ensure_config_short(exp_dir)
                    resp = inject_prompt(slop, prompt, rank)
                    with inj_path.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                            f"INJECT rank={rank} resp={resp} prompt={prompt}\n"
                        )
                    log(log_path, f"INJECT rank={rank} ok resp={resp}")
                    # patch new items infinity false
                    try:
                        qpath = exp_dir / "queue.json"
                        q = json.loads(qpath.read_text())
                        for it in q:
                            if isinstance(it, dict) and it.get("status") in (
                                "pending",
                                "working",
                            ):
                                it["infinity"] = False
                                cs = it.setdefault("config_snapshot", {})
                                cs["infinity_mode"] = False
                                cs["frames"] = 17
                        qpath.write_text(json.dumps(q, indent=2))
                    except Exception:
                        pass
                    rank += 1
                    state_path.write_text(
                        json.dumps({"rank": rank, "ts": time.time()}, indent=2)
                    )
                except Exception as e:
                    with inj_path.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                            f"INJECT_FAIL rank={rank} err={e} prompt={prompt}\n"
                        )
                    log(log_path, f"INJECT fail rank={rank}: {e}")

        # sleep remainder of sample interval
        elapsed = time.time() - t_loop
        time.sleep(max(1.0, float(args.sample_s) - elapsed))

    log(log_path, f"DONE samples≈{sample_i} final_rank={rank}")
    with snap.open("a", encoding="utf-8") as f:
        f.write(f"\n=== end {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        try:
            f.write(sm.sample_to_csv_row(sm.read_meminfo()) + "\n")
        except Exception as e:
            f.write(f"mem err {e}\n")
    report = write_report(scratch)
    log(log_path, f"report written {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
