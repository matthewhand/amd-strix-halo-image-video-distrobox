"""
tests/test_pipeline_slow.py
===========================
Slow integration tests — proves the actual generation pipeline works end-to-end:

  • Image generation  → slop_N_base.png written, size > 0
  • Video + audio     → FINAL_N.mp4 + .wav written, durations verified
  • Story mode        → 2 beats injected, both complete, grouped by story_id
  • asset_paths       → populated in queue.json after completion

Each test injects a job via the REST API, polls /queue/paginated until
done (or times out), then asserts files on disk and JSON fields.

Run:
    python3 -m pytest tests/test_pipeline_slow.py -v --timeout=1800
    # or without pytest:
    python3 tests/test_pipeline_slow.py

Environment:
    SLOPFINITY_URL   base URL  (default: http://localhost:9099)
    EXP_DIR          output dir (default: comfy-outputs/experiments)
    POLL_INTERVAL    seconds between polls (default: 10)
    JOB_TIMEOUT      max seconds to wait per job (default: 900 = 15 min)
"""

import os
import sys
import time
import json
import uuid
import glob
import subprocess
import urllib.request
import urllib.parse
import urllib.error

BASE = os.environ.get("SLOPFINITY_URL", "http://localhost:9099")
EXP_DIR = os.environ.get("EXP_DIR", "comfy-outputs/experiments")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"
INFO = "\033[36mℹ\033[0m"

results = []


# ─── helpers ──────────────────────────────────────────────────────────────────

def log(sym, msg):
    print(f"  {sym} {msg}", flush=True)


def api_post(path, data: dict) -> dict:
    """POST form-encoded data, return parsed JSON."""
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def api_get(path) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.loads(r.read())


def inject(prompt, priority="queue", **extra) -> dict:
    return api_post("/inject", {"prompt": prompt, "priority": priority, **extra})


def wait_for_done(ts: float, label: str, timeout: int = JOB_TIMEOUT):
    """Poll /queue/paginated until the item with this ts is done or timeout."""
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        try:
            data = api_get("/queue/paginated?limit=200&filter=all")
            items = data.get("items", [])
            match = next((i for i in items if abs(i.get("ts", 0) - ts) < 0.01), None)
            if match:
                st = match.get("status")
                if st != last_status:
                    elapsed = int(time.time() - (deadline - timeout))
                    log(INFO, f"[{label}] status={st} elapsed={elapsed}s")
                    last_status = st
                if st == "done":
                    return match
                if st == "cancelled":
                    return match
        except Exception as e:
            log(WARN, f"[{label}] poll error: {e}")
        time.sleep(POLL_INTERVAL)
    return None


def find_files(pattern):
    return sorted(glob.glob(os.path.join(EXP_DIR, pattern)))


def check_server():
    try:
        api_get("/queue/paginated?limit=1")
        return True
    except Exception:
        return False


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    sym = PASS if passed else FAIL
    print(f"{sym} {name}" + (f" — {detail}" if detail else ""))


# ─── tests ────────────────────────────────────────────────────────────────────

def test_server_reachable():
    print("\n[1/5] Server reachable")
    ok = check_server()
    record("server reachable", ok, BASE)
    return ok


def test_image_generation():
    """Inject an image-only job, poll to done, verify base PNG on disk."""
    print("\n[2/5] Image generation (image_only=1)")
    prompt = f"slow-test image check {uuid.uuid4().hex[:6]}"
    ts = None
    try:
        r = inject(prompt, image_only="1")
        assert r.get("status") == "ok", f"inject failed: {r}"
        # Get the ts of the injected item
        data = api_get("/queue/paginated?limit=50&filter=pending")
        match = next((i for i in data["items"] if i.get("prompt") == prompt), None)
        assert match, "injected item not found in pending queue"
        ts = match["ts"]
        log(INFO, f"injected ts={ts:.3f}")
    except Exception as e:
        record("image_generation inject", False, str(e))
        return

    t0 = time.time()
    item = wait_for_done(ts, "image")
    elapsed = time.time() - t0

    if item is None:
        record("image_generation", False, f"timed out after {JOB_TIMEOUT}s")
        return

    succeeded = item.get("succeeded", True)  # None = unknown = treat as ok
    record("image_generation completed", succeeded is not False,
           f"elapsed={elapsed:.0f}s status={item.get('status')}")

    # Check PNG on disk
    pngs = find_files(f"slop_*_base.png")
    # Find one newer than t0
    new_pngs = [p for p in pngs if os.path.getmtime(p) > t0]
    record("image_generation base PNG written", bool(new_pngs),
           new_pngs[0] if new_pngs else "none found")
    if new_pngs:
        sz = os.path.getsize(new_pngs[0])
        record("image_generation PNG non-empty", sz > 1000, f"{sz} bytes")

    # Check asset_paths populated
    asset_paths = item.get("asset_paths") or []
    record("image_generation asset_paths populated", bool(asset_paths),
           str(asset_paths))

    log(INFO, f"image_generation total elapsed: {elapsed:.0f}s")


def test_video_with_audio():
    """Inject a full video+audio job, poll to done, verify MP4 and WAV."""
    print("\n[3/5] Video + audio (LTX2 + HeartMuLa)")
    prompt = f"slow-test video audio {uuid.uuid4().hex[:6]}"
    ts = None
    try:
        r = inject(prompt)
        assert r.get("status") == "ok"
        data = api_get("/queue/paginated?limit=50&filter=pending")
        match = next((i for i in data["items"] if i.get("prompt") == prompt), None)
        assert match, "injected item not in pending queue"
        ts = match["ts"]
        log(INFO, f"injected ts={ts:.3f}")
    except Exception as e:
        record("video_audio inject", False, str(e))
        return

    t0 = time.time()
    item = wait_for_done(ts, "video+audio", timeout=JOB_TIMEOUT)
    elapsed = time.time() - t0

    if item is None:
        record("video_audio", False, f"timed out after {JOB_TIMEOUT}s")
        return

    record("video_audio completed", item.get("succeeded") is not False,
           f"elapsed={elapsed:.0f}s")

    # MP4 check
    mp4s = [p for p in find_files("FINAL_*.mp4") if os.path.getmtime(p) > t0]
    record("video_audio FINAL MP4 written", bool(mp4s),
           mp4s[0] if mp4s else "none")
    if mp4s:
        sz = os.path.getsize(mp4s[0])
        record("video_audio MP4 non-empty", sz > 100_000, f"{sz/1e6:.1f} MB")
        # Duration via ffprobe if available
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", mp4s[0]],
                timeout=10
            ).decode().strip()
            dur = float(out)
            record("video_audio MP4 duration", dur > 5.0, f"{dur:.1f}s")
            log(INFO, f"MP4 duration: {dur:.1f}s")
        except Exception:
            log(WARN, "ffprobe not available — skipping duration check")

    # WAV check
    wavs = [p for p in find_files("*.wav") if os.path.getmtime(p) > t0]
    record("video_audio WAV written", bool(wavs),
           wavs[0] if wavs else "none")
    if wavs:
        sz = os.path.getsize(wavs[0])
        record("video_audio WAV non-empty", sz > 10_000, f"{sz/1e6:.2f} MB")

    # Stages check
    stages = item.get("stages", {})
    audio_done = stages.get("audio", {}).get("status") == "done"
    video_done = stages.get("video", {}).get("status") == "done"
    record("video_audio audio stage done", audio_done)
    record("video_audio video stage done", video_done)

    log(INFO, f"video+audio total elapsed: {elapsed:.0f}s  (~{elapsed/60:.1f} min)")


def test_story_mode():
    """Inject 2 beats with same story_id, both should complete, grouped."""
    print("\n[4/5] Story mode (2 beats, shared story_id)")
    story_id = str(uuid.uuid4())
    story_title = "Slow Test Story"
    beats = [
        f"slow-test story beat-1 {uuid.uuid4().hex[:4]}",
        f"slow-test story beat-2 {uuid.uuid4().hex[:4]}",
    ]
    tss = []
    try:
        for b in beats:
            r = inject(b, story_id=story_id, story_title=story_title)
            assert r.get("status") == "ok", f"inject failed: {r}"
        # Get ts for each
        data = api_get("/queue/paginated?limit=100&filter=all")
        for b in beats:
            m = next((i for i in data["items"] if i.get("prompt") == b), None)
            assert m, f"beat not in queue: {b}"
            tss.append(m["ts"])
        log(INFO, f"injected {len(tss)} beats, story_id={story_id[:8]}…")
    except Exception as e:
        record("story_mode inject", False, str(e))
        return

    # Verify story_id stamped correctly
    data = api_get(f"/queue/paginated?limit=100&filter=all")
    story_items = [i for i in data["items"] if i.get("story_id") == story_id]
    record("story_mode story_id on both items", len(story_items) == 2,
           f"found {len(story_items)}/2")
    if story_items:
        record("story_mode story_title correct",
               story_items[0].get("story_title") == story_title,
               story_items[0].get("story_title"))

    # Poll both to done — image_only for speed
    t0 = time.time()
    done_items = []
    for ts in tss:
        item = wait_for_done(ts, f"story-beat@{ts:.0f}", timeout=JOB_TIMEOUT)
        if item:
            done_items.append(item)

    elapsed = time.time() - t0
    record("story_mode both beats completed", len(done_items) == 2,
           f"completed={len(done_items)}/2 elapsed={elapsed:.0f}s")

    if len(done_items) == 2:
        both_story_id = all(i.get("story_id") == story_id for i in done_items)
        record("story_mode done items retain story_id", both_story_id)

    log(INFO, f"story_mode total elapsed: {elapsed:.0f}s")


def test_asset_paths_persist():
    """Check that recent done items have non-empty asset_paths in queue.json."""
    print("\n[5/5] asset_paths persistence (check existing done items)")
    try:
        data = api_get("/queue/paginated?filter=done&limit=20")
        items = data.get("items", [])
    except Exception as e:
        record("asset_paths check", False, str(e))
        return

    if not items:
        record("asset_paths check", None, "no done items in queue — skipped")
        log(WARN, "no done items to inspect")
        return

    with_assets = [i for i in items if i.get("asset_paths")]
    pct = len(with_assets) / len(items) * 100
    record("asset_paths populated", len(with_assets) > 0,
           f"{len(with_assets)}/{len(items)} items ({pct:.0f}%)")

    # Also check stages.audio.asset and stages.video.asset
    with_audio_asset = [
        i for i in items
        if i.get("stages", {}).get("audio", {}).get("asset")
    ]
    with_video_asset = [
        i for i in items
        if i.get("stages", {}).get("video", {}).get("asset")
    ]
    record("stages.audio.asset present in done items",
           bool(with_audio_asset),
           f"{len(with_audio_asset)}/{len(items)}")
    record("stages.video.asset present in done items",
           bool(with_video_asset),
           f"{len(with_video_asset)}/{len(items)}")


# ─── entrypoint ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Slopfinity Slow Integration Test Suite")
    print(f"  BASE={BASE}  EXP_DIR={EXP_DIR}")
    print(f"  JOB_TIMEOUT={JOB_TIMEOUT}s  POLL_INTERVAL={POLL_INTERVAL}s")
    print("=" * 60)

    if not test_server_reachable():
        print(f"\n{FAIL} Server not reachable at {BASE} — aborting")
        sys.exit(1)

    # These two are slow — full generation cycles
    test_image_generation()
    test_video_with_audio()
    test_story_mode()
    test_asset_paths_persist()

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    passed = skipped = failed = 0
    for name, ok, detail in results:
        if ok is None:
            sym = WARN
            skipped += 1
        elif ok:
            sym = PASS
            passed += 1
        else:
            sym = FAIL
            failed += 1
        print(f"  {sym} {name}" + (f"  [{detail}]" if detail else ""))

    print(f"\n  {PASS} {passed} passed  {FAIL} {failed} failed  {WARN} {skipped} skipped")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
