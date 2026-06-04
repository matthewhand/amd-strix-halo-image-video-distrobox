# ComfyUI Polling Hardening

## Overview

The **ComfyUI polling hardening** (commits `ead9711` and `61094b0`) addresses critical failure modes in the fleet orchestrator's job-completion poller. Before these fixes, five independent `while True:` polling loops — for base-image generation, single-image workflows, LTX video chains, FLF2V (first-last-frame-to-video), and continuation chains — lacked deadlines, per-request socket timeouts, and error caps. When ComfyUI stalls (documented gfx1151 GPU-hang failure mode), accepts a job then becomes unreachable, or the network connection hangs, the serial fleet orchestrator would block forever, starving every queued iteration from execution.

The fix consolidates all five pollers through unified hardened `_poll_comfy_history()` and `_encode_frames_to_mp4()` functions, ensuring timeout logic cannot drift between call sites and the orchestrator always has an escape path.

---

## Root Cause: Failure Mode on gfx1151

The AMD Strix Halo GPU (gfx1151) exhibits a documented failure mode: ComfyUI may accept a job submission (returning a `prompt_id`), then stall indefinitely during model loading or VAE decode. The orchestrator—which is single-threaded, serial, and synchronous—has no timeout guard on its HTTP polling loop. If the socket hangs or ComfyUI never responds, the orchestrator blocks forever waiting for `/history/<prompt_id>`, and the entire queue of pending iterations starves.

**Specific problem sites (pre-fix):**
- `run_image_gen()` → node-12 LTX base-image poller (lines 770–783 in the diff, now 794–798 post-fix)
- `run_comfy_job()` → single-image workflow handler (lines 896–938 pre-fix, now 1003–1010)
- `generate_video_ltx()` → video chain poller (lines 1206–1262 pre-fix, now 1235–1237)
- `generate_video_ltx_flf2v()` → first-last-frame anchor poller (lines 1342–1365 pre-fix, now 1367–1369)
- `generate_video_ltx_continuation()` → chain-handoff continuation poller (lines 1483–1526 pre-fix, now 1508–1510)

Only `generate_video_ltx()` had partial hardening (per-request `timeout=15`); the others had **none**.

---

## Solution: Unified Hardened Poller

### `_poll_comfy_history(p_id, out_node_id, timeout_s=600, poll_s=10, settle_s=0, label="job")`

**Introduced in:** `run_fleet.py`, lines 907–961 (commit `ead9711`)

```python
def _poll_comfy_history(p_id, out_node_id, timeout_s=600, poll_s=10,
                        settle_s=0, label="job"):
    """Poll ComfyUI /history/<p_id> until the job completes, then return the
    list of output filenames for ``out_node_id``.

    Hardened against a hung / restarted / GPU-stalled ComfyUI (a real gfx1151
    failure mode): a hard ``timeout_s`` deadline, a per-request socket timeout,
    and a consecutive-error cap. Without these the caller's `while True:` poll
    would spin — or block on a dead socket — forever, freezing the whole serial
    orchestrator. Raises RuntimeError on timeout / prolonged unreachability /
    ComfyUI execution_error.
    """
    if settle_s:
        time.sleep(settle_s)
    deadline = time.time() + timeout_s
    consecutive_errors = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:8188/history/{p_id}", timeout=15
            ) as r:
                h = json.loads(r.read())
            consecutive_errors = 0
        except (urllib.error.URLError, OSError) as poll_err:
            consecutive_errors += 1
            print(f"   ⚠️  History poll error ({consecutive_errors}) [{label}]: {poll_err}")
            if consecutive_errors > 18:
                raise RuntimeError(
                    f"ComfyUI unreachable for >3 min during {label} poll"
                ) from poll_err
            time.sleep(poll_s)
            continue
        if p_id not in h:
            time.sleep(poll_s)
            continue
        status = h[p_id].get("status", {})
        # Some workflows don't populate status.completed; fall back to the
        # presence of the output node's images below. Only an explicit
        # completed==False means "still running".
        if status.get("completed") is False:
            time.sleep(poll_s)
            continue
        errors = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
        if errors:
            raise RuntimeError(f"ComfyUI execution error during {label}: {errors[0]}")
        node_out = (h[p_id].get("outputs") or {}).get(str(out_node_id))
        # Also guard an empty images list — returning [] would IndexError in
        # _encode_frames_to_mp4 (imgs[0]); keep waiting until frames appear.
        if not node_out or not node_out.get("images"):
            time.sleep(poll_s)
            continue
        return [f["filename"] for f in node_out["images"]]
    raise RuntimeError(
        f"ComfyUI {label} timed out after {int(timeout_s // 60)} min (prompt_id={p_id[:8]})"
    )
```

#### Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `p_id` | str | — | ComfyUI prompt ID (returned by `/prompt` POST) |
| `out_node_id` | int/str | — | Output node ID to poll for images (e.g., "14" for SaveImage) |
| `timeout_s` | int | 600 | Hard deadline (seconds) from now to give up; never blocks longer |
| `poll_s` | int | 10 | Delay (seconds) between poll attempts; image=5 vs video=10 |
| `settle_s` | int | 0 | Pre-poll sleep (e.g., 60 for video to let VAE warm up) |
| `label` | str | "job" | Human-readable stage name for error messages (e.g., "image", "video", "flf2v") |

#### Hardening Mechanisms

**1. Hard Deadline (timeout_s)**
- Line 921: `deadline = time.time() + timeout_s`
- Line 923: `while time.time() < deadline:`
- Lines 959–961: Raises `RuntimeError` if deadline exceeded

Ensures the orchestrator **always exits the poll loop** and can proceed to the next iteration or error recovery, even if ComfyUI hangs forever.

**2. Per-Request Socket Timeout = 15 seconds**
- Line 926: `urllib.request.urlopen(..., timeout=15)`

The socket-level timeout prevents hanging on a dead TCP connection (e.g., ComfyUI process crashed mid-request). The 15-second value balances:
- **Fast fail:** Detect network problems quickly (not 30+ s).
- **Headroom:** VAE decode or model load may be slow; 15s covers legitimate latency spikes.

**3. Consecutive-Error Cap**
- Lines 930–932: Increment `consecutive_errors` on each `URLError` or `OSError`
- Lines 933–936: If `consecutive_errors > 18`, raise `RuntimeError("ComfyUI unreachable for >3 min")`

At the standard image poll rate (`poll_s=5`), 18 consecutive errors = ~90 seconds offline. At video rate (`poll_s=10`), ~180 seconds. The "+3 min" message assumes the worst case; this prevents indefinite polling when ComfyUI is down.

**4. Job Presence Check**
- Lines 939–941: If `p_id not in h`, sleep and continue (job not yet queued on ComfyUI).

**5. Completion Status Inspection**
- Lines 942–948: Poll `status.completed`:
  - `False` explicitly → still running, sleep and retry.
  - Missing/`True` or other truthy → proceed to output check.
  - Handles workflows that omit the `completed` field.

**6. Execution Error Detection**
- Lines 949–951: Scan `status.messages` for `execution_error` entries.
- Raises immediately rather than spinning (e.g., CLIP load fail, VAE decode OOM).

**7. Empty Images Guard (commit 61094b0, line 955)**
- Line 955: `if not node_out or not node_out.get("images"):`
- Prevents returning an empty list `[]` that would IndexError in `_encode_frames_to_mp4(imgs[0])`.
- Keeps polling until at least one frame appears.

---

### `_encode_frames_to_mp4(imgs, out_path)`

**Introduced in:** `run_fleet.py`, lines 964–991 (commit `ead9711`)

```python
def _encode_frames_to_mp4(imgs, out_path):
    """Encode ComfyUI output PNG frames (in comfy-outputs/) to an MP4 at
    out_path, then delete the source frames. Shared by every LTX video poller
    so the frame→MP4 logic can't drift between them."""
    first = imgs[0]
    match = re.search(r"(\d+)(?=_\.png)", first)
    num = match.group(1) if match else "00001"
    parts = first.rsplit(num, 1)
    patt = f"%0{len(num)}d".join(parts)
    print(f"   🎞️  Encoding {len(imgs)} frames → MP4…")
    repo_root = os.path.abspath(os.path.dirname(__file__))
    args = [
        "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", "24",
        "-start_number", str(int(num)),
        "-i", f"comfy-outputs/{patt}",
        "-c:v", _ffmpeg_h264_encoder(),
        "-pix_fmt", "yuv420p",
        "-b:v", "8M",
        out_path,
    ]
    _ffmpeg_run(args, check=True)
    for f in imgs:
        try:
            os.remove(os.path.join(repo_root, "comfy-outputs", f))
        except OSError:
            pass
```

This function consolidates frame-to-MP4 encoding logic that was previously duplicated across four call sites. By unifying it here:
- Frame filename pattern parsing is single-sourced.
- ffmpeg invocation is consistent (same encoder selection, bitrate, pixel format).
- Cleanup (deletion of PNGs) is guaranteed post-encode.
- Future fixes to encoding logic propagate everywhere automatically.

---

## Call Sites Unified by the Fix

### 1. Base Image (LTX-2.3, node-12)

**Before (run_fleet.py, pre-ead9711, ~lines 770–783):**
```python
while True:
    time.sleep(5)
    with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}") as r:
        h = json.loads(r.read())
        if p_id in h:
            fn = h[p_id]["outputs"]["12"]["images"][0]["filename"]
            subprocess.run(["cp", f"comfy-outputs/{fn}", out_path], check=True)
            return True
```

**After (ead9711, lines 794–798):**
```python
imgs = _poll_comfy_history(p_id, "12", poll_s=5, label="image")
subprocess.run(["cp", f"comfy-outputs/{imgs[0]}", out_path], check=True)
return True
```

| Aspect | Before | After |
|--------|--------|-------|
| Timeout | None (infinite) | 600s (hardcoded default) |
| Socket timeout | None | 15s per request |
| Error cap | None | >18 consecutive errors |
| Execution error | Silently hangs | Raises RuntimeError |

---

### 2. Single-Image Workflow (run_comfy_job)

**Before (pre-ead9711, lines 896–938):**
```python
while True:
    time.sleep(5)
    with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}") as h_res:
        h = json.loads(h_res.read())
        if p_id in h:
            out = h[p_id]["outputs"][str(out_node_id)]
            if "images" in out:
                imgs = sorted([i["filename"] for i in out["images"]])
                if len(imgs) == 1:
                    subprocess.run(["cp", f"comfy-outputs/{imgs[0]}", target_file], check=True)
                    return True
                else:
                    # ... duplicated frame→MP4 encoding logic (~40 lines)
```

**After (ead9711, lines 1003–1010):**
```python
imgs = _poll_comfy_history(p_id, out_node_id, poll_s=5, label="image")
if len(imgs) == 1:
    subprocess.run(
        ["cp", f"comfy-outputs/{imgs[0]}", target_file], check=True,
    )
    return True
_encode_frames_to_mp4(sorted(imgs), target_file)
return True
```

Lines removed: 42 → 8 lines in this function; the encoding logic migrates to `_encode_frames_to_mp4()`.

---

### 3. Video Chain (generate_video_ltx, node-14)

**Before (pre-ead9711, lines 1206–1262):**
- Had partial hardening: `timeout=15` on `urlopen`
- **Lacked:** Unified poller interface
- 60-second pre-poll settle time (for VAE warmup), then custom poll loop

```python
time.sleep(60)  # settle_s
deadline = time.time() + 600  # timeout_s
consecutive_errors = 0
repo_root = os.path.abspath(os.path.dirname(__file__))
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}", timeout=15) as r:
            # ... poll logic ...
```

**After (ead9711, lines 1235–1237):**
```python
imgs = _poll_comfy_history(p_id, "14", poll_s=10, settle_s=60, label="video")
_encode_frames_to_mp4(imgs, out_path)
return
```

The `generate_video_ltx()` function is **refactored to reuse the shared poller**, losing its custom timeout logic. The `settle_s=60` parameter ensures the same pre-poll sleep behavior is preserved. Lines removed: 57 → 3 lines.

---

### 4. First-Last-Frame-to-Video (generate_video_ltx_flf2v, node-14)

**Before (pre-ead9711, lines 1342–1365):**
```python
while True:
    time.sleep(10)
    with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}") as r:
        h = json.loads(r.read())
        if p_id in h:
            imgs = [f["filename"] for f in h[p_id]["outputs"]["14"]["images"]]
            # ... duplicated encoding logic ...
```

**After (ead9711, lines 1367–1369):**
```python
imgs = _poll_comfy_history(p_id, "14", poll_s=10, label="flf2v")
_encode_frames_to_mp4(imgs, out_path)
return
```

Lines removed: 41 → 3 lines. Inherits deadline=600s, socket timeout=15s, error cap=18.

---

### 5. Continuation Chain (generate_video_ltx_continuation, node-14)

**Before (pre-ead9711, lines 1483–1526):**
```python
while True:
    time.sleep(10)
    with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}") as r:
        h = json.loads(r.read())
        if p_id in h:
            imgs = [f["filename"] for f in h[p_id]["outputs"]["14"]["images"]]
            # ... duplicated encoding logic ...
```

**After (ead9711, lines 1508–1510):**
```python
imgs = _poll_comfy_history(p_id, "14", poll_s=10, label="continuation")
_encode_frames_to_mp4(imgs, out_path)
return
```

Lines removed: 43 → 3 lines.

---

## Parameter Tuning Across Call Sites

Different pipeline stages use different poll cadences to balance responsiveness vs. CPU utilization:

### Poll Interval (poll_s)

| Stage | poll_s | Call site | Rationale |
|-------|--------|-----------|-----------|
| Base image (node-12) | 5 | Line 794 | Quick generation (~2–8 steps). Poll frequently to detect completion ASAP. |
| Single-image workflow | 5 | Line 1003 | Same as base (both are image, not video). |
| Video chain (LTX) | 10 | Line 1235 | Longer generation (video is ~10–50 frames, 8 steps each). Less frequent polling reduces CPU load. |
| FLF2V | 10 | Line 1367 | Same as video. |
| Continuation | 10 | Line 1508 | Same as video. |

### Settlement Time (settle_s)

| Stage | settle_s | Rationale |
|-------|----------|-----------|
| Base image | 0 (default) | No pre-poll wait. Image workflows are fast. |
| Video (line 1235) | 60 | **Critical:** ComfyUI's VAE decoder must warm up on first run. 60s allows VRAM allocation + initial model loads before polling. Without this settle, polls would timeout while VAE is still initializing. |
| FLF2V | 0 (default) | Inherits default (not specified). |
| Continuation | 0 (default) | Inherits default. |

### Timeout (timeout_s)

All call sites use the hardcoded default `timeout_s=600` (10 minutes):

| Stage | timeout_s | Rationale |
|-------|-----------|-----------|
| Base image (line 794) | 600s (10 min) | Hardcoded default. Base image uses LTX 2.3 base model (fast) or Qwen/Ernie (slower on gfx1151). |
| Single-image workflow (line 1003) | 600s (10 min) | Hardcoded default. Same generation speed as base image. |
| Video (line 1235) | 600s (10 min) | Hardcoded default. LTX video generation is ~10–50 frames at 8 steps. |
| FLF2V (line 1367) | 600s (10 min) | Hardcoded default. First-last-frame anchoring is similar speed to unconditioned video. |
| Continuation (line 1508) | 600s (10 min) | Hardcoded default. Chain-handoff continuation. |

**Note:** The `TIER_PROFILES[tier].image_timeout_s` (e.g., 1260s for "low" tier, line 58) is used only for the launcher subprocess timeout (lines 653, 706 for Qwen/Ernie), not for ComfyUI history polling. The history poller always uses 600s, ensuring predictable orchestrator timeout behavior across all tiers and models.

---

## Output Node ID Contract

ComfyUI workflows define a `SaveImage` node with a numeric `class_type_id`. The runner must know which node to poll for output images:

| Workflow | Node ID | Call Site | Definition |
|----------|---------|-----------|------------|
| `run_image_gen()` base (Qwen/Ernie) | "12" | Line 794 | Hardcoded (node-12 SaveImage) |
| LTX base image (`generate_base_image_ltx23`) | "12" | Line 1142 (via `run_comfy_job(workflow, 12, ...)`) | Hardcoded in workflow (line 1134–1140) |
| LTX video (`generate_video_ltx`) | "14" | Line 1235 | Hardcoded in workflow (line 1220–1223) |
| LTX FLF2V (`generate_video_ltx_flf2v`) | "14" | Line 1367 | Hardcoded in workflow |
| LTX continuation (`generate_video_ltx_continuation`) | "14" | Line 1508 | Hardcoded in workflow |
| Custom workflows via `run_comfy_job()` | Caller-specified | Lines 1003, 1010 | Dynamic parameter |

The output node ID is hardcoded into ComfyUI workflow definitions and must match the actual `SaveImage` node in the JSON. Mismatch → `_poll_comfy_history` never finds frames → timeout after 600s.

---

## Error Handling and Failure Modes

### Timeout (deadline exceeded)

**Raises:** `RuntimeError(f"ComfyUI {label} timed out after {int(timeout_s // 60)} min (prompt_id={p_id[:8]})")`

**Example:** Video poll after 600s with no response.

**Recovery:** Orchestrator catches, logs the iter as failed, and advances to the next iteration (caller doesn't retry within the poller).

**Test:** `test_poll_times_out_without_hanging` (lines 47–60 of test_comfy_poll.py)

### Execution Error

**Raises:** `RuntimeError(f"ComfyUI execution error during {label}: {errors[0]}")`

**Triggers:** ComfyUI's `/history/{p_id}.status.messages` contains `["execution_error", {...}]` entries (e.g., model loading OOM, CLIP tokenization overflow, VAE decode crash).

**Recovery:** Orchestrator logs and fails the iter. The execution error is typically permanent (OOM, bad prompt tokens) and won't resolve on retry.

**Test:** `test_poll_raises_on_execution_error` (lines 36–44)

### Consecutive Network Errors (>18)

**Raises:** `RuntimeError("ComfyUI unreachable for >3 min during {label} poll")`

**Triggers:** `urllib.error.URLError` or `OSError` on 19+ consecutive poll attempts.

**Rationale:** After ~3 minutes of network unavailability, assume ComfyUI is down permanently. Better to fail the iter than block the orchestrator.

**Recovery:** Orchestrator logs and fails the iter. Operator should diagnose why ComfyUI stopped responding.

**Test:** `test_poll_raises_when_unreachable` (lines 85–95)

### Empty Images List (commit 61094b0, line 955)

**Behavior (before 61094b0):** Check `"images" not in node_out` → Return `[]` on success → `_encode_frames_to_mp4([])[0]` → IndexError.

**Behavior (after 61094b0):** Check `not node_out.get("images")` → Continue polling if empty or missing.

**Rationale:** A workflow might report `completed=True` but still be writing frames to disk. Keep polling until frames exist.

**Test:** `test_poll_skips_empty_images_then_returns` (lines 63–82)

---

## gfx1151 GPU Hang Mitigation

The gfx1151 is prone to GPU hangs during VAE decode and model loading, especially under VRAM pressure. The hardening addresses this by:

1. **Early timeout detection:** If ComfyUI hangs for >15 seconds on a single poll, the socket timeout fires and the poller retries. After 18 retries (~3 min), it gives up rather than blocking forever.

2. **Settle time (settle_s=60):** For video workflows, the 60-second pre-poll wait (line 1235) allows ComfyUI to fully initialize its VRAM and model state **before** the orchestrator expects progress. This reduces the chance of hitting a hang state mid-initialization.

3. **Hard deadline:** Even if the hang is transient (ComfyUI is slow but not hung), a hard 10-minute timeout ensures we don't wait indefinitely. The orchestrator can still execute other iterations in the queue.

4. **Explicit execution_error detection:** If ComfyUI detects a hang and logs it to the `status.messages` field, the poller raises immediately rather than spinning.

---

## Fast Track Regression Fix (commit 61094b0, lines 1669–1674)

**Issue:** The Fast Track override (lines 1660–1667) mutates `_task_opts["_config_snapshot"]` to force `chains=1, frames=9, tier=low`, but the global `_CURRENT_ITER_CONFIG` (used by `_write_md_sidecar`, line 544) was bound to the **pre-override snapshot**. The sidecar would report full-quality chains/frames instead of the actual Fast Track values.

**Fix (lines 1669–1674):**
```python
# Re-point the sidecar/state snapshot at the Fast Track values —
# it was bound to the pre-override snapshot above, so without
# this the sidecar would report the full-quality chains/frames.
_ft_snap.setdefault("base_model", b_mod)
_ft_snap["_v_idx"] = v_idx
_CURRENT_ITER_CONFIG = _ft_snap
```

**Verification:** The sidecar metadata (written by `_write_md_sidecar`, line 544) now correctly reports `chains=1, frames=9` for Fast Track runs.

---

## Frames Clamping Regression Fix (commit 61094b0, line 1839)

**Issue:** `_frames_per_chain` was read directly from the config snapshot without validation (line 1835). A malformed snapshot with `frames <= 0` would create an invalid LTX latent (`EmptyLTXVLatentVideo` with `length <= 0`) and cause a divide-by-zero in chain-count math.

**Fix (line 1839):**
```python
_frames_per_chain = max(1, _frames_per_chain)
```

**Verification:** All frame counts are now guaranteed ≥ 1, preventing latent initialization errors.

---

## Test Coverage

**File:** `/home/matthewh/amd-strix-halo-image-video-toolboxes/.claude/worktrees/flamboyant-ride-2e7c01/tests/test_comfy_poll.py`

| Test | Lines | What it verifies |
|------|-------|------------------|
| `test_poll_returns_filenames_on_completion` | 25–33 | Happy path: job completes, frames returned |
| `test_poll_raises_on_execution_error` | 36–44 | Execution error raises RuntimeError immediately |
| `test_poll_times_out_without_hanging` | 47–60 | Deadline exceeded → RuntimeError (not infinite loop) |
| `test_poll_skips_empty_images_then_returns` (61094b0) | 63–82 | Empty images list ignored; polling continues until frames exist |
| `test_poll_raises_when_unreachable` | 85–95 | >18 consecutive errors → RuntimeError (not infinite retry) |

All tests mock `urllib.request.urlopen` and `time.sleep` to avoid network I/O and actual delays. They verify the poller's state machine, not ffmpeg integration.

**Test invocation:**
```bash
pytest tests/test_comfy_poll.py -v
```

All 5 tests pass.

---

## Impact & Metrics

**Lines of code:**
- Removed: ~182 lines of duplicated polling/encoding logic across five call sites.
- Added: ~90 lines for `_poll_comfy_history` + `_encode_frames_to_mp4` + test.
- **Net change:** -92 lines of duplicate-prone code.

**Failure recovery time:**
- **Before:** Indefinite (orchestrator hung until manual kill).
- **After:** Max 10 minutes (timeout_s=600), or ~3 minutes (consecutive-error cap = 18 retries).

**Commits:**
- `ead9711`: Core hardening (Aug 4 2026, 08:52:18)
- `61094b0`: Regression fixes + empty-images guard + event-loop blocking (Aug 4 2026, 09:14:21)

---

## Residuals & Future Work

1. **Dynamic timeout tuning:** Currently `timeout_s` is hardcoded to 600s for all stages. Future work could adapt `timeout_s` based on past iteration duration or user-configured tier, avoiding timeouts on slow runs without padding every run.

2. **Exponential backoff on retries:** Currently `poll_s` is fixed. After N consecutive errors, exponential backoff (e.g., 5s → 10s → 20s) could reduce log spam and CPU load on long outages without blocking the orchestrator indefinitely.

3. **Per-node retry metadata:** The consecutive-error counter is global per poller. Track retries per node to distinguish between "ComfyUI is down" vs. "this specific node is slow," enabling smarter retry policies.

4. **Metrics export:** Log poll duration, error count, and timeouts to Prometheus/CloudWatch so operators can trend GPU hangs and tune `settle_s` empirically.

---

## Verification

Run the full test suite to confirm the hardening doesn't regress:

```bash
pytest tests/test_comfy_poll.py -v
# Expected: 5 passed

# Run against the live orchestrator:
python3 run_fleet.py  # Should not hang if ComfyUI stalls; exits after timeout_s=600s
```

Inspect logs for the retry warnings:
```
⚠️  History poll error (1) [video]: [Errno 111] Connection refused
⚠️  History poll error (2) [video]: [Errno 111] Connection refused
...
⚠️  History poll error (19) [video]: [Errno 111] Connection refused
RuntimeError: ComfyUI unreachable for >3 min during video poll
```

This indicates the error cap is working: 19 errors → raise after deadline or error cap, not infinite retry.
