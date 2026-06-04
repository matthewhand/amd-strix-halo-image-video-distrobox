# run_fleet iter lifecycle

## Overview

The `run_fleet.py` orchestrator implements a serial, event-driven architecture that runs one iteration at a time: prompt generation → audio (optional) → base image → video chains (with multi-frame handoff) → final merge + optional audio mux. Each iteration (v_idx) processes a single queue item claimed atomically under the queue lock, applies per-task snapshot overrides on top of global config, and records the result as a done archive (for audit) with optional requeue (for infinity mode).

This document covers the per-iteration lifecycle: how configuration is snapshotted and applied, how the matrix mode and fast-track overrides work, how frames-per-chain and effective size are derived and clamped, how seed modes work, how stage prompts are applied at each generation call, how the chain loop with multi-frame handoff executes, and how iter failure and cancellation are handled.

---

## Iter Configuration Snapshot & Opts Dict

### The `_task_opts` Dictionary

The per-iter configuration originates in `generate_prompt()` (lines 283–470) and returns a 3-tuple when a queue item is claimed: `(prompt_text, base_model, opts_dict)`. The opts dict is the single source of truth for per-iter state that outlives the `generate_prompt()` call.

**Key fields in `_task_opts`** (built at lines 310–352):

```python
opts = {
    "image_only": bool(task.get("image_only")),
    "skip_video": bool(task.get("skip_video")),
    "infinity": bool(task.get("infinity")),
    "fast_track": bool(task.get("fast_track")),
    "seed_image": task.get("seed_image") or "",
    "seed_images": list(task.get("seed_images") or []),
    "seeds_mode": (task.get("seeds_mode") or "").strip().lower(),
    "stage_prompts": dict(task.get("stage_prompts") or {}),
    "polymorphic": bool(task.get("polymorphic", task.get("chaos", True))),
    "chaos": bool(task.get("chaos", task.get("polymorphic", True))),
    "_seed_prompt": task.get("seed_prompt") or task["prompt"],
    "_orig_task_ts": task.get("ts"),
    "_config_snapshot": task.get("config_snapshot"),
    "_priority": task.get("priority", "next"),
    "_when_idle": bool(task.get("when_idle", False)),
}
```

All downstream snapshot reads (chains, frames, size, tier, audio model, tts model) use the **effective snapshot** `_eff_snap`, which prefers `_config_snapshot` and falls back to the global `config`:

```python
# Lines 1649–1652
_eff_snap = dict(_task_opts.get("_config_snapshot") or config or {})
_eff_snap.setdefault("base_model", b_mod)
_eff_snap["_v_idx"] = v_idx
_CURRENT_ITER_CONFIG = _eff_snap
```

This snapshot is then **mutated in-place** by Fast Track and Matrix overrides (see sections below), and persists through to sidecar generation via the module-global `_CURRENT_ITER_CONFIG` variable (line 477).

### `_config_snapshot` Origin & Persistence

When the user injects a task via the `/inject` endpoint, they select values from the slopfinity Settings panel (chains, frames, size, tier, audio_model, etc.). The API captures this as `task["config_snapshot"]` — a deep copy of the global config at inject time. This snapshot survives queue persistence (`queue.json`): when the same task is requeued for infinity mode, its snapshot is read back and applied so every cycle uses the **same configuration the user originally chose**, not the global config (which may have drifted while the task was cycling).

**Commit d60afcc** made this snapshot actually honored: before this fix, video generators were passed `config["frames"]` and `config["size"]` from the global config, regardless of snapshot values. Now they receive `_eff_size` and `_frames_per_chain` derived from the snapshot.

### Reading Config Snapshot at Chain/Frame/Tier Points

**Frames (lines 1834–1839)**:
```python
_frames_per_chain = int(
    (_task_opts.get("_config_snapshot") or config or {}).get("frames", 49)
)
_frames_per_chain = max(1, _frames_per_chain)
```

**Size (lines 1843–1846)**:
```python
_eff_size = (
    (_task_opts.get("_config_snapshot") or config or {}).get("size")
    or config.get("size")
)
```

**Tier (lines 1774–1781, d60afcc):**
```python
_snap_tier = (_task_opts.get("_config_snapshot") or {}).get("tier")
tier = "low" if _task_opts.get("fast_track") else (
    _snap_tier if _snap_tier in ("low", "med", "high") else pick_tier(v_idx)
)
_CURRENT_ITER_CONFIG["tier"] = tier
```

**Audio model (lines 1704–1706)**:
```python
_audio_model = (_task_opts.get("_config_snapshot") or config or {}).get(
    "audio_model", "none"
) or "none"
```

**TTS model (lines 1748–1750)**:
```python
_tts_model = (_task_opts.get("_config_snapshot") or config or {}).get(
    "tts_model", "none"
) or "none"
```

---

## Matrix Mode & Forced Model Combo Application

### MATRIX_PHASES & pick_matrix_combo

Matrix mode (enabled via `FLEET_MATRIX=1` env var) cycles through three model combos to verify the end-to-end pipeline across different base/video/audio pairs (lines 1521–1532):

```python
MATRIX_PHASES = [
    ("qwen", "ltx-2.3", "none"),
    ("ernie", "ltx-2.3", "none"),
    ("ernie", "ltx-2.3", "heartmula"),
]

def pick_matrix_combo(v_idx):
    """Return (base, video, audio) for this video index under matrix mode."""
    per = int(os.environ.get("FLEET_MATRIX_PER", "3"))
    phase_idx = ((v_idx - 1) // per) % len(MATRIX_PHASES)
    return MATRIX_PHASES[phase_idx]
```

**Default:** 3 iterations per phase (configurable via `FLEET_MATRIX_PER` env var). For example, v1–v3 run qwen→ltx+none, v4–v6 run ernie→ltx+none, v7–v9 run ernie→ltx+heartmula.

### Apply Forced Combo to Iter Snapshot (d60afcc)

**Pre-fix behavior:** Matrix mode called `pick_matrix_combo(v_idx)` and printed the forced models (lines 1629–1633) but never applied them. The global config's audio_model was used, so audio generation was skipped unless the global config had it set.

**Post-fix behavior (lines 1628–1642):**
```python
if matrix_mode:
    b_mod, v_mod_forced, a_mod_forced = pick_matrix_combo(v_idx)
    print(
        f"[MATRIX] v{v_idx}: {b_mod} → {v_mod_forced} + audio={a_mod_forced}",
        flush=True,
    )
    # Actually APPLY the forced combo (previously only printed): write
    # the models into this iter's effective snapshot so audio selection
    # (which gates heartmula on audio_model) and the sidecar honor it.
    # Built from config so frames/chains/tier/size aren't dropped.
    _m_snap = dict(_task_opts.get("_config_snapshot") or config or {})
    _m_snap["base_model"] = b_mod
    _m_snap["video_model"] = v_mod_forced
    _m_snap["audio_model"] = a_mod_forced
    _task_opts["_config_snapshot"] = _m_snap
```

**Invariant:** The forced combo is written into the snapshot BEFORE it's used to populate `_CURRENT_ITER_CONFIG` (line 1649). All downstream audio selection and sidecar generation read from this updated snapshot.

---

## Fast Track Override

Fast Track is a user-facing flag (set via `task["fast_track"]` in the queue) that forces the lowest-quality budget for fast validation: **1 chain × 9 frames × tier=low, skip audio + TTS + upscale**. Targets ~3 min/clip on AMD Strix Halo.

**Application (lines 1659–1679):**
```python
_is_fast = _task_opts.get("fast_track") or (_task_opts.get("_config_snapshot") or {}).get("fast_track") or (_task_opts.get("config_snapshot") or {}).get("fast_track")
if _is_fast:
    _ft_snap = dict((_task_opts.get("_config_snapshot") or config) or {})
    _ft_snap["chains"] = 1
    _ft_snap["frames"] = 9
    _ft_snap["tier"] = "low"
    _ft_snap["audio_model"] = "none"
    _ft_snap["tts_model"] = "none"
    _ft_snap["upscale_model"] = "none"
    _task_opts["_config_snapshot"] = _ft_snap
    # Re-point the sidecar/state snapshot at the Fast Track values —
    # it was bound to the pre-override snapshot above, so without
    # this the sidecar would report the full-quality chains/frames.
    _ft_snap.setdefault("base_model", b_mod)
    _ft_snap["_v_idx"] = v_idx
    _CURRENT_ITER_CONFIG = _ft_snap
    print(
        f"[FLEET] 🏃 Fast Track v{v_idx}: chains=1 frames=9 "
        f"tier=low audio/tts/upscale skipped",
        flush=True,
    )
```

**Scope:** This override ONLY affects the current iteration's snapshot; the global config is untouched. It's checked three times:
1. Directly in `_task_opts` (queue item flag)
2. In the config snapshot (user set it globally)
3. In the legacy `config_snapshot` field name (backwards compat)

**Interaction with tier selection:** Fast Track always forces `tier="low"` (line 1775) independently, so tier derivation respects this (see Tier Selection section).

**Note:** The docstring at lines 314–318 of `generate_prompt()` states outdated values (chains=2, frames=17); the actual implementation uses chains=1, frames=9 (changed in ddab557).

---

## Tier Selection: pick_tier vs Snapshot Tier

### Default Rotating Tier (pick_tier function)

In the absence of a user-selected snapshot tier, the runner rotates through low/med/high based on video index to validate incrementally before running expensive high-quality passes (lines 117–125):

```python
def pick_tier(v_idx):
    forced = os.environ.get("FLEET_TIER", "").strip().lower()
    if forced in TIER_PROFILES:
        return forced
    if v_idx <= 2:
        return "low"
    if v_idx <= 5:
        return "med"
    return "high"
```

Override via env var: `FLEET_TIER=high` pins all iters to high-quality.

### Snapshot Tier Override (d60afcc)

**Pre-fix:** Image gen always used `pick_tier(v_idx)`, ignoring any user-selected snapshot.tier.

**Post-fix (lines 1774–1781):**
```python
_snap_tier = (_task_opts.get("_config_snapshot") or {}).get("tier")
tier = "low" if _task_opts.get("fast_track") else (
    _snap_tier if _snap_tier in ("low", "med", "high") else pick_tier(v_idx)
)
_CURRENT_ITER_CONFIG["tier"] = tier
```

**Precedence:**
1. Fast Track: always forces `low`
2. Snapshot tier: used if set to a valid value
3. pick_tier(v_idx): rotating default fallback

The resolved `tier` is immediately reflected back into `_CURRENT_ITER_CONFIG` so the markdown sidecar writer reports what was actually used, not the stale snapshot value.

---

## Frame & Size Derivation with Clamping

### _frames_per_chain (Base Derivation)

**Snapshot read (lines 1834–1839)**:
```python
_frames_per_chain = int(
    (_task_opts.get("_config_snapshot") or config or {}).get("frames", 49)
)
_frames_per_chain = max(1, _frames_per_chain)
```

- Defaults to 49 (high-quality) when unset
- Clamped to ≥1 to avoid divide-by-zero in chain math and malformed LTX latents

### _eff_size (Effective Render Size)

**Snapshot read (lines 1843–1846)**:
```python
_eff_size = (
    (_task_opts.get("_config_snapshot") or config or {}).get("size")
    or config.get("size")
)
```

Size can be an aspect ratio string ("16:9", "1:1") or a pixel spec ("1280*720"). When passed to video generators, it's resolved to pixel form via `_resolve_size()` (lines 1018–1033).

### FLF2V 9-Frame Minimum (ddab557)

When per-chain seed mode activates FLF2V (First-Last-Frame-to-Video), the end keyframe is placed at frame index `max(8, ((frames - 1) // 8) * 8)`. If the total latent is fewer than 9 frames, this index lands outside the valid range [0, frames-1], producing a malformed ComfyUI graph.

**Guard (lines 1885–1892)**:
```python
_flf2v_active = len(_per_chain_seeds) >= 2
if _flf2v_active:
    # FLF2V puts its end keyframe at frame index max(8, …); a latent
    # shorter than 9 frames places that guide out of [0, frames-1]
    # and yields a malformed graph / wrong output. Enforce the LTX
    # 9-frame minimum (only reachable via a hand-set frames<9 + per-
    # chain seeds; default/Fast-Track frame counts are already ≥9).
    _frames_per_chain = max(9, _frames_per_chain)
```

This is the only scenario where frames are bumped up: when FLF2V is active, frames are never allowed below 9. Default and Fast-Track values (49 and 9 respectively) already satisfy this.

---

## Seed Modes: Per-Task vs Per-Chain

### Per-Task Seed (Single Image Mode)

A user can upload a base image that replaces the generated base (lines 1782–1801):

```python
_seed_for_base = _task_opts.get("seed_image") or ""
if not _seed_for_base and (_task_opts.get("seeds_mode") == "per-chain"):
    _seed_list = _task_opts.get("seed_images") or []
    if _seed_list:
        _seed_for_base = _seed_list[0]
if _seed_for_base:
    _seed_src = os.path.join(OUTPUT_DIR, _seed_for_base)
    if not os.path.exists(_seed_src):
        raise FileNotFoundError(
            f"seed image vanished from outputs dir: {_seed_src}"
        )
    os.makedirs(os.path.dirname(in_img) or ".", exist_ok=True)
    subprocess.run(["cp", _seed_src, in_img], check=True)
    print(
        f"[FLEET] 🌱 seed-as-base: copied {_seed_for_base} → {in_img} (skipping generate_base)",
        flush=True,
    )
```

**Field:** `task["seed_image"]` (string, single filename).

### Per-Chain Seed (Multi-Keyframe Mode)

When a user uploads N≥2 seed images with `seeds_mode="per-chain"`, each chain c spans from seed[c-1] to seed[c] using FLF2V, forcing N-1 chains (lines 1870–1882):

```python
_per_chain_seeds = (
    _task_opts.get("seed_images") or []
    if _task_opts.get("seeds_mode") == "per-chain" else []
)
if len(_per_chain_seeds) >= 2:
    _n_chains = len(_per_chain_seeds) - 1
    print(
        f"[FLEET] 🌱 per-chain FLF2V: {len(_per_chain_seeds)} seeds → "
        f"{_n_chains} chains (seed[i]→seed[i+1])",
        flush=True,
    )
```

**Fields:** `task["seed_images"]` (list of filenames), `task["seeds_mode"]` ("per-chain" or falsy).

**Chain Binding (lines 1931–1948):**
For chain c:
- START keyframe: `seed_images[c-1]`
- END keyframe: `seed_images[c]`
```python
_start_seed = _per_chain_seeds[c_idx - 1]
_end_seed = _per_chain_seeds[c_idx]
_start_fn = f"{_stem}_kf_start_c{c_idx}.png"
_end_fn = f"{_stem}_kf_end_c{c_idx}.png"
subprocess.run(
    ["cp", os.path.join(OUTPUT_DIR, _start_seed),
     f"comfy-input/{_start_fn}"], check=True,
)
subprocess.run(
    ["cp", os.path.join(OUTPUT_DIR, _end_seed),
     f"comfy-input/{_end_fn}"], check=True,
)
generate_video_ltx_flf2v(
    _start_fn, _end_fn, _vid_prompt, seg,
    _eff_size, _frames_per_chain,
)
```

### Edge Case: Single Seed in Per-Chain Mode (ddab557)

**Pre-fix:** If the user set `seeds_mode="per-chain"` but only uploaded one seed, the code would fail silently or attempt to span seed[0]→seed[0] (invalid).

**Post-fix:** Detection at /inject time (the router, not shown here) demotes single-seed per-chain tasks to per-task mode so the one seed still serves as a base image.

---

## Stage Prompts Application (d9becc4)

### Per-Stage Prompt Overrides

The user can inject stage-specific prompt overrides via the Raw-mode or multi-beat form: separate texts for {image, video, music, tts} stages. These override the main prompt for that stage only; filename slug and sidecars keep the main prompt for stable identity.

**Extraction (lines 1688–1695)**:
```python
_stage_prompts = _task_opts.get("stage_prompts") or {}
_img_prompt = (_stage_prompts.get("image") or "").strip() or p
_vid_prompt = (_stage_prompts.get("video") or "").strip() or p
_music_prompt = (_stage_prompts.get("music") or "").strip() or p
```

Fallback to main prompt `p` when the stage override is empty/missing.

### Application at Each Generation Call

Each generation function receives its stage-specific prompt:

**Image (lines 1802–1805)**:
```python
elif b_mod in ["qwen", "ernie"]:
    run_image_gen(b_mod, _img_prompt, in_img, tier=tier, size_str=config.get("size"))
else:
    generate_base_image_ltx23(p, in_img, config["size"])
```

**Video (chain c, lines 1949–1984)**:
```python
generate_video_ltx_flf2v(
    _start_fn, _end_fn, _vid_prompt, seg,
    _eff_size, _frames_per_chain,
)
# ... or ...
generate_video_ltx_continuation(
    _handoff_frames, _vid_prompt, seg,
    _eff_size, _frames_per_chain,
)
# ... or ...
generate_video_ltx(
    os.path.basename(in_img), _vid_prompt, seg, _eff_size, _frames_per_chain
)
```

**Music (lines 1724)**:
```python
ok = heartmula_wav(_music_prompt, audio_wav, duration_s=target_dur)
```

**History:** These fields were persisted to `queue.json` since round-3 (d0c4f1e) but never consumed. **d9becc4** added the read-and-apply path.

---

## Chain Loop & Multi-Frame Handoff

### Chain Count Derivation

**Audio-driven chains (lines 1847–1861):**
When `config.audio_driven_chains` is true AND audio was generated, chain count is sized to span the audio duration:
```python
_audio_driven = bool(
    (_task_opts.get("_config_snapshot") or config or {}).get(
        "audio_driven_chains"
    )
)
if _audio_driven and audio_duration_s > 0 and _frames_per_chain > 0:
    _chain_seconds = _frames_per_chain / 24.0
    _n_chains = max(1, int(math.ceil(audio_duration_s / _chain_seconds)))
    _n_chains = min(_n_chains, 30)  # safety cap
```

**Default (lines 1863–1868)**:
```python
else:
    _n_chains = int(
        (_task_opts.get("_config_snapshot") or config or {}).get(
            "chains", 10
        )
        or 10
    )
```

**Per-chain seed override (lines 1876–1877)**:
If `len(seed_images) >= 2`, it forces `_n_chains = len(seed_images) - 1`.

### Chain Handoff Keyframes (K)

**Derivation (lines 1896–1901)**:
```python
_handoff_k = int(
    (_task_opts.get("_config_snapshot") or config or {}).get(
        "chain_handoff_keyframes", 4
    ) or 4
)
_handoff_k = max(1, min(_handoff_k, 8))
```

Default K=4; clamped to [1, 8]. K=1 reverts to legacy single-frame handoff; K>1 anchors the next chain's first K frames to the previous chain's last K frames.

### Loop Body: Per-Chain Generation (lines 1904–2040)

```
for c_idx in range(1, _n_chains + 1):
  1. Check cancel.flag (lines 1910–1920)
  2. Generate chain c via:
     - FLF2V (per-chain seeds) OR
     - Continuation (K>1 multi-frame) OR
     - I2V (first chain, K=1, or continuation inactive)
  3. Extract handoff frames for c+1 (lines 1994–2040)
     - If c < _n_chains and K>1: ffmpeg extract last K frames + margin
     - If c < _n_chains and K=1: ffmpeg extract last frame
```

### Cancel Flag Check (ddab557)

**At each chain boundary (lines 1910–1920)**:
```python
_cf = os.path.join(OUTPUT_DIR, "cancel.flag")
try:
    if os.path.exists(_cf) and os.path.getmtime(_cf) >= _iter_started_ts:
        print(
            f"[FLEET] cancel.flag — aborting iter v{v_idx} at chain {c_idx}/{_n_chains}",
            flush=True,
        )
        _iter_cancelled = True
        break
except OSError:
    pass
```

**Semantics:** The mtime check gates against `_iter_started_ts` so a stale flag from a prior iter is ignored. When the cancel flag is fresher than iter start, we break the loop and set `_iter_cancelled = True` to unwind via the `_IterCancelled` exception (see Iter Failure & Cancellation section).

### FLF2V Chain Binding

When per-chain seeds are active (lines 1931–1962):
```python
_start_seed = _per_chain_seeds[c_idx - 1]
_end_seed = _per_chain_seeds[c_idx]
_start_fn = f"{_stem}_kf_start_c{c_idx}.png"
_end_fn = f"{_stem}_kf_end_c{c_idx}.png"
subprocess.run([...cp... _start_seed ... _start_fn...], check=True)
subprocess.run([...cp... _end_seed ... _end_fn...], check=True)
generate_video_ltx_flf2v(
    _start_fn, _end_fn, _vid_prompt, seg,
    _eff_size, _frames_per_chain,
)
```

Chain 1 spans seed[0]→seed[1], chain 2 spans seed[1]→seed[2], etc. All keyframes are copied into `comfy-input/` before the workflow is submitted to ComfyUI.

### Multi-Frame Continuation (K>1)

When c>1, K>1, and not FLF2V (lines 1963–1978):
```python
elif c_idx > 1 and _handoff_k > 1 and _handoff_frames:
    generate_video_ltx_continuation(
        _handoff_frames, _vid_prompt, seg,
        _eff_size, _frames_per_chain,
    )
```

The `_handoff_frames` list was populated after chain c-1 (see below).

### Handoff Frame Extraction (lines 1994–2040)

**Multi-frame case (K>1, lines 1995–2025)**:
```python
if _handoff_k > 1:
    sec = max(0.4, (_handoff_k + 2) / 24.0)
    handoff_pattern = f"comfy-input/{_stem}_h{c_idx}_%03d.png"
    _ffmpeg_run(
        ["-y", "-sseof", f"-{sec}", "-i", seg,
         "-vsync", "0", handoff_pattern],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    extracted = sorted(
        [f for f in os.listdir("comfy-input")
         if f.startswith(stem_prefix)],
        key=lambda x: int(
            re.search(r"_(\d+)\.png$", x).group(1)
        ),
    )
    _handoff_frames = extracted[-_handoff_k:]
```

Extract the last K frames from the chain video using `ffmpeg -sseof -<sec>` (seek from end of file). A margin of 2+ frames is included (`-handoff_k + 2`), then only the trailing K are kept. This anchors the next chain's continuation.

**Legacy single-frame case (K=1, lines 2026–2040)**:
```python
else:
    next_in = f"comfy-input/{_stem}_f{c_idx}.png"
    _ffmpeg_run(
        ["-y", "-sseof", "-1", "-i", seg,
         "-update", "1", "-q:v", "1", next_in],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    in_img = next_in
    subprocess.run(
        ["cp", next_in, f"{OUTPUT_DIR}/{_stem}_f{c_idx}.png"],
        check=True,
    )
```

Extracts the last frame only; used for the next chain's I2V input.

### Handoff Frame Cleanup (lines 2042–2055)

After the chain loop completes, all transient handoff frames are deleted from `comfy-input/`:
```python
try:
    for _hf in os.listdir("comfy-input"):
        if _hf.startswith(f"{_stem}_h") or _hf.startswith(f"{_stem}_f"):
            try:
                os.remove(os.path.join("comfy-input", _hf))
            except OSError:
                pass
except OSError:
    pass
```

These are intermediate files (prefixed `_h<c_idx>_` and `_f<c_idx>_`) not needed once chaining is complete. Cleanup prevents unbounded growth of `comfy-input/` across thousands of iterations.

---

## Iter Failure & Cancellation (ddab557)

### _IterCancelled Exception Class (line 38–41)

```python
class _IterCancelled(Exception):
    """Raised to unwind the current iter when the user cancels the running item
    mid-flight (dashboard writes cancel.flag). Handled distinctly from real
    errors so a cancel isn't logged/recorded as a failure."""
```

### Cancellation Unwind

When `cancel.flag` is detected at a chain boundary (lines 1910–1920), `_iter_cancelled` is set to True and the loop breaks. After the loop, if no chains finished, `_IterCancelled` is raised (lines 2057–2061):

```python
if _iter_cancelled and not chain_vids:
    raise _IterCancelled(f"iter v{v_idx} cancelled (no chains rendered)")
```

If ≥1 chain rendered, finalization proceeds to give the user the partial result.

### Exception Handling (lines 2141–2165)

```python
except Exception as e:
    if isinstance(e, _IterCancelled) or _iter_cancelled:
        # User-initiated cancel of the running item — not a failure.
        # The queue already marks it cancelled (so the archive path
        # below won't requeue it); just update state and move on.
        print(f"[FLEET] ⏹ {e}", flush=True)
        try:
            update_state(mode="Cancelled", step="User cancelled",
                         video=v_idx, total=1000, prompt=p)
        except Exception:
            pass
    else:
        iter_failed = True
        import traceback

        print(f"[FLEET] ❌ Error Video {v_idx}: {e!r}", flush=True)
        traceback.print_exc()
        # Reflect the failure in dashboard state immediately
        try:
            update_state(mode="Completed", step="Failed (see log)",
                         video=v_idx, total=1000, prompt=p)
        except Exception:
            pass
        time.sleep(10)
```

**Cancellation path:** Sets `iter_failed = False` and updates dashboard to "Cancelled" state. The queue already has `status="cancelled"` (written by the dashboard when the user clicked cancel), so the requeue logic (lines 2263–2316) skips requeue for cancelled items.

**Failure path:** Sets `iter_failed = True`, prints traceback, and updates dashboard to "Failed (see log)". This update was added in d60afcc so the UI doesn't hang on the previous step (e.g. "Rendering") forever.

### State Updates in Exception Path (d60afcc)

**Pre-fix:** When an iter failed, the dashboard state would show the last step before the crash (e.g. "Rendering") until the next successful iter started. If the crash was on the final item, the UI would hang indefinitely.

**Post-fix:** Both cancellation and failure paths call `update_state()` in the except block, so the UI reflects the terminal state immediately.

---

## _CURRENT_ITER_CONFIG & Markdown Sidecar

### Module-Global Snapshot Tracker (line 477)

```python
_CURRENT_ITER_CONFIG: dict = {}
```

This is a **module-level tracker** populated at the top of each iter loop (lines 1649–1674) so `_write_md_sidecar()` can read iter metadata without threading it through every function call.

### Sidecar Generation (lines 532–598)

The markdown sidecar (`.md` file next to FINAL outputs) embeds pipeline metadata:
- Prompt and model
- Tier, size, frames, chain count
- Base/video/audio/TTS model selection
- Component list (all slop_* and FINAL_* files from the iter)

**Example output:**
```markdown
# FINAL_42_abstract_garden_audio.mp4

**Prompt:** An abstract garden of digital flowers…
**Model:** ltx-2.3  •  **Kind:** final-with-audio  •  **Tier:** high
**Generated:** 2026-06-04T12:34:56

## Pipeline
- Base image: qwen
- Video model: ltx-2.3
- Audio model: heartmula
- TTS: none
- Size: 1280*720  •  Frames/part: 49  •  Chains: 3

## Components
- slop_42_abstract_garden_base.png
- slop_42_abstract_garden_c1.mp4
- slop_42_abstract_garden_c2.mp4
- slop_42_abstract_garden_c3.mp4
- FINAL_42_abstract_garden.mp4
- FINAL_42_abstract_garden_audio.mp4
```

### Tier Sync Back to Snapshot (d60afcc)

Since the tier can be either snapshot-driven or dynamically derived via `pick_tier(v_idx)`, the resolved tier is synced back into `_CURRENT_ITER_CONFIG` (line 1781) so the sidecar reports the actual tier used:

```python
_CURRENT_ITER_CONFIG["tier"] = tier
```

---

## Iter Asset Tracking & Archive

### _iter_assets List

Each iter maintains a list of generated asset basenames (relative to `OUTPUT_DIR`) (lines 1620, 1734, 1812, 2098, 2138):

```python
_iter_assets: list = []
# Populated as each stage writes:
_iter_assets.append(os.path.basename(audio_wav))
_iter_assets.append(os.path.basename(_out_base))
_iter_assets.append(os.path.basename(final_silent))
_iter_assets.append(os.path.basename(final_audio))
```

### Done Archive & Requeue (lines 2184–2319)

After iter completion (success or failure), the queue is atomically updated under `cfg.queue_lock()`:

1. **Pull live record:** Find and remove the in-flight `working` item, preserving any mid-flight toggles (infinity, polymorphic, priority).

2. **Done archive:** Append a `done` record (audit log) with metadata (lines 2240–2259):
   ```python
   q_now.append({
       "prompt": _task_opts["_seed_prompt"],
       "status": "done",
       "succeeded": not iter_failed,
       "ts": orig_ts or _iter_started_ts,
       "started_ts": _iter_started_ts,
       "completed_ts": time.time(),
       "duration_s": time.time() - _iter_started_ts,
       "v_idx": v_idx - 1,
       "image_only": eff_image_only,
       "infinity": eff_infinity,
       "chaos": eff_chaos,
       "config_snapshot": eff_config_snapshot,
       "assets": list(_iter_assets),
   })
   ```

3. **Requeue (if infinity AND not cancelled):** Lines 2261–2306.
   - **Polymorphic:** Re-append the seed so the next cycle runs the LLM rewriter again.
   - **Fixed-prompt:** Cache the rewritten text so future cycles skip the LLM.

4. **Atomic save:** `cfg.save_queue(q_now)` commits the done + (maybe) requeue + working-row drop in a single call.

---

## Data Flow Diagram

```
iter v_idx starts
  ↓
generate_prompt(m_id, v_idx)
  → claim queue item under lock
  → return (prompt, base_model, opts_dict)
  ↓
_eff_snap = dict(opts._config_snapshot or config)
_CURRENT_ITER_CONFIG = _eff_snap
  ↓
[MATRIX MODE]
  → _m_snap["base_model/video_model/audio_model"] = forced
  → opts._config_snapshot = _m_snap
  ↓
[FAST TRACK]
  → _ft_snap["chains/frames/tier/audio/tts"] = overrides
  → opts._config_snapshot = _ft_snap
  → _CURRENT_ITER_CONFIG = _ft_snap
  ↓
Audio stage
  → _audio_model from snapshot
  → heartmula_wav(_music_prompt, ...)
  ↓
Base image stage
  → tier = snapshot.tier | pick_tier(v_idx)
  → _CURRENT_ITER_CONFIG["tier"] = tier
  → run_image_gen(b_mod, _img_prompt, ...)
  ↓
Chain loop: for c_idx in range(1, _n_chains+1)
  → check cancel.flag (mtime >= iter_start)
  → if cancel && no chains: raise _IterCancelled
  ↓
  FLF2V path (per-chain seeds)
    → generate_video_ltx_flf2v(seed[c-1], seed[c], _vid_prompt, ...)
  ↓
  Continuation path (K>1 multi-frame)
    → extract last K frames from chain c-1
    → generate_video_ltx_continuation(_handoff_frames, _vid_prompt, ...)
  ↓
  I2V path (first chain or K=1)
    → generate_video_ltx(in_img, _vid_prompt, ...)
  ↓
Clean up handoff frames from comfy-input/
  ↓
Final merge (concat all chains)
  ↓
[if audio generated]
  → ffmpeg mux audio onto final_silent → final_audio
  ↓
_write_md_sidecar reads _CURRENT_ITER_CONFIG
  ↓
[exception]
  → if _IterCancelled: update_state("Cancelled")
  → else: update_state("Failed")
  ↓
Archive done record + requeue (if infinity && !cancelled)
  ↓
v_idx += 1
```

---

## Failure Modes & Edge Cases

### Config Snapshot Lost Update (d60afcc)

**Symptom:** User injected a task with `frames=17, size="1:1"`, but video generators received global config values instead.

**Root cause:** Snapshot was computed but not applied. Video generators used `config["frames"]` and `config["size"]` directly, ignoring the snapshot's per-task overrides.

**Fix:** Introduce `_eff_size` and `_frames_per_chain` derived from snapshot, pass them to all three video generators (FLF2V, continuation, I2V).

**Residuals:** LTX-2.3 base image generation (line 1805, `generate_base_image_ltx23(p, in_img, config["size"])`) still uses global config["size"]. This is intentional: the base image is not snapshot-aware, only the chains are. Consider adding snapshot-aware size to base LTX if needed.

### Matrix Mode Never Applied (d60afcc)

**Symptom:** Matrix mode printed the forced models but heartmula never ran; audio_model still came from global config.

**Root cause:** `pick_matrix_combo()` was called and printed, but the returned models were never written back into the snapshot.

**Fix:** After `pick_matrix_combo()`, write the forced base/video/audio models into `_task_opts["_config_snapshot"]` before it's used to populate `_CURRENT_ITER_CONFIG`. All downstream audio selection and sidecar generation now read from the updated snapshot.

### Tier Staleness in Sidecar (d60afcc)

**Symptom:** MP4 sidecar reported a different tier than what was actually used for image gen (e.g., sidecar says "med" but image was generated at "low" due to pick_tier rotating).

**Root cause:** Tier was computed for image gen but not synced back into `_CURRENT_ITER_CONFIG`. The sidecar writer read the stale snapshot value.

**Fix:** Immediately after tier selection (line 1781), sync it back: `_CURRENT_ITER_CONFIG["tier"] = tier`.

### FLF2V Keyframe Out of Bounds (ddab557)

**Symptom:** `frames=5` + FLF2V → last keyframe at index `max(8, ((5-1)//8)*8) = 0` (only the start frame, no interpolation).

**Root cause:** FLF2V's `frame_idx` must be a multiple of 8 per LTX's temporal token grid. If the latent is shorter than 9 frames, the computed index lands outside [0, frames-1], yielding a malformed ComfyUI graph (e.g., "image index out of range").

**Fix:** Clamp `_frames_per_chain = max(9, _frames_per_chain)` when FLF2V is active. Default and Fast-Track values already satisfy this; only hand-set low values trigger the clamp.

### Stale terminate.flag (ddab557)

**Symptom:** Fleet was terminated via dashboard, then restarted. It exited immediately and was un-restartable without manual flag removal.

**Root cause:** `terminate.flag` was written by the dashboard but never cleared. The next iter loop checked the file, found it, and exited before any queue items could be processed.

**Fix:** At startup (lines 1575–1584), detect and remove any stale `terminate.flag` from a previous run. The flag is now ephemeral: it exists only for the duration of the current run.

### cancel.flag Never Read (ddab557)

**Symptom:** User clicked "cancel" on a running item; the item kept rendering, and only the requeue was skipped.

**Root cause:** The dashboard's `/queue/cancel` endpoint wrote `cancel.flag`, but `run_fleet.py` never read it. The flag was ignored.

**Fix:** At each chain boundary (lines 1910–1920), check if `cancel.flag` exists and its mtime ≥ iter_start. If so, set `_iter_cancelled = True` and break. After the loop, if no chains finished, raise `_IterCancelled` to unwind and record the item as cancelled (not failed).

### Per-Chain Seed with Single Seed

**Symptom:** User set `seeds_mode="per-chain"` but only uploaded one seed image.

**Root cause:** The per-chain code tries to span seed[i] → seed[i+1]; with a single seed, this is impossible.

**Fix:** At `/inject` time (router logic, not run_fleet.py), detect single-seed per-chain and demote to `seeds_mode="per-task"`. The one seed still serves as a base image via the per-task seed logic (lines 1786–1789).

### Stage Prompts Never Applied (d9becc4)

**Symptom:** User provided separate prompts for image/video/music stages via `/inject` form. All stages used the main prompt instead.

**Root cause:** The fields were persisted to `queue.json` (since round-3 fixes) but `generate_prompt()` and the main loop never read them.

**Fix:** Add `"stage_prompts": dict(task.get("stage_prompts") or {})` to opts (line 335), extract them in the main loop (lines 1688–1695), and pass the stage-specific prompts to each generation function.

### Iter Failure Hangs Dashboard (d60afcc)

**Symptom:** An iter crashed; the UI showed "Rendering" forever (or until the next iter, which may never come if there's only one item in the queue).

**Root cause:** Exception handling printed the error but didn't call `update_state()`. The dashboard was left showing the last state before the crash.

**Fix:** In the except block (lines 2162–2165), call `update_state(mode="Completed", step="Failed (see log)", ...)` so the UI reflects the terminal state immediately.

---

## Verification

### Test Suite Coverage

- `tests/test_config_extras.py` — config snapshot round-trip, queue lock atomicity, extra-field persistence (seed_image, stage_prompts, seed_images, etc.)
- `tests/test_server_queue.py` — queue operations and state transitions
- Full test suite is passing (commits d9becc4, ddab557, d60afcc)

### Key Test Scenarios (covered by test_config_extras.py and integration testing)

1. **Snapshot tier override:** Inject a task with `tier="med"`, verify image gen uses that tier instead of pick_tier(v_idx).
2. **Matrix mode forced combo:** Enable matrix mode, run 3 iters, verify audio_model is forced to none, heartmula, none per phase.
3. **Fast Track budget:** Inject with `fast_track=True`, verify chains=1, frames=9, tier=low, audio/tts skipped.
4. **FLF2V 9-frame clamp:** Set `frames=5, seeds_mode="per-chain", seed_images=[a,b,c]`, verify _frames_per_chain bumped to 9.
5. **Cancel.flag mid-flight:** Set cancel.flag, wait for chain boundary, verify iter unwound via _IterCancelled and recorded as cancelled.
6. **Per-stage prompts:** Inject `stage_prompts={image: "...", video: "..."}`, verify image and video stages use their overrides, slug uses main prompt.
7. **Extra field persistence:** Verify seed_image, seed_images, stage_prompts, seed_prompt, requeued_from_ts survive queue DB round-trip.

---

## Residuals & Future Work

1. **Base image size not snapshot-aware:** Line 1805 uses global `config["size"]` for LTX-2.3 base. Consider accepting a size_str parameter and passing snapshot size to `generate_base_image_ltx23()` so the entire iter (base + chains) respects snapshot size.

2. **TTS stage stubbed:** Lines 1742–1765 print a warning but skip TTS. Wire `scripts/qwen_tts_serve.py` or kokoro path to actually produce TTS output.

3. **Handoff frame margin heuristic:** Line 1998 computes `sec = max(0.4, (_handoff_k + 2) / 24.0)`. The +2 frames is a hand-tuned margin; no datasheet validation. Could use adaptive margin based on frame drops observed.

4. **Chain count cap at 30:** Line 1856 caps audio-driven chains at 30. Document the rationale (memory, render time) or make configurable.

5. **LTX quantization hardcoded:** Lines 1070, 1154, 1269 hardcode the checkpoint name (`ltx-2.3-22b-distilled-fp8.safetensors`). Consider making the quantization variant user-selectable (e.g., full precision, int8).

6. **No per-chain TTS:** TTS is stubbed globally. When wired, consider per-chain TTS (different voices per segment) as a per-stage feature similar to prompts.

7. **Fast Track docstring stale:** Lines 314–318 of `generate_prompt()` state outdated values (chains=2, frames=17). Update to chains=1, frames=9.

---

## Commits Referenced

- **d60afcc**: `fix(run_fleet): honor per-task snapshot for frames/size/tier + apply matrix combo` — Fixes snapshot application, tier synchronization, matrix mode forcing, and iter-failure state update.
- **d9becc4**: `fix: round-5 batch 2 — apply per-stage prompts + engine-aware TTS default voice` — Wires stage_prompts through opts and applies them at each generation call.
- **ddab557**: `fix: round-5 batch 1 — flag IPC, settings round-trip, FLF2V/seed guards` — Adds cancel.flag reading and _IterCancelled unwinding, clears stale terminate.flag, enforces FLF2V 9-frame minimum, demotes single-seed per-chain to per-task.
