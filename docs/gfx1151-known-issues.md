# gfx1151 (Strix Halo) known issues & mitigations

Hardware-specific quirks found while validating the slop pipeline on AMD
Ryzen AI Max "Strix Halo" (gfx1151) with the TheRock ROCm nightly stack
(`torch 2.11.0+rocm7.13.0a20260424`). Single source of truth for the guard
logic is [`slopfinity/compat.py`](../slopfinity/compat.py).

---

## ERNIE-Image GPU hang above 512² — **mitigated, not fixed**

**Symptom.** `ernie_launcher.py` at 1024×1024 completes its 8 diffusion steps,
then hangs the GPU during the VAE decode:

```
MIOpen(HIP): Warning [OpenRuntimeLibraryForDevice] CK grouped conv library
  not found for device gfx1151: libMIOpenCKGroupedConv_gfx1151.so: cannot open
  shared object file
HW Exception by GPU node-1 ... reason :GPU Hang
```

The hang wedges the GPU for ~minutes; the driver recovers and ComfyUI survives,
but the iteration is lost and the GPU is briefly unusable for everything.

### What we tried

1. **Lower resolution → works.** `--width 512 --height 512` decodes cleanly
   every time. This is the shipped mitigation.
2. **Chased the "CK grouped conv library not found" warning.** Found a genuine
   **dangling-symlink packaging bug** in the ROCm wheel: the loader wants
   `libMIOpenCKGroupedConv_gfx1151.so`, but in
   `_rocm_sdk_libraries_gfx1151/lib/` only `…so.1.0` exists — the unversioned
   `.so` symlink was never created (exactly 1 lib affected; all others are fine).
3. **Repaired the symlink and re-tested at 1024².**
   `ln -sf libMIOpenCKGroupedConv_gfx1151.so.1.0 …/libMIOpenCKGroupedConv_gfx1151.so`
   — the library then loaded (the "not found" warning disappeared) **but the
   GPU still hung at the 1024² decode.**

### Conclusion (root cause)

The hang is in the **CK grouped-conv GPU kernel itself on gfx1151**, not in
library resolution. It hangs whether MIOpen falls back (lib missing) *or* runs
the real CK kernel (lib present). The dangling symlink is a real-but-separate
cosmetic bug; fixing it does **not** help, so we did **not** bake it in.

### Mitigation (shipped)

`slopfinity/compat.py` caps ERNIE to `ERNIE_MAX_DIM = 512`. `run_fleet.py` and
`slopfinity/worker_sh.py` pass `--width 512 --height 512`; `routers/config.py`
surfaces a UI warning when ERNIE is selected (toasted by `app.js`).

### Outstanding

- **1024² ERNIE is unresolved.** The only plausible lever is a *different*
  ROCm/MIOpen build where the gfx1151 CK grouped-conv kernel is fixed — e.g. a
  rebuild onto a newer nightly (`torch 2.10.0+rocm7.13.0a20260513`, the
  maintained release track). **Unproven and a low-odds gamble:** same `7.13.0a`
  ROCm line, gfx1151 kernel fixes aren't guaranteed in a 3-week-newer nightly,
  and it's a full container recompile that risks the currently-working stack.
  Only chase it if 1024² ERNIE is specifically needed; test *only* against this
  exact 1024² decode before trusting it.
- For higher-res images today, prefer **Qwen-Image** (stable to ~1664×928) or
  **LTX-2.3** rather than ERNIE.

---

## Other known-broken states (in `compat.py`)

See also the full [Compatibility Matrix](../COMPATIBILITY_MATRIX.md) for
supported resolutions / aspect ratios per image & video model.

| State | Severity | Note |
| --- | --- | --- |
| `base_model=ernie` | danger | GPU-hangs >512² (above); auto-capped to 512². |
| `video_model=wan2.2 / wan2.5` | warning | Unreliable on this hardware — ComfyUI timeouts. Use LTX-2.3. |

`tts_model=qwen-tts` was previously listed here as broken — it now **works** after
the `HSA_OVERRIDE_GFX_VERSION` 11.0.0→11.5.1 + device-sync fix (see the Qwen-TTS
section in `pipeline-validation-log.md`), so it was removed from the registry.

---

## Unrelated fixes made alongside

- **`paths.py` honors `SLOPFINITY_STATE_DIR`.** It previously hardcoded
  `/workspace` (root-owned on host) and crashed the dashboard on host-native
  startup. Now matches `config.py`/`db.py`, with a writability fallback.
- **Default LLM = `gemma4:26b` via ollama** (`DEFAULT_LLM_CONFIG`) — the same
  model the host's hermes agent uses, so the box shares one warm 26B. Cold
  enhance calls can exceed the 60s timeout and fall back to the raw prompt.

---

## ROCm stack note

The GPU stack is **not apt-managed** — it's TheRock nightly wheels
(`Dockerfile:42-45`, `pip --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/`).
Installed `torch 2.11.0` is one of only ~5 dev-HEAD nightlies ever published
(Apr 23–26, then pulled); the maintained tracks are `2.10.0` / `2.9.1`. A lower
version number here means *more* supported, not older.
