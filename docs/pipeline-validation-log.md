# Slop-pipeline validation log (gfx1151 / Strix Halo)

What each pipeline stage was *proven* to do on this hardware, with evidence.
Companion to [`gfx1151-known-issues.md`](./gfx1151-known-issues.md) (the hardware
quirks) and [`../COMPATIBILITY_MATRIX.md`](../COMPATIBILITY_MATRIX.md) (supported
resolutions). Verdicts: ✅ proven · ⚠️ works-with-caveat · ❌ disproven · ⬜ not yet validated.

## 2026-06-03 — Image stage + app end-to-end

| Stage | Engine / model | Verdict | Evidence |
| --- | --- | --- | --- |
| Image | **Qwen-Image** (`--fast`, 8-step) | ✅ | 1664×928 PNG |
| Image | **ERNIE-Image-Turbo** | ⚠️ ≤512² only | 512² PNG clean; **GPU-hangs at 1024²** (CK grouped-conv kernel) |
| Image | **LTX-2.3** (still) | ✅ | 768×512 via ComfyUI `frames=1` + audio-latent on; no native still launcher |
| App E2E | `/inject` → queue → `run_fleet` → qwen → gallery | ✅ | cyberpunk/cabin/astronaut rendered, served at `/assets` |
| LLM (concept rewrite) | `gemma4:26b` via ollama | ⚠️ | cold enhance exceeds 60s → falls back to raw prompt (still generates) |

Key proofs:
- **qwen `--out` is broken** in this launcher (writes to `~/.qwen-image-studio/`); the
  worker's `--out` is a no-op. HF cache needs `HF_HOME` + `HF_HUB_OFFLINE` or it stalls re-downloading.
- **ERNIE 1024² hang is a kernel bug, not a lib-path bug** — repairing the dangling
  `libMIOpenCKGroupedConv_gfx1151.so` symlink made the lib load but the GPU still hung. Capped to 512².
- **`paths.py` host-native startup fix** — it hardcoded a root-owned `/workspace`; now honors `SLOPFINITY_STATE_DIR`.

## Pending

| Stage | Engine / model | Verdict |
| --- | --- | --- |
| Music | **Heartmula** | ⬜ to validate |
| TTS | **Kokoro** (`af_heart`, `am_michael`, `am_puck`, …) | ⬜ to validate |
| TTS | **Qwen-TTS** | ⬜ (known-broken on gfx1151 per `compat.py`) |
| TTS | **DramaBox** (plain / expressive / narrator voices) | ⬜ to validate |
| Video | LTX-2.3 / WAN 2.x | ⬜ not validated this session |
