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

## 2026-06-03 — Music + TTS stage

| Stage | Engine / model | Verdict | Evidence |
| --- | --- | --- | --- |
| Music | **Heartmula** (3B) | ✅ | 20.1s 48kHz **stereo** WAV, 3.86 MB (gen ~4 min: 250-step gen + 10-step codec decode) |
| TTS | **Kokoro** (`af_heart` / `am_michael` / `am_puck`) | ✅ | 3 real WAVs ~200–240 KB, **~1–2s each** via `:8010/tts` |
| TTS | **DramaBox** | ⚠️ blocked | engine works (prior 2.4 MB outputs exist) but its disk guard trips: needs 4 GB on `/opt/dramabox-hf` (`/mnt/downloads`), only **3.5 GB free** |
| TTS | **Qwen-TTS** | ❌ broken | `ModuleNotFoundError: qwen_tts` (matches the known-broken status in `compat.py`) |

Invocation gotchas:
- TTS server `:8010` (`strix-halo-qwen-tts`) mounts the **main repo's**
  `comfy-outputs/experiments` as `/workspace`, so WAVs land there and the
  `/files/tts/` fetch 404s from a worktree — the files are real, just served
  from the main tree.
- Heartmula needs the model bind-mounted at `/mnt/downloads/comfy-models/HeartMuLa`
  (`-v …:ro`) **and** `HSA_OVERRIDE_GFX_VERSION=11.5.1` — without the mount it
  exits `FATAL: model root not found`.

## Pending

| Stage | Engine / model | Verdict |
| --- | --- | --- |
| Video | LTX-2.3 / WAN 2.x | ⬜ not validated this session (WAN flaky per `compat.py`) |
| TTS | DramaBox at 1024-equivalent | blocked on disk — free ≥0.6 GB on `/mnt/downloads` to retry |
