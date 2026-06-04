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
| TTS | **DramaBox** | ✅ (after disk reclaim) | 1.67 MB WAV, **215s cold** (Gemma load) — was disk-blocked until ~153 GB freed (see below) |
| TTS | **Qwen-TTS** | 🚫 won't-fix | needs the `qwen-tts` pip package (absent — only Alibaba's cloud `dashscope` SDK present) **and** the Qwen3-TTS model (not downloaded); uncertain on gfx1151 and redundant with Kokoro/DramaBox |

## 2026-06-03 — disk reclaim (unblocked DramaBox)

Audit of `/mnt/downloads` (848 G, was 3.5 G free). **Nothing is hardlinked** —
every LTX/WAN weight is a distinct `links=1` copy; the redundancy is *version*
redundancy, not duplicates. Removed (user-approved) old/dev LTX + comfy-models
WAN ⇒ **3.5 G → 157 G free** (~153 G reclaimed). Kept the active
`ltx-2.3-22b-distilled-fp8` stack. **Still pending:** `/mnt/downloads/wan-models/`
(~40 G) is root-owned — needs `sudo rm -rf` by the operator.

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
| Video | LTX-2.3 | ⬜ not validated this session (WAN-2.x removed during reclaim — flaky on gfx1151) |
