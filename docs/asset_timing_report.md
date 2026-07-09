# Asset timing report — all STAGE_BUDGETS models

**Generated:** 2026-07-09 01:57:01 +0000
**Assets:** `/tmp/grok-goal-baa73c3b43f0/implementer/assets`
**CSV:** `/tmp/grok-goal-baa73c3b43f0/implementer/asset_timing_report_all_models.csv`
**Inventory:** every `(stage, model)` in `slopfinity/scheduler.py` `STAGE_BUDGETS`, plus merge/`ffmpeg_mux`

Each **model** has **small** and **large** rows (duration, path, bytes). Failures are honest (no invented sizes).

## Model inventory measured

| Stage | Models |
|-------|--------|
| tts | kokoro, qwen-tts |
| image | qwen, ernie, ltx-2.3 |
| audio | heartmula |
| video | ltx-2.3, wan2.2, wan2.5 |
| upscale | ltx-spatial |
| merge | ffmpeg_mux |

## Results

| Kind | Model | Tier | Duration (s) | Size (bytes) | Filename | OK | Notes |
|------|-------|------|-------------:|-------------:|----------|----|-------|
| tts | `kokoro` | small | 8.843 | 61484 | `tts_kokoro_small_tts_kokoro_af_heart_1783557748353_75a437.wav` | True | url=/files/tts/tts_kokoro_af_heart_1783557748353_75a437 |
| tts | `kokoro` | large | 12.503 | 330796 | `tts_kokoro_large_tts_kokoro_af_heart_1783557757313_984f37.wav` | True | url=/files/tts/tts_kokoro_af_heart_1783557757313_984f37 |
| tts | `qwen-tts` | small | 0.239 | — | `—` | False | ModuleNotFoundError: No module named 'qwen_tts' in TTS  |
| tts | `qwen-tts` | large | 0.102 | — | `—` | False | ModuleNotFoundError: No module named 'qwen_tts' in TTS  |
| image | `ernie` | small | 568.122 | 1038343 | `image_ernie_small.png` | True | rc=0 |
| image | `ernie` | large | 773.401 | 1769128 | `image_ernie_large.png` | True | rc=0 |
| image | `ltx-2.3` | small | 34.379 | — | `—` | False | rc=2 host_exists=False host=/home/matthewh/amd-strix-ha |
| image | `ltx-2.3` | large | 38.977 | — | `—` | False | rc=2 host_exists=False host=/home/matthewh/amd-strix-ha |
| image | `qwen` | small | 362.665 | — | `—` | False | timeout job=d5d0f66c-0e8b-40a5-ab0b-955258806e7c |
| image | `qwen` | large | 484.639 | — | `—` | False | timeout job=f3968d64-b0c2-4f82-aa99-2061bbc4c71d |
| audio | `heartmula` | small | 233.671 | 783404 | `audio_heartmula_small_hm_1783558622_ecafa955.wav` | True | url=/files/music/hm_1783558622_ecafa955.wav |
| audio | `heartmula` | large | 227.959 | 1935404 | `audio_heartmula_large_hm_1783558855_90ac6751.wav` | True | url=/files/music/hm_1783558855_90ac6751.wav |
| video | `ltx-2.3` | small | 0.219 | — | `—` | False | Comfy/host gen failed; comfy_up=False; rc=2 python3: ca |
| video | `ltx-2.3` | large | 0.22 | — | `—` | False | Comfy/host gen failed; comfy_up=False; rc=2 python3: ca |
| video | `wan2.2` | small | 21.332 | — | `—` | False | Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FR |
| video | `wan2.2` | large | 20.272 | — | `—` | False | Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FR |
| video | `wan2.5` | small | 20.146 | — | `—` | False | Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FR |
| video | `wan2.5` | large | 20.42 | — | `—` | False | Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FR |
| upscale | `ltx-spatial` | small | 0.0 | — | `—` | False | no completed video asset to upscale; STAGE_BUDGETS list |
| upscale | `ltx-spatial` | large | 0.0 | — | `—` | False | no completed video asset to upscale; STAGE_BUDGETS list |
| merge | `ffmpeg_mux` | small | 0.286 | 7411 | `merge_out_small.mp4` | True |  |
| merge | `ffmpeg_mux` | large | 0.143 | 32095 | `merge_out_large.mp4` | True |  |

## Full paths

- **tts/kokoro/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/tts_kokoro_small_tts_kokoro_af_heart_1783557748353_75a437.wav` — **61484 bytes**, **8.843 s**
- **tts/kokoro/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/tts_kokoro_large_tts_kokoro_af_heart_1783557757313_984f37.wav` — **330796 bytes**, **12.503 s**
- **tts/qwen-tts/small**: **FAILED** — ModuleNotFoundError: No module named 'qwen_tts' in TTS container (HF path fixed; package not installed)
- **tts/qwen-tts/large**: **FAILED** — ModuleNotFoundError: No module named 'qwen_tts' in TTS container (HF path fixed; package not installed)
- **image/ernie/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/image_ernie_small.png` — **1038343 bytes**, **568.122 s**
- **image/ernie/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/image_ernie_large.png` — **1769128 bytes**, **773.401 s**
- **image/ltx-2.3/small**: **FAILED** — rc=2 host_exists=False host=/home/matthewh/amd-strix-halo-image-video-toolboxes/comfy-outputs/experiments/ltx_timing_small.png
- **image/ltx-2.3/large**: **FAILED** — rc=2 host_exists=False host=/home/matthewh/amd-strix-halo-image-video-toolboxes/comfy-outputs/experiments/ltx_timing_large.png
- **image/qwen/small**: **FAILED** — timeout job=d5d0f66c-0e8b-40a5-ab0b-955258806e7c
- **image/qwen/large**: **FAILED** — timeout job=f3968d64-b0c2-4f82-aa99-2061bbc4c71d
- **audio/heartmula/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/audio_heartmula_small_hm_1783558622_ecafa955.wav` — **783404 bytes**, **233.671 s**
- **audio/heartmula/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/audio_heartmula_large_hm_1783558855_90ac6751.wav` — **1935404 bytes**, **227.959 s**
- **video/ltx-2.3/small**: **FAILED** — Comfy/host gen failed; comfy_up=False; rc=2 python3: can't open file '/opt/ltx_launcher.py': [Errno 2] No such file or directory

- **video/ltx-2.3/large**: **FAILED** — Comfy/host gen failed; comfy_up=False; rc=2 python3: can't open file '/opt/ltx_launcher.py': [Errno 2] No such file or directory

- **video/wan2.2/small**: **FAILED** — Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FRAMES]
                       [--vae_tiling] [--vae_tile_px VAE_TILE_PX]
wan_launcher.py: error: unrecognized arguments: --out /out/video_wan2_2_small.mp4 --model wan2.2

- **video/wan2.2/large**: **FAILED** — Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FRAMES]
                       [--vae_tiling] [--vae_tile_px VAE_TILE_PX]
wan_launcher.py: error: unrecognized arguments: --out /out/video_wan2_2_large.mp4 --model wan2.2

- **video/wan2.5/small**: **FAILED** — Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FRAMES]
                       [--vae_tiling] [--vae_tile_px VAE_TILE_PX]
wan_launcher.py: error: unrecognized arguments: --out /out/video_wan2_5_small.mp4 --model wan2.5

- **video/wan2.5/large**: **FAILED** — Comfy/host gen failed; comfy_up=False; rc=2 es INFER_FRAMES]
                       [--vae_tiling] [--vae_tile_px VAE_TILE_PX]
wan_launcher.py: error: unrecognized arguments: --out /out/video_wan2_5_large.mp4 --model wan2.5

- **upscale/ltx-spatial/small**: **FAILED** — no completed video asset to upscale; STAGE_BUDGETS lists ltx-spatial
- **upscale/ltx-spatial/large**: **FAILED** — no completed video asset to upscale; STAGE_BUDGETS lists ltx-spatial
- **merge/ffmpeg_mux/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/merge_out_small.mp4` — **7411 bytes**, **0.286 s**
- **merge/ffmpeg_mux/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/assets/merge_out_large.mp4` — **32095 bytes**, **0.143 s**
