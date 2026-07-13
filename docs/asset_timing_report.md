# Widened asset timing report (re-run)

**Generated:** 2026-07-09 06:06:16 +0000
**Assets:** `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets`
**CSV:** `/tmp/grok-goal-baa73c3b43f0/implementer/widen/report.csv`
**Summary:** 18/35 succeeded

Widened matrix: 3 Kokoro voices × 3 text tiers; HeartMuLa × 3 durations; Ernie × 3 steps; Qwen image retry after OOM free; LTX/WAN entrypoints; merge × 3 clip lengths.

## Results

| Kind | Model | Tier | Duration (s) | Size (bytes) | Filename | OK | Notes |
|------|-------|------|-------------:|-------------:|----------|----|-------|
| tts | `kokoro:af_heart` | small | 41.037 | 42028 | `tts_af_heart_small_tts_kokoro_af_heart_1783571032469_443143.wav` | True | cold start |
| tts | `kokoro:af_heart` | medium | 3.787 | 203820 | `tts_af_heart_medium_tts_kokoro_af_heart_1783571073498_644767.wav` | True |  |
| tts | `kokoro:af_heart` | large | 5.368 | 527404 | `tts_af_heart_large_tts_kokoro_af_heart_1783571077286_a8282c.wav` | True |  |
| tts | `kokoro:am_michael` | small | 2.544 | 49196 | `tts_am_michael_small_tts_kokoro_am_michael_1783571082653_57aeb6.wav` | True |  |
| tts | `kokoro:am_michael` | medium | 3.608 | 220204 | `tts_am_michael_medium_tts_kokoro_am_michael_1783571085197_e6598b.wav` | True |  |
| tts | `kokoro:am_michael` | large | 5.989 | 570412 | `tts_am_michael_large_tts_kokoro_am_michael_1783571088806_0ab99c.wav` | True |  |
| tts | `kokoro:bf_emma` | small | 2.896 | 59436 | `tts_bf_emma_small_tts_kokoro_bf_emma_1783571094795_4fccd8.wav` | True |  |
| tts | `kokoro:bf_emma` | medium | 3.523 | 188460 | `tts_bf_emma_medium_tts_kokoro_bf_emma_1783571097691_6d9fff.wav` | True |  |
| tts | `kokoro:bf_emma` | large | 5.036 | 477228 | `tts_bf_emma_large_tts_kokoro_bf_emma_1783571101215_600f8e.wav` | True |  |
| tts | `qwen-tts` | small | 4.963 | — | `—` | False | launcher failed (qwen_tts package / HF) |
| tts | `qwen-tts` | medium | 3.891 | — | `—` | False | launcher failed (qwen_tts package / HF) |
| tts | `qwen-tts` | large | 3.746 | — | `—` | False | launcher failed (qwen_tts package / HF) |
| image | `ernie` | small | 569.533 | 1426240 | `image_ernie_small.png` | True | rc=0 |
| image | `ernie` | medium | 680.996 | 1805610 | `image_ernie_medium.png` | True | rc=0 |
| image | `ernie` | large | 770.629 | 1929941 | `image_ernie_large.png` | True | rc=0 |
| image | `ltx-2.3` | small | 4.021 | — | `—` | False | rc=2 missing /opt/ltx_launcher.py |
| image | `ltx-2.3` | medium | 0.753 | — | `—` | False | rc=2 missing /opt/ltx_launcher.py |
| image | `ltx-2.3` | large | 0.772 | — | `—` | False | rc=2 missing /opt/ltx_launcher.py |
| image | `qwen` | small | 901.717 | — | `—` | False | timeout job=3ff695ad-8c80-4340-a204-6fd6b61cd04b |
| image | `qwen` | medium | 0.0 | — | `—` | False | skipped after small timeout |
| image | `qwen` | large | 0.0 | — | `—` | False | skipped after small timeout |
| audio | `heartmula` | small | 231.868 | 583724 | `audio_heartmula_small_hm_1783572312_4200ff52.wav` | True |  |
| audio | `heartmula` | medium | 223.355 | 1167404 | `audio_heartmula_medium_hm_1783572544_71b509f2.wav` | True |  |
| audio | `heartmula` | large | 221.504 | 860204 | `audio_heartmula_large_hm_1783572767_dbbec0d6.wav` | True |  |
| video | `ltx-2.3` | small | 211.433 | — | `—` | False | rc=2 |
| video | `ltx-2.3` | large | 180.946 | — | `—` | False | rc=2 |
| video | `wan2.2` | small | 204.724 | — | `—` | False | rc=2 |
| video | `wan2.2` | large | 200.571 | — | `—` | False | rc=2 |
| video | `wan2.5` | small | 200.592 | — | `—` | False | rc=2 |
| video | `wan2.5` | large | 200.788 | — | `—` | False | rc=2 |
| upscale | `ltx-spatial` | small | 0.0 | — | `—` | False | no video seed |
| upscale | `ltx-spatial` | large | 0.0 | — | `—` | False | no video seed |
| merge | `ffmpeg_mux` | small | 0.382 | 7421 | `merge_out_small.mp4` | True |  |
| merge | `ffmpeg_mux` | medium | 0.281 | 17369 | `merge_out_medium.mp4` | True |  |
| merge | `ffmpeg_mux` | large | 0.363 | 41969 | `merge_out_large.mp4` | True |  |

## Full paths

- **tts/kokoro:af_heart/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_af_heart_small_tts_kokoro_af_heart_1783571032469_443143.wav` — **42028 bytes**, **41.037 s**
- **tts/kokoro:af_heart/medium**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_af_heart_medium_tts_kokoro_af_heart_1783571073498_644767.wav` — **203820 bytes**, **3.787 s**
- **tts/kokoro:af_heart/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_af_heart_large_tts_kokoro_af_heart_1783571077286_a8282c.wav` — **527404 bytes**, **5.368 s**
- **tts/kokoro:am_michael/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_am_michael_small_tts_kokoro_am_michael_1783571082653_57aeb6.wav` — **49196 bytes**, **2.544 s**
- **tts/kokoro:am_michael/medium**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_am_michael_medium_tts_kokoro_am_michael_1783571085197_e6598b.wav` — **220204 bytes**, **3.608 s**
- **tts/kokoro:am_michael/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_am_michael_large_tts_kokoro_am_michael_1783571088806_0ab99c.wav` — **570412 bytes**, **5.989 s**
- **tts/kokoro:bf_emma/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_bf_emma_small_tts_kokoro_bf_emma_1783571094795_4fccd8.wav` — **59436 bytes**, **2.896 s**
- **tts/kokoro:bf_emma/medium**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_bf_emma_medium_tts_kokoro_bf_emma_1783571097691_6d9fff.wav` — **188460 bytes**, **3.523 s**
- **tts/kokoro:bf_emma/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/tts_bf_emma_large_tts_kokoro_bf_emma_1783571101215_600f8e.wav` — **477228 bytes**, **5.036 s**
- **tts/qwen-tts/small**: **FAILED** — launcher failed (qwen_tts package / HF)
- **tts/qwen-tts/medium**: **FAILED** — launcher failed (qwen_tts package / HF)
- **tts/qwen-tts/large**: **FAILED** — launcher failed (qwen_tts package / HF)
- **image/ernie/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/image_ernie_small.png` — **1426240 bytes**, **569.533 s**
- **image/ernie/medium**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/image_ernie_medium.png` — **1805610 bytes**, **680.996 s**
- **image/ernie/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/image_ernie_large.png` — **1929941 bytes**, **770.629 s**
- **image/ltx-2.3/small**: **FAILED** — rc=2 missing /opt/ltx_launcher.py
- **image/ltx-2.3/medium**: **FAILED** — rc=2 missing /opt/ltx_launcher.py
- **image/ltx-2.3/large**: **FAILED** — rc=2 missing /opt/ltx_launcher.py
- **image/qwen/small**: **FAILED** — timeout job=3ff695ad-8c80-4340-a204-6fd6b61cd04b
- **image/qwen/medium**: **FAILED** — skipped after small timeout
- **image/qwen/large**: **FAILED** — skipped after small timeout
- **audio/heartmula/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/audio_heartmula_small_hm_1783572312_4200ff52.wav` — **583724 bytes**, **231.868 s**
- **audio/heartmula/medium**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/audio_heartmula_medium_hm_1783572544_71b509f2.wav` — **1167404 bytes**, **223.355 s**
- **audio/heartmula/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/audio_heartmula_large_hm_1783572767_dbbec0d6.wav` — **860204 bytes**, **221.504 s**
- **video/ltx-2.3/small**: **FAILED** — rc=2
- **video/ltx-2.3/large**: **FAILED** — rc=2
- **video/wan2.2/small**: **FAILED** — rc=2
- **video/wan2.2/large**: **FAILED** — rc=2
- **video/wan2.5/small**: **FAILED** — rc=2
- **video/wan2.5/large**: **FAILED** — rc=2
- **upscale/ltx-spatial/small**: **FAILED** — no video seed
- **upscale/ltx-spatial/large**: **FAILED** — no video seed
- **merge/ffmpeg_mux/small**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/merge_out_small.mp4` — **7421 bytes**, **0.382 s**
- **merge/ffmpeg_mux/medium**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/merge_out_medium.mp4` — **17369 bytes**, **0.281 s**
- **merge/ffmpeg_mux/large**: `/tmp/grok-goal-baa73c3b43f0/implementer/widen/assets/merge_out_large.mp4` — **41969 bytes**, **0.363 s**
