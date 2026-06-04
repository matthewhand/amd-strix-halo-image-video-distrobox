# Asset evidence matrix

**Goal:** keep **at least one** working example of **every configuration permutation**
for each asset type, at all times — as evidence the pipeline works — **without
duplicates** (one canonical file per permutation, not five copies of the same thing).

- **Files:** `evidence/{image,video,music,tts}/` (canonical name = the permutation).
- **Generator:** `scripts/build_evidence_tts.py` — **idempotent/dedup-aware**: it
  skips any permutation that already has a file and only generates what's missing.
- **Manifest:** `evidence/tts/manifest.json` (machine-readable per-voice status).
- **Slop view:** the dashboard loads `evidence/` into the gallery on startup **if it
  exists** (see `slopfinity/routers/assets.py`, `SLOPFINITY_EVIDENCE_DIR`).

Total permutations: **71** = 3 image + 1 video + 1 music + 66 TTS.

---

## Image (3) — `evidence/image/`

| base_model | File | Notes |
| --- | --- | --- |
| qwen | `qwen.png` | Qwen-Image, ~1664×928 |
| ernie | `ernie.png` | ERNIE-Image-Turbo, **512²** (hangs above — see gfx1151-known-issues) |
| ltx-2.3 | `ltx-2.3.png` | LTX-2.3 still (ComfyUI frames=1 + audio) |

## Video (1) — `evidence/video/`

| video_model | File | Notes |
| --- | --- | --- |
| ltx-2.3 | `ltx-2.3.mp4` | T2V, 768×512 + audio track. (WAN 2.x removed — flaky on gfx1151) |

## Music (1) — `evidence/music/`

| audio_model | File | Notes |
| --- | --- | --- |
| heartmula | `heartmula.wav` | HeartMuLa 3B, 48 kHz stereo |

## TTS (66) — `evidence/tts/`  → `{engine}_{voice}.wav`

### Kokoro (54) — by language (a=US, b=British, e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Chinese; f/m = gender)
- **US female (11):** af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
- **US male (9):** am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa
- **British (8):** bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
- **Spanish (3):** ef_dora, em_alex, em_santa
- **French (1):** ff_siwis
- **Hindi (4):** hf_alpha, hf_beta, hm_omega, hm_psi
- **Italian (2):** if_sara, im_nicola
- **Japanese (5):** jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
- **Portuguese (3):** pf_dora, pm_alex, pm_santa
- **Chinese (8):** zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi, zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang

### Qwen-TTS (9) — premium timbres
aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian

### DramaBox (3) — cloned/dramatic
kid, narrator-female, narrator-male

---

## Live status

The per-asset checklist with length/size is generated and updated as the build runs.
See the **final tabulation** appended below once `build_evidence_tts.py` completes,
and `evidence/tts/manifest.json` for the authoritative state.

<!-- EVIDENCE_TABLE_START -->
_(populated on build completion)_
<!-- EVIDENCE_TABLE_END -->
