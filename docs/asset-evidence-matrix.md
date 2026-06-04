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
**Final tabulation — 71/71 permutations, one canonical file each.** Base dir: `evidence/`.

| Asset | Permutation | File | Length | Size |
| --- | --- | --- | --- | --- |
| image | qwen | `image/qwen.png` | — | 1492 KB |
| image | ernie | `image/ernie.png` | — | 441 KB |
| image | ltx-2.3 | `image/ltx-2.3.png` | — | 535 KB |
| video | ltx-2.3 | `video/ltx-2.3.mp4` | 2.0s | 631 KB |
| music | heartmula | `music/heartmula.wav` | 20.1s | 3765 KB |
| tts·kokoro | af_alloy | `tts/kokoro_af_alloy.wav` | 3.5s | 163 KB |
| tts·kokoro | af_aoede | `tts/kokoro_af_aoede.wav` | 2.8s | 133 KB |
| tts·kokoro | af_bella | `tts/kokoro_af_bella.wav` | 4.6s | 216 KB |
| tts·kokoro | af_heart | `tts/kokoro_af_heart.wav` | 4.5s | 209 KB |
| tts·kokoro | af_jessica | `tts/kokoro_af_jessica.wav` | 2.9s | 136 KB |
| tts·kokoro | af_kore | `tts/kokoro_af_kore.wav` | 3.0s | 142 KB |
| tts·kokoro | af_nicole | `tts/kokoro_af_nicole.wav` | 4.5s | 211 KB |
| tts·kokoro | af_nova | `tts/kokoro_af_nova.wav` | 3.3s | 156 KB |
| tts·kokoro | af_river | `tts/kokoro_af_river.wav` | 2.8s | 132 KB |
| tts·kokoro | af_sarah | `tts/kokoro_af_sarah.wav` | 2.9s | 138 KB |
| tts·kokoro | af_sky | `tts/kokoro_af_sky.wav` | 3.1s | 144 KB |
| tts·kokoro | am_adam | `tts/kokoro_am_adam.wav` | 3.0s | 139 KB |
| tts·kokoro | am_echo | `tts/kokoro_am_echo.wav` | 3.4s | 160 KB |
| tts·kokoro | am_eric | `tts/kokoro_am_eric.wav` | 2.9s | 136 KB |
| tts·kokoro | am_fenrir | `tts/kokoro_am_fenrir.wav` | 2.8s | 133 KB |
| tts·kokoro | am_liam | `tts/kokoro_am_liam.wav` | 2.8s | 132 KB |
| tts·kokoro | am_michael | `tts/kokoro_am_michael.wav` | 5.0s | 233 KB |
| tts·kokoro | am_onyx | `tts/kokoro_am_onyx.wav` | 3.6s | 168 KB |
| tts·kokoro | am_puck | `tts/kokoro_am_puck.wav` | 3.0s | 142 KB |
| tts·kokoro | am_santa | `tts/kokoro_am_santa.wav` | 13.3s | 622 KB |
| tts·kokoro | bf_alice | `tts/kokoro_bf_alice.wav` | 3.2s | 150 KB |
| tts·kokoro | bf_emma | `tts/kokoro_bf_emma.wav` | 4.1s | 194 KB |
| tts·kokoro | bf_isabella | `tts/kokoro_bf_isabella.wav` | 3.1s | 144 KB |
| tts·kokoro | bf_lily | `tts/kokoro_bf_lily.wav` | 3.2s | 148 KB |
| tts·kokoro | bm_daniel | `tts/kokoro_bm_daniel.wav` | 3.0s | 140 KB |
| tts·kokoro | bm_fable | `tts/kokoro_bm_fable.wav` | 3.6s | 169 KB |
| tts·kokoro | bm_george | `tts/kokoro_bm_george.wav` | 3.7s | 174 KB |
| tts·kokoro | bm_lewis | `tts/kokoro_bm_lewis.wav` | 3.5s | 162 KB |
| tts·kokoro | ef_dora | `tts/kokoro_ef_dora.wav` | 2.7s | 128 KB |
| tts·kokoro | em_alex | `tts/kokoro_em_alex.wav` | 2.8s | 129 KB |
| tts·kokoro | em_santa | `tts/kokoro_em_santa.wav` | 2.8s | 130 KB |
| tts·kokoro | ff_siwis | `tts/kokoro_ff_siwis.wav` | 3.0s | 140 KB |
| tts·kokoro | hf_alpha | `tts/kokoro_hf_alpha.wav` | 3.2s | 151 KB |
| tts·kokoro | hf_beta | `tts/kokoro_hf_beta.wav` | 2.9s | 136 KB |
| tts·kokoro | hm_omega | `tts/kokoro_hm_omega.wav` | 3.2s | 152 KB |
| tts·kokoro | hm_psi | `tts/kokoro_hm_psi.wav` | 3.2s | 152 KB |
| tts·kokoro | if_sara | `tts/kokoro_if_sara.wav` | 2.8s | 130 KB |
| tts·kokoro | im_nicola | `tts/kokoro_im_nicola.wav` | 3.1s | 144 KB |
| tts·kokoro | jf_alpha | `tts/kokoro_jf_alpha.wav` | 4.4s | 206 KB |
| tts·kokoro | jf_gongitsune | `tts/kokoro_jf_gongitsune.wav` | 3.9s | 185 KB |
| tts·kokoro | jf_nezumi | `tts/kokoro_jf_nezumi.wav` | 3.8s | 180 KB |
| tts·kokoro | jf_tebukuro | `tts/kokoro_jf_tebukuro.wav` | 3.7s | 174 KB |
| tts·kokoro | jm_kumo | `tts/kokoro_jm_kumo.wav` | 3.2s | 150 KB |
| tts·kokoro | pf_dora | `tts/kokoro_pf_dora.wav` | 2.8s | 129 KB |
| tts·kokoro | pm_alex | `tts/kokoro_pm_alex.wav` | 2.8s | 130 KB |
| tts·kokoro | pm_santa | `tts/kokoro_pm_santa.wav` | 2.9s | 137 KB |
| tts·kokoro | zf_xiaobei | `tts/kokoro_zf_xiaobei.wav` | 3.3s | 157 KB |
| tts·kokoro | zf_xiaoni | `tts/kokoro_zf_xiaoni.wav` | 2.9s | 134 KB |
| tts·kokoro | zf_xiaoxiao | `tts/kokoro_zf_xiaoxiao.wav` | 2.8s | 130 KB |
| tts·kokoro | zf_xiaoyi | `tts/kokoro_zf_xiaoyi.wav` | 2.7s | 127 KB |
| tts·kokoro | zm_yunjian | `tts/kokoro_zm_yunjian.wav` | 2.8s | 132 KB |
| tts·kokoro | zm_yunxi | `tts/kokoro_zm_yunxi.wav` | 2.9s | 134 KB |
| tts·kokoro | zm_yunxia | `tts/kokoro_zm_yunxia.wav` | 2.8s | 130 KB |
| tts·kokoro | zm_yunyang | `tts/kokoro_zm_yunyang.wav` | 2.8s | 129 KB |
| tts·qwen | aiden | `tts/qwen_aiden.wav` | 3.0s | 138 KB |
| tts·qwen | dylan | `tts/qwen_dylan.wav` | 5.0s | 232 KB |
| tts·qwen | eric | `tts/qwen_eric.wav` | 4.0s | 187 KB |
| tts·qwen | ono_anna | `tts/qwen_ono_anna.wav` | 2.6s | 120 KB |
| tts·qwen | ryan | `tts/qwen_ryan.wav` | 4.5s | 210 KB |
| tts·qwen | serena | `tts/qwen_serena.wav` | 5.3s | 247 KB |
| tts·qwen | sohee | `tts/qwen_sohee.wav` | 5.0s | 236 KB |
| tts·qwen | uncle_fu | `tts/qwen_uncle_fu.wav` | 6.6s | 311 KB |
| tts·qwen | vivian | `tts/qwen_vivian.wav` | 4.1s | 191 KB |
| tts·dramabox | kid | `tts/dramabox_kid.wav` | 8.3s | 1561 KB |
| tts·dramabox | narrator-female | `tts/dramabox_narrator-female.wav` | 9.7s | 1816 KB |
| tts·dramabox | narrator-male | `tts/dramabox_narrator-male.wav` | 7.0s | 1314 KB |
<!-- EVIDENCE_TABLE_END -->
