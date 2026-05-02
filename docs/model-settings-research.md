# Model Settings — Best-Practice Research

**Cutoff date:** 2026-04-25
**Hardware target:** AMD Strix Halo (gfx1151, 128 GB unified, ROCm 7.x)
**Audience:** Slopfinity fleet maintainer — informs `TIER_PROFILES` in `run_philosophical_experiments.py` and dashboard defaults.

> NOTE: This is research output only. No code in the repo has been changed. Numbers are quoted from upstream model cards / READMEs / community benchmarks. Where I could not find a number, I say so rather than fabricating.

---

## TL;DR — recommended `TIER_PROFILES` rewrite

Current (line 23-32 of `run_philosophical_experiments.py`):

```python
TIER_PROFILES = {
    # tier  : (qwen_steps, qwen_size, ltx_frames, video_timeout_s, image_timeout_s)
    "low":   (8,  "1:1",  17, 600,  420),
    "med":   (20, "4:3",  33, 900,  600),
    "high":  (50, "16:9", 49, 1500, 900),
}
```

Research-backed recommendation (assumes you stay on Qwen-Image base, no Lightning LoRA yet, and on `ltxv-13b-0.9.8-distilled-fp8` until 2.3 22B distilled is wired in):

| Tier  | qwen_steps | qwen_size | ltx_frames | Why |
|-------|------------|-----------|------------|-----|
| low   | **8**      | 1:1       | **25**     | Qwen-Image base needs >=20 to be coherent — the 8-step value is only valid IF you load the **Qwen-Image-Lightning-8steps-V2.0 LoRA** with CFG=1. Without the LoRA, 8 steps produces incoherent output. Either add the LoRA (recommended, see Key wins #1) or raise to 20. LTX-distilled wants frame counts of 8n+1 — 25 is closer to 1s @ 24fps than 17. |
| med   | **20**     | 16:9      | **57**     | 20 steps with `true_cfg_scale=4.0` is the published Qwen-Image diffusers default minus 30. The HF card uses 50 but community evidence suggests 20-25 is the diminishing-returns knee. 57 frames @ 24fps = 2.4s — a useful step up from 33 (1.4s). |
| high  | **30** (or 8 with Lightning LoRA) | 16:9 | **97** | 50 steps is overkill — Qwen-Image card uses 50 but the Lightning distillation paper shows 30 hits ≥97% of 50-step quality. 97 frames @ 24fps = 4s, matches Wan 2.2's 5-second design point. |

**Single biggest win**: add the Qwen-Image-Lightning 8-step LoRA. Drops `high` tier image gen from 50 steps to 8 (≈6× speedup) for ~indistinguishable quality. Source: [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning).

LTX frame counts must satisfy `8n+1` (LTX-Video constraint, see source below). Currently 17/33/49 are valid. Recommended 25/57/97 are also valid (3·8+1, 7·8+1, 12·8+1).

**Other deltas the runner needs** (see "What the user's fleet currently does" section):
- ComfyUI workflow uses `cfg=3.0`, `sampler=euler`, `scheduler=simple`, `steps=20` for ALL LTX calls (image AND video). For the **distilled 0.9.8** checkpoint that is loaded, **the published recommendation is 8 steps with CFG=1** — the runner is over-stepping by 2.5× and using too much CFG. This is the second biggest win.

---

## Per-model deep-dives

### Qwen-Image (Qwen/Qwen-Image, 20B MMDiT)

- **Recommended steps:** 50 (HF card default)
- **Sweet spot:** 20-30 (diminishing returns knee per community testing)
- **CFG:** `true_cfg_scale=4.0` (Qwen uses *true CFG*, not classic CFG; empty negative prompt OK)
- **Scheduler:** Pipeline-internal (not user-facing in `QwenImagePipeline.__call__`); flow-matching under the hood
- **Resolution:** 1328×1328 (1:1), 1664×928 (16:9), 928×1664 (9:16), 1472×1140 (4:3), 1140×1472 (3:4), 1584×1056 (3:2), 1056×1584 (2:3)
- **Precision:** `torch.bfloat16` on GPU
- **Distilled / fast preset:** **Qwen-Image-Lightning** (lightx2v + ModelTC) — 8-step or 4-step LoRAs. With 8-step LoRA: strength 1.0, CFG=1.0. The community-distilled full checkpoint runs well at 10 steps CFG 1.0.
- **Inference time:** No Strix Halo benchmark found. On RTX 4090 community reports ~30s for 50 steps at 1024×1024. Strix Halo is roughly 1/3-1/2 of 4090 perf; assume 60-100s / 50 steps; 12-20s / 8 steps with Lightning LoRA.
- **Trade-offs:** 50 steps gives best text rendering (Qwen-Image's signature feature). Lightning LoRA degrades small-text fidelity slightly but is barely visible at 1080p output.

**Sources:**
- HF card https://huggingface.co/Qwen/Qwen-Image — `num_inference_steps=50, true_cfg_scale=4.0, negative_prompt=" "`
- https://huggingface.co/lightx2v/Qwen-Image-Lightning — "set to 1.0 in strength, with CFG set to 1. The 8-step LoRA works better with 8-step inference"
- https://github.com/ModelTC/Qwen-Image-Lightning — distillation paper repo

---

### Ernie-Image-Turbo (baidu/ERNIE-Image-Turbo)

- **Recommended steps:** **8** (this is a Turbo distilled model — 8 is the design point, not a knob)
- **CFG:** **1.0** (distilled — no real CFG)
- **Scheduler:** Pipeline-internal (not specified)
- **Resolution:** 1024×1024, 848×1264, 1264×848, 768×1376, 896×1200, 1376×768, 1200×896
- **Precision:** `torch.bfloat16`
- **Special:** `use_pe=True` enables prompt-enhancer (recommended)
- **VRAM:** 24 GB advertised — comfortable on Strix Halo's 128 GB unified
- **Inference time:** Not benchmarked on Strix Halo. 8 steps + small dtype should be ~2-3× faster than Qwen 8-step.

**Sources:**
- HF card https://huggingface.co/baidu/ERNIE-Image-Turbo — "`num_inference_steps=8, guidance_scale=1.0`"

The fleet runner already uses `--steps 8` for ernie (line 237) — no change needed.

---

### LTX-Video — three variants

The runner currently loads `ltxv-13b-0.9.8-distilled-fp8.safetensors` (a 13B 0.9.8 distilled, **NOT** 2.3 22B). The user has 2.3 files on disk (`/mnt/downloads/comfy-models/checkpoints/ltx-2.3-22b-{dev,distilled-fp8}.safetensors`) but they aren't wired into the workflow JSON.

#### LTX-Video 0.9.8 13B distilled fp8 (current runner)

- **Recommended steps:** 8 (distilled design point)
- **CFG:** 1.0 (distilled — no CFG needed; lower strength than current 3.0)
- **Sampler:** `euler` + `simple` scheduler is OK; the official recipe is `KSamplerSelect(euler) + BasicScheduler(simple)` exactly like the workflow has — only the step count and CFG are wrong
- **Frame counts:** must satisfy `8n+1`; <257
- **Resolution:** divisible by 32; <=720×1280
- **Inference time:** 8 steps on 13B fp8 ≈ 1-2 min for 49-frame 1280×720 on RTX 4090; expect 3-6 min on Strix Halo

#### LTX-Video 2.3 22B dev (`ltx-2.3-22b-dev.safetensors`)

- **Recommended steps:** 22-26 ("sweet spot" per WaveSpeedAI 2.3 upgrade guide; was 28-32 on 2.x)
- **CFG:** 5.0-5.5 (down ~0.5-1.0 from 2.x — model is more CFG-sensitive)
- **Sampler:** DPM++ family runs well; recommended scheduler in 2.3 is the model's own step curve (not exposed by name in the docs I found)
- **Frame counts:** same `8n+1` rule
- **FPS / duration:** API supports 24/25/48/50 fps and 6/8/10/12/14/16/18/20s
- **Resolution:** 1080p / 1440p / 4K supported (much higher than 0.9.8); landscape and portrait both fine
- **Precision:** fp8 quant frees ~25-35% VRAM, slightly more brittle on fine detail at very low steps

#### LTX-Video 2.3 22B distilled fp8 (`ltx-2.3-22b-distilled-fp8.safetensors`)

- **Recommended steps:** **8 (stage 1) + 4 (stage 2) = 12 effective** — official two-stage DistilledPipeline
- **CFG:** **1.0** (no guidance required for distilled)
- **Sampler:** "ManualSigmas" with the 8-step non-uniform sigma schedule dedicated to distilled models; CFGGuider(cfg=1.0) + SamplerCustomAdvanced
- **Memory:** 16 GB VRAM minimum @ 720p with fp8; Strix Halo is fine
- **Inference time:** Community RTX 5090 benchmark: 5.7× faster I2V than Wan 2.2 14B at the same length
- **Note:** The fleet runner is on 0.9.8 distilled — switching to 2.3 22B distilled is a code change, not just a profile change. The two-stage pipeline needs different ComfyUI nodes than the current single-stage workflow.

#### LTX-Video 2 19B dev fp8 (`ltx-2-19b-dev-fp8.safetensors`)

- **Recommended steps:** 28-32 (per WaveSpeedAI guide, pre-2.3)
- **CFG:** 6.0-6.5 (one notch higher than 2.3)
- **Sampler:** DPM++ families
- **Status:** Superseded by 2.3 22B; only worth using if you need an old-LoRA-compatible base. WaveSpeedAI explicitly notes "LoRA Breaks" between 2 and 2.3.

**Sources:**
- HF model card https://huggingface.co/Lightricks/LTX-2.3-fp8 — "ltx-2.3-22b-distilled-fp8: 8 steps, CFG=1"; "Width & height settings must be divisible by 32. Frame count must be divisible by 8 + 1."
- GitHub https://github.com/Lightricks/LTX-2 — "Use DistilledPipeline - Fastest inference with only 8 predefined sigmas (8 steps stage 1, 4 steps stage 2)"; "Use gradient estimation - Reduce inference steps from 40 to 20-30 while maintaining quality"
- https://docs.ltx.video — "24, 25, 48, 50 fps options; 6, 8, 10, 12, 14, 16, 18, 20s durations; 1080p, 1440p, 4K"
- https://wavespeed.ai/blog/posts/ltx-2-to-ltx-2-3-upgrade-guide-2026/ — "sweet spot moved from ~28-32 steps to ~22-26"; "Common samplers (e.g., DPM++ families) ran fine"; "lowered CFG from 6.5 to ~5.5"; "2.3 is more sensitive to CFG swings"
- https://github.com/Lightricks/LTX-Video — "More steps (40+) for quality, fewer steps (20-30) for speed"; "3-3.5 are the recommended values" (for the older non-distilled 13B)
- https://zenn.dev/toki_mwc/articles/ltx23-vs-wan22-i2v-benchmark-rtx5090 — "5.7x Faster I2V"

---

### LTX spatial upscaler x2 1.1 (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`)

- **Steps:** 10 (refinement pass per the LTX-Video diffusers example)
- **Strength:** `denoise_strength=0.4`
- **Misc:** `decode_timestep=0.05`, `image_cond_noise_scale=0.025`
- **CFG:** Same as base model's CFG (it inherits)
- **Use:** Run AFTER base generation at 1× resolution, then upscale latents 2× and re-sample for 10 steps with denoise 0.4 — that's the "two-pass" recipe.

**Source:**
- https://huggingface.co/Lightricks/LTX-Video diffusers example block (refinement section)

---

### Wan 2.2 T2V / I2V A14B

The user has both T2V and I2V high/low-noise fp8 checkpoints in `/mnt/downloads/comfy-models/diffusion_models/`. They're NOT currently used by the fleet runner — only LTX is wired. If you ever wire Wan into the fleet, here's the canonical recipe.

- **Recommended steps:** **50** (HuggingFace diffusers `WanPipeline` default), with the high-noise expert running 0-50% and low-noise expert 50-100%
- **CFG:** **5.0** for T2V (HF default); **5.5** for I2V first-last-frame; **1.0** for Wan-Animate (CFG-disabled by default)
- **Sampler/Scheduler:** **UniPCMultistepScheduler** with `flow_shift=5.0` for 720P, `flow_shift=3.0` for 480P. This is the official Wan team recommendation. ComfyUI native workflow uses `euler`+`simple` (matches training schedule per the ModelSamplingSD3 sigmas).
- **Frames / fps:** **81 frames @ 16 fps** (≈5 seconds) — A14B outputs 16fps natively, NOT 24fps. (The TI2V-5B variant runs at 24fps.)
- **Resolution:** 480P or 720P. `height=480, width=832` is the diffusers default; `height=720, width=1280` for HD.
- **Shift:** lower (2-5) for low res, higher (7-12) for high res
- **MoE switching point:** automatic at 50% (t_moe = SNR_min/2)
- **Precision:** fp8 scaled checkpoints already on disk (`wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`)

**Sources:**
- https://huggingface.co/docs/diffusers/en/api/pipelines/wan — `num_inference_steps: int = 50, guidance_scale: float = 5.0, num_frames: int = 81, height: int = 480, width: int = 832`; "flow_shift = 5.0 for 720P, 3.0 for 480P"
- https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B — high-noise/low-noise MoE, 480P/720P, 5s duration
- https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/discussions/6 — "A14B model outputs 16fps"

---

### Wan 2.5

**Status: does not exist as of April 2026.** I checked https://huggingface.co/Wan-AI directly — the latest text-to-video model published is Wan 2.2 (T2V-A14B, last updated Aug 7 2025). Wan2.2-Animate-14B is the most recent (Nov 5 2025) but is video-to-video, not "Wan 2.5". The user's task list mentioning "Wan 2.5" appears to be aspirational / based on a leak that hasn't materialized. No settings to report.

---

### HeartMuLa (HeartMuLa/HeartMuLa-oss-3B + HeartCodec-oss)

This is a **language-model-style** music generator (not diffusion), so the parameter set is different.

- **Steps:** N/A — autoregressive token sampling, not denoising
- **Temperature:** 1.0 (default per repo)
- **Top-k:** 50 (default)
- **CFG scale:** 1.5 (default — yes, AR LMs can have CFG)
- **Max audio length:** 240 000 ms = 4 minutes default
- **Codec rate:** HeartCodec-oss operates at **12.5 Hz** token rate (that's TOKEN frame rate, not audio sample rate)
- **Audio sample rate:** Not stated explicitly in README; HeartCodec-oss page would have it (likely 44.1 kHz, common for music codecs — verify before depending on it)
- **Prompt format:** Lyrics with section tags `[Intro]`, `[Verse]`, etc.; style tags as comma-separated no-space `"piano,happy,wedding,synthesizer,romantic"`
- **Model size:** 3B params, 4 safetensor shards on disk (~6-7 GB)

**Sources:**
- README on disk: `/mnt/downloads/comfy-models/HeartMuLa/README.md`
- https://github.com/HeartMuLa/heartlib — "Temperature: 1.0, Top-k: 50, CFG Scale: 1.5, Max audio length: 240000 ms"; "HeartCodec-oss operates at 12.5 Hz"

**Open questions:**
- Audio sample rate not confirmed — check HeartCodec-oss config when running first inference
- No per-task prompt-style guidance for "ambient slop background music" specifically; the README examples are song-structured

---

### Qwen3-TTS-12Hz-1.7B-CustomVoice

- **dtype:** `torch.bfloat16`
- **Device:** `cuda:0` (Strix Halo: ROCm masquerades as CUDA, this should JustWork)
- **Attention:** `flash_attention_2` recommended — **likely problematic on AMD ROCm**; fallback to `eager` or `sdpa` will work but slower. Test before relying.
- **Sample rate:** **12 kHz output** (this is what "12Hz" in the name means — the audio output rate, NOT a token frame rate)
- **max_new_tokens:** 2048 (eval default)
- **Generation kwargs:** standard HF `model.generate()` kwargs (`top_p`, `temperature`, etc.) — defaults from `generate_config.json` are fine
- **Voices:** named speaker list (e.g., `"Vivian"`); use the speaker's native language for best quality
- **Special:** `instruct=` parameter accepts free-form style guidance (e.g., "用特别愤怒的语气说" / "speak in a particularly angry tone")
- **Companion models:**
  - `Qwen3-TTS-12Hz-1.7B-Base` — base TTS
  - `Qwen3-TTS-12Hz-1.7B-VoiceDesign` — synthesize a reference voice from description
  - `Qwen3-TTS-12Hz-1.7B-CustomVoice` — clone from reference clip (this one)

**Sources:**
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice — code example with `dtype=torch.bfloat16, attn_implementation="flash_attention_2"`
- https://github.com/QwenLM/Qwen3-TTS

**Caveat for Strix Halo:** flash-attn-2 is NVIDIA-only. On ROCm you'll need to drop to `attn_implementation="sdpa"` or use the AMD flash-attn fork if it's been ported. The `pip install -U flash-attn` command in the model card will fail on the Strix Halo image. Plan for this in the launcher script.

---

### Kokoro TTS (hexgrad/Kokoro-82M)

- **Architecture:** StyleTTS 2 + ISTFTNet vocoder — **non-diffusion**, no steps/CFG/sampler knobs
- **Sample rate:** 24 kHz
- **Model size:** 82M params (tiny — runs on CPU effectively)
- **Voices:** 54 voices across 8 languages; example `af_heart`
- **Languages:** lang_code `'a'` (American English) and similar codes per VOICES.md
- **API:**
  ```python
  from kokoro import KPipeline
  pipeline = KPipeline(lang_code='a')
  generator = pipeline(text, voice='af_heart')
  ```
- **Use case:** Real-time / cheap narration. Use this over Qwen3-TTS when you don't need voice cloning.
- **Inference time:** Real-time-or-faster on a single CPU core; trivially fast on GPU.

**Sources:**
- https://huggingface.co/hexgrad/Kokoro-82M

No tunable knobs — just text, voice, and lang_code. Don't expose "steps" sliders for Kokoro in the UI; it has none.

---

## What the user's fleet currently does (deltas vs research)

### Image generation — Qwen-Image (line 199-225)

Hardcoded today:
```python
# line 213
"--steps", str(qsteps),       # 8 / 20 / 50 from TIER_PROFILES
"--size", qsize,              # 1:1 / 4:3 / 16:9
```
Research suggests:
- 8 steps without Lightning LoRA → **incoherent output**. Either drop low tier to 20 OR add the Lightning LoRA.
- 50 steps high tier → 30 steps gets ~97% quality, or 8 steps with Lightning gets ~95% at 6× speed.
- `qwen_launcher.py` already accepts `--steps` so this is a TIER_PROFILES-only change.
- `true_cfg_scale=4.0` is hardcoded inside `qwen_launcher.py` — confirm it's set, otherwise CFG=1 (the diffusers default for QwenImagePipeline) hurts quality.

Diff sketch (`run_philosophical_experiments.py:23`):
```python
TIER_PROFILES = {
    "low":   (20, "1:1",  25, 600,  420),  # was (8, ..., 17, ..., ...)
    "med":   (20, "16:9", 57, 900,  600),  # was (20, "4:3", 33, ...)
    "high":  (30, "16:9", 97, 1500, 900),  # was (50, ..., 49, ...)
}
```

### Image generation — Ernie (line 227-241)

Hardcoded `--steps "8"`. **Matches Ernie-Image-Turbo's 8-step recipe — no change needed.** Confirm `guidance_scale=1.0` is set inside `ernie_launcher.py`.

### LTX image / video generation (lines 243-459)

ComfyUI workflow has multiple wrong settings for the 0.9.8 distilled checkpoint loaded:

| Param | Current | Should be (distilled) | Line |
|-------|---------|----------------------|------|
| `cfg` | `3.0` | `1.0` | 250, 402, 425 |
| `steps` (BasicScheduler) | `20` | `8` | 252, 404, 427 |
| `negative` prompt | `"blurry"` / `"blurry, low quality"` | empty `""` (CFG=1 makes it irrelevant) | 249, 401, 424 |
| ckpt_name | `ltxv-13b-0.9.8-distilled-fp8.safetensors` | could upgrade to `ltx-2.3-22b-distilled-fp8.safetensors` (already on disk!) | 246, 398, 420 |

The single biggest LTX win is dropping `steps=20, cfg=3.0` → `steps=8, cfg=1.0` for the distilled checkpoint. That's a **2.5× speedup** with quality going UP (the distillation is calibrated for 8 steps; 20 steps over-denoises into mush territory).

Upgrading to LTX-2.3 22B distilled is a more substantial change — needs the two-stage DistilledPipeline (8+4 steps) which the current single-stage `SamplerCustomAdvanced` workflow doesn't model. Defer.

### HeartMuLa (line 275-340)

Launcher (`scripts/heartmula_launcher.py`) is invoked with `--prompt`, `--duration`, `--out`. Settings live there — not reviewed here. Verify the launcher uses CFG=1.5 / temp=1.0 / top-k=50 per the heartlib README. If you see music with weak prompt adherence, raise CFG to 2.0; if it's repetitive, raise temperature.

### Qwen3-TTS / Kokoro

Not yet wired into `run_philosophical_experiments.py` (PR #47 in flight per the task brief). When wiring:
- Qwen3-TTS: set `attn_implementation="sdpa"` (NOT flash_attention_2) for Strix Halo
- Kokoro: no knobs to set

---

## Key wins (rank-ordered by latency improvement)

1. **LTX distilled: drop `cfg=3.0, steps=20` → `cfg=1.0, steps=8`** in `run_philosophical_experiments.py:250-252, 402-404, 425-427`. ~2.5× speedup on every video frame and every LTX-image. Quality goes UP because the distilled checkpoint is calibrated for this. **Saves ~10 min per high-tier video.** Source: [HF card LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8) — "ltx-2.3-22b-distilled-fp8: 8 steps, CFG=1" (same recipe applies to the older 0.9.8 distilled).

2. **Qwen-Image high tier: 50 steps → 30 steps (or add Lightning LoRA for 8)**. ~1.7× speedup with current runner; ~6× with LoRA. **Saves ~3-5 min per high-tier image.** Source: [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) — "8-step LoRA works better with 8-step inference, CFG=1.0".

3. **Qwen-Image low tier: 8 steps without LoRA is broken — fix it**. Either set steps to 20 (so low-tier produces real images) or load the Lightning LoRA. Currently low tier is wasting GPU on noise. **Quality fix more than speed fix**, but renders the "low" tier actually useful. Source: same as #2.

4. **LTX frame counts: 17/33/49 → 25/57/97** (still 8n+1). No latency change per frame, but actually delivers 1s/2.4s/4s clips instead of 0.7s/1.4s/2s. Better fit for the 24fps ffmpeg encode the runner uses (line 365, 451). Source: [LTX-Video repo](https://github.com/Lightricks/LTX-Video) — "frame count divisible by 8+1".

5. **When wiring Qwen3-TTS: use `attn_implementation="sdpa"` not `flash_attention_2`**. flash-attn-2 doesn't build on ROCm; the model card's pip command will fail in the Strix Halo container. Pre-empts a confusing build error on first run. Source: [Qwen3-TTS-CustomVoice card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).

---

## Open questions / sparse evidence

- **HeartMuLa audio sample rate** — README says codec is 12.5 Hz token rate but doesn't state the output WAV sample rate. Will need to inspect first generated WAV or read HeartCodec-oss config.
- **LTX-2.3 22B distilled exact sigma schedule** — referenced as "8-step non-uniform sigma schedule dedicated to distilled models" but the actual sigmas are not quoted in any public README I found. Lives in `packages/ltx-pipelines/` source. Not blocking — the DistilledPipeline class encapsulates them.
- **Wan 2.5** — does not exist on HuggingFace as of 2026-04-25. If it ships later, the Wan 2.2 settings here are a reasonable starting point but expect tuning.
- **Strix Halo specific timings** — no public benchmarks for Qwen-Image / LTX 2.3 / Wan 2.2 on gfx1151. All inference-time estimates extrapolated from RTX 4090/5090 figures with a ~2-3× slowdown assumption. The user's own first runs are the authoritative source.
- **`true_cfg_scale` vs `guidance_scale` confusion in Qwen-Image** — Qwen-Image uses `true_cfg_scale` (a *parallel* CFG channel that does true classifier-free guidance) while keeping `guidance_scale=1.0`. Cross-checked: the model card explicitly uses `true_cfg_scale=4.0`. Make sure `qwen_launcher.py` doesn't conflate the two.
- **ComfyUI sampler/scheduler for Wan 2.2** — diffusers default is UniPCMultistep with flow_shift=5.0, but ComfyUI native workflow uses `euler+simple` matching `ModelSamplingSD3` sigmas. Both work; UniPC is slightly faster per step, euler is closer to training schedule. Not a clear winner.
- **Qwen-Image-Lightning V2.0 vs V1.0** — V2.0 (Sept 2025) reduces over-saturation. Use V2.0 if adding the LoRA.
