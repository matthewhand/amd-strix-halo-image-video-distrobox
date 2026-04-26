#!/usr/bin/env python3
"""run_fleet.py — general fleet runner for the Strix Halo image/video toolbox.

This is the canonical fleet orchestrator: it drives the Qwen -> LTX -> mux
pipeline across tiered quality ramps and feeds the slopfinity dashboard.
Originally named ``run_philosophical_experiments.py`` (the seed-prompt
fallback still uses a philosophical-style scene template -- kept intentionally
as a creative default), but the script is no longer scoped to philosophy.
The file-level identity is "fleet runner"; subject matter is data-driven.
"""

import urllib.request
import json
import subprocess
import time
import os
import sys
import random
import argparse
import re
import glob
import math
import fleet_config as cfg

# Default Configuration
OUTPUT_DIR = "comfy-outputs/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("comfy-input", exist_ok=True)

# ---- Tiered quality ramp ---------------------------------------------------
# Incrementally ramp up framerate + resolution across videos so we validate
# the pipeline on cheap passes before committing to slow max-quality runs.
# Override any tier with FLEET_TIER env var ("low" | "med" | "high").
TIER_PROFILES = {
    # tier  : (qwen_steps, qwen_size, ltx_frames, video_timeout_s, image_timeout_s)
    # Image budgets bumped 3× (2026-04-26) — Ernie on Strix Halo runs
    # ~62-80 s/step under tier=low, so 420s timed out at ~7/8 steps. The
    # 1260s headroom covers loading + 8 × 80s + VAE without black-holing
    # the queue. v_idx still advances on timeout via the main-loop guard.
    "low": (8, "1:1", 17, 600, 1260),
    "med": (20, "4:3", 33, 900, 1800),
    "high": (50, "16:9", 49, 1500, 2700),
}

_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "by",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "from",
    "into",
    "over",
    "under",
    "very",
    "really",
    "just",
    "some",
    "any",
    "all",
}


def slugify_prompt(text: str, max_words: int = 5, max_len: int = 40) -> str:
    """Derive a filesystem-safe slug from an LLM-rewritten prompt so output
    files carry semantic hints instead of opaque v1_base.png. Drops stopwords,
    keeps alnum, caps to N words / M chars."""
    if not text:
        return "untitled"
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    keep = [t for t in toks if t not in _STOPWORDS] or toks
    slug = "_".join(keep[:max_words])[:max_len].strip("_")
    return slug or "untitled"


def pick_tier(v_idx):
    forced = os.environ.get("FLEET_TIER", "").strip().lower()
    if forced in TIER_PROFILES:
        return forced
    if v_idx <= 2:
        return "low"
    if v_idx <= 5:
        return "med"
    return "high"


def run_with_timeout(cmd, timeout_s, label=""):
    """subprocess.run with a hard timeout and a clear message on kill."""
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"⏱  {label or cmd[0]} exceeded {timeout_s}s timeout")


# ─── ffmpeg backend resolver ────────────────────────────────────────────────
# The fleet calls ffmpeg in 4 places (frame-seq → mp4, chain bridge extract,
# concat demuxer, music mux). Originally these assumed host ffmpeg; that
# silently broke when ffmpeg was uninstalled from the host. This resolver
# probes once at import time and dispatches every ffmpeg invocation through
# `_ffmpeg_run()`, falling back to `docker exec strix-halo-comfyui ffmpeg`
# when the host has none. Container paths are translated via the known
# bind-mount prefixes; calls that reference unmounted paths (e.g. concat.txt
# in cwd) get a clear error rather than a silent FileNotFoundError.
_FFMPEG_BACKEND = None  # 'host' | 'docker' | None (resolved on first call)
_DOCKER_FFMPEG_CTR = "strix-halo-comfyui"
# host_path_prefix → (container_path_prefix) for the strix-halo-comfyui mounts.
_DOCKER_PATH_MAP = [
    (os.path.abspath("comfy-outputs"), "/opt/ComfyUI/output"),
    (os.path.abspath("comfy-input"), "/opt/ComfyUI/input"),
]


def _resolve_ffmpeg_backend():
    global _FFMPEG_BACKEND
    if _FFMPEG_BACKEND is not None:
        return _FFMPEG_BACKEND
    # Host first — fastest, no path translation, full encoder set.
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _FFMPEG_BACKEND = "host"
        print("[FLEET] ffmpeg backend: host")
        return _FFMPEG_BACKEND
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # Container fallback — works for inputs/outputs that live under the
    # strix-halo-comfyui bind-mounts. libx264 NOT available; libopenh264 is.
    try:
        subprocess.run(
            ["docker", "exec", _DOCKER_FFMPEG_CTR, "ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _FFMPEG_BACKEND = "docker"
        print(
            f"[FLEET] ffmpeg backend: docker exec {_DOCKER_FFMPEG_CTR} (libopenh264 only)"
        )
        return _FFMPEG_BACKEND
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    _FFMPEG_BACKEND = None
    return None


def _ffmpeg_h264_encoder():
    """Return the H.264 encoder name for the active backend. Host ffmpeg
    typically ships libx264; the strix-halo-comfyui container ships
    libopenh264 (Cisco) but not libx264 (Fedora's `ffmpeg-free` excludes it)."""
    return "libx264" if _resolve_ffmpeg_backend() == "host" else "libopenh264"


def _docker_translate_path(p):
    """Map a host path to its container counterpart via the known mounts.
    Returns the unchanged path if no mapping applies — caller may then
    raise a clear error rather than feeding the docker exec a path it
    can't see."""
    ap = os.path.abspath(p)
    for host_prefix, ctr_prefix in _DOCKER_PATH_MAP:
        if ap == host_prefix or ap.startswith(host_prefix + os.sep):
            return ctr_prefix + ap[len(host_prefix) :]
    return p


def _ffmpeg_run(args, *, translate_paths=None, **subprocess_kwargs):
    """Run ffmpeg via the resolved backend. `args` is the argv tail (without
    the leading 'ffmpeg'). When backend is 'docker', `translate_paths` lists
    the indices into `args` that are FILE PATHS and need host→container
    rewriting. Defaults to translating any arg that looks like a path
    relative to cwd. Pass `check=True` etc. via kwargs as usual."""
    backend = _resolve_ffmpeg_backend()
    if backend is None:
        raise RuntimeError(
            "No ffmpeg available. Install on host (`sudo apt install -y ffmpeg`) "
            f"or ensure docker container '{_DOCKER_FFMPEG_CTR}' is running."
        )
    if backend == "host":
        return subprocess.run(["ffmpeg", *args], **subprocess_kwargs)
    # docker backend — translate args that are paths
    new_args = list(args)
    if translate_paths is None:
        # Heuristic: any arg containing 'comfy-outputs/' or 'comfy-input/'
        # gets translated. Conservative — only rewrites paths we KNOW are
        # mounted, leaving everything else (codec names, flags) alone.
        for i, a in enumerate(new_args):
            if isinstance(a, str) and ("comfy-outputs" in a or "comfy-input" in a):
                new_args[i] = _docker_translate_path(a)
    else:
        for i in translate_paths:
            new_args[i] = _docker_translate_path(new_args[i])
    return subprocess.run(
        ["docker", "exec", _DOCKER_FFMPEG_CTR, "ffmpeg", *new_args],
        **subprocess_kwargs,
    )


def update_state(
    mode="Idle", step="Waiting", video=0, total=0, chain=0, total_chains=0, prompt=""
):
    cfg.set_state(mode, step, video, total, chain, total_chains, prompt)


def get_lmstudio_model():
    """Resolve the prompt-rewriter model id. Order:
    1. Trust config.llm.model_id if set (slopfinity Settings).
    2. Otherwise poll the OpenAI-compat endpoint at :11434 for ≤30s.
    3. Give up — return None and let generate_prompt fall back to its
       VOID-style template instead of the fleet hanging forever.
    """
    cfg_model = (cfg.load_config().get("llm") or {}).get("model_id")
    if cfg_model:
        print(f"🔍 LLM: using configured {cfg_model}", flush=True)
        return cfg_model
    print("🔍 No configured LLM — polling :11434 for ≤30s...", flush=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/v1/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
                models = [
                    m["id"] for m in data["data"] if "embed" not in m["id"].lower()
                ]
                if models:
                    print(f"🔍 LLM: discovered {models[0]}", flush=True)
                    return models[0]
        except Exception:
            pass
        time.sleep(5)
    print("⚠️  No LLM reachable — running with VOID-style fallback prompts", flush=True)
    return None


def generate_prompt(model_id, v_idx):
    q = cfg.get_queue()
    config_now = cfg.load_config()
    default_base = config_now.get("base_model", "qwen")
    # Pop the FIRST pending item (skipping any done/cancelled at the front).
    # done/cancelled records stay in queue.json for the audit log.
    pending_idx = next(
        (i for i, x in enumerate(q) if x.get("status") in (None, "pending")), -1
    )
    if pending_idx >= 0:
        task = q.pop(pending_idx)
        # Stamp a sentinel "working" record back onto the queue so:
        #   1. The dashboard can show that this item is mid-flight.
        #   2. The /queue/toggle-infinity and /queue/toggle-polymorphic
        #      endpoints can edit the in-flight flags (honoured at
        #      requeue time).
        #   3. The requeue path re-reads this record to pick up any
        #      mid-flight toggles before deciding whether to re-append.
        working = dict(task)
        working["status"] = "working"
        working["started_ts"] = time.time()
        q.append(working)
        cfg.save_queue(q)
        opts = {
            "image_only": bool(task.get("image_only")),
            "skip_video": bool(task.get("skip_video")),
            "infinity": bool(task.get("infinity")),
            # Fast Track flag — when set, the per-iter chains/frames/tier
            # readers below override config to chains=2 / frames=17 /
            # tier="low" and skip audio + tts for fastest possible
            # turnaround (~3 min/clip on Strix Halo). Used to verify a
            # model is working before committing to a full-quality run.
            "fast_track": bool(task.get("fast_track")),
            # Seed images — user-uploaded starting frames staged from the
            # Subjects-card seed picker. Two consumption modes:
            #   per-task   → task carries a single ``seed_image`` (string).
            #                Iter copies the seed to comfy-input/<stem>_base.png
            #                in place of generate_base_image_*.
            #   per-chain  → task carries ``seed_images`` (list[str]) of >=2 names.
            #                Chain c spans seed_images[c-1] → seed_images[c]
            #                via LTX FLF2V (LTXVAddGuide first+last). N_chains
            #                forced to len(seed_images) - 1.
            "seed_image": task.get("seed_image") or "",
            "seed_images": list(task.get("seed_images") or []),
            "seeds_mode": (task.get("seeds_mode") or "").strip().lower(),
            # `polymorphic` (UI toggle, also persisted as `chaos` on
            # injected tasks for backwards compat):
            # True  → LLM rewrites the seed afresh on EVERY cycle
            # False → LLM rewrites once, the rewritten text is cached and
            #         reused verbatim on subsequent cycles.
            "polymorphic": bool(task.get("polymorphic", task.get("chaos", True))),
            # Preserve the chaos field so the requeued item shows the
            # polymorphic badge in the UI (which reads `q.chaos`).
            "chaos": bool(task.get("chaos", task.get("polymorphic", True))),
            "_seed_prompt": task.get("seed_prompt") or task["prompt"],
            "_orig_task_ts": task.get("ts"),
            # Carry forward the config snapshot so the requeued cycle
            # uses the same model selections the user picked when they
            # first injected the prompt.
            "_config_snapshot": task.get("config_snapshot"),
            "_priority": task.get("priority", "next"),
            "_when_idle": bool(task.get("when_idle", False)),
        }
        # Cached rewrite path — non-polymorphic infinity on cycle 2+.
        if task.get("pre_rewritten"):
            return task["prompt"], task.get("base_model", default_base), opts
        seed = opts["_seed_prompt"]
        base_for_default = task.get("base_model", default_base)
        if model_id:
            try:
                # Honour Settings → Prompts overrides where set. The system
                # prompt comes from `philosophical_prompt`, the user-message
                # template from `fleet_user_prompt_template` (`{seed}` slot).
                # Both fall back to built-in defaults when blank/None.
                sys_p = cfg.get_philosophical_prompt(config_now)
                user_tmpl = cfg.get_fleet_user_prompt_template(config_now)
                try:
                    user_p = user_tmpl.format(seed=seed)
                except Exception:
                    # Malformed user template — keep their text intact and
                    # append the seed so the LLM still has a subject.
                    user_p = f"{user_tmpl}\nSubject: {seed}"
                # Pull temperature from slopfinity config so users can tune.
                # Default 0.7 — gemma4 in particular is too creative at 1.0
                # and produces drifting, off-topic prose.
                temp = float((config_now.get("llm") or {}).get("temperature", 0.7))
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    "temperature": temp,
                }
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    content = json.loads(response.read())["choices"][0]["message"][
                        "content"
                    ].strip()
                    opts["_rewritten_prompt"] = content
                    return content, base_for_default, opts
            except Exception as e:
                print(
                    f"[FLEET] LLM enhance failed for queue item: {e!r} — using raw prompt",
                    flush=True,
                )
        return seed, base_for_default, opts

    config = cfg.load_config()
    if config.get("infinity_mode"):
        themes = config["infinity_themes"]
        idx = config.get("infinity_index", 0) % len(themes)
        theme = themes[idx]
        config["infinity_index"] = idx + 1
        cfg.save_config(config)

        # Honour Settings → Prompts overrides for the infinity-mode call.
        # System prompt = philosophical_prompt; user template uses {theme}.
        system_prompt = cfg.get_philosophical_prompt(config)
        infinity_tmpl = cfg.get_infinity_user_prompt_template(config)
        try:
            user_prompt = infinity_tmpl.format(theme=theme)
        except Exception:
            user_prompt = f"{infinity_tmpl}\nTheme: {theme}"
        try:
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            req = urllib.request.Request(
                "http://127.0.0.1:11434/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                content = json.loads(response.read())["choices"][0]["message"][
                    "content"
                ].strip()
                return content, config.get("base_model", "qwen")
        except Exception as e:
            print(
                f"[FLEET] infinity LLM failed: {e!r} — using theme as raw prompt",
                flush=True,
            )
            # KEEP the theme so the slug carries semantic info instead of
            # collapsing to a generic VOID-style template.
            return theme, config.get("base_model", "qwen")

    # No-LLM fallback only used when there's no theme to retain (e.g. a
    # raw queue item with LLM unavailable AND no infinity_themes seed).
    # The {style} slot is rotated through a small list so repeated outages
    # don't produce identical imagery.
    styles = ["clay animation", "classic sketchy cartoon", "3d digital cel shaded"]
    style = random.choice(styles)
    void_tmpl = cfg.get_void_fallback_template(config)
    try:
        prompt = void_tmpl.format(style=style)
    except Exception:
        prompt = void_tmpl
    return prompt, config.get("base_model", "qwen")


def _write_sidecar(out_path, **fields):
    """Write a small JSON next to an asset capturing prompt + model + ts so the
    slopfinity asset-info modal can show truthful metadata. Cheap; failure is
    non-fatal — the asset still exists if the sidecar can't be written."""
    try:
        side = out_path + ".json"
        with open(side, "w") as f:
            json.dump({**fields, "ts": time.time()}, f)
    except Exception as e:
        print(f"   ⚠️  sidecar write failed for {out_path}: {e}")


def run_image_gen(model, prompt, out_path, tier="high"):
    qsteps, qsize, _frames, _vto, ito = TIER_PROFILES[tier]
    print(
        f"🖼️  Image Gen [{model}] tier={tier} steps={qsteps} size={qsize} timeout={ito}s"
    )
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    qwen_out = os.path.expanduser("~/.qwen-image-studio")
    os.makedirs(qwen_out, exist_ok=True)
    if model == "qwen":
        # FIX: mount ~/.qwen-image-studio so Qwen's PNGs (which it writes to
        # /root/.qwen-image-studio/ inside the container) survive --rm.
        cmd = [
            "docker",
            "run",
            "--rm",
            "-e",
            "PYTHONPATH=/opt/qwen-image-studio/src",
            "-v",
            f"{os.getcwd()}:/workspace",
            "-v",
            f"{hf_cache}:/root/.cache/huggingface",
            "-v",
            f"{qwen_out}:/root/.qwen-image-studio",
            "-w",
            "/workspace",
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            "amd-strix-halo-image-video-toolbox:latest",
            "python3",
            "/opt/qwen_launcher.py",
            "generate",
            "--prompt",
            prompt,
            "--steps",
            str(qsteps),
            "--size",
            qsize,
        ]
        run_with_timeout(cmd, ito, label="qwen_launcher")
        # Pick up the newest PNG from the persisted output dir.
        pngs = glob.glob(os.path.join(qwen_out, "*.png"))
        if not pngs:
            raise Exception(
                "Qwen generated no image (nothing in ~/.qwen-image-studio)."
            )
        latest = max(pngs, key=os.path.getctime)
        subprocess.run(["mv", latest, out_path], check=True)
        _write_sidecar(out_path, prompt=prompt, model=model, tier=tier, kind="image")
        print(f"   ✅ Image saved to {out_path}")
        return True
    elif model == "ernie":
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.getcwd()}:/workspace",
            "-v",
            f"{hf_cache}:/root/.cache/huggingface",
            "-w",
            "/workspace",
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            "amd-strix-halo-image-video-toolbox:latest",
            "python3",
            "/opt/ernie_launcher.py",
            "--prompt",
            prompt,
            "--model",
            "baidu/ERNIE-Image-Turbo",
            "--steps",
            "8",
            "--out",
            out_path,
        ]
        run_with_timeout(cmd, ito, label="ernie_launcher")
        return True
    else:
        # LTX-2.3 Image Workflow
        seed = random.randint(1, 1000000)
        workflow = {
            "1": {
                "class_type": "LowVRAMCheckpointLoader",
                "inputs": {"ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors"},
            },
            "2": {
                "class_type": "LTXVGemmaCLIPModelLoader",
                "inputs": {
                    "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
                    "ltxv_path": "ltx-2.3-22b-distilled-fp8.safetensors",
                    "max_length": 1024,
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "blurry", "clip": ["2", 0]},
            },
            "5": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "cfg": 1.0,
                },
            },
            "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "7": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["1", 0],
                    "scheduler": "simple",
                    "steps": 8,
                    "denoise": 1.0,
                },
            },
            "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "9": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": 1280, "height": 720, "length": 1, "batch_size": 1},
            },
            "10": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["8", 0],
                    "guider": ["5", 0],
                    "sampler": ["6", 0],
                    "sigmas": ["7", 0],
                    "latent_image": ["9", 0],
                },
            },
            "11": {
                "class_type": "LTXVVAEDecode",
                "inputs": {
                    "vae": ["1", 2],
                    "latents": ["10", 0],
                },
            },
            "12": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"ltx_base_{int(time.time())}",
                    "images": ["11", 0],
                },
            },
        }
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8188/prompt",
                data=json.dumps({"prompt": workflow}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                p_id = json.loads(r.read())["prompt_id"]
            while True:
                time.sleep(5)
                with urllib.request.urlopen(
                    f"http://127.0.0.1:8188/history/{p_id}"
                ) as r:
                    h = json.loads(r.read())
                    if p_id in h:
                        fn = h[p_id]["outputs"]["12"]["images"][0]["filename"]
                        subprocess.run(
                            ["cp", f"comfy-outputs/{fn}", out_path], check=True
                        )
                        return True
        except Exception as e:
            if hasattr(e, "read"):
                print(f"❌ ComfyUI Error Body: {e.read().decode('utf-8')}")
            raise e


def heartmula_wav(prompt, out_wav_host, duration_s, timeout_s=600):
    """Generate a HeartMuLa music track to `out_wav_host` (host path under
    OUTPUT_DIR). Returns True on success. Pure WAV producer — no muxing.
    Used by the new pre-image audio stage (so we know the final track's
    length before the video chain loop starts) AND by `run_heartmula_gen`
    when the runner still wants a one-shot generate-then-mux flow."""
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    out_wav_ctr = f"/workspace/{out_wav_host}"
    print(f"🎵 Heartmula music: duration={duration_s:.1f}s prompt={prompt[:60]!r}")
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{os.getcwd()}:/workspace",
        "-v",
        f"{hf_cache}:/root/.cache/huggingface",
        "-v",
        "/mnt/downloads/comfy-models:/mnt/downloads/comfy-models:ro",
        "-w",
        "/workspace",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "-e",
        "HSA_OVERRIDE_GFX_VERSION=11.5.1",
        "amd-strix-halo-image-video-toolbox:latest",
        "python3",
        "/workspace/scripts/heartmula_launcher.py",
        "--prompt",
        prompt,
        "--duration",
        f"{max(1.0, duration_s):.1f}",
        "--out",
        out_wav_ctr,
        "--real",
    ]
    try:
        run_with_timeout(cmd, timeout_s, label="heartmula_launcher")
    except Exception as e:
        print(f"   ⚠️  Heartmula generation failed: {e}")
        return False
    if not os.path.exists(out_wav_host):
        print(f"   ⚠️  Heartmula produced no WAV at {out_wav_host}")
        return False
    return True


def run_heartmula_gen(prompt, video_path, final_out_path, duration_s, timeout_s=600):
    """Legacy generate-then-mux entry point. Generates the WAV via
    `heartmula_wav`, then muxes onto `video_path` to `final_out_path`.
    The new main-loop path generates the WAV up front (pre-image) and
    muxes after Final Merge — but this helper stays for back-compat."""
    music_wav_host = os.path.join(OUTPUT_DIR, f"_tmp_music_{int(time.time())}.wav")
    if not heartmula_wav(prompt, music_wav_host, duration_s, timeout_s=timeout_s):
        return False

    print(f"   🎼 Muxing {music_wav_host} onto {video_path} -> {final_out_path}")
    mux_args = [
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        music_wav_host,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        final_out_path,
    ]
    try:
        _ffmpeg_run(mux_args, check=True)
    except Exception as e:
        print(f"   ⚠️  ffmpeg mux failed: {e}")
        return False
    finally:
        try:
            os.remove(music_wav_host)
        except OSError:
            pass

    print(f"   ✅ Muxed MP4+audio saved to {final_out_path}")
    return True


def run_comfy_job(workflow, out_node_id, target_file):
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            p_id = json.loads(r.read())["prompt_id"]
        while True:
            time.sleep(5)
            with urllib.request.urlopen(
                f"http://127.0.0.1:8188/history/{p_id}"
            ) as h_res:
                h = json.loads(h_res.read())
                if p_id in h:
                    out = h[p_id]["outputs"][str(out_node_id)]
                    if "images" in out:
                        imgs = sorted([i["filename"] for i in out["images"]])
                        if len(imgs) == 1:
                            subprocess.run(
                                ["cp", f"comfy-outputs/{imgs[0]}", target_file],
                                check=True,
                            )
                            return True
                        else:
                            first = imgs[0]
                            match = re.search(r"(\d+)(?=_\.png)", first)
                            num = match.group(1) if match else "00001"
                            patt = (
                                first.rsplit(num, 1)[0]
                                + f"%0{len(num)}d"
                                + first.rsplit(num, 1)[1]
                            )
                            print(f"   🎞️  Encoding {len(imgs)} frames...")
                            args = [
                                "-y",
                                "-framerate",
                                "24",
                                "-start_number",
                                str(int(num)),
                                "-i",
                                f"comfy-outputs/{patt}",
                                "-c:v",
                                _ffmpeg_h264_encoder(),
                                "-pix_fmt",
                                "yuv420p",
                                "-b:v",
                                "8M",
                                target_file,
                            ]
                            _ffmpeg_run(args, check=True)
                            for f in imgs:
                                try:
                                    os.remove(f"comfy-outputs/{f}")
                                except:
                                    pass
                            return True
    except Exception as e:
        if hasattr(e, "read"):
            print(f"❌ ComfyUI Error Body: {e.read().decode('utf-8')}")
        raise e


def _resolve_size(size_str, default="1280*720"):
    """Translate the slopfinity config `size` (which may be an aspect ratio
    like '1:1' / '4:3' / '16:9' OR a literal 'WIDTH*HEIGHT') into a
    'WIDTH*HEIGHT' string LTX/Comfy can parse."""
    if not size_str:
        return default
    if "*" in size_str:
        return size_str
    aspects = {
        "1:1": "1024*1024",
        "4:3": "1152*864",
        "16:9": "1280*720",
        "9:16": "720*1280",
        "3:4": "864*1152",
    }
    return aspects.get(size_str.strip(), default)


def generate_base_image_ltx23(prompt, output_path, size_str):
    size_str = _resolve_size(size_str)
    w, h = map(int, size_str.split("*"))
    seed = random.randint(1, 1000000000)
    workflow = {
        "1": {
            "class_type": "LowVRAMCheckpointLoader",
            "inputs": {"ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors"},
        },
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {
                "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
                "ltxv_path": "ltx-2.3-22b-distilled-fp8.safetensors",
                "max_length": 1024,
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality", "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "cfg": 1.0,
            },
        },
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "7": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": w, "height": h, "length": 1, "batch_size": 1},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["5", 0],
                "sampler": ["6", 0],
                "sigmas": ["7", 0],
                "latent_image": ["9", 0],
            },
        },
        "11": {
            "class_type": "LTXVVAEDecode",
            "inputs": {
                "vae": ["1", 2],
                "latents": ["10", 0],
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"base_{int(time.time())}",
                "images": ["11", 0],
            },
        },
    }
    return run_comfy_job(workflow, 12, output_path)


def generate_video_ltx(image_fn, prompt, out_path, size_str, frames):
    size_str = _resolve_size(size_str)
    w, h = map(int, size_str.split("*"))
    print(f"🎬 Video Gen [LTX-2.3]...")
    seed = random.randint(1, 1000000)
    prefix = f"vid_{int(time.time())}"
    workflow = {
        "1": {
            "class_type": "LowVRAMCheckpointLoader",
            "inputs": {"ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors"},
        },
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {
                "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
                "ltxv_path": "ltx-2.3-22b-distilled-fp8.safetensors",
                "max_length": 1024,
            },
        },
        "3": {"class_type": "LoadImage", "inputs": {"image": image_fn}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "cfg": 1.0,
            },
        },
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": w, "height": h, "length": frames, "batch_size": 1},
        },
        "11": {
            "class_type": "LTXVImgToVideoConditionOnly",
            "inputs": {
                "vae": ["1", 2],
                "image": ["3", 0],
                "latent": ["10", 0],
                "strength": 1.0,
            },
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["6", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["11", 0],
            },
        },
        "13": {
            "class_type": "LTXVVAEDecode",
            "inputs": {"vae": ["1", 2], "latents": ["12", 0]},
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": prefix, "images": ["13", 0]},
        },
    }
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            p_id = json.loads(r.read())["prompt_id"]
        while True:
            time.sleep(10)
            with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}") as r:
                h = json.loads(r.read())
                if p_id in h:
                    imgs = [f["filename"] for f in h[p_id]["outputs"]["14"]["images"]]
                    first = imgs[0]
                    match = re.search(r"(\d+)(?=_\.png)", first)
                    num = match.group(1) if match else "00001"
                    parts = first.rsplit(num, 1)
                    patt = f"%0{len(num)}d".join(parts)
                    print(f"   🎞️  Encoding...")
                    # ffmpeg lives inside the strix-halo-comfyui container (host
                    # has none installed). The container's /opt/ComfyUI/output
                    # mounts ./comfy-outputs/ on host, so paths translate cleanly.
                    repo_root = os.path.abspath(os.path.dirname(__file__))
                    out_rel = os.path.relpath(
                        out_path, os.path.join(repo_root, "comfy-outputs")
                    )
                    # Frame-seq → MP4. The ffmpeg backend resolver picks the
                    # H.264 encoder available in the active backend (libx264 on
                    # host, libopenh264 in the strix-halo-comfyui container).
                    args = [
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        "24",
                        "-start_number",
                        str(int(num)),
                        "-i",
                        f"comfy-outputs/{patt}",
                        "-c:v",
                        _ffmpeg_h264_encoder(),
                        "-pix_fmt",
                        "yuv420p",
                        "-b:v",
                        "8M",
                        out_path,
                    ]
                    _ffmpeg_run(args, check=True)
                    for f in imgs:
                        try:
                            os.remove(os.path.join(repo_root, "comfy-outputs", f))
                        except:
                            pass
                    return
    except urllib.error.HTTPError as e:
        print(f"❌ ComfyUI HTTP Error: {e.code} {e.reason}")
        print(f"   Body: {e.read().decode('utf-8')}")
        raise e


def generate_video_ltx_flf2v(start_image_fn, end_image_fn, prompt, out_path, size_str, frames):
    """LTX 2.3 First-Last-Frame to Video.

    Generates a clip whose first frame matches ``start_image_fn`` and
    last frame matches ``end_image_fn``. Produces dramatically more
    consistent motion than chained image-to-video extension because LTX
    interpolates between two anchor frames in latent space rather than
    drifting via last-frame-handoff.

    Implementation uses two ``LTXVAddGuide`` nodes — one at frame_idx=0
    pinning the start image, another at the highest valid index pinning
    the end image. ``frame_idx`` MUST be a multiple of 8 per LTX's
    temporal token grid; we round the last-frame index down accordingly.
    """
    size_str = _resolve_size(size_str)
    w, h = map(int, size_str.split("*"))
    print(f"🎬 Video Gen [LTX-2.3 FLF2V] {start_image_fn} → {end_image_fn}")
    seed = random.randint(1, 1000000)
    prefix = f"flf2v_{int(time.time())}"
    # frame_idx must be multiple of 8. For 49 frames last=48; for 17 frames
    # last=16; for 9 frames last=8. Floor division pins the end frame to
    # the nearest valid token boundary at-or-before the final frame.
    last_frame_idx = max(8, ((frames - 1) // 8) * 8)
    workflow = {
        "1": {
            "class_type": "LowVRAMCheckpointLoader",
            "inputs": {"ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors"},
        },
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {
                "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
                "ltxv_path": "ltx-2.3-22b-distilled-fp8.safetensors",
                "max_length": 1024,
            },
        },
        "3a": {"class_type": "LoadImage", "inputs": {"image": start_image_fn}},
        "3b": {"class_type": "LoadImage", "inputs": {"image": end_image_fn}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality", "clip": ["2", 0]},
        },
        "10": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": w, "height": h, "length": frames, "batch_size": 1},
        },
        # First keyframe at t=0
        "11a": {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "vae": ["1", 2],
                "latent": ["10", 0],
                "image": ["3a", 0],
                "frame_idx": 0,
                "strength": 1.0,
            },
        },
        # Last keyframe at the highest valid token-aligned index
        "11b": {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": ["11a", 0],
                "negative": ["11a", 1],
                "vae": ["1", 2],
                "latent": ["11a", 2],
                "image": ["3b", 0],
                "frame_idx": last_frame_idx,
                "strength": 1.0,
            },
        },
        "6": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["11b", 0],
                "negative": ["11b", 1],
                "cfg": 1.0,
            },
        },
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["6", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["11b", 2],
            },
        },
        "13": {
            "class_type": "LTXVVAEDecode",
            "inputs": {"vae": ["1", 2], "latents": ["12", 0]},
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": prefix, "images": ["13", 0]},
        },
    }
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            p_id = json.loads(r.read())["prompt_id"]
        while True:
            time.sleep(10)
            with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}") as r:
                h = json.loads(r.read())
                if p_id in h:
                    imgs = [f["filename"] for f in h[p_id]["outputs"]["14"]["images"]]
                    first = imgs[0]
                    match = re.search(r"(\d+)(?=_\.png)", first)
                    num = match.group(1) if match else "00001"
                    parts = first.rsplit(num, 1)
                    patt = f"%0{len(num)}d".join(parts)
                    print(f"   🎞️  Encoding...")
                    repo_root = os.path.abspath(os.path.dirname(__file__))
                    args = [
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        "24",
                        "-start_number",
                        str(int(num)),
                        "-i",
                        f"comfy-outputs/{patt}",
                        "-c:v",
                        _ffmpeg_h264_encoder(),
                        "-pix_fmt",
                        "yuv420p",
                        "-b:v",
                        "8M",
                        out_path,
                    ]
                    _ffmpeg_run(args, check=True)
                    for f in imgs:
                        try:
                            os.remove(os.path.join(repo_root, "comfy-outputs", f))
                        except OSError:
                            pass
                    return
    except urllib.error.HTTPError as e:
        print(f"❌ ComfyUI HTTP Error: {e.code} {e.reason}")
        print(f"   Body: {e.read().decode('utf-8')}")
        raise e


# ---- Pipeline matrix --------------------------------------------------------
# User-requested ramp: first prove Qwen → LTX 2.3, then Ernie → LTX 2.3, then
# Ernie → LTX 2.3 + Heartmula music. Enabled with FLEET_MATRIX=1. Counts per
# phase configurable via FLEET_MATRIX_PER=N (default 3).
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


def main():
    m_id = get_lmstudio_model()
    v_idx = 1
    matrix_mode = os.environ.get("FLEET_MATRIX", "").strip() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if matrix_mode:
        print(
            "[FLEET] MATRIX MODE enabled — cycling through: "
            + ", ".join(
                f"{b}->{v}{'+' + a if a != 'none' else ''}" for b, v, a in MATRIX_PHASES
            )
        )
    # Sweep stale `working` sentinels from a previous fleet run. The fleet
    # runner stamps a working row on every pop and removes it at requeue
    # time; a hard crash mid-iter would leave one stranded. On startup,
    # promote them to `cancelled` so the dashboard reflects "the prior
    # run died" rather than showing a phantom in-flight item.
    try:
        q0 = cfg.get_queue()
        cleaned = False
        for it in q0:
            if it.get("status") == "working":
                it["status"] = "cancelled"
                it["cancelled_ts"] = time.time()
                it["infinity"] = False
                cleaned = True
        if cleaned:
            cfg.save_queue(q0)
            print(
                "[FLEET] swept stale `working` sentinels from previous run", flush=True
            )
    except Exception as e:
        print(f"[FLEET] startup working-sweep failed: {e!r}", flush=True)
    while True:
        config = cfg.load_config()
        if not config.get("infinity_mode") and v_idx > 1000:
            break
        update_state(mode="Thinking", step="Concept", video=v_idx, total=1000)
        _iter_started_ts = time.time()
        # Per-iter list of generated asset basenames (relative to OUTPUT_DIR /
        # /workspace) — populated as each stage writes its file. Persisted on
        # the done record so the dashboard's expanded done card can render
        # thumbnails / mini-players for every produced output.
        _iter_assets: list = []
        _gp = generate_prompt(m_id, v_idx)
        if len(_gp) == 3:
            p, b_mod, _task_opts = _gp
        else:
            p, b_mod = _gp
            _task_opts = {}
        print(f"[FLEET] >>> Iter v{v_idx} prompt='{p[:80]}...'", flush=True)
        if matrix_mode:
            b_mod, v_mod_forced, a_mod_forced = pick_matrix_combo(v_idx)
            print(
                f"[MATRIX] v{v_idx}: {b_mod} → {v_mod_forced} + audio={a_mod_forced}",
                flush=True,
            )
        iter_failed = False
        try:
            # Fast Track override — mutate this iter's snapshot only so all
            # downstream snapshot reads (chains/frames/tier/audio/tts) pick
            # up the lower-quality budget. Global config.json untouched.
            # Targets ~3 min/clip on Strix Halo for "does the model work?"
            # smoke runs.
            if _task_opts.get("fast_track"):
                _ft_snap = dict((_task_opts.get("_config_snapshot") or config) or {})
                _ft_snap["chains"] = 2
                _ft_snap["frames"] = 17
                _ft_snap["tier"] = "low"
                _ft_snap["audio_model"] = "none"
                _ft_snap["tts_model"] = "none"
                _ft_snap["upscale_model"] = "none"
                _task_opts["_config_snapshot"] = _ft_snap
                print(
                    f"[FLEET] 🏃 Fast Track v{v_idx}: chains=2 frames=17 "
                    f"tier=low audio/tts/upscale skipped",
                    flush=True,
                )

            _slug = slugify_prompt(p)
            # Filename stem: "slop_<idx>_<slug>". The slop_ prefix replaces
            # the older "v<idx>_" form so output names read coherently with
            # the dashboard's "Slop" branding while still carrying the
            # iteration index for chronological grouping + uniqueness.
            _stem = f"slop_{v_idx}_{_slug}"

            # ─── Audio (Heartmula) — runs BEFORE Base Image ──────────────
            # Honors config.audio_model (no longer matrix-mode-gated). When
            # audio_driven_chains is enabled, the resulting WAV's duration
            # determines how many video chains we generate so the final cut
            # matches audio length.
            audio_wav = None
            audio_duration_s = 0.0
            _audio_model = (_task_opts.get("_config_snapshot") or config or {}).get(
                "audio_model", "none"
            ) or "none"
            if (
                _audio_model
                and _audio_model != "none"
                and not _task_opts.get("image_only")
                and not _task_opts.get("skip_video")
            ):
                update_state(
                    mode="Composing", step="Audio", video=v_idx, total=1000, prompt=p
                )
                target_dur = float(
                    (_task_opts.get("_config_snapshot") or config or {}).get(
                        "audio_duration_s", 30.0
                    )
                )
                if _audio_model == "heartmula":
                    audio_wav = f"{OUTPUT_DIR}/{_stem}_music.wav"
                    try:
                        ok = heartmula_wav(p, audio_wav, duration_s=target_dur)
                        if ok and os.path.exists(audio_wav):
                            audio_duration_s = target_dur
                            _write_sidecar(
                                audio_wav,
                                prompt=p,
                                model="heartmula",
                                kind="music",
                                duration_s=target_dur,
                            )
                            _iter_assets.append(os.path.basename(audio_wav))
                        else:
                            audio_wav = None
                    except Exception as e:
                        print(f"[FLEET] ⚠️  Audio (heartmula) failed: {e!r}", flush=True)
                        audio_wav = None
                # else: other audio models (none today) — silently skip.

            # ─── TTS (Qwen-TTS / Kokoro) — placeholder: NOT YET IMPLEMENTED ─
            # The dashboard exposes tts_model but run_fleet.py doesn't have
            # a TTS code path yet (the slopfinity/workers/tts.py worker is
            # part of the unwired StageWorker pattern). Skip silently for
            # now; the heartbeat still flips through TTS so the UI shows
            # the stage as visited but no asset is produced.
            _tts_model = (_task_opts.get("_config_snapshot") or config or {}).get(
                "tts_model", "none"
            ) or "none"
            tts_wav = None
            if (
                _tts_model
                and _tts_model != "none"
                and not _task_opts.get("image_only")
                and not _task_opts.get("skip_video")
            ):
                update_state(
                    mode="Voicing", step="TTS", video=v_idx, total=1000, prompt=p
                )
                # TODO: wire scripts/qwen_tts_serve.py / kokoro path here.
                print(
                    f"[FLEET] ⚠️  TTS stage requested (model={_tts_model}) but not yet implemented in run_fleet.py",
                    flush=True,
                )

            in_img = f"comfy-input/{_stem}_base.png"
            update_state(
                mode="Rendering", step="Base Image", video=v_idx, total=1000, prompt=p
            )
            tier = "low" if _task_opts.get("fast_track") else pick_tier(v_idx)
            # Seed-image short-circuit: when the task carries a user-supplied
            # seed (per-task) OR per-chain mode (where chain 0 starts from
            # seed_images[0]), copy the seed to in_img and skip generate_base_*.
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
            elif b_mod in ["qwen", "ernie"]:
                run_image_gen(b_mod, p, in_img, tier=tier)
            else:
                generate_base_image_ltx23(p, in_img, config["size"])

            _out_base = f"{OUTPUT_DIR}/{_stem}_base.png"
            subprocess.run(["cp", in_img, _out_base], check=True)
            # Mirror the sidecar so the slopfinity asset-info modal can show
            # the prompt for the *outputs* copy too.
            _write_sidecar(_out_base, prompt=p, model=b_mod, kind="image", v_idx=v_idx)
            _iter_assets.append(os.path.basename(_out_base))

            # Per-task image_only flag: skip every stage after Base Image —
            # no chains, no ffmpeg, no audio. Use this when you just want
            # quick image renders without the heavy LTX video pipeline.
            if _task_opts.get("image_only") or _task_opts.get("skip_video"):
                update_state(
                    mode="Completed",
                    step="Image Only",
                    video=v_idx,
                    total=1000,
                    prompt=p,
                )
                print(
                    f"[FLEET] image_only=True — skipping chain video stage", flush=True
                )
                v_idx += 1
                continue

            # Chain count: when audio_driven_chains is enabled AND we have an
            # audio duration, size the loop so total video duration ≥ audio
            # duration. Otherwise honor config.chains (legacy default 10).
            _frames_per_chain = int(
                (_task_opts.get("_config_snapshot") or config or {}).get("frames", 49)
            )
            _audio_driven = bool(
                (_task_opts.get("_config_snapshot") or config or {}).get(
                    "audio_driven_chains"
                )
            )
            if _audio_driven and audio_duration_s > 0 and _frames_per_chain > 0:
                _chain_seconds = _frames_per_chain / 24.0
                _n_chains = max(1, int(math.ceil(audio_duration_s / _chain_seconds)))
                _n_chains = min(
                    _n_chains, 30
                )  # safety cap so a long track can't run away
                print(
                    f"[FLEET] audio_driven_chains: audio={audio_duration_s:.1f}s · chain={_chain_seconds:.2f}s → {_n_chains} chains",
                    flush=True,
                )
            else:
                _n_chains = int(
                    (_task_opts.get("_config_snapshot") or config or {}).get(
                        "chains", 10
                    )
                    or 10
                )

            # Per-chain seed mode — N seeds force N-1 FLF2V chains spanning
            # seed[i] → seed[i+1]. Overrides config.chains and audio-driven.
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

            chain_vids = []
            _flf2v_active = len(_per_chain_seeds) >= 2
            for c_idx in range(1, _n_chains + 1):
                update_state(
                    mode="Rendering",
                    step="Video Chains",
                    video=v_idx,
                    total=1000,
                    chain=c_idx,
                    total_chains=_n_chains,
                    prompt=p,
                )
                seg = f"{OUTPUT_DIR}/{_stem}_c{c_idx}.mp4"
                if _flf2v_active:
                    # Stage both keyframes into comfy-input. The base-stage
                    # already copied seed[0] → in_img; for chain c we need
                    # the END frame = seed_images[c]. The START is seed[c-1]
                    # — which equals in_img on chain 1, and the previous
                    # chain's end keyframe on subsequent chains.
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
                        _start_fn, _end_fn, p, seg,
                        config["size"], config["frames"],
                    )
                    _write_sidecar(
                        seg,
                        prompt=p,
                        model="ltx-2.3-flf2v",
                        kind="video",
                        part=c_idx,
                        of=_n_chains,
                        kf_start=_start_seed,
                        kf_end=_end_seed,
                    )
                else:
                    generate_video_ltx(
                        os.path.basename(in_img), p, seg, config["size"], config["frames"]
                    )
                    _write_sidecar(
                        seg,
                        prompt=p,
                        model="ltx-2.3",
                        kind="video",
                        part=c_idx,
                        of=_n_chains,
                    )
                chain_vids.append(seg)
                if c_idx < _n_chains and not _flf2v_active:
                    next_in = f"comfy-input/{_stem}_f{c_idx}.png"
                    _ffmpeg_run(
                        [
                            "-y",
                            "-sseof",
                            "-1",
                            "-i",
                            seg,
                            "-update",
                            "1",
                            "-q:v",
                            "1",
                            next_in,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    in_img = next_in
                    subprocess.run(
                        ["cp", next_in, f"{OUTPUT_DIR}/{_stem}_f{c_idx}.png"],
                        check=True,
                    )

            update_state(
                mode="Finalizing", step="Final Merge", video=v_idx, total=1000, prompt=p
            )
            # Write the concat list inside comfy-outputs so the docker backend
            # (when host ffmpeg is missing) can see it via the bound mount. Each
            # entry is relative to comfy-outputs/ — ffmpeg's concat demuxer
            # resolves them relative to the list file's directory by default.
            concat_path = f"{OUTPUT_DIR}/_concat_{v_idx}.txt"
            with open(concat_path, "w") as f:
                for v in chain_vids:
                    # concat demuxer needs paths relative to the list file
                    f.write(f"file '{os.path.basename(v)}'\n")
            final_silent = f"{OUTPUT_DIR}/FINAL_{v_idx}_{_slug}.mp4"
            _ffmpeg_run(
                [
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_path,
                    "-c",
                    "copy",
                    final_silent,
                ],
                check=True,
            )
            _write_sidecar(
                final_silent,
                prompt=p,
                model="ltx-2.3",
                kind="final",
                parts=len(chain_vids),
            )
            _iter_assets.append(os.path.basename(final_silent))
            os.remove(concat_path)

            # Mux: if Audio ran successfully up front, lay the WAV onto the
            # silent FINAL_*.mp4 and emit FINAL_*_audio.mp4 alongside.
            # Replaces the old matrix-mode-only heartmula path; now any
            # iteration with audio_model != "none" gets a muxed final.
            if audio_wav and os.path.exists(audio_wav):
                final_audio = f"{OUTPUT_DIR}/FINAL_{v_idx}_{_slug}_audio.mp4"
                mux_args = [
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    final_silent,
                    "-i",
                    audio_wav,
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    final_audio,
                ]
                try:
                    _ffmpeg_run(mux_args, check=True)
                    _write_sidecar(
                        final_audio,
                        prompt=p,
                        model="heartmula",
                        kind="final-with-audio",
                        parts=len(chain_vids),
                        audio_duration_s=audio_duration_s,
                    )
                    _iter_assets.append(os.path.basename(final_audio))
                except Exception as e:
                    print(f"[FLEET] ⚠️  ffmpeg mux failed: {e!r}", flush=True)
        except Exception as e:
            iter_failed = True
            import traceback

            print(f"[FLEET] ❌ Error Video {v_idx}: {e!r}", flush=True)
            traceback.print_exc()
            time.sleep(10)
        # ALWAYS advance v_idx — even on failure — so every iter gets a unique
        # filename. Otherwise repeated chain failures keep clobbering v1_base.png.
        v_idx += 1
        if iter_failed:
            print(f"[FLEET] ↪ continuing to v{v_idx}", flush=True)
        # ARCHIVE + REQUEUE for an infinity item:
        #   1. Pull the in-flight `working` record (so we honour any
        #      mid-flight toggles the user made via the toggle endpoints).
        #   2. Drop that working record from the queue.
        #   3. Append a `done` record (audit log).
        #   4. If the item was flagged infinity AND wasn't cancelled
        #      mid-flight, append a fresh pending item with all original
        #      flags + config_snapshot preserved.
        #
        # Both writes happen under a single `cfg.save_queue` call so a
        # crash mid-flow can't leave the queue with a stranded `working`
        # row.
        try:
            if _task_opts.get("_seed_prompt"):
                q_now = cfg.get_queue()
                # Pull (and remove) the in-flight row matching this iter
                # so we can read the user's most recent toggle / cancel
                # state. The row may be `status=working` (untouched) OR
                # `status=cancelled` (user clicked cancel mid-flight) —
                # both share the same ts as the originally-popped item.
                orig_ts = _task_opts.get("_orig_task_ts")
                live_record = None
                kept = []
                for it in q_now:
                    if (
                        it.get("ts") == orig_ts
                        and it.get("status") in ("working", "cancelled")
                        and live_record is None
                    ):
                        live_record = it
                        continue  # drop the in-flight sentinel
                    kept.append(it)
                q_now = kept

                # Effective flags: the live record's flags win over the
                # snapshot in `_task_opts` so the user's mid-flight
                # toggles take effect. Cancellation also sticks: a user
                # who clicked "cancel" on the active item set the live
                # record's status to "cancelled" — that disables requeue.
                live = live_record or {}
                eff_infinity = bool(live.get("infinity", _task_opts.get("infinity")))
                eff_polymorphic = bool(
                    live.get(
                        "polymorphic",
                        live.get("chaos", _task_opts.get("polymorphic", True)),
                    )
                )
                eff_chaos = bool(
                    live.get(
                        "chaos", live.get("polymorphic", _task_opts.get("chaos", True))
                    )
                )
                eff_image_only = bool(
                    live.get("image_only", _task_opts.get("image_only"))
                )
                eff_skip_video = bool(
                    live.get("skip_video", _task_opts.get("skip_video"))
                )
                eff_when_idle = bool(
                    live.get("when_idle", _task_opts.get("_when_idle", False))
                )
                eff_priority = live.get("priority", _task_opts.get("_priority", "next"))
                eff_config_snapshot = live.get("config_snapshot") or _task_opts.get(
                    "_config_snapshot"
                )
                cancelled_mid_flight = live.get("status") == "cancelled"

                # 1) Done archive — audit log entry.
                q_now.append(
                    {
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
                        # Asset basenames produced this iter, in the order they
                        # were written (base image → final mp4 → muxed audio mp4).
                        "assets": list(_iter_assets),
                    }
                )

                # 2) Requeue — only if the user still wants infinity AND
                # they didn't cancel mid-flight.
                if eff_infinity and not cancelled_mid_flight:
                    if eff_polymorphic:
                        # Re-append the SEED so the next pop runs the LLM
                        # rewriter again — every cycle gets fresh prose.
                        new_task = {
                            "prompt": _task_opts["_seed_prompt"],
                            "polymorphic": True,
                            "chaos": True,
                        }
                    else:
                        # Cache the rewrite on the task so future cycles
                        # skip the LLM round-trip entirely.
                        rewritten = (
                            _task_opts.get("_rewritten_prompt")
                            or _task_opts["_seed_prompt"]
                        )
                        new_task = {
                            "prompt": rewritten,
                            "polymorphic": False,
                            "chaos": False,
                            "pre_rewritten": True,
                            "seed_prompt": _task_opts["_seed_prompt"],
                        }
                    requeued = {
                        **new_task,
                        "priority": eff_priority,
                        "status": "pending",
                        # +1µs offset so the new pending row can never
                        # share a ts with the done archive we just wrote
                        # (the queue keys are de facto (ts) primaries).
                        "ts": time.time() + 1e-6,
                        "infinity": True,
                        "when_idle": eff_when_idle,
                        "image_only": eff_image_only,
                        "skip_video": eff_skip_video,
                        "config_snapshot": eff_config_snapshot,
                        "requeued_from_ts": orig_ts,
                    }
                    q_now.append(requeued)
                    tag = "polymorphic" if eff_polymorphic else "fixed-prompt"
                    print(
                        f"[FLEET] ♾  re-queued infinity prompt ({tag}) to back of queue",
                        flush=True,
                    )
                elif _task_opts.get("infinity") and cancelled_mid_flight:
                    print(
                        f"[FLEET] ♾  infinity item cancelled mid-flight — NOT re-queueing",
                        flush=True,
                    )
                elif _task_opts.get("infinity") and not eff_infinity:
                    print(
                        f"[FLEET] ♾  user toggled infinity OFF mid-flight — NOT re-queueing",
                        flush=True,
                    )

                # Single commit of done + (maybe) requeued + working-row drop.
                cfg.save_queue(q_now)
        except Exception as e:
            print(f"[FLEET] archive/requeue failed: {e!r}", flush=True)
    update_state(mode="Completed")


if __name__ == "__main__":
    main()
