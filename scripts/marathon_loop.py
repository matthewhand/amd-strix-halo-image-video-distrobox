#!/usr/bin/env python3
"""8-hour multi-modal generation marathon: Mage images, TTS, music, video.

GPU-aware rotation (single UMA GPU): Mage bursts + TTS (CPU/kokoro), then
optional video (Comfy LTX or HOMIE) and HeartMuLa music with exclusive handoff.
Append-only manifests under MAGE_MARATHON_SCRATCH (default: ./marathon-run).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("MAGE_MARATHON_SCRATCH", ROOT / "marathon-run"))
DURATION_S = float(os.environ.get("MARATHON_DURATION_S", str(8 * 3600)))
MAGE_URL = os.environ.get("MAGE_URL", "http://127.0.0.1:8181")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:8010")
MUSIC_URL = os.environ.get("MUSIC_URL", "http://127.0.0.1:8011")
COMFY_URL = os.environ.get("SLOPFINITY_COMFY_URL", "http://127.0.0.1:8188")
HOMIE_URL = os.environ.get("HOMIE_URL", "http://127.0.0.1:8192")

OUT_IMG = Path(os.environ.get("MARATHON_IMG_DIR", ROOT / "comfy-outputs" / "mage" / "marathon"))
OUT_VID = Path(os.environ.get("MARATHON_VID_DIR", ROOT / "comfy-outputs" / "mage" / "marathon_video"))
OUT_TTS = Path(os.environ.get("MARATHON_TTS_DIR", ROOT / "comfy-outputs" / "experiments" / "tts_marathon"))
OUT_MUSIC = Path(os.environ.get("MARATHON_MUSIC_DIR", ROOT / "comfy-outputs" / "experiments" / "music_marathon"))

for d in (SCRATCH, OUT_IMG, OUT_VID, OUT_TTS, OUT_MUSIC):
    d.mkdir(parents=True, exist_ok=True)

MANIFEST = SCRATCH / "marathon_manifest.jsonl"
UPTIME = SCRATCH / "uptime.jsonl"
VISUAL = SCRATCH / "visual_qa.jsonl"
STATE = SCRATCH / "marathon_state.json"

# Expanding prompt bank: base themes + combinatorial axes that grow over time
SUBJECTS = [
    "headphones", "ramen bowl", "red fox", "lighthouse", "neon alley",
    "hot air balloon", "steam locomotive", "desert caravan", "underwater ruins",
    "mountain shrine", "coffee mug", "origami crane", "glass greenhouse",
    "samurai armor", "bioluminescent jellyfish", "vintage typewriter",
    "ice palace", "market spice stalls", "robot gardener", "paper lanterns",
    "coral reef diver", "observatory dome", "bamboo forest path", "clay teapot",
]
STYLES = [
    "photoreal", "cinematic teal-orange", "watercolor", "oil painting",
    "anime cel shade", "isometric diorama", "macro photography", "noir B&W",
    "impressionist", "cyberpunk neon", "soft studio catalog", "infrared look",
    "tilt-shift miniature", "ukiyo-e woodblock", "brutalist architecture photo",
]
SETTINGS = [
    "at dawn mist", "in heavy rain", "under aurora borealis", "golden hour",
    "moonlit night", "sandstorm haze", "snowy clearing", "crowded festival",
    "abandoned factory", "floating islands", "subway platform", "rooftop garden",
]
MOODS = [
    "serene", "melancholic", "triumphant", "uncanny", "cozy", "epic wide shot",
    "intimate close-up", "documentary candid", "dreamlike", "high energy",
]
TTS_LINES = [
    "The red fox waits at the edge of the snow.",
    "Steam rises from the bowl as chopsticks rest nearby.",
    "Neon rain paints the alley in pink and cyan.",
    "A lighthouse beam cuts through the storm.",
    "Morning light fills the quiet loft with plants.",
    "Somewhere beyond the pine mist, a shrine waits.",
    "Welcome to the marathon of generative scenes.",
    "This is a spoken sample for the audio pipeline check.",
]
MUSIC_TAGS = [
    "lofi chill beats, rainy window, soft piano",
    "epic orchestral, stormy sea, brass and drums",
    "cyberpunk synthwave, night drive, arpeggios",
    "acoustic folk, campfire, warm guitar",
    "ambient drones, deep space, sparse bells",
    "jazz trio, late night bar, brushed snare",
]


def now() -> float:
    return time.time()


def iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def log_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 120):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return {}
        return json.loads(body.decode())


def health(url: str, timeout: float = 3.0) -> bool:
    try:
        http_json("GET", url.rstrip("/") + "/health", timeout=timeout)
        return True
    except Exception:
        try:
            # Comfy uses /system_stats
            req = urllib.request.Request(url.rstrip("/") + "/system_stats")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False


def compose(*args: str) -> int:
    cmd = ["docker", "compose", *args]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        log_jsonl(UPTIME, {
            "ts": iso(), "event": "compose_fail", "cmd": args,
            "stderr": (r.stderr or "")[-500:],
        })
    return r.returncode


def ensure_mage() -> bool:
    if health(MAGE_URL):
        return True
    log_jsonl(UPTIME, {"ts": iso(), "event": "restart_mage"})
    compose("--profile", "mage-image", "up", "-d", "mage-image-service")
    for _ in range(40):
        if health(MAGE_URL):
            # wait for loaded
            try:
                h = http_json("GET", f"{MAGE_URL}/health", timeout=5)
                if h.get("loaded"):
                    return True
            except Exception:
                pass
        time.sleep(3)
    return health(MAGE_URL)


def ensure_tts() -> bool:
    if health(TTS_URL):
        return True
    log_jsonl(UPTIME, {"ts": iso(), "event": "restart_tts"})
    compose("--profile", "qwen-tts", "up", "-d", "qwen-tts-service")
    for _ in range(20):
        if health(TTS_URL):
            return True
        time.sleep(2)
    return health(TTS_URL)


def ensure_music() -> bool:
    if health(MUSIC_URL):
        return True
    log_jsonl(UPTIME, {"ts": iso(), "event": "restart_music"})
    compose("--profile", "heartmula", "up", "-d", "heartmula-service")
    for _ in range(20):
        if health(MUSIC_URL):
            return True
        time.sleep(2)
    return health(MUSIC_URL)


def stop_gpu_peers_for(target: str) -> None:
    """Stop exclusive GPU workers when switching heavy modalities."""
    # Always try stop; ignore failures
    if target != "mage":
        subprocess.run(["docker", "stop", "strix-halo-mage-image"], capture_output=True)
    if target != "comfy":
        subprocess.run(["docker", "stop", "strix-halo-comfyui"], capture_output=True)
    if target != "homie":
        subprocess.run(["docker", "stop", "strix-halo-homie-video"], capture_output=True)
    if target != "music":
        # HeartMuLa uses GPU for real gens
        pass


def is_noise_arr(mean: float, std: float) -> bool:
    return 90.0 < mean < 140.0 and std < 40.0


def image_stats(path: Path) -> dict:
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    mean, std = float(a.mean()), float(a.std())
    return {
        "mean": mean,
        "std": std,
        "min": float(a.min()),
        "max": float(a.max()),
        "noise": is_noise_arr(mean, std),
        "wh": list(Image.open(path).size),
    }


def diversifying_prompt(tick: int) -> tuple[str, str]:
    """Return (slug, prompt) with increasing combinatorial variety."""
    rng = random.Random(tick * 9973 + int(time.time()) // 60)
    # Grow axes used over time
    n_subj = min(len(SUBJECTS), 4 + tick // 5)
    n_style = min(len(STYLES), 3 + tick // 7)
    n_set = min(len(SETTINGS), 3 + tick // 6)
    n_mood = min(len(MOODS), 3 + tick // 8)
    subj = rng.choice(SUBJECTS[:n_subj])
    style = rng.choice(STYLES[:n_style])
    setting = rng.choice(SETTINGS[:n_set])
    mood = rng.choice(MOODS[:n_mood])
    extra = ""
    if tick > 20:
        extra = f", unique detail seed {tick}-{rng.randint(1000,9999)}"
    if tick > 50:
        extra += f", composition rule of thirds, {rng.choice(['wide', 'telephoto', 'dutch angle'])}"
    prompt = f"{subj} {setting}, {mood}, {style}{extra}"
    slug = f"t{tick:05d}_{subj.replace(' ', '_')[:24]}"
    return slug, prompt


def gen_mage_image(tick: int) -> dict | None:
    if not ensure_mage():
        log_jsonl(UPTIME, {"ts": iso(), "event": "mage_down"})
        return None
    slug, prompt = diversifying_prompt(tick)
    seed = 10000 + tick
    t0 = now()
    try:
        body = http_json(
            "POST",
            f"{MAGE_URL}/api/generate",
            {
                "prompt": prompt,
                "width": 768,
                "height": 768,
                "steps": 4,
                "cfg": 1.0,
                "seed": seed,
                "model": "mage-turbo",
            },
            timeout=180,
        )
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "mage_gen_err", "error": str(e)})
        return None
    if not body.get("ok"):
        return {
            "modality": "image", "model": "mage-turbo", "ok": False,
            "prompt": prompt, "error": body, "ts": iso(),
        }
    cpath = body["path"]
    host = cpath.replace("/opt/ComfyUI/output/mage", str(ROOT / "comfy-outputs" / "mage"))
    src = Path(host)
    dest = OUT_IMG / f"{slug}.png"
    try:
        dest.write_bytes(src.read_bytes())
    except Exception:
        dest = src
    try:
        stats = image_stats(dest)
    except Exception as e:
        return {
            "modality": "image", "model": "mage-turbo", "ok": False,
            "prompt": prompt, "error": f"stats:{e}", "ts": iso(),
        }
    accepted = not stats["noise"] and dest.stat().st_size > 20000
    if not accepted:
        reject = OUT_IMG / "rejected"
        reject.mkdir(exist_ok=True)
        try:
            dest.rename(reject / dest.name)
            dest = reject / dest.name
        except Exception:
            pass
    row = {
        "modality": "image",
        "model": "mage-turbo",
        "ok": accepted,
        "noise": stats["noise"],
        "prompt": prompt,
        "slug": slug,
        "path": str(dest),
        "mean": stats["mean"],
        "std": stats["std"],
        "seed": seed,
        "elapsed_s": round(now() - t0, 2),
        "ts": iso(),
        "tick": tick,
    }
    log_jsonl(MANIFEST, row)
    return row


def gen_tts(tick: int) -> dict | None:
    if not ensure_tts():
        return None
    text = TTS_LINES[tick % len(TTS_LINES)]
    # diversify text
    text = f"{text} Sample number {tick}."
    t0 = now()
    try:
        body = http_json(
            "POST",
            f"{TTS_URL}/tts",
            {"text": text, "voice": "af_heart", "engine": "kokoro"},
            timeout=120,
        )
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "tts_err", "error": str(e)})
        return None
    # Resolve output if path given
    path = body.get("path") or body.get("url") or ""
    ok = bool(body.get("ok", True)) if isinstance(body, dict) else False
    # Some TTS returns file bytes path under /tmp
    for key in ("path", "out", "file"):
        if body.get(key) and Path(str(body[key])).is_file():
            src = Path(str(body[key]))
            dest = OUT_TTS / f"tts_{tick:05d}_{src.name}"
            try:
                dest.write_bytes(src.read_bytes())
                path = str(dest)
                ok = dest.stat().st_size > 1000
            except Exception:
                path = str(src)
                ok = src.stat().st_size > 1000
            break
    row = {
        "modality": "tts",
        "model": "kokoro",
        "ok": ok,
        "prompt": text,
        "path": path,
        "elapsed_s": round(now() - t0, 2),
        "ts": iso(),
        "tick": tick,
        "raw_keys": list(body.keys()) if isinstance(body, dict) else [],
    }
    log_jsonl(MANIFEST, row)
    return row


def gen_music(tick: int) -> dict | None:
    """HeartMuLa uses GPU — stop Mage first."""
    stop_gpu_peers_for("music")
    if not ensure_music():
        ensure_mage()
        return None
    tags = MUSIC_TAGS[tick % len(MUSIC_TAGS)]
    t0 = now()
    try:
        body = http_json(
            "POST",
            f"{MUSIC_URL}/music",
            {"prompt": tags, "duration": 8.0},
            timeout=900,
        )
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "music_err", "error": str(e)})
        ensure_mage()
        return None
    ok = bool(body.get("ok"))
    url = body.get("url", "")
    row = {
        "modality": "music",
        "model": "heartmula",
        "ok": ok,
        "prompt": tags,
        "path": url,
        "elapsed_s": round(now() - t0, 2),
        "ts": iso(),
        "tick": tick,
        "error": body.get("error") if not ok else None,
    }
    log_jsonl(MANIFEST, row)
    # restore Mage for image priority
    ensure_mage()
    return row


def gen_video_from_still(tick: int, still: Path, prompt: str) -> dict | None:
    """Prefer LTX via host python slopfinity.ltx_comfy; fallback HOMIE HTTP."""
    stop_gpu_peers_for("comfy")
    out = OUT_VID / f"vid_{tick:05d}_{still.stem[:40]}.mp4"
    t0 = now()
    # Try start Comfy
    compose("--profile", "comfyui", "up", "-d", "comfyui-service")
    comfy_up = False
    for _ in range(30):
        if health(COMFY_URL):
            comfy_up = True
            break
        time.sleep(3)
    if comfy_up:
        env = os.environ.copy()
        env["SLOPFINITY_COMFY_URL"] = COMFY_URL
        env["COMFY_OUTPUTS"] = str(ROOT / "comfy-outputs")
        env["COMFY_INPUT"] = str(ROOT / "comfy-input")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "ltx_launcher.py"),
            "--mode", "video",
            "--prompt", prompt[:200],
            "--image", str(still),
            "--out", str(out),
            "--width", "640",
            "--height", "384",
            "--frames", "25",
            "--timeout", "1200",
        ]
        log_path = SCRATCH / f"i2v_ltx_{tick:05d}.log"
        with log_path.open("w") as lf:
            r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        ok = r.returncode == 0 and out.is_file() and out.stat().st_size > 5000
        row = {
            "modality": "video",
            "model": "ltx-comfy",
            "ok": ok,
            "prompt": prompt,
            "seed_image": str(still),
            "path": str(out) if out.exists() else "",
            "size": out.stat().st_size if out.exists() else 0,
            "elapsed_s": round(now() - t0, 2),
            "ts": iso(),
            "tick": tick,
            "log": str(log_path),
            "rc": r.returncode,
        }
        if ok:
            # duration via ffprobe
            try:
                pr = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
                    capture_output=True, text=True,
                )
                row["duration_s"] = float(pr.stdout.strip() or 0)
            except Exception:
                row["duration_s"] = None
            log_jsonl(MANIFEST, row)
            ensure_mage()
            return row
        log_jsonl(MANIFEST, row)

    # HOMIE fallback
    stop_gpu_peers_for("homie")
    compose("--profile", "homie-video", "up", "-d", "homie-video-service")
    for _ in range(40):
        if health(HOMIE_URL):
            break
        time.sleep(3)
    if health(HOMIE_URL):
        try:
            # Map host path into container mount (comfy-outputs → /opt/ComfyUI/output)
            still_s = str(still.resolve())
            host_out = str((ROOT / "comfy-outputs").resolve())
            if still_s.startswith(host_out):
                cont_img = "/opt/ComfyUI/output" + still_s[len(host_out) :]
            else:
                cont_img = still_s
            body = http_json(
                "POST",
                f"{HOMIE_URL}/api/generate",
                {
                    "prompt": prompt,
                    "ref_image": cont_img,
                    "image": cont_img,
                    "frames": 25,
                    "steps": 20,
                    "size": "832*480",
                },
                timeout=1800,
            )
            row = {
                "modality": "video",
                "model": "homie",
                "ok": bool(body.get("ok")),
                "prompt": prompt,
                "seed_image": still_s,
                "container_image": cont_img,
                "path": body.get("path") or body.get("url") or "",
                "elapsed_s": round(now() - t0, 2),
                "ts": iso(),
                "tick": tick,
                "raw": {k: body.get(k) for k in list(body)[:10]} if isinstance(body, dict) else {},
            }
            # Copy container output if under /opt/ComfyUI/output
            if row["ok"] and body.get("path"):
                bp = str(body["path"])
                if bp.startswith("/opt/ComfyUI/output/"):
                    host_p = ROOT / "comfy-outputs" / bp[len("/opt/ComfyUI/output/") :]
                    if host_p.is_file():
                        dest = OUT_VID / host_p.name
                        try:
                            dest.write_bytes(host_p.read_bytes())
                            row["path"] = str(dest)
                            row["size"] = dest.stat().st_size
                        except Exception:
                            row["path"] = str(host_p)
            log_jsonl(MANIFEST, row)
            ensure_mage()
            return row
        except Exception as e:
            log_jsonl(UPTIME, {"ts": iso(), "event": "homie_err", "error": str(e)})
            row = {
                "modality": "video",
                "model": "homie",
                "ok": False,
                "prompt": prompt,
                "seed_image": str(still),
                "error": str(e),
                "elapsed_s": round(now() - t0, 2),
                "ts": iso(),
                "tick": tick,
            }
            log_jsonl(MANIFEST, row)

    ensure_mage()
    row = {
        "modality": "video",
        "model": "none",
        "ok": False,
        "prompt": prompt,
        "error": "ltx_and_homie_failed",
        "ts": iso(),
        "tick": tick,
        "elapsed_s": round(now() - t0, 2),
    }
    log_jsonl(MANIFEST, row)
    return row


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"tick": 0, "started_at": iso(), "started_ts": now(), "good_images": []}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


def main() -> int:
    st = load_state()
    if "started_ts" not in st:
        st["started_ts"] = now()
        st["started_at"] = iso()
    end_ts = st["started_ts"] + DURATION_S
    # allow resume to still target full duration from original start
    log_jsonl(UPTIME, {
        "ts": iso(),
        "event": "marathon_start",
        "duration_s": DURATION_S,
        "end_ts": end_ts,
        "resume_tick": st.get("tick", 0),
    })
    print(f"[marathon] duration={DURATION_S}s scratch={SCRATCH}", flush=True)

    tick = int(st.get("tick", 0))
    good_images: list[dict] = list(st.get("good_images", []))

    while now() < end_ts:
        remaining = end_ts - now()
        log_jsonl(UPTIME, {
            "ts": iso(),
            "event": "heartbeat",
            "tick": tick,
            "remaining_s": round(remaining),
            "mage": health(MAGE_URL),
            "tts": health(TTS_URL),
            "music": health(MUSIC_URL),
        })

        cycle = int(st.get("cycle", 0))

        # --- Priority: Mage images (2 per cycle) ---
        for _ in range(2):
            if now() >= end_ts:
                break
            row = gen_mage_image(tick)
            tick += 1
            if row and row.get("ok"):
                good_images.append({"path": row["path"], "prompt": row["prompt"]})
                good_images = good_images[-40:]
            st["tick"] = tick
            st["good_images"] = good_images
            save_state(st)

        # --- TTS (CPU) every cycle ---
        if now() < end_ts:
            gen_tts(tick)
            tick += 1
            st["tick"] = tick
            save_state(st)

        # --- Music every 3rd cycle (exclusive GPU handoff) ---
        if cycle % 3 == 1 and now() < end_ts:
            gen_music(tick)
            tick += 1
            st["tick"] = tick
            save_state(st)

        # --- Video every 4th cycle from a good still ---
        if cycle % 4 == 2 and good_images and now() < end_ts:
            still_info = random.choice(good_images[-10:])
            still = Path(still_info["path"])
            if still.is_file():
                gen_video_from_still(tick, still, still_info.get("prompt") or "cinematic motion")
            tick += 1
            st["tick"] = tick
            save_state(st)

        # Harvest TTS files from container into host dir (best-effort)
        try:
            subprocess.run(
                [
                    "docker", "cp",
                    "strix-halo-qwen-tts:/tmp/slopfinity-tts/.",
                    str(OUT_TTS) + "/",
                ],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass

        cycle += 1
        st["cycle"] = cycle
        st["tick"] = tick
        save_state(st)

        # brief pause to avoid hammering
        time.sleep(2)

    # finalize summary
    elapsed = now() - st["started_ts"]
    summary = {
        "started_at": st.get("started_at"),
        "ended_at": iso(),
        "duration_s": elapsed,
        "target_s": DURATION_S,
        "ticks": tick,
        "manifest": str(MANIFEST),
    }
    (SCRATCH / "marathon_summary.json").write_text(json.dumps(summary, indent=2))
    log_jsonl(UPTIME, {"ts": iso(), "event": "marathon_end", **summary})
    print(f"[marathon] done elapsed={elapsed:.0f}s ticks={tick}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
