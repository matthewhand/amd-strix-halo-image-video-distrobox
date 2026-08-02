#!/usr/bin/env python3
"""Philosopher short pipeline (Slopfinity-oriented):

  a) 30–60s TTS of a philosopher's teaching (Kokoro via :8010 / Slopfinity /tts)
  b) Mage still matching the teaching's imagery
  c) LTX I2V from that still
  d) Loop/extend video to TTS length and mux → FINAL_philosopher_*.mp4

Also configures Slopfinity (mage + ltx + kokoro, no music/upscale) and can
queue the same job via POST /inject for the coordinator.

Usage:
  python3 scripts/philosopher_slop_pipeline.py
  python3 scripts/philosopher_slop_pipeline.py --philosopher epictetus --also-inject
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(os.environ.get("SLOPFINITY_EXP_DIR") or ROOT / "comfy-outputs" / "experiments")
WORK = ROOT / "comfy-outputs" / "experiments" / "philosopher_shorts"
EXP.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

SLOP = os.environ.get("SLOPFINITY_URL", "http://127.0.0.1:9099").rstrip("/")
MAGE = os.environ.get("MAGE_URL", "http://127.0.0.1:8181").rstrip("/")
TTS = os.environ.get("TTS_URL", "http://127.0.0.1:8010").rstrip("/")
COMFY = os.environ.get("SLOPFINITY_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

# ~180–220 words → ~45–60s at Kokoro pace on this host (~3.5–4 w/s)
PHILOSOPHERS = {
    "marcus_aurelius": {
        "name": "Marcus Aurelius",
        "teaching": (
            "You have power over your mind, not outside events. Realize this, and you will find strength. "
            "The happiness of your life depends upon the quality of your thoughts. "
            "Waste no more time arguing about what a good person should be. Be one. "
            "When you arise in the morning, think of what a privilege it is to be alive, to think, to enjoy, to love. "
            "Accept the things to which fate binds you, and love the people with whom fate brings you together, "
            "but do so with all your heart. "
            "If it is not right, do not do it. If it is not true, do not say it. "
            "The best revenge is not to be like your enemy. "
            "Look well into yourself. There is a source of strength which will always spring up if you will always look. "
            "It is not death that a man should fear, but he should fear never beginning to live. "
            "Dwell on the beauty of life. Watch the stars, and see yourself running with them."
        ),
        "image": (
            "Marcus Aurelius as a contemplative Roman emperor in a quiet marble colonnade at dawn mist, "
            "serene, soft window light, photoreal cinematic, intimate portrait, palette of warm stone and sky blue"
        ),
        "video": (
            "slow push-in on stoic emperor in colonnade, morning light shifting, gentle dust motes, "
            "calm breathing stillness, cinematic"
        ),
    },
    "epictetus": {
        "name": "Epictetus",
        "teaching": (
            "Make the best use of what is in your power, and take the rest as it happens. "
            "Some things are up to us, and some things are not up to us. "
            "It's not what happens to you, but how you react to it that matters. "
            "First say to yourself what you would be, and then do what you have to do. "
            "No person is free who is not master of themselves. "
            "Wealth consists not in having great possessions, but in having few wants. "
            "If you want to improve, be content to be thought foolish and stupid. "
            "Don't explain your philosophy. Embody it. "
            "He is a wise man who does not grieve for the things which he has not, "
            "but rejoices for those which he has. "
            "Circumstances do not make the man, they only reveal him to himself. "
            "Freedom is the only worthy goal in life. It is won by disregarding things that lie beyond our control."
        ),
        "image": (
            "ancient stoic teacher Epictetus in a simple stone room with a single oil lamp, "
            "students' silhouettes, warm candlelight, oil painting, intimate close-up, palette of amber and charcoal"
        ),
        "video": (
            "lamp flame flicker, slow camera drift around teacher figure, contemplative mood, "
            "soft shadows moving, cinematic documentary feel"
        ),
    },
    "laozi": {
        "name": "Laozi",
        "teaching": (
            "The Tao that can be told is not the eternal Tao. The name that can be named is not the eternal name. "
            "Nature does not hurry, yet everything is accomplished. "
            "When I let go of what I am, I become what I might be. "
            "A good traveler has no fixed plans, and is not intent on arriving. "
            "Knowing others is intelligence; knowing yourself is true wisdom. "
            "Mastering others is strength; mastering yourself is true power. "
            "The softest things in the world overcome the hardest things in the world. "
            "Be content with what you have; rejoice in the way things are. "
            "When you realize there is nothing lacking, the whole world belongs to you. "
            "Do the difficult things while they are easy, and do the great things while they are small. "
            "A journey of a thousand miles begins with a single step."
        ),
        "image": (
            "Laozi as an old sage walking a mountain path in dawn mist among pine trees, "
            "serene, ink wash sumi-e meets photoreal, wide landscape, palette of soft jade and mist grey"
        ),
        "video": (
            "mist drifting through pines, sage walking slowly along path, gentle wind in robes, "
            "aerial pull-back revealing mountains, serene"
        ),
    },
}


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 180):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body.decode()) if body else {}


def http_form(url: str, fields: dict[str, str], timeout: float = 60) -> dict:
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body.decode()) if body else {}


def compose(*args: str) -> int:
    return subprocess.run(
        ["docker", "compose", *args], cwd=str(ROOT), capture_output=True, text=True
    ).returncode


def wait_ok(url: str, tries: int = 40, sleep_s: float = 3.0) -> bool:
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(sleep_s)
    return False


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
    ).strip()
    return float(out or 0)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def gen_tts(text: str, out_wav: Path) -> float:
    """Return duration seconds."""
    # Prefer direct TTS worker; also try Slopfinity proxy
    body = None
    for url in (f"{TTS}/tts", f"{SLOP}/tts"):
        try:
            body = http_json(
                "POST", url,
                {"text": text, "voice": "af_heart", "engine": "kokoro"},
                timeout=300,
            )
            break
        except Exception as e:
            print(f"  tts try {url} failed: {e}", flush=True)
    if not body or not body.get("ok", True):
        raise RuntimeError(f"tts failed: {body}")

    # Resolve file from JSON envelope
    src = None
    for key in ("audio_path", "path", "file"):
        p = body.get(key)
        if p and Path(str(p)).is_file():
            src = Path(str(p))
            break
    url = body.get("url") or body.get("audio_path") or ""
    leaf = ""
    if isinstance(url, str) and "/files/" in url:
        leaf = url.split("/files/")[-1]
    if not src and leaf:
        for base in (EXP, EXP / "tts", ROOT / "workspace", WORK):
            cand = base / leaf if not leaf.startswith("tts/") else base / leaf
            if cand.is_file():
                src = cand
                break
            cand2 = base / Path(leaf).name
            if cand2.is_file():
                src = cand2
                break
    if not src:
        # harvest from container
        harvest = WORK / "tts_harvest"
        harvest.mkdir(exist_ok=True)
        subprocess.run(
            ["docker", "cp", "strix-halo-qwen-tts:/tmp/slopfinity-tts/.", str(harvest) + "/"],
            capture_output=True,
            timeout=60,
        )
        cands = sorted(harvest.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if cands:
            src = cands[-1]
    if not src or not src.is_file():
        raise RuntimeError(f"tts file not found for {body}")
    out_wav.write_bytes(src.read_bytes())
    # also publish under EXP/tts for Slopfinity
    pub = EXP / "tts" / out_wav.name
    pub.parent.mkdir(exist_ok=True)
    try:
        shutil.copy2(out_wav, pub)
    except Exception:
        pass
    dur = ffprobe_duration(out_wav)
    print(f"  tts {out_wav.name} duration={dur:.1f}s words={len(text.split())}", flush=True)
    if dur < 25:
        print("  warning: TTS shorter than 30s target; teaching may need more words", flush=True)
    return dur


def gen_mage(prompt: str, out_png: Path, seed: int = 42) -> Path:
    subprocess.run(["docker", "stop", "strix-halo-comfyui", "strix-halo-qwen-image"], capture_output=True)
    compose("--profile", "mage-image", "up", "-d", "mage-image-service")
    # wait loaded
    for _ in range(50):
        try:
            h = http_json("GET", f"{MAGE}/health", timeout=5)
            if h.get("loaded") or h.get("ok"):
                break
        except Exception:
            pass
        time.sleep(3)
    exp_name = out_png.name
    container_out = f"/opt/ComfyUI/output/experiments/{exp_name}"
    body = http_json(
        "POST",
        f"{MAGE}/api/generate",
        {
            "prompt": prompt,
            "width": 768,
            "height": 768,
            "steps": 4,
            "cfg": 1.0,
            "seed": seed,
            "model": "mage-turbo",
            "out": container_out,
        },
        timeout=180,
    )
    if not body.get("ok"):
        raise RuntimeError(f"mage failed: {body}")
    host = EXP / exp_name
    if not host.is_file():
        cpath = str(body.get("path") or "")
        mapped = cpath.replace("/opt/ComfyUI/output/", str(ROOT / "comfy-outputs") + "/")
        if Path(mapped).is_file():
            shutil.copy2(mapped, out_png)
        else:
            raise RuntimeError(f"mage output missing: {body}")
    else:
        shutil.copy2(host, out_png)
        if host.resolve() != out_png.resolve():
            pass
    # ensure in EXP with sidecar
    dest = EXP / out_png.name
    if not dest.is_file():
        shutil.copy2(out_png, dest)
    (Path(str(dest) + ".json")).write_text(json.dumps({
        "prompt": prompt,
        "model": "mage-turbo",
        "image_model": "mage-turbo",
        "kind": "image",
        "via": "philosopher_slop_pipeline",
    }, indent=2) + "\n")
    print(f"  mage {dest.name} size={dest.stat().st_size}", flush=True)
    return dest


def gen_i2v(still: Path, prompt: str, out_mp4: Path) -> Path:
    subprocess.run(["docker", "stop", "strix-halo-mage-image", "strix-halo-qwen-image"], capture_output=True)
    compose("--profile", "comfyui", "up", "-d", "comfyui-service")
    if not wait_ok(f"{COMFY}/system_stats", tries=40):
        raise RuntimeError("comfy not ready")
    env = os.environ.copy()
    env["SLOPFINITY_COMFY_URL"] = COMFY
    env["COMFY_OUTPUTS"] = str(ROOT / "comfy-outputs")
    env["COMFY_INPUT"] = str(ROOT / "comfy-input")
    frames = int(os.environ.get("PHILO_VIDEO_FRAMES", "49"))
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ltx_launcher.py"),
        "--mode", "video",
        "--prompt", prompt[:220],
        "--image", str(still),
        "--out", str(out_mp4),
        "--width", "640",
        "--height", "384",
        "--frames", str(frames),
        "--timeout", "1200",
    ]
    log = WORK / f"{out_mp4.stem}.log"
    with log.open("w") as lf:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not out_mp4.is_file() or out_mp4.stat().st_size < 5000:
        raise RuntimeError(f"i2v failed rc={r.returncode} log={log}")
    print(f"  video {out_mp4.name} size={out_mp4.stat().st_size} dur={ffprobe_duration(out_mp4):.2f}s", flush=True)
    return out_mp4


def stitch(video: Path, audio: Path, out_final: Path, audio_dur: float) -> Path:
    """Loop video to match TTS length, mux audio → FINAL mp4 in EXP_DIR."""
    # re-encode looped video + audio, cut to audio duration
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(video),
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{audio_dur:.3f}",
        "-shortest",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_final),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_final.is_file():
        raise RuntimeError(f"ffmpeg stitch failed: {(r.stderr or '')[-400:]}")
    print(f"  FINAL {out_final.name} dur={ffprobe_duration(out_final):.1f}s size={out_final.stat().st_size}", flush=True)
    return out_final


def configure_slopfinity() -> None:
    cfg = {
        "base_model": "mage",
        "video_model": "ltx-2.3",
        "audio_model": "none",
        "tts_model": "kokoro",
        "tts_voice": "af_heart",
        "upscale_model": "none",
        "tier": "low",
        "frames": 49,
        "chains": 1,
    }
    try:
        http_json("POST", f"{SLOP}/config", cfg, timeout=15)
        print("  slopfinity config updated:", cfg, flush=True)
    except Exception as e:
        print(f"  config warn: {e}", flush=True)
    try:
        http_json("POST", f"{SLOP}/coordinator/start", timeout=30)
        print("  coordinator started", flush=True)
    except Exception as e:
        print(f"  coordinator warn: {e}", flush=True)


def inject_job(ph: dict) -> dict:
    stages = {
        "image": ph["image"],
        "video": ph["video"],
        "music": "",
        "tts": ph["teaching"],
    }
    return http_form(
        f"{SLOP}/inject",
        {
            "prompt": f"{ph['name']}: {ph['teaching'][:200]}",
            "priority": "0",
            "stage_prompts": json.dumps(stages),
            "fast_track": "0",
        },
        timeout=30,
    )


def write_sidecar(final: Path, meta: dict) -> None:
    Path(str(final) + ".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def run_one(key: str, also_inject: bool = False) -> dict:
    ph = PHILOSOPHERS[key]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"philosopher_{slug(key)}_{stamp}"
    print(f"\n=== {ph['name']} ===", flush=True)

    wav = WORK / f"{base}.wav"
    png = WORK / f"{base}_mage.png"
    # also publish image name into EXP
    png_exp = EXP / f"marathon_{base}_mage.png"
    vid = WORK / f"{base}_i2v.mp4"
    final = EXP / f"FINAL_{base}.mp4"

    dur = gen_tts(ph["teaching"], wav)
    # extend text if too short
    if dur < 30:
        extended = ph["teaching"] + " " + ph["teaching"]
        print("  extending teaching for longer TTS…", flush=True)
        dur = gen_tts(extended, wav)
        teaching_used = extended
    else:
        teaching_used = ph["teaching"]

    still = gen_mage(ph["image"], png_exp, seed=hash(key) % 100000)
    shutil.copy2(still, png)
    raw = gen_i2v(still, ph["video"], vid)
    stitch(raw, wav, final, dur)

    meta = {
        "prompt": ph["image"],
        "tts_text": teaching_used,
        "video_prompt": ph["video"],
        "philosopher": ph["name"],
        "model": "mage-turbo+ltx-2.3+kokoro",
        "image_model": "mage-turbo",
        "video_model": "ltx-2.3",
        "tts_model": "kokoro",
        "audio_model": None,
        "kind": "video",
        "duration_s": ffprobe_duration(final),
        "tts_duration_s": dur,
        "seed_image": str(still),
        "via": "philosopher_slop_pipeline",
        "final": True,
        "workflow": ["tts", "mage_image", "ltx_i2v", "ffmpeg_stitch"],
    }
    write_sidecar(final, meta)

    inject_resp = None
    if also_inject:
        try:
            inject_resp = inject_job({**ph, "teaching": teaching_used})
            print(f"  inject: {inject_resp}", flush=True)
        except Exception as e:
            print(f"  inject failed: {e}", flush=True)

    return {
        "philosopher": ph["name"],
        "ok": final.is_file(),
        "final": final.name,
        "url": f"/files/{final.name}",
        "duration_s": meta["duration_s"],
        "tts_duration_s": dur,
        "image": still.name,
        "inject": inject_resp,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--philosopher", default="marcus_aurelius", choices=list(PHILOSOPHERS))
    ap.add_argument("--all", action="store_true", help="run all philosophers")
    ap.add_argument("--also-inject", action="store_true", help="also queue via Slopfinity /inject")
    args = ap.parse_args()

    print("Configuring Slopfinity for mage + ltx + kokoro (no music/upscale)…", flush=True)
    configure_slopfinity()

    keys = list(PHILOSOPHERS) if args.all else [args.philosopher]
    results = []
    for k in keys:
        try:
            results.append(run_one(k, also_inject=args.also_inject))
        except Exception as e:
            print(f"FAILED {k}: {e}", flush=True)
            results.append({"philosopher": k, "ok": False, "error": str(e)})

    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "note": "Intermediates assets OFF → only FINAL_*.mp4 visible in gallery",
    }
    (EXP / "philosopher_pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if any(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
