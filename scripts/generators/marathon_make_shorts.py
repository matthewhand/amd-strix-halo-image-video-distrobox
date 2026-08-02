#!/usr/bin/env python3
"""Turn Mage marathon stills into short FINAL_*.mp4 clips (video + music|tts).

Pipeline per still:
  1. LTX I2V from the still (Comfy)
  2. HeartMuLa music OR Kokoro TTS (alternating)
  3. ffmpeg mux → comfy-outputs/experiments/FINAL_short_{n}_{slug}.mp4
  4. Sidecar with prompt + model metadata for Slopfinity gallery

Usage:
  python3 scripts/marathon_make_shorts.py --n 3
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
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(os.environ.get("SLOPFINITY_EXP_DIR") or ROOT / "comfy-outputs" / "experiments")
OUT_VID = ROOT / "comfy-outputs" / "mage" / "marathon_video" / "shorts"
EXP.mkdir(parents=True, exist_ok=True)
OUT_VID.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from slopfinity.workers import ffmpeg_mux  # noqa: E402


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 120):
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
    import urllib.parse

    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body.decode()) if body else {}


def compose(*args: str) -> int:
    r = subprocess.run(["docker", "compose", *args], cwd=str(ROOT), capture_output=True, text=True)
    return r.returncode


def wait_url(url: str, tries: int = 40, sleep_s: float = 3.0) -> bool:
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(sleep_s)
    return False


def pick_stills(n: int) -> list[Path]:
    # Prefer arranged compare mage stills, then general marathon
    cands = sorted(
        (ROOT / "comfy-outputs/mage/marathon").glob("cmp_b*_p*_mage_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(cands) < n:
        more = sorted(
            (ROOT / "comfy-outputs/mage/marathon").glob("t*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in more:
            if p not in cands:
                cands.append(p)
    return cands[:n]


def prompt_for_still(still: Path) -> str:
    # sidecar next to published exp file
    for name in (f"marathon_{still.name}", still.name):
        sc = EXP / f"{name}.json"
        if sc.is_file():
            try:
                return json.loads(sc.read_text()).get("prompt") or still.stem
            except Exception:
                pass
    sc2 = Path(str(still) + ".json")
    if sc2.is_file():
        try:
            return json.loads(sc2.read_text()).get("prompt") or still.stem
        except Exception:
            pass
    return still.stem.replace("_", " ")


def slugify(s: str, max_len: int = 32) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return (s[:max_len] or "short")


def gen_video(still: Path, prompt: str, out_mp4: Path) -> bool:
    # warm comfy, park mage/qwen
    subprocess.run(["docker", "stop", "strix-halo-mage-image"], capture_output=True)
    subprocess.run(["docker", "stop", "strix-halo-qwen-image"], capture_output=True)
    compose("--profile", "comfyui", "up", "-d", "comfyui-service")
    if not wait_url("http://127.0.0.1:8188/system_stats", tries=40):
        print("comfy not ready", flush=True)
        return False
    env = os.environ.copy()
    env["SLOPFINITY_COMFY_URL"] = "http://127.0.0.1:8188"
    env["COMFY_OUTPUTS"] = str(ROOT / "comfy-outputs")
    env["COMFY_INPUT"] = str(ROOT / "comfy-input")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ltx_launcher.py"),
        "--mode", "video",
        "--prompt", prompt[:200],
        "--image", str(still),
        "--out", str(out_mp4),
        "--width", "640",
        "--height", "384",
        "--frames", "25",
        "--timeout", "1200",
    ]
    log = OUT_VID / f"{out_mp4.stem}.log"
    with log.open("w") as lf:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and out_mp4.is_file() and out_mp4.stat().st_size > 5000
    print(f"  video rc={r.returncode} ok={ok} size={out_mp4.stat().st_size if out_mp4.exists() else 0}", flush=True)
    return ok


def gen_music(prompt: str, out_wav: Path) -> bool:
    compose("--profile", "heartmula", "up", "-d", "heartmula-service")
    if not wait_url("http://127.0.0.1:8011/health", tries=20):
        return False
    tags = f"cinematic short film score, {prompt[:80]}, soft pulse"
    try:
        body = http_json(
            "POST",
            "http://127.0.0.1:8011/music",
            {"prompt": tags, "duration": 8.0},
            timeout=300,
        )
    except Exception as e:
        print(f"  music err {e}", flush=True)
        return False
    if not body.get("ok"):
        print(f"  music fail {body}", flush=True)
        return False
    # resolve path/url
    for key in ("path", "out", "file"):
        p = body.get(key)
        if p and Path(str(p)).is_file():
            out_wav.write_bytes(Path(str(p)).read_bytes())
            return out_wav.stat().st_size > 1000
    url = body.get("url") or ""
    if url.startswith("/files/"):
        host = EXP / url.split("/files/")[-1]
        if host.is_file():
            out_wav.write_bytes(host.read_bytes())
            return True
    # scan music dir
    music_dir = ROOT / "comfy-outputs/experiments/music"
    if music_dir.is_dir():
        cands = sorted(music_dir.glob("hm_*.wav"), key=lambda p: p.stat().st_mtime)
        if cands:
            out_wav.write_bytes(cands[-1].read_bytes())
            return True
    return False


def gen_tts(text: str, out_wav: Path) -> bool:
    try:
        body = http_json(
            "POST",
            "http://127.0.0.1:9099/tts",
            {"text": text, "voice": "af_heart", "engine": "kokoro"},
            timeout=120,
        )
    except Exception:
        try:
            body = http_json(
                "POST",
                "http://127.0.0.1:8010/tts",
                {"text": text, "voice": "af_heart", "engine": "kokoro"},
                timeout=120,
            )
        except Exception as e:
            print(f"  tts err {e}", flush=True)
            return False
    for key in ("path", "out", "file", "audio_path"):
        p = body.get(key)
        if p and Path(str(p)).is_file():
            out_wav.write_bytes(Path(str(p)).read_bytes())
            return out_wav.stat().st_size > 1000
    url = body.get("url") or ""
    if isinstance(url, str) and url.startswith("/files/"):
        host = EXP / url.split("/files/")[-1]
        if host.is_file():
            out_wav.write_bytes(host.read_bytes())
            return True
    # docker harvest
    try:
        subprocess.run(
            ["docker", "cp", "strix-halo-qwen-tts:/tmp/slopfinity-tts/.", str(OUT_VID) + "/"],
            capture_output=True,
            timeout=30,
        )
        cands = sorted(OUT_VID.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if cands:
            out_wav.write_bytes(cands[-1].read_bytes())
            return True
    except Exception:
        pass
    return False


def write_sidecar(final_path: Path, meta: dict) -> None:
    sc = Path(str(final_path) + ".json")
    sc.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of shorts")
    ap.add_argument("--with-tts", action="store_true", help="prefer TTS over music on odd indices")
    args = ap.parse_args()

    stills = pick_stills(args.n)
    if not stills:
        print("no stills found", file=sys.stderr)
        return 1
    print(f"making {len(stills)} shorts from:", flush=True)
    for s in stills:
        print(f"  {s.name}", flush=True)

    results = []
    for i, still in enumerate(stills):
        prompt = prompt_for_still(still)
        slug = slugify(still.stem.replace("cmp_b", "").replace("mage_", ""))
        print(f"\n[{i+1}/{len(stills)}] {still.name}", flush=True)
        print(f"  prompt: {prompt[:100]}", flush=True)

        raw_vid = OUT_VID / f"raw_{i:02d}_{slug}.mp4"
        audio = OUT_VID / f"aud_{i:02d}_{slug}.wav"
        final_name = f"FINAL_short_{i:02d}_{slug}.mp4"
        final_path = EXP / final_name

        if not gen_video(still, prompt, raw_vid):
            results.append({"still": still.name, "ok": False, "error": "video"})
            continue

        use_tts = args.with_tts or (i % 2 == 1)
        audio_ok = False
        audio_kind = "none"
        if use_tts:
            tts_text = f"A short scene: {prompt[:120]}"
            audio_ok = gen_tts(tts_text, audio)
            audio_kind = "tts" if audio_ok else "none"
        if not audio_ok:
            audio_ok = gen_music(prompt, audio)
            audio_kind = "music" if audio_ok else audio_kind

        if audio_ok and audio.is_file():
            ok = ffmpeg_mux.mux(str(raw_vid), str(audio), str(final_path), pad_to_video=True)
            mode = f"video+{audio_kind}"
        else:
            shutil.copy2(raw_vid, final_path)
            ok = final_path.is_file()
            mode = "video-only"

        meta = {
            "prompt": prompt,
            "model": "ltx-2.3+heartmula" if audio_kind == "music" else (
                "ltx-2.3+kokoro" if audio_kind == "tts" else "ltx-2.3"
            ),
            "image_model": "mage",
            "video_model": "ltx-2.3",
            "audio_model": "heartmula" if audio_kind == "music" else (
                "kokoro" if audio_kind == "tts" else None
            ),
            "kind": "video",
            "seed_image": str(still),
            "mode": mode,
            "via": "marathon_make_shorts",
            "final": True,
        }
        write_sidecar(final_path, meta)
        # also copy raw video as intermediate with sidecar (optional)
        print(f"  FINAL {final_name} ok={ok} mode={mode} size={final_path.stat().st_size if final_path.exists() else 0}", flush=True)
        results.append({
            "still": still.name,
            "ok": ok,
            "final": final_name,
            "url": f"/files/{final_name}",
            "mode": mode,
            "prompt": prompt,
        })

    summary = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}
    (EXP / "marathon_shorts_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Toggle Intermediates → assets OFF to see only FINAL_*.mp4", flush=True)
    return 0 if any(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
