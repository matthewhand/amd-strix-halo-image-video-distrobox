#!/usr/bin/env python3
"""8-hour multi-modal generation marathon via Slopfinity (+ Mage/TTS/music/video).

Primary path: Slopfinity dashboard endpoints on :9099
  - POST /services/{id}/warm  — start workers through service_registry
  - POST /inject (image_only)  — queue image gens through ImageWorker/coordinator
  - POST /tts                  — TTS proxy
  - GET  /assets /files/*      — gallery visibility (EXP_DIR)

Fallback: direct Mage/TTS HTTP when inject is unavailable, still publishing
every artifact into comfy-outputs/experiments as marathon_* so the dashboard
Live Gallery picks them up.

GPU-aware rotation on single UMA: image bursts + TTS, then music/video handoff.
Append-only manifests under MAGE_MARATHON_SCRATCH (default: ./marathon-run).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("MAGE_MARATHON_SCRATCH", ROOT / "marathon-run"))
DURATION_S = float(os.environ.get("MARATHON_DURATION_S", str(8 * 3600)))
SLOPFINITY_URL = os.environ.get("SLOPFINITY_URL", "http://127.0.0.1:9099").rstrip("/")
VIA_SLOPFINITY = os.environ.get("MARATHON_VIA_SLOPFINITY", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
# Same-prompt multi-model comparison: 3 prompts on model A, then same 3 on model B…
COMPARE_MODE = os.environ.get("MARATHON_COMPARE_MODE", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
COMPARE_BATCH = max(1, int(os.environ.get("MARATHON_COMPARE_BATCH", "3")))
COMPARE_MODELS = [
    m.strip().lower()
    for m in os.environ.get("MARATHON_COMPARE_MODELS", "mage,qwen").split(",")
    if m.strip()
]
MAGE_URL = os.environ.get("MAGE_URL", "http://127.0.0.1:8181")
QWEN_URL = os.environ.get("IMAGE_API_URL", os.environ.get("QWEN_URL", "http://127.0.0.1:8180"))
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:8010")
MUSIC_URL = os.environ.get("MUSIC_URL", "http://127.0.0.1:8011")
COMFY_URL = os.environ.get("SLOPFINITY_COMFY_URL", "http://127.0.0.1:8188")
HOMIE_URL = os.environ.get("HOMIE_URL", "http://127.0.0.1:8192")

# model_id -> (service_id for warm/park, default base URL)
IMAGE_MODEL_SERVICES: dict[str, tuple[str, str]] = {
    "mage": ("mage-image", MAGE_URL),
    "mage-turbo": ("mage-image", MAGE_URL),
    "qwen": ("qwen-image", QWEN_URL),
    "ltx": ("comfyui", COMFY_URL),
    "ltx-2.3": ("comfyui", COMFY_URL),
}
EXP_DIR = Path(
    os.environ.get("SLOPFINITY_EXP_DIR")
    or os.environ.get("SLOPFINITY_STATE_DIR")
    or (ROOT / "comfy-outputs" / "experiments")
)

OUT_IMG = Path(os.environ.get("MARATHON_IMG_DIR", ROOT / "comfy-outputs" / "mage" / "marathon"))
OUT_VID = Path(os.environ.get("MARATHON_VID_DIR", ROOT / "comfy-outputs" / "mage" / "marathon_video"))
OUT_TTS = Path(os.environ.get("MARATHON_TTS_DIR", ROOT / "comfy-outputs" / "experiments" / "tts_marathon"))
OUT_MUSIC = Path(os.environ.get("MARATHON_MUSIC_DIR", ROOT / "comfy-outputs" / "experiments" / "music_marathon"))

for d in (SCRATCH, OUT_IMG, OUT_VID, OUT_TTS, OUT_MUSIC, EXP_DIR):
    d.mkdir(parents=True, exist_ok=True)

MANIFEST = SCRATCH / "marathon_manifest.jsonl"
UPTIME = SCRATCH / "uptime.jsonl"
VISUAL = SCRATCH / "visual_qa.jsonl"
STATE = SCRATCH / "marathon_state.json"

# Expanding prompt bank: large combinatorial axes; diversifying_prompt unlocks more over time
# and avoids recent subject/style repeats so prompts stay increasingly different.
SUBJECTS = [
    "headphones", "ramen bowl", "red fox", "lighthouse", "neon alley",
    "hot air balloon", "steam locomotive", "desert caravan", "underwater ruins",
    "mountain shrine", "coffee mug", "origami crane", "glass greenhouse",
    "samurai armor", "bioluminescent jellyfish", "vintage typewriter",
    "ice palace", "market spice stalls", "robot gardener", "paper lanterns",
    "coral reef diver", "observatory dome", "bamboo forest path", "clay teapot",
    "clockwork owl", "stained glass cathedral", "copper airship", "kintsugi vase",
    "moonlit koi pond", "volcanic forge", "crystal cavern", "skybridge city",
    "abandoned carousel", "library of vines", "mechanical stag", "porcelain mask",
    "solar sail yacht", "frozen waterfall", "spice market camel", "ink-black raven",
    "greenhouse tram", "obsidian throne", "lantern festival boat", "rust desert rover",
    "tea ceremony room", "cloud temple steps", "neon ramen cart", "storm-chaser van",
    "glass violin", "ember phoenix", "subway mural cat", "tidepool starfish",
    "alpine cable car", "silk weaving loom", "midnight bakery window", "harbor tugboat",
    "quantum garden maze", "paper dragon kite", "bronze sundial plaza", "fog ferry dock",
    "coral bone shipwreck", "starlit yurt", "chrome scooter alley", "rainy bookshop ladder",
    "sapphire ice rink", "mushroom village path", "gilded elevator cage", "windmill orchard",
    "deep-sea bathysphere", "calligraphy desk", "train station clock", "cactus flower crown",
]
STYLES = [
    "photoreal", "cinematic teal-orange", "watercolor", "oil painting",
    "anime cel shade", "isometric diorama", "macro photography", "noir B&W",
    "impressionist", "cyberpunk neon", "soft studio catalog", "infrared look",
    "tilt-shift miniature", "ukiyo-e woodblock", "brutalist architecture photo",
    "art nouveau poster", "vaporwave gradient", "charcoal sketch", "gouache illustration",
    "long-exposure light trails", "documentary 35mm film grain", "stained-glass mosaic",
    "claymation stop-motion still", "holographic chrome render", "pastel storybook",
    "high-key fashion editorial", "low-key Rembrandt lighting photo", "pixel-art scene",
    "linocut print", "matte painting concept art", "fisheye street photo",
    "infrared false-color landscape", "polaroid faded edges", "ink wash sumi-e",
    "cross-processed film", "studio product shot", "aerial orthographic map art",
    "risograph print", "baroque oil drama", "synthwave grid horizon",
]
SETTINGS = [
    "at dawn mist", "in heavy rain", "under aurora borealis", "golden hour",
    "moonlit night", "sandstorm haze", "snowy clearing", "crowded festival",
    "abandoned factory", "floating islands", "subway platform", "rooftop garden",
    "inside a crystal cave", "on a glass skybridge", "beside a volcanic crater",
    "at a night market", "in a flooded cathedral", "above cloud forests",
    "on a frozen harbor", "in a neon arcade", "along a bamboo aqueduct",
    "in zero-gravity greenhouse", "under monsoon clouds", "at a desert oasis",
    "inside a museum after hours", "on tidal mudflats", "in a redwood canopy",
    "at an alpine pass", "beside a humming reactor core", "in a rain-soaked alley",
    "on a lantern-lit canal", "inside a clock tower", "on a monorail platform",
    "at a spice caravan camp", "in a bioluminescent swamp", "on a rooftop helipad",
    "inside a stained-glass atrium", "at a foggy pier", "on a mesa overlook",
    "in a winter greenhouse", "beside a waterfall elevator", "in an underground bazaar",
]
MOODS = [
    "serene", "melancholic", "triumphant", "uncanny", "cozy", "epic wide shot",
    "intimate close-up", "documentary candid", "dreamlike", "high energy",
    "quiet solitude", "awe and scale", "playful whimsy", "tense anticipation",
    "nostalgic warmth", "cold clinical precision", "sacred stillness",
    "chaotic carnival energy", "lonely highway mood", "mythic grandeur",
    "tender domestic calm", "industrial grit", "ethereal weightlessness",
    "urgent storm chase", "meditative stillness", "electric nightlife pulse",
]
CAMERAS = [
    "wide", "telephoto", "dutch angle", "overhead top-down", "worm's-eye view",
    "35mm street", "85mm portrait", "panoramic anamorphic", "macro extreme close-up",
    "drone aerial", "through-window framed", "split diopter feel", "shallow bokeh",
]
LIGHTING = [
    "soft window light", "harsh noon sun", "neon rim light", "candlelit glow",
    "overcast diffuse", "golden backlight", "blue hour ambience", "spotlight drama",
    "bioluminescent fill", "lightning flash freeze", "campfire warmth", "clinic fluorescents",
]
MATERIALS = [
    "brushed brass details", "wet asphalt reflections", "frosted glass surfaces",
    "weathered wood grain", "silk fabric folds", "oxidized copper", "polished marble",
    "paper texture edges", "moss-covered stone", "iridescent scales", "matte ceramic glaze",
]
COLOR_PALETTES = [
    "emerald and gold", "magenta and cyan", "sepia monochrome", "pastel mint coral",
    "charcoal and amber", "ice blue silver", "rust orange teal", "lavender dusk tones",
    "forest green ochre", "pure monochrome high contrast",
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
    "Copper airships drift above the glass city.",
    "The koi pond mirrors a sky full of lanterns.",
    "Thunder rolls over the frozen waterfall.",
    "Ink black ravens gather on the cathedral spire.",
    "A mechanical stag steps through the bamboo path.",
    "The midnight bakery window glows with warm bread light.",
    "Beneath the harbor, a coral shipwreck sleeps.",
    "Paper dragons kite above the spice market.",
    "The quantum garden maze rearranges itself at dusk.",
    "A fog ferry leaves the dock without a sound.",
    "In the clock tower, gears keep a patient rhythm.",
    "Storm-chasers park on the mesa as lightning blooms.",
]
MUSIC_TAGS = [
    "lofi chill beats, rainy window, soft piano",
    "epic orchestral, stormy sea, brass and drums",
    "cyberpunk synthwave, night drive, arpeggios",
    "acoustic folk, campfire, warm guitar",
    "ambient drones, deep space, sparse bells",
    "jazz trio, late night bar, brushed snare",
    "taiko drums, bamboo flute, mountain shrine",
    "desert blues slide guitar, wind, hand claps",
    "glitch hop, neon arcade, broken beats",
    "baroque harpsichord, cathedral reverb",
    "tropical house, shoreline waves, steel drum",
    "dark techno, industrial warehouse, pulse",
    "celtic harp, morning fog, soft whistle",
    "cinematic hybrid trailer, rising strings, hits",
    "bossa nova, rainy cafe, nylon guitar",
    "chiptune melody, pixel adventure, 8-bit drums",
    "gamelan metallophones, incense haze, soft gong",
    "nordic folk fiddle, snow forest, frame drum",
    "soulful R&B, midnight city, warm bass",
    "post-rock guitars, vast plains, slow build",
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


def http_form(url: str, fields: dict[str, str], timeout: float = 60) -> dict:
    """POST application/x-www-form-urlencoded (Slopfinity /inject)."""
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return {}
        return json.loads(body.decode())


def write_slopfinity_sidecar(
    dest: Path,
    *,
    prompt: str | None = None,
    model: str | None = None,
    kind: str | None = None,
    extra: dict | None = None,
) -> Path | None:
    """Write ``<filename>.json`` next to an EXP_DIR asset so /asset/{file} returns prompt.

    Slopfinity ``routers/assets.py`` reads ``filename + '.json'`` for the prompt
    shown in the Live Gallery detail pane.
    """
    if not dest:
        return None
    dest = Path(dest)
    if kind is None:
        suf = dest.suffix.lower()
        if suf in (".png", ".jpg", ".jpeg", ".webp"):
            kind = "image"
        elif suf in (".mp4", ".webm", ".mov"):
            kind = "video"
        elif suf in (".wav", ".mp3", ".ogg", ".flac"):
            kind = "audio"
        else:
            kind = "other"
    meta = {
        "prompt": prompt or "",
        "model": model,
        "image_model": model,  # explicit for Slopfinity /asset metadata
        "kind": kind,
    }
    if extra:
        meta.update(extra)
        # Keep image_model in sync if caller only set model
        if extra.get("model") and not extra.get("image_model"):
            meta["image_model"] = extra["model"]
    sidecar = Path(str(dest) + ".json")
    try:
        sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        return sidecar
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "sidecar_write_err", "error": str(e), "dest": str(dest)})
        return None


def publish_to_slopfinity(
    src: Path,
    name: str | None = None,
    *,
    prompt: str | None = None,
    model: str | None = None,
    kind: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Hardlink/copy into EXP_DIR so dashboard /assets and /files see it.

    Also writes a prompt sidecar (``filename.json``) when prompt is provided —
    required for the Live Gallery to show the generation prompt.
    """
    if not src or not Path(src).is_file():
        return {"ok": False, "error": "missing_src"}
    src = Path(src)
    dest_name = name or f"marathon_{src.name}"
    if not dest_name.startswith("marathon_"):
        dest_name = f"marathon_{dest_name}"
    dest = EXP_DIR / dest_name
    try:
        if dest.exists():
            if dest.stat().st_ino == src.stat().st_ino:
                action = "exists"
            else:
                try:
                    dest.unlink()
                except OSError:
                    pass
                try:
                    os.link(src, dest)
                    action = "hardlink"
                except OSError:
                    shutil.copy2(src, dest)
                    action = "copy"
        else:
            try:
                os.link(src, dest)
                action = "hardlink"
            except OSError:
                shutil.copy2(src, dest)
                action = "copy"
        if prompt is not None or model is not None or extra:
            write_slopfinity_sidecar(
                dest, prompt=prompt, model=model, kind=kind, extra=extra,
            )
        return {
            "ok": True,
            "path": str(dest),
            "url": f"/files/{dest.name}",
            "action": action,
            "sidecar": str(dest) + ".json",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def ensure_slopfinity_coordinator() -> bool:
    """Start Phase-4 coordinator if not already running."""
    try:
        st = http_json("GET", f"{SLOPFINITY_URL}/coordinator/status", timeout=5)
        if st.get("running"):
            return True
    except Exception:
        pass
    try:
        st = http_json("POST", f"{SLOPFINITY_URL}/coordinator/start", timeout=30)
        return bool(st.get("ok") or st.get("running"))
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "coordinator_start_err", "error": str(e)})
        return False


def slopfinity_warm(service_id: str) -> bool:
    """Warm a pipeline worker through Slopfinity service_registry."""
    try:
        body = http_json(
            "POST",
            f"{SLOPFINITY_URL}/services/{service_id}/warm",
            timeout=180,
        )
        return bool(body.get("ok"))
    except Exception as e:
        log_jsonl(UPTIME, {
            "ts": iso(), "event": "slopfinity_warm_err",
            "service": service_id, "error": str(e)[:200],
        })
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
    # Prefer Slopfinity warm path (same as ImageWorker / service_registry)
    if VIA_SLOPFINITY and slopfinity_warm("mage-image"):
        for _ in range(40):
            if health(MAGE_URL):
                try:
                    h = http_json("GET", f"{MAGE_URL}/health", timeout=5)
                    if h.get("loaded"):
                        return True
                except Exception:
                    pass
            time.sleep(3)
        if health(MAGE_URL):
            return True
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
    if VIA_SLOPFINITY and slopfinity_warm("qwen-tts"):
        for _ in range(20):
            if health(TTS_URL):
                return True
            time.sleep(2)
        if health(TTS_URL):
            return True
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
    if VIA_SLOPFINITY and slopfinity_warm("heartmula"):
        for _ in range(20):
            if health(MUSIC_URL):
                return True
            time.sleep(2)
        if health(MUSIC_URL):
            return True
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


# Recent picks shared within a process for anti-repeat diversity
_RECENT_SUBJECTS: list[str] = []
_RECENT_STYLES: list[str] = []
_RECENT_SETTINGS: list[str] = []
_RECENT_PROMPTS_HASH: list[str] = []
_RECENT_MAX = 48


def _pick_fresh(rng: random.Random, pool: list[str], recent: list[str], unlock: int) -> str:
    """Pick from unlocked prefix of pool, strongly avoiding recent choices."""
    unlocked = pool[: max(1, min(len(pool), unlock))]
    avoid = set(recent[-min(len(recent), max(6, unlock // 3)) :])
    candidates = [x for x in unlocked if x not in avoid] or unlocked
    choice = rng.choice(candidates)
    recent.append(choice)
    if len(recent) > _RECENT_MAX:
        del recent[: len(recent) - _RECENT_MAX]
    return choice


def diversifying_prompt(tick: int, recent_store: dict | None = None) -> tuple[str, str]:
    """Return (slug, prompt) with always-increasing combinatorial variety.

    Axes unlock over tick so early runs stay simple; later ticks add camera,
    lighting, materials, and color palettes. Anti-repeat lists push each new
    prompt away from the last several dozen subjects/styles/settings.
    """
    rng = random.Random(tick * 9973 + int(time.time() * 1000) % 1_000_000)
    # Grow unlock windows so more of each bank becomes available over time
    n_subj = min(len(SUBJECTS), 8 + tick // 3)
    n_style = min(len(STYLES), 6 + tick // 4)
    n_set = min(len(SETTINGS), 6 + tick // 4)
    n_mood = min(len(MOODS), 5 + tick // 5)
    n_cam = min(len(CAMERAS), 2 + tick // 10)
    n_light = min(len(LIGHTING), 1 + tick // 12)
    n_mat = min(len(MATERIALS), 1 + tick // 15)
    n_pal = min(len(COLOR_PALETTES), 1 + tick // 18)

    recent_subj = _RECENT_SUBJECTS
    recent_style = _RECENT_STYLES
    recent_set = _RECENT_SETTINGS
    if recent_store is not None:
        recent_subj = recent_store.setdefault("recent_subjects", [])
        recent_style = recent_store.setdefault("recent_styles", [])
        recent_set = recent_store.setdefault("recent_settings", [])

    subj = _pick_fresh(rng, SUBJECTS, recent_subj, n_subj)
    style = _pick_fresh(rng, STYLES, recent_style, n_style)
    setting = _pick_fresh(rng, SETTINGS, recent_set, n_set)
    mood = rng.choice(MOODS[:n_mood])

    parts = [f"{subj} {setting}", mood, style]
    # Layer extra axes as tick grows — always more different than earlier ones
    if tick >= 10:
        parts.append(f"{rng.choice(CAMERAS[:n_cam])} framing")
    if tick >= 25:
        parts.append(rng.choice(LIGHTING[:n_light]))
    if tick >= 40:
        parts.append(rng.choice(MATERIALS[:n_mat]))
    if tick >= 55:
        parts.append(f"palette of {rng.choice(COLOR_PALETTES[:n_pal])}")
    if tick >= 15:
        parts.append(f"unique detail seed {tick}-{rng.randint(1000, 9999)}")
    # Force structural novelty every N ticks
    if tick % 7 == 0:
        parts.append(rng.choice([
            "rule of thirds composition",
            "centered symmetry",
            "leading lines toward vanishing point",
            "negative space dominant",
            "layered foreground bokeh",
            "diagonal dynamic composition",
        ]))
    if tick % 11 == 0:
        parts.append(rng.choice([
            "single subject isolation",
            "crowd of secondary figures",
            "no humans, pure environment",
            "tiny human for scale",
            "animal companion present",
        ]))

    prompt = ", ".join(parts)
    # Final anti-dup: if hash of prompt core recently seen, shuffle mood+style
    core = f"{subj}|{setting}|{style}"
    h = hashlib.md5(core.encode()).hexdigest()[:10]
    if h in _RECENT_PROMPTS_HASH:
        style = rng.choice(STYLES[:n_style])
        mood = rng.choice(MOODS[:n_mood])
        parts[1] = mood
        parts[2] = style
        prompt = ", ".join(parts)
        h = hashlib.md5(f"{subj}|{setting}|{style}".encode()).hexdigest()[:10]
    _RECENT_PROMPTS_HASH.append(h)
    if len(_RECENT_PROMPTS_HASH) > _RECENT_MAX * 2:
        del _RECENT_PROMPTS_HASH[: len(_RECENT_PROMPTS_HASH) - _RECENT_MAX * 2]

    slug = f"t{tick:05d}_{subj.replace(' ', '_')[:24]}"
    return slug, prompt


def _finalize_image_row(
    tick: int,
    slug: str,
    prompt: str,
    dest: Path,
    seed: int,
    t0: float,
    via: str,
) -> dict:
    try:
        stats = image_stats(dest)
    except Exception as e:
        return {
            "modality": "image", "model": "mage-turbo", "ok": False,
            "prompt": prompt, "error": f"stats:{e}", "ts": iso(), "tick": tick,
            "via": via,
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
    pub = {}
    if accepted:
        pub = publish_to_slopfinity(
            dest,
            f"marathon_{slug}.png",
            prompt=prompt,
            model="mage-turbo",
            kind="image",
            extra={"tick": tick, "slug": slug, "via": via, "seed": seed},
        )
    row = {
        "modality": "image",
        "model": "mage-turbo",
        "ok": accepted,
        "noise": stats["noise"],
        "prompt": prompt,
        "slug": slug,
        "path": str(dest),
        "slopfinity_url": pub.get("url") if accepted else None,
        "slopfinity_path": pub.get("path") if accepted else None,
        "mean": stats["mean"],
        "std": stats["std"],
        "seed": seed,
        "elapsed_s": round(now() - t0, 2),
        "ts": iso(),
        "tick": tick,
        "via": via,
    }
    log_jsonl(MANIFEST, row)
    return row


def gen_mage_image_via_inject(tick: int, slug: str, prompt: str, seed: int, t0: float) -> dict | None:
    """Queue image_only job on Slopfinity; ImageWorker calls Mage and writes EXP_DIR."""
    ensure_slopfinity_coordinator()
    slopfinity_warm("mage-image")
    # Align pipeline config so inject uses Mage turbo
    try:
        http_json(
            "POST",
            f"{SLOPFINITY_URL}/config",
            {
                "base_model": "mage",
                "video_model": "none",
                "audio_model": "none",
                "tts_model": "none",
                "tier": "low",
            },
            timeout=15,
        )
    except Exception:
        pass
    try:
        inj = http_form(
            f"{SLOPFINITY_URL}/inject",
            {
                "prompt": prompt,
                "priority": "0",
                "image_only": "1",
                "fast_track": "1",
            },
            timeout=30,
        )
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "inject_err", "error": str(e)})
        return None
    if inj.get("status") not in ("ok", None) and inj.get("status") == "blocked":
        log_jsonl(UPTIME, {"ts": iso(), "event": "inject_blocked", "body": inj})
        return None

    # Poll queue for this prompt
    deadline = now() + float(os.environ.get("MARATHON_INJECT_TIMEOUT_S", "240"))
    found = None
    while now() < deadline:
        try:
            q = http_json("GET", f"{SLOPFINITY_URL}/queue/paginated?limit=30", timeout=10)
        except Exception:
            time.sleep(2)
            continue
        items = q.get("items") or q.get("queue") or []
        for it in items:
            if (it.get("prompt") or "") != prompt:
                continue
            stages = it.get("stages") or {}
            img = stages.get("image") or {}
            st = img.get("status") or it.get("status")
            if st in ("done",) or it.get("succeeded") is True:
                found = it
                break
            if st in ("failed", "error") or it.get("succeeded") is False:
                log_jsonl(UPTIME, {
                    "ts": iso(), "event": "inject_image_failed",
                    "error": img.get("error") or it.get("error"),
                    "id": it.get("id"),
                })
                return None
        if found:
            break
        time.sleep(2)

    if not found:
        log_jsonl(UPTIME, {"ts": iso(), "event": "inject_timeout", "prompt": prompt[:80]})
        return None

    stages = found.get("stages") or {}
    img = stages.get("image") or {}
    asset = img.get("asset") or img.get("output") or ""
    # asset_paths may also hold base
    ap = found.get("asset_paths") or {}
    if not asset and isinstance(ap, dict):
        asset = ap.get("base") or ap.get("image") or ""
    src = Path(str(asset)) if asset else None
    if not src or not src.is_file():
        # ImageWorker writes mage_*_base.png into EXP_DIR — pick newest matching
        cands = sorted(EXP_DIR.glob("mage_*_base.png"), key=lambda p: p.stat().st_mtime)
        src = cands[-1] if cands else None
    if not src or not src.is_file():
        return {
            "modality": "image", "model": "mage-turbo", "ok": False,
            "prompt": prompt, "error": "inject_done_but_no_asset",
            "ts": iso(), "tick": tick, "via": "slopfinity_inject",
        }

    dest = OUT_IMG / f"{slug}.png"
    try:
        dest.write_bytes(src.read_bytes())
    except Exception:
        dest = src
    # Also publish under stable marathon_ name
    return _finalize_image_row(tick, slug, prompt, dest, seed, t0, via="slopfinity_inject")


def gen_mage_image_direct(tick: int, slug: str, prompt: str, seed: int, t0: float) -> dict | None:
    """Direct Mage HTTP (ImageWorker contract) + publish into Slopfinity EXP_DIR."""
    if not ensure_mage():
        log_jsonl(UPTIME, {"ts": iso(), "event": "mage_down"})
        return None
    # Write into EXP_DIR via container mount path so /files serves it immediately
    exp_name = f"marathon_{slug}.png"
    container_out = f"/opt/ComfyUI/output/experiments/{exp_name}"
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
                "out": container_out,
            },
            timeout=180,
        )
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "mage_gen_err", "error": str(e)})
        return None
    if not body.get("ok"):
        return {
            "modality": "image", "model": "mage-turbo", "ok": False,
            "prompt": prompt, "error": body, "ts": iso(), "tick": tick,
            "via": "mage_direct",
        }
    # Prefer EXP_DIR result, else map container path
    exp_host = EXP_DIR / exp_name
    cpath = body.get("path") or ""
    host = cpath.replace("/opt/ComfyUI/output/mage", str(ROOT / "comfy-outputs" / "mage"))
    host = host.replace("/opt/ComfyUI/output/", str(ROOT / "comfy-outputs") + "/")
    src = exp_host if exp_host.is_file() else Path(host)
    dest = OUT_IMG / f"{slug}.png"
    try:
        if src.is_file():
            dest.write_bytes(src.read_bytes())
        else:
            dest.write_bytes(Path(host).read_bytes())
    except Exception:
        if src.is_file():
            dest = src
    # _finalize_image_row publishes + writes prompt sidecar for Live Gallery
    return _finalize_image_row(tick, slug, prompt, dest, seed, t0, via="mage_direct_slopfinity_publish")


def gen_mage_image(tick: int, recent_store: dict | None = None) -> dict | None:
    slug, prompt = diversifying_prompt(tick, recent_store=recent_store)
    seed = 10000 + tick
    t0 = now()
    use_inject = os.environ.get("MARATHON_USE_INJECT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if VIA_SLOPFINITY and use_inject:
        # Full dashboard queue path (coordinator ImageWorker). Slower; enable
        # explicitly with MARATHON_USE_INJECT=1.
        try:
            row = gen_mage_image_via_inject(tick, slug, prompt, seed, t0)
            if row and row.get("ok"):
                return row
        except Exception as e:
            log_jsonl(UPTIME, {"ts": iso(), "event": "inject_exception", "error": str(e)})
    # Default: ImageWorker-equivalent Mage HTTP, writing into EXP_DIR so
    # Slopfinity /assets and /files serve the result immediately.
    return gen_mage_image_direct(tick, slug, prompt, seed, t0)


def gen_tts(tick: int) -> dict | None:
    if not ensure_tts():
        return None
    # Rotate lines and inject tick + secondary fragment so every sample differs
    base = TTS_LINES[tick % len(TTS_LINES)]
    alt = TTS_LINES[(tick * 3 + 5) % len(TTS_LINES)]
    text = f"{base} {alt} Marathon spoken sample {tick}."
    t0 = now()
    body: dict = {}
    via = "tts_direct"
    # Prefer Slopfinity TTS proxy (dashboard contract)
    if VIA_SLOPFINITY:
        try:
            body = http_json(
                "POST",
                f"{SLOPFINITY_URL}/tts",
                {"text": text, "voice": "af_heart", "engine": "kokoro"},
                timeout=120,
            )
            via = "slopfinity_tts"
        except Exception as e:
            log_jsonl(UPTIME, {"ts": iso(), "event": "slopfinity_tts_err", "error": str(e)})
            body = {}
    if not body:
        try:
            body = http_json(
                "POST",
                f"{TTS_URL}/tts",
                {"text": text, "voice": "af_heart", "engine": "kokoro"},
                timeout=120,
            )
            via = "tts_direct"
        except Exception as e:
            log_jsonl(UPTIME, {"ts": iso(), "event": "tts_err", "error": str(e)})
            return None
    # Resolve output if path given
    path = body.get("path") or body.get("url") or body.get("audio_path") or ""
    ok = bool(body.get("ok", True)) if isinstance(body, dict) else False
    # Some TTS returns file bytes path under /tmp
    for key in ("path", "out", "file", "audio_path"):
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
    # URL-only responses: try download from Slopfinity /files/
    pub_url = None
    if ok and path:
        p = Path(str(path).replace("/files/", "")) if str(path).startswith("/files/") else Path(str(path))
        if p.is_file():
            pub = publish_to_slopfinity(
                p,
                prompt=text,
                model="kokoro",
                kind="audio",
                extra={"tick": tick, "via": via, "modality": "tts"},
            )
            pub_url = pub.get("url")
        elif str(path).startswith("/files/"):
            pub_url = str(path)
            # Ensure sidecar exists for URL-only EXP_DIR files
            leaf = str(path).split("/files/")[-1]
            exp_p = EXP_DIR / leaf
            if exp_p.is_file():
                write_slopfinity_sidecar(
                    exp_p, prompt=text, model="kokoro", kind="audio",
                    extra={"tick": tick, "via": via, "modality": "tts"},
                )
    elif isinstance(body.get("url"), str) and body["url"].startswith("/files/"):
        pub_url = body["url"]
        ok = True
        path = pub_url
    row = {
        "modality": "tts",
        "model": "kokoro",
        "ok": ok,
        "prompt": text,
        "path": path,
        "slopfinity_url": pub_url,
        "elapsed_s": round(now() - t0, 2),
        "ts": iso(),
        "tick": tick,
        "via": via,
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
    base = MUSIC_TAGS[tick % len(MUSIC_TAGS)]
    twist = MUSIC_TAGS[(tick * 5 + 2) % len(MUSIC_TAGS)].split(",")[0]
    tags = f"{base}, with hint of {twist}, variation {tick}"
    t0 = now()
    try:
        body = http_json(
            "POST",
            f"{MUSIC_URL}/music",
            {"prompt": tags, "duration": 8.0},
            # Keep bounded so music handoff cannot stall the multi-modal loop
            timeout=float(os.environ.get("MARATHON_MUSIC_TIMEOUT_S", "240")),
        )
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "music_err", "error": str(e)[:300]})
        ensure_mage()
        return None
    ok = bool(body.get("ok"))
    url = body.get("url", "")
    path = body.get("path") or url
    pub_url = None
    # Resolve host path if under known mounts
    for key in ("path", "out", "file"):
        cand = body.get(key)
        if cand and Path(str(cand)).is_file():
            pub = publish_to_slopfinity(
                Path(str(cand)),
                prompt=tags,
                model="heartmula",
                kind="audio",
                extra={"tick": tick, "modality": "music"},
            )
            if pub.get("ok"):
                path = pub["path"]
                pub_url = pub["url"]
            break
    if isinstance(url, str) and url.startswith("/files/"):
        pub_url = url
    row = {
        "modality": "music",
        "model": "heartmula",
        "ok": ok,
        "prompt": tags,
        "path": path,
        "slopfinity_url": pub_url,
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
            pub = publish_to_slopfinity(
                out,
                prompt=prompt,
                model="ltx-comfy",
                kind="video",
                extra={"tick": tick, "seed_image": str(still), "modality": "video"},
            )
            row["slopfinity_url"] = pub.get("url")
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


def slopfinity_park(service_id: str) -> bool:
    try:
        body = http_json(
            "POST",
            f"{SLOPFINITY_URL}/services/{service_id}/park",
            timeout=120,
        )
        return bool(body.get("ok"))
    except Exception as e:
        log_jsonl(UPTIME, {
            "ts": iso(), "event": "slopfinity_park_err",
            "service": service_id, "error": str(e)[:200],
        })
        return False


def ensure_image_model(model: str) -> bool:
    """Warm the service for this image model; park exclusive peers first."""
    model = (model or "mage").lower().strip()
    svc_id, base = IMAGE_MODEL_SERVICES.get(model, ("mage-image", MAGE_URL))
    # Park other uma-heavy image services so only one model is warm
    peers = {
        "mage-image": MAGE_URL,
        "qwen-image": QWEN_URL,
        "comfyui": COMFY_URL,
    }
    for peer, _ in peers.items():
        if peer != svc_id:
            # Don't park tts/heartmula here — only exclusive image peers
            if peer in ("mage-image", "qwen-image", "comfyui"):
                if VIA_SLOPFINITY:
                    slopfinity_park(peer)
                else:
                    cname = {
                        "mage-image": "strix-halo-mage-image",
                        "qwen-image": "strix-halo-qwen-image",
                        "comfyui": "strix-halo-comfyui",
                    }.get(peer)
                    if cname:
                        subprocess.run(["docker", "stop", cname], capture_output=True)

    if VIA_SLOPFINITY:
        if slopfinity_warm(svc_id):
            pass
        else:
            # fallback compose
            profile = {
                "mage-image": ("mage-image", "mage-image-service"),
                "qwen-image": ("qwen-image", "qwen-image-service"),
                "comfyui": ("comfyui", "comfyui-service"),
            }.get(svc_id)
            if profile:
                compose("--profile", profile[0], "up", "-d", profile[1])
    else:
        profile = {
            "mage-image": ("mage-image", "mage-image-service"),
            "qwen-image": ("qwen-image", "qwen-image-service"),
            "comfyui": ("comfyui", "comfyui-service"),
        }.get(svc_id)
        if profile:
            compose("--profile", profile[0], "up", "-d", profile[1])

    # health wait
    for _ in range(50):
        if model.startswith("mage"):
            if health(MAGE_URL):
                try:
                    h = http_json("GET", f"{MAGE_URL}/health", timeout=5)
                    if h.get("loaded", True):
                        return True
                except Exception:
                    return True
        elif model.startswith("qwen"):
            # qwen health is often /docs
            try:
                req = urllib.request.Request(f"{QWEN_URL.rstrip('/')}/docs")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                try:
                    req = urllib.request.Request(f"{QWEN_URL.rstrip('/')}/")
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
        elif model.startswith("ltx"):
            if health(COMFY_URL):
                return True
        time.sleep(3)
    log_jsonl(UPTIME, {"ts": iso(), "event": "image_model_not_ready", "model": model})
    return False


def _gen_qwen_image(prompt: str, seed: int, out_hint: Path) -> Path | None:
    """Qwen Image Studio: form POST /api/generate → poll /api/jobs → PNG on disk.

    Studio expects application/x-www-form-urlencoded (not JSON). Results land
    under ``qwen-outputs/jobs/<job_id>/image-*.png``.
    """
    base = QWEN_URL.rstrip("/")
    fields = {
        "prompt": prompt,
        "steps": str(int(os.environ.get("QWEN_STEPS", "8"))),
        "seed": str(int(seed)),
        "num_images": "1",
        "ultra_fast": "true",
        "fast": "false",
        "size": os.environ.get("QWEN_SIZE", "1:1"),
    }
    try:
        body = http_form(f"{base}/api/generate", fields, timeout=60)
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "qwen_gen_err", "error": str(e)})
        return None
    job_id = body.get("job_id") or body.get("id")
    if not job_id:
        log_jsonl(UPTIME, {"ts": iso(), "event": "qwen_no_job", "body": body})
        return None

    def _find_job_png(jid: str) -> Path | None:
        job_dir = ROOT / "qwen-outputs" / "jobs" / jid
        if job_dir.is_dir():
            cands = sorted(job_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
            if cands:
                return cands[-1]
        # container home mapping
        alt = Path.home() / ".qwen-image-studio" / "jobs" / jid
        if alt.is_dir():
            cands = sorted(alt.glob("*.png"), key=lambda p: p.stat().st_mtime)
            if cands:
                return cands[-1]
        return None

    def _job_status(jid: str) -> dict:
        # Prefer bulk jobs map
        try:
            bulk = http_json("GET", f"{base}/api/jobs", timeout=15)
        except Exception:
            bulk = {}
        jobs = {}
        if isinstance(bulk, dict):
            jobs = bulk.get("jobs") or bulk
            if isinstance(jobs, dict) and jid in jobs and isinstance(jobs[jid], dict):
                return jobs[jid]
            # sometimes list
            if isinstance(bulk.get("jobs"), list):
                for j in bulk["jobs"]:
                    if isinstance(j, dict) and j.get("id") == jid or j.get("job_id") == jid:
                        return j
        # on-disk jobs.json (host bind)
        jf = ROOT / "qwen-outputs" / "jobs.json"
        if jf.is_file():
            try:
                data = json.loads(jf.read_text())
                jmap = data.get("jobs") if isinstance(data, dict) else {}
                if isinstance(jmap, dict) and jid in jmap:
                    return jmap[jid]
            except Exception:
                pass
        return {}

    deadline = now() + float(os.environ.get("QWEN_JOB_TIMEOUT_S", "600"))
    while now() < deadline:
        # file may appear before status flips
        png = _find_job_png(job_id)
        if png and png.is_file() and png.stat().st_size > 10000:
            return png
        st = _job_status(job_id)
        status = (st.get("status") or st.get("stage") or "").lower()
        if status in ("completed", "done", "success"):
            png = _find_job_png(job_id)
            if png:
                return png
            # paths in job record
            for key in ("path", "output", "image", "file", "filename", "result_path"):
                v = st.get(key)
                if v and Path(str(v)).is_file():
                    return Path(str(v))
            log_jsonl(UPTIME, {"ts": iso(), "event": "qwen_done_no_path", "job": job_id})
            return None
        if status in ("failed", "error"):
            log_jsonl(UPTIME, {
                "ts": iso(), "event": "qwen_failed",
                "error": st.get("error") or st.get("message"),
                "job": job_id,
            })
            return None
        time.sleep(3)
    log_jsonl(UPTIME, {"ts": iso(), "event": "qwen_timeout", "job": job_id})
    return None


def _gen_mage_to_path(prompt: str, seed: int, exp_name: str) -> Path | None:
    # Caller (compare batch / ensure_image_model) must warm Mage first.
    if not health(MAGE_URL):
        if not ensure_image_model("mage"):
            return None
    container_out = f"/opt/ComfyUI/output/experiments/{exp_name}"
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
                "out": container_out,
            },
            timeout=180,
        )
    except Exception as e:
        log_jsonl(UPTIME, {"ts": iso(), "event": "mage_gen_err", "error": str(e)})
        return None
    if not body.get("ok"):
        return None
    exp_host = EXP_DIR / exp_name
    if exp_host.is_file():
        return exp_host
    cpath = body.get("path") or ""
    host = cpath.replace("/opt/ComfyUI/output/mage", str(ROOT / "comfy-outputs" / "mage"))
    host = host.replace("/opt/ComfyUI/output/", str(ROOT / "comfy-outputs") + "/")
    if host and Path(host).is_file():
        return Path(host)
    return None


def gen_compare_image(
    *,
    model: str,
    prompt: str,
    seed: int,
    tick: int,
    batch_id: int,
    prompt_idx: int,
    slug: str,
) -> dict:
    """Generate one image for a compare batch; publish with peer-friendly name."""
    model = model.lower().strip()
    t0 = now()
    # cmp_b0001_p0_mage_t00680_subject.png
    base_slug = slug if slug.startswith("t") else f"t{tick:05d}_{slug}"
    exp_name = f"cmp_b{batch_id:04d}_p{prompt_idx}_{model}_{base_slug}.png"
    dest = OUT_IMG / exp_name
    src: Path | None = None

    if model.startswith("mage"):
        src = _gen_mage_to_path(prompt, seed, exp_name)
    elif model.startswith("qwen"):
        src = _gen_qwen_image(prompt, seed, dest)
    else:
        row = {
            "modality": "image", "model": model, "ok": False,
            "prompt": prompt, "error": f"unsupported_compare_model:{model}",
            "ts": iso(), "tick": tick, "batch_id": batch_id,
            "prompt_idx": prompt_idx, "via": "compare",
        }
        log_jsonl(MANIFEST, row)
        return row

    if not src or not src.is_file():
        row = {
            "modality": "image", "model": model, "ok": False,
            "prompt": prompt, "error": "no_output",
            "ts": iso(), "tick": tick, "batch_id": batch_id,
            "prompt_idx": prompt_idx, "via": "compare",
        }
        log_jsonl(MANIFEST, row)
        return row

    try:
        dest.write_bytes(src.read_bytes())
    except Exception:
        dest = src

    try:
        stats = image_stats(dest)
    except Exception as e:
        row = {
            "modality": "image", "model": model, "ok": False,
            "prompt": prompt, "error": f"stats:{e}",
            "ts": iso(), "tick": tick, "batch_id": batch_id,
            "prompt_idx": prompt_idx, "via": "compare",
        }
        log_jsonl(MANIFEST, row)
        return row

    accepted = not stats["noise"] and dest.stat().st_size > 20000
    if not accepted:
        reject = OUT_IMG / "rejected"
        reject.mkdir(exist_ok=True)
        try:
            dest.rename(reject / dest.name)
            dest = reject / dest.name
        except Exception:
            pass

    peers = [
        f"cmp_b{batch_id:04d}_p{prompt_idx}_{m}_{base_slug}.png"
        for m in COMPARE_MODELS
    ]
    pub = {}
    if accepted:
        pub = publish_to_slopfinity(
            dest,
            exp_name,
            prompt=prompt,
            model=model,
            kind="image",
            extra={
                "tick": tick,
                "slug": base_slug,
                "batch_id": batch_id,
                "prompt_idx": prompt_idx,
                "compare_group": f"b{batch_id:04d}_p{prompt_idx}",
                "compare_peers": peers,
                "compare_models": COMPARE_MODELS,
                "seed": seed,
                "via": "compare",
            },
        )
    row = {
        "modality": "image",
        "model": model,
        "ok": accepted,
        "noise": stats["noise"],
        "prompt": prompt,
        "slug": base_slug,
        "path": str(dest),
        "slopfinity_url": pub.get("url") if accepted else None,
        "slopfinity_path": pub.get("path") if accepted else None,
        "mean": stats["mean"],
        "std": stats["std"],
        "seed": seed,
        "elapsed_s": round(now() - t0, 2),
        "ts": iso(),
        "tick": tick,
        "batch_id": batch_id,
        "prompt_idx": prompt_idx,
        "compare_group": f"b{batch_id:04d}_p{prompt_idx}",
        "compare_peers": peers,
        "via": "compare",
    }
    log_jsonl(MANIFEST, row)
    return row


def arrange_compare_gallery_order(
    batch_id: int,
    *,
    models: list[str] | None = None,
    n_prompts: int | None = None,
    batch_files: list[dict] | None = None,
) -> int:
    """Set mtimes so Live Gallery (mtime desc) shows like-scenes adjacent.

    Order (newest first): p0_model0, p0_model1, p1_model0, p1_model1, ...
    Also touches sibling sidecars so they don't look "newer" than the PNG.
    """
    models = models or COMPARE_MODELS
    n_prompts = n_prompts if n_prompts is not None else COMPARE_BATCH
    touched = 0
    # Highest mtime first in gallery → assign descending times
    t = time.time()
    step = 0

    def _touch(path: Path, when: float) -> None:
        nonlocal touched
        if not path or not path.is_file():
            return
        os.utime(path, (when, when))
        sc = Path(str(path) + ".json")
        if sc.is_file():
            os.utime(sc, (when, when))
        # grid sidecar if any
        g = Path(str(path) + ".grid.json")
        if g.is_file():
            os.utime(g, (when, when))
        touched += 1

    # Prefer explicit batch_files (has paths), else glob
    ordered_paths: list[Path] = []
    if batch_files:
        for pi in range(n_prompts):
            for model in models:
                hit = next(
                    (
                        f for f in batch_files
                        if f.get("prompt_idx") == pi and f.get("model") == model and f.get("ok")
                    ),
                    None,
                )
                if not hit:
                    continue
                for cand in (
                    hit.get("slopfinity_path"),
                    hit.get("path"),
                    str(EXP_DIR / (hit.get("file") or "")),
                    str(EXP_DIR / f"marathon_{hit.get('file')}") if hit.get("file") else "",
                ):
                    if cand and Path(cand).is_file():
                        ordered_paths.append(Path(cand))
                        break
    else:
        for pi in range(n_prompts):
            for model in models:
                # published as marathon_cmp_... or cmp_...
                patterns = [
                    f"marathon_cmp_b{batch_id:04d}_p{pi}_{model}_*.png",
                    f"cmp_b{batch_id:04d}_p{pi}_{model}_*.png",
                ]
                found = []
                for pat in patterns:
                    found.extend(EXP_DIR.glob(pat))
                    found.extend(OUT_IMG.glob(pat.replace("marathon_", "") if pat.startswith("marathon_") else pat))
                # unique by inode
                seen: set[int] = set()
                for p in found:
                    try:
                        ino = p.stat().st_ino
                    except OSError:
                        continue
                    if ino in seen:
                        continue
                    seen.add(ino)
                    ordered_paths.append(p)

    for i, path in enumerate(ordered_paths):
        _touch(path, t - i)
        # If hardlink pair exists under the other dir, same inode already touched
        # but also touch non-hardlinked copy if present
        alt_names = {path.name}
        if path.name.startswith("marathon_"):
            alt_names.add(path.name[len("marathon_"):])
        else:
            alt_names.add(f"marathon_{path.name}")
        for name in alt_names:
            for base in (EXP_DIR, OUT_IMG):
                alt = base / name
                if alt.is_file() and alt.resolve() != path.resolve():
                    try:
                        if alt.stat().st_ino != path.stat().st_ino:
                            _touch(alt, t - i)
                    except OSError:
                        pass

    # Bump batch index slightly older than last image so it doesn't float above the strip
    for idx_name in (f"cmp_b{batch_id:04d}_index.json", "cmp_latest_index.json"):
        for base in (EXP_DIR, SCRATCH):
            ip = base / idx_name
            if ip.is_file():
                os.utime(ip, (t - len(ordered_paths) - 1, t - len(ordered_paths) - 1))

    log_jsonl(UPTIME, {
        "ts": iso(),
        "event": "compare_gallery_arranged",
        "batch_id": batch_id,
        "touched": touched,
        "order": [p.name for p in ordered_paths[:12]],
    })
    return touched


def run_compare_batch(st: dict, tick: int) -> tuple[int, list[dict]]:
    """Generate COMPARE_BATCH prompts, then each model runs the same prompts.

    Returns (new_tick, good_image_rows).
    """
    batch_id = int(st.get("compare_batch_id", 0)) + 1
    st["compare_batch_id"] = batch_id
    prompts: list[tuple[str, str, int]] = []  # slug, prompt, seed
    for i in range(COMPARE_BATCH):
        slug, prompt = diversifying_prompt(tick + i, recent_store=st)
        seed = 10000 + tick + i  # same seed across models for fair-ish compare
        prompts.append((slug, prompt, seed))

    log_jsonl(UPTIME, {
        "ts": iso(),
        "event": "compare_batch_start",
        "batch_id": batch_id,
        "models": COMPARE_MODELS,
        "n_prompts": len(prompts),
        "prompts": [p[:80] for _, p, _ in prompts],
    })

    good: list[dict] = []
    batch_files: list[dict] = []

    for model in COMPARE_MODELS:
        log_jsonl(UPTIME, {
            "ts": iso(), "event": "compare_model_start",
            "batch_id": batch_id, "model": model,
        })
        if not ensure_image_model(model):
            log_jsonl(UPTIME, {
                "ts": iso(), "event": "compare_model_skip",
                "batch_id": batch_id, "model": model, "reason": "not_ready",
            })
            for prompt_idx, (slug, prompt, seed) in enumerate(prompts):
                row = {
                    "modality": "image", "model": model, "ok": False,
                    "prompt": prompt, "error": "model_not_ready",
                    "ts": iso(), "tick": tick, "batch_id": batch_id,
                    "prompt_idx": prompt_idx, "via": "compare",
                }
                log_jsonl(MANIFEST, row)
                tick += 1
                batch_files.append({
                    "model": model, "prompt_idx": prompt_idx, "prompt": prompt,
                    "seed": seed, "ok": False, "path": None, "slopfinity_url": None,
                    "file": None,
                })
            st["tick"] = tick
            save_state(st)
            continue
        # First image after a model warm can include cold-start load time.
        if model.startswith("qwen"):
            os.environ["QWEN_JOB_TIMEOUT_S"] = os.environ.get(
                "QWEN_FIRST_TIMEOUT_S",
                os.environ.get("QWEN_JOB_TIMEOUT_S", "600"),
            )
        for prompt_idx, (slug, prompt, seed) in enumerate(prompts):
            row = gen_compare_image(
                model=model,
                prompt=prompt,
                seed=seed,
                tick=tick,
                batch_id=batch_id,
                prompt_idx=prompt_idx,
                slug=slug,
            )
            # Subsequent qwen images after first can use shorter timeout
            if model.startswith("qwen") and prompt_idx == 0:
                os.environ["QWEN_JOB_TIMEOUT_S"] = os.environ.get(
                    "QWEN_JOB_TIMEOUT_S_STEADY", "300"
                )
            tick += 1
            batch_files.append({
                "model": model,
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "seed": seed,
                "ok": row.get("ok"),
                "path": row.get("path"),
                "slopfinity_url": row.get("slopfinity_url"),
                "file": Path(row["path"]).name if row.get("path") else None,
            })
            if row.get("ok"):
                good.append({"path": row["path"], "prompt": prompt, "model": model})
            st["tick"] = tick
            save_state(st)

    # Batch index for humans + gallery tooling
    index = {
        "batch_id": batch_id,
        "ts": iso(),
        "models": COMPARE_MODELS,
        "n_prompts": COMPARE_BATCH,
        "prompts": [
            {"idx": i, "slug": s, "prompt": p, "seed": seed}
            for i, (s, p, seed) in enumerate(prompts)
        ],
        "files": batch_files,
        "by_prompt": {
            str(i): {
                m: next(
                    (
                        f.get("slopfinity_url") or f.get("file")
                        for f in batch_files
                        if f.get("prompt_idx") == i and f.get("model") == m
                    ),
                    None,
                )
                for m in COMPARE_MODELS
            }
            for i in range(COMPARE_BATCH)
        },
    }
    idx_path = EXP_DIR / f"cmp_b{batch_id:04d}_index.json"
    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    # Also keep a rolling latest pointer
    (EXP_DIR / "cmp_latest_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    (SCRATCH / "cmp_latest_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    # Gallery is mtime-desc: force same-scene model variants to sit next to each other
    arrange_compare_gallery_order(
        batch_id,
        models=COMPARE_MODELS,
        n_prompts=COMPARE_BATCH,
        batch_files=batch_files,
    )
    log_jsonl(UPTIME, {
        "ts": iso(), "event": "compare_batch_done",
        "batch_id": batch_id,
        "ok": sum(1 for f in batch_files if f.get("ok")),
        "total": len(batch_files),
        "index": str(idx_path),
    })
    return tick, good


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
    force_new = os.environ.get("MARATHON_FORCE_NEW", "").strip() in ("1", "true", "yes")
    if force_new or "started_ts" not in st:
        # Keep tick / good_images for unique filenames and I2V seeds; reset clock
        prev_tick = int(st.get("tick", 0))
        st = {
            "tick": prev_tick,
            "cycle": 0,
            "started_at": iso(),
            "started_ts": now(),
            "good_images": list(st.get("good_images", []))[-40:],
            "recent_subjects": list(st.get("recent_subjects", [])),
            "recent_styles": list(st.get("recent_styles", [])),
            "recent_settings": list(st.get("recent_settings", [])),
            "compare_batch_id": int(st.get("compare_batch_id", 0) or 0),
            "phase": os.environ.get("MARATHON_PHASE", "fresh"),
            "prev_tick_at_restart": prev_tick,
        }
        save_state(st)
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
        "force_new": force_new,
        "phase": st.get("phase"),
    })
    print(
        f"[marathon] duration={DURATION_S}s scratch={SCRATCH} "
        f"force_new={force_new} tick={st.get('tick', 0)} "
        f"via_slopfinity={VIA_SLOPFINITY} slop={SLOPFINITY_URL} exp={EXP_DIR} "
        f"compare={COMPARE_MODE} batch={COMPARE_BATCH} models={COMPARE_MODELS}",
        flush=True,
    )
    if VIA_SLOPFINITY:
        ensure_slopfinity_coordinator()
        slopfinity_warm("qwen-tts")
        if not COMPARE_MODE:
            slopfinity_warm("mage-image")

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
            "compare_mode": COMPARE_MODE,
            "compare_batch_id": st.get("compare_batch_id", 0),
        })

        cycle = int(st.get("cycle", 0))

        # --- Images: same-prompt multi-model compare batches ---
        if COMPARE_MODE and now() < end_ts:
            tick, batch_good = run_compare_batch(st, tick)
            for g in batch_good:
                good_images.append(g)
            good_images = good_images[-40:]
            st["tick"] = tick
            st["good_images"] = good_images
            save_state(st)
        else:
            # Legacy: 2 Mage images per cycle
            for _ in range(2):
                if now() >= end_ts:
                    break
                row = gen_mage_image(tick, recent_store=st)
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

        # --- Music every 5th cycle (exclusive GPU handoff; bounded timeout) ---
        if cycle % 5 == 1 and now() < end_ts:
            gen_music(tick)
            tick += 1
            st["tick"] = tick
            save_state(st)

        # --- Video every 5th cycle (offset) from a good still ---
        if cycle % 5 == 3 and good_images and now() < end_ts:
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
