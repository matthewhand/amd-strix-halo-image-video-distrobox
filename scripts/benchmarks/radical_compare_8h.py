#!/usr/bin/env python3
"""Compare image models × aspect ratios × LTX video via Slopfinity inject.

For each radical subject+look pair:
  - pick an aspect ratio (portrait / landscape / ultrawide / square / …)
  - for each base image model (mage, qwen, ltx-2.3):
      inject image + video (LTX i2v) with audio/tts/upscale off
    sequenced for single-GPU UMA (warm image model, then comfy for video)

Content rules (hard — category purity):
  - NSFW is allowed as SEPARATE lanes: gore OR erotic OR other.
  - NEVER mix gore/viscera with nudity/sexual content in one generation.
  - Subjects, aesthetics, motion, and fences are lane-private (no shared bank).
  - Positive wording only in image prompts (do not name banned content to forbid it).
  - Erotic: mature adult humans only, stated positively; no body-horror language.
  - Gore: clothed/clinical violence only; no sensual or nude framing.
  - Other: surreal/political; no gore spectacle, no nudity.

Queue title is human-only; stage prompts are clean art/motion text (no seq numbers).
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = os.environ.get("SLOPFINITY_URL", "http://127.0.0.1:9099")
_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "COMPARE_OUT",
    str(_ROOT / "marathon-run" / "radical_8h"),
))
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "manifest.jsonl"
STATE = OUT / "state.json"
PID_FILE = OUT / "compare.pid"

DURATION_S = float(os.environ.get("COMPARE_DURATION_S", str(8 * 3600)))
POLL_S = float(os.environ.get("COMPARE_POLL_S", "8"))
# Image-only jobs finish in ~1–3 min; full i2v can take 20–45+ min.
JOB_TIMEOUT_S = float(os.environ.get("COMPARE_JOB_TIMEOUT_S", "900"))
# Default IMAGE-ONLY for volume. Set COMPARE_IMAGE_ONLY=0 to restore per-still LTX.
IMAGE_ONLY = os.environ.get("COMPARE_IMAGE_ONLY", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
# Optional: animate every Nth successful still (0 = never). Only when IMAGE_ONLY.
VIDEO_EVERY_N = int(os.environ.get("COMPARE_VIDEO_EVERY_N", "0"))
VIDEO_MODEL = os.environ.get("COMPARE_VIDEO_MODEL", "ltx-2.3")
# Drop chronically-422 qwen by default for throughput; override via env.
MODELS = [
    m.strip()
    for m in os.environ.get("COMPARE_MODELS", "mage,ltx-2.3").split(",")
    if m.strip()
]
# Fixed prompt set size: generate N unique prompts, then run ALL of them on
# model A (warm once), then the SAME N on model B, etc. Default 10.
PROMPT_BATCH = max(1, int(os.environ.get("COMPARE_PROMPT_BATCH", "10")))
# After a set finishes stills, animate the best image per prompt (~10s LTX).
ANIMATE_BEST = os.environ.get("COMPARE_ANIMATE_BEST", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
# Prefer this model when both stills succeed and sizes are close (tie-break).
ANIMATE_PREFER = (os.environ.get("COMPARE_ANIMATE_PREFER", "mage") or "mage").strip().lower()
VIDEO_FRAMES = max(25, min(241, int(os.environ.get("COMPARE_VIDEO_FRAMES", "241"))))
VIDEO_FPS = max(8, min(30, int(os.environ.get("COMPARE_VIDEO_FPS", "24"))))
# Wall budget for one 10s i2v (241 frames can take 20–40+ min on UMA).
VIDEO_TIMEOUT_S = float(os.environ.get("COMPARE_VIDEO_TIMEOUT_S", "3600"))
EXP_DIR = Path(
    os.environ.get("SLOPFINITY_EXP_DIR")
    or os.environ.get("SLOPFINITY_STATE_DIR")
    or (_ROOT / "comfy-outputs" / "experiments")
)
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Motion lines are lane-private (erotic must not inherit "violence" energy wording).
MOTION_GORE = [
    "slow clinical camera drift, blood mist shifting under fluorescent buzz",
    "steady push-in on industrial violence, sparks and spray settling",
    "cold orbit, steam rising from steel, tension held without sensual framing",
]
MOTION_EROTIC = [
    "slow intimate camera drift, soft breath and fabric motion, warm haze",
    "gentle push-in, skin highlights shifting, tender living motion",
    "parallax of silk and light, sensual micro-movement, no violence",
]
MOTION_OTHER = [
    "slow drifting camera, subtle living motion, atmospheric haze shifting",
    "gentle push-in, elements settling, cinematic micro-movement",
    "parallax drift, light flickering, soft secondary motion in fabrics",
]

# (name, width, height) — all multiples of 16
RATIOS = [
    ("square", 1024, 1024),
    ("portrait_3_4", 768, 1024),
    ("portrait_9_16", 768, 1344),
    ("landscape_4_3", 1024, 768),
    ("landscape_16_9", 1280, 720),
    ("ultrawide_21_9", 1344, 576),
    ("cinema_2_1", 1280, 640),
    ("tall_phone", 720, 1280),
    ("square_sm", 768, 768),
    ("landscape_sm", 896, 512),
]

# Lane-private looks. Gore aesthetics never used for erotic and vice versa.
AESTHETICS_GORE = [
    "crime-scene flash, hospital-green and arterial-red only, CCTV fish-eye, clinical harsh light",
    "military thermal false-color white-hot amber and ice-blue, crosshairs, misaligned cyanotype",
    "biopsy-slide eosin pink and hematoxylin purple as full-scene light, shattered glass shards",
    "damaged VHS pause-frame, tracking tears, bruise-purple and spoiled-milk yellow, industrial grit",
    "wet-plate collodion silver-black, cracked varnish, cold morgue fluorescence",
    "xerox zine grit, misregistered CMYK, punk flyer layout, maximum print damage, no glamour",
    "ulcerated chrome and rust, extreme Dutch angle, anti-beautiful medical clarity",
    "deep-fried compression, oversharpen violence, radioactive saturation, battlefield ugliness",
]

AESTHETICS_EROTIC = [
    "high-key fashion editorial, soft studio key light, satin highlights, clean color grade",
    "warm tungsten bedroom glow, shallow depth of field, creamy bokeh, intimate proximity",
    "liquid chrome and rose-gold reflections, glossy magazine polish, sensual sheen",
    "oil-painting sfumato, baroque drapery, honeyed skin tones, classical beauty lighting",
    "neon pink and violet club light, soft bloom, glamorous night palette, no medical tones",
    "infrared false-color fashion shoot, electric magenta and cyan, stylish not clinical",
    "velvet darkness with single rim light, elegant silhouette, adult glamour still",
    "pastel storybook softness applied to adult sensual scene, painterly, tender",
]

AESTHETICS_OTHER = [
    "brutalist architecture photo, flat grey sky, hard geometry, documentary 35mm grain",
    "isometric diorama, acid lime and burnt magenta game lighting, zero negative space",
    "wet-plate silver-black infected by RGB LED slime, cracked varnish, surreal museum",
    "anaglyph red-cyan misregistered, double exposure of eras, deliberately unresolvable",
    "xerox zine grit, punk flyer layout, bureaucratic paperwork texture",
    "deep-fried meme compression, oversharpen halos, radioactive saturation, absurd internet",
    "matte painting concept art, vast scale, theatrical sky, no visceral detail",
    "synthwave grid horizon, neon lines, clean digital dusk, no body horror",
]

# Category-pure subjects. Gore and erotic NEVER share a generation.
# Gore: clothed/clinical figures only — no pageant, bridal, or sensual framing.
GORE = [
    "a surgeon in scrubs calmly arranging a living torso like a floral centerpiece, organs as roses, blood as varnish, clinical setting, fully clothed medical staff",
    "armored gladiators mid-dismemberment in a neon abattoir cathedral, bones as architecture, clothed adult combatants",
    "a cloaked saint whose halo is spinning circular saws, martyrdom as industrial accident, fully robed adult figure",
    "generals in dress uniform at a banquet eating maps of fictional countries, flesh and cartography, grotesque baroque satire",
    "soldiers in uniform planting flags of peeled skin in a field of broken clocks, anti-war allegory, adult figures, fictional",
    "office workers in business attire commuting through a river of blood, fluorescent banal evil, fully clothed",
    "an executioner in a black coat polishing a guillotine blade like a violin, audience of empty suits",
    "cathedral nave paved with teeth, congregation of flayed statues in ceremonial robes, wet enamel light",
    "a battlefield kitchen where chefs in whites carve medals from bone, industrial stoves, purely utilitarian staging",
    "forensic technicians in coveralls mapping a crime scene of impossible geometry, blood as diagram lines, clinical",
]

# Erotic: mature adults, sensual/explicit — no blood, viscera, wounds, surgery, or body-horror unzipping.
EROTIC = [
    "two consenting adults mid-embrace as a melting marble statue, full adult nudity, classical sculpture softening into oil paint",
    "an adult fashion model nude except for a living city-map of moving traffic lights across skin, strange runway tiles, glamorous",
    "adult lovers kissing under sheets of translucent silk in a flooded library, books as pillows, painterly explicit, mature adults",
    "consensual adult sexual ritual staged as high-fashion editorial, velvet and gold, all mature adults, sensual not medical",
    "adult orgy reinterpreted as a luminous Renaissance painting that is also a glowing motherboard, bodies as soft circuitry, mature adults only",
    "a nude adult woman and a nude adult man in formal black-and-white half-costumes slow-dancing, mature adults only, desire as visual paradox",
    "a nude adult figure reclining as a landscape of highways and soft hills, map-erotic pastoral, mature form, serene",
    "two adult faces pressed together until features gently trade places, intimate and uncanny, sensual, unbroken skin",
    "adult strip club dancers as living neon calligraphy of desire, clothed in light alone, 18+ only, glamorous club energy",
    "consenting adult partners tangled in fiber-optic cable as jewelry, explicit mature bodies, cyber-sensual, no wounds",
]

# Other: radical surreal/political — no viscera spectacle, no nudity.
OTHER = [
    "politicians as transparent aquariums of knife-fish debating with torn flags, parliament as a throat, satire of power, clothed figures",
    "a CEO pinned to a quarterly-earnings chart like a butterfly specimen, worshippers in business-casual eating ticker-tape communion",
    "adult figures in formal vestments performing maintenance on a still-living corporate logo, liturgical circuitry, server-rack cathedral",
    "adult Icarus with mechanical wings of burning hard drives, coolant raining upward, sun a melting corporate eye, clothed",
    "black-hole waiters with constellation faces carving roast stars for unfinished equations in orbital-debris gowns",
    "a city where silence is a capital crime, noise police harvest shouts into neon jars, mayor a choir of mouths",
    "dinner guests with furniture for heads, polite conversation on a breathing wood table, fully dressed",
    "a stapler become a religious relic under wet chrome enamel light, kneeling bishop of static electricity",
    "office workers commuting through impossible sideways gravity, fluorescent hell of bureaucracy, business attire",
    "a border wall of sleeping billionaires' open mouths, migrants as light, no real likenesses, allegorical",
    "two skyscrapers leaning together under inverted weather, steel tenderness, no human bodies required",
    "a funeral where the guest of honor is still giving a TED talk, pie charts as wreaths, audience in mourning clothes",
    "the end of the world drawn by a furious adult outsider artist then hung under museum spotlights",
    "runway model whose gown is living ethernet cables, audience of CRT faces, fully dressed fashion surreal",
    "a subway car that is also a cathedral choir loft, commuters singing quarterly reports as hymns, clothed",
    "a weather map that has become a living ocean inside a boardroom fish tank, executives as divers in suits",
    "adult museum guards bowing to an empty frame that stares back with a thousand tiny security cameras",
    "a library where every book is a locked door and every door is a sentence, mature adult patrons in coats",
    "an airport departure board listing emotions instead of cities, travelers with luggage of fog, clothed",
    "a wedding cake made of tiny skyscrapers collapsing in slow-motion frosting, adult guests in formal wear applauding",
    "clocks growing roots through asphalt, time as an invasive plant in a downtown plaza, no bodies needed",
    "a courtroom where the jury is a single rotating prism, light verdicts, adult lawyers as long shadows",
    "a diner at 3am where the menu is written in extinct languages and the cook is a lighthouse",
    "adult street magicians pulling entire bus routes out of a top hat made of wet newspaper, street clothes",
]

LANES = ["gore", "gore", "erotic", "erotic", "other", "other"]

IRONIC_WORDS = {
    "gore": ["GENTLE", "MERCY", "HYGIENE", "PEACE", "SOFT", "WELCOME", "RELAX", "KINDNESS"],
    "erotic": ["CORPORATE", "COMPLIANCE", "TAX", "HR", "BUDGET", "AGENDA", "SOBER", "MODEST"],
    "other": ["FINE", "NORMAL", "OKAY", "CHILL", "BASIC", "MEH", "SURE", "LOVELY"],
}

# Positive-only lane stamps (do not list forbidden content — that summons it).
# Keep wording free of cross-lane keywords so categories stay pure.
FENCE_GORE = (
    "Category: clinical horror and industrial violence only. "
    "All human figures are fully clothed adults or armored clinical forms. "
    "Mood is medical, industrial, and anti-glamour."
)
FENCE_EROTIC = (
    "Category: glamorous mature adult intimacy only. "
    "Only fully grown adult human bodies, consenting and alive, unbroken skin. "
    "Mood is warm, polished, and desire-forward."
)
FENCE_OTHER = (
    "Category: surreal political and absurd scene only. "
    "Figures are clothed adults or non-human symbols. "
    "Mood is satirical and strange."
)


def req(method, path, data=None, form=None, timeout=180):
    url = BASE + path
    headers, body = {}, None
    if form is not None:
        body = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        code = e.code
    except Exception as e:
        return 0, {"error": str(e)}
    try:
        return code, json.loads(raw.decode()) if raw else {}
    except Exception:
        return code, {"_raw": raw[:400].decode("utf-8", "replace")}


def log(rec: dict) -> None:
    with MANIFEST.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2, default=str))


def model_to_service(model: str) -> str | None:
    m = model.lower()
    if m.startswith("mage"):
        return "mage-image"
    if m.startswith("qwen") or m == "ernie":
        return "qwen-image" if "ernie" not in m else "qwen-image"  # ernie via qwen studio or separate
    if m.startswith("ltx"):
        return "comfyui"
    return None


def _available_gb() -> float:
    code, body = req("GET", "/system/ram", timeout=15)
    if code == 200 and isinstance(body, dict):
        try:
            return float(body.get("available_gb") or 0)
        except (TypeError, ValueError):
            pass
    return 0.0


def _need_gb_for_model(model: str, *, for_video: bool = False) -> float:
    """Conservative peak GB before we allow warm/start."""
    m = (model or "").lower()
    if for_video or m.startswith("ltx"):
        return 48.0  # LTX video peak on UMA
    if m.startswith("qwen"):
        return 28.0
    if m.startswith("mage"):
        return 22.0
    return 30.0


def wait_for_headroom(need_gb: float, safety_gb: float = 12.0, timeout_s: float = 300) -> bool:
    """Block until MemAvailable >= need+safety, reclaiming via /free + peer park."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        avail = _available_gb()
        if avail <= 0:
            return True  # can't measure — don't hard-block forever
        if avail >= need_gb + safety_gb:
            print(
                f"  headroom ok avail={avail:.1f}GB need={need_gb:.1f}+{safety_gb:.1f}",
                flush=True,
            )
            return True
        print(
            f"  headroom LOW avail={avail:.1f}GB need={need_gb:.1f}+{safety_gb:.1f} — reclaim…",
            flush=True,
        )
        # Park heavies and free comfy cache
        for sid in ("mage-image", "qwen-image", "comfyui", "heartmula", "homie-video"):
            req("POST", f"/services/{sid}/park", timeout=120)
        req("POST", "/free", timeout=60)
        time.sleep(8)
    avail = _available_gb()
    print(f"  headroom TIMEOUT avail={avail:.1f}GB", flush=True)
    return avail >= need_gb + safety_gb


def ensure_model(model: str, *, for_video: bool = False) -> bool:
    """Warm one model only after RAM headroom check. Returns False if refused."""
    need = _need_gb_for_model(model, for_video=for_video)
    if not wait_for_headroom(need):
        log({
            "event": "headroom_refused",
            "model": model,
            "need_gb": need,
            "available_gb": _available_gb(),
        })
        return False
    svc = model_to_service(model)
    if svc:
        code, body = req("POST", f"/services/{svc}/warm", timeout=300)
        if code != 200 or (isinstance(body, dict) and body.get("ok") is False):
            print(f"  warm {svc} failed http={code} {body}", flush=True)
            log({"event": "warm_failed", "service": svc, "http": code, "body": body})
            return False
    if for_video or (model or "").lower().startswith("ltx"):
        if not wait_for_headroom(48.0):
            return False
        code, body = req("POST", "/services/comfyui/warm", timeout=300)
        if code != 200 or (isinstance(body, dict) and body.get("ok") is False):
            print(f"  warm comfy failed http={code} {body}", flush=True)
            return False
    _, st = req("GET", "/coordinator/status")
    if not (isinstance(st, dict) and st.get("running")):
        req("POST", "/coordinator/start", timeout=30)
    return True


def _stage_map(it: dict) -> dict:
    stages = it.get("stages") or {}
    return {s: (st or {}).get("status") for s, st in stages.items() if isinstance(st, dict)}


def _is_terminal(it: dict) -> bool:
    sm = _stage_map(it)
    if not sm:
        return it.get("status") in ("done", "failed", "cancelled")
    return all(v in ("done", "failed", "skipped") for v in sm.values() if v)


def wait_for_prompt(
    queue_prompt: str,
    timeout_s: float = 1500,
    *,
    image_only: bool = False,
) -> dict | None:
    """Wait until job is terminal.

    image_only=True: return as soon as the image stage is done/failed
    (do not wait on video/merge — used for high-volume stills).
    """
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < timeout_s:
        _, q = req("GET", "/queue/paginated?limit=40")
        for it in (q.get("items") or []):
            if (it.get("prompt") or "") != queue_prompt:
                continue
            sm = _stage_map(it)
            now = time.time()
            if now - last_print > 20:
                print(f"    … {int(now - t0)}s stages={sm}", flush=True)
                last_print = now
            if image_only:
                img_st = sm.get("image")
                if img_st in ("done", "failed", "skipped"):
                    return it
            elif _is_terminal(it):
                return it
        time.sleep(POLL_S)
    return None


def build_pair(rng: random.Random, lane: str, *, with_quoted_word: bool):
    """Build a category-pure subject×aesthetic pair.

    Hard rule: gore and erotic never share subject, look, motion, or fence text.
    """
    ratio_name, w, h = rng.choice(RATIOS)
    quoted_word = None
    lane = (lane or "other").lower().strip()

    if lane == "gore":
        subject = rng.choice(GORE)
        look = rng.choice(AESTHETICS_GORE)
        motion = rng.choice(MOTION_GORE)
        fence = FENCE_GORE
        title_lane = "gore"
    elif lane == "erotic":
        subject = rng.choice(EROTIC)
        look = rng.choice(AESTHETICS_EROTIC)
        motion = rng.choice(MOTION_EROTIC)
        fence = FENCE_EROTIC
        title_lane = "erotic"
    else:
        subject = rng.choice(OTHER)
        look = rng.choice(AESTHETICS_OTHER)
        motion = rng.choice(MOTION_OTHER)
        fence = FENCE_OTHER
        title_lane = "other"

    if with_quoted_word:
        quoted_word = rng.choice(IRONIC_WORDS.get(title_lane, IRONIC_WORDS["other"]))
        text_clause = (
            f'Include the single word "{quoted_word}" as clearly readable lettering '
            f"somewhere in the scene (sign, banner, neon, carved stone, embroidery, or label). "
            f"Only that word as text — no other words, no numbers, no logos."
        )
    else:
        # Positive: empty frame text, do not list forbidden glyphs.
        text_clause = "the frame contains no written words or digits"

    image_prompt = f"{subject}. Look: {look}. {fence} {text_clause}."
    video_prompt = (
        f"{subject}. {motion}. "
        f"Same look and mood as the still. {fence}"
    )
    return {
        "subject": subject,
        "look": look,
        "ratio_name": ratio_name,
        "width": w,
        "height": h,
        "image_prompt": image_prompt,
        "video_prompt": video_prompt,
        "lane": title_lane,
        "quoted_word": quoted_word,
        "with_quoted_word": with_quoted_word,
    }


def _file_score(path: str | None) -> int:
    """Weak quality proxy: larger successful PNG usually has more detail."""
    if not path or not os.path.isfile(path):
        return -1
    try:
        return int(os.path.getsize(path))
    except OSError:
        return -1


def pick_best_stills(
    results_by_prompt: dict[int, list[dict]],
    *,
    prefer: str = ANIMATE_PREFER,
) -> list[dict]:
    """One winner per prompt_idx among successful stills.

    Score = file size; prefer-model wins ties (and small size gaps <5%).
    """
    winners: list[dict] = []
    for pidx in sorted(results_by_prompt.keys()):
        cands = [r for r in results_by_prompt[pidx] if r.get("ok") and r.get("asset")]
        if not cands:
            continue
        prefer_l = (prefer or "").lower()

        def sort_key(r: dict):
            sc = _file_score(r.get("asset"))
            is_pref = 1 if (r.get("model") or "").lower().startswith(prefer_l[:4]) else 0
            return (sc, is_pref)

        best = max(cands, key=sort_key)
        # If prefer model is within 5% size of best, take prefer.
        pref_cands = [
            r for r in cands
            if (r.get("model") or "").lower().startswith(prefer_l[:4])
        ]
        if pref_cands:
            pref = max(pref_cands, key=lambda r: _file_score(r.get("asset")))
            bsc, psc = _file_score(best.get("asset")), _file_score(pref.get("asset"))
            if bsc > 0 and psc >= 0.95 * bsc:
                best = pref
        winners.append(best)
    return winners


def animate_still(
    *,
    still_path: str,
    video_prompt: str,
    pair: dict,
    set_id: int,
    prompt_idx: int,
    model_src: str,
    salt: str,
) -> dict:
    """~10s LTX i2v from a winning still. Uses slopfinity.ltx_comfy directly."""
    import sys

    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    w, h = int(pair["width"]), int(pair["height"])
    # LTX is happier with even dims already multiples of 16.
    out_name = (
        f"anim_set{set_id:03d}_p{prompt_idx:02d}_{model_src}_{salt}_"
        f"{w}x{h}_{VIDEO_FRAMES}f.mp4"
    )
    out_path = EXP_DIR / out_name
    t0 = time.time()
    rec: dict = {
        "event": "animate",
        "set_id": set_id,
        "prompt_idx": prompt_idx,
        "model_src": model_src,
        "seed_image": still_path,
        "frames": VIDEO_FRAMES,
        "fps": VIDEO_FPS,
        "width": w,
        "height": h,
        "out": str(out_path),
        "lane": pair.get("lane"),
    }
    print(
        f"  ANIMATE p{prompt_idx:02d} from {model_src} "
        f"{VIDEO_FRAMES}f@{VIDEO_FPS}fps (~{VIDEO_FRAMES/VIDEO_FPS:.1f}s) "
        f"← {os.path.basename(still_path)}",
        flush=True,
    )
    if not wait_for_headroom(48.0, safety_gb=12.0, timeout_s=400):
        rec.update({"ok": False, "error": "headroom for LTX"})
        log(rec)
        return rec
    # Warm Comfy only (park image peers).
    for sid in ("mage-image", "qwen-image", "homie-video", "heartmula"):
        req("POST", f"/services/{sid}/park", timeout=120)
    code, body = req("POST", "/services/comfyui/warm", timeout=300)
    if code != 200 or (isinstance(body, dict) and body.get("ok") is False):
        rec.update({"ok": False, "error": f"comfy warm failed {body}"})
        log(rec)
        return rec

    try:
        from slopfinity import ltx_comfy
    except Exception as exc:
        rec.update({"ok": False, "error": f"import ltx_comfy: {exc}"})
        log(rec)
        return rec

    try:
        rc = ltx_comfy.generate_video(
            video_prompt or pair.get("image_prompt") or "",
            str(out_path),
            image_path=still_path,
            width=w,
            height=h,
            frames=VIDEO_FRAMES,
            fps=VIDEO_FPS,
            timeout_s=VIDEO_TIMEOUT_S,
        )
    except Exception as exc:
        rec.update({
            "ok": False,
            "error": str(exc),
            "elapsed_s": round(time.time() - t0, 1),
        })
        log(rec)
        return rec

    ok = rc == 0 and out_path.is_file() and out_path.stat().st_size > 5000
    rec.update({
        "ok": ok,
        "rc": rc,
        "size": out_path.stat().st_size if out_path.is_file() else 0,
        "elapsed_s": round(time.time() - t0, 1),
        "duration_target_s": round(VIDEO_FRAMES / float(VIDEO_FPS), 2),
    })
    if ok:
        try:
            import subprocess
            pr = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(out_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            rec["duration_s"] = float((pr.stdout or "").strip() or 0)
        except Exception:
            rec["duration_s"] = None
        print(
            f"  ANIM OK p{prompt_idx:02d} {out_path.name} "
            f"size={rec['size']} dur={rec.get('duration_s')}",
            flush=True,
        )
    else:
        print(f"  ANIM FAIL p{prompt_idx:02d} rc={rc}", flush=True)
    log(rec)
    return rec


def _run_one_still(
    *,
    model: str,
    pair: dict,
    salt: str,
    prompt_idx: int,
    set_id: int,
    want_video: bool,
    t0: float,
    counters: dict,
) -> dict | None:
    """Inject + wait one still. Updates counters; returns result record or None."""
    w, h = pair["width"], pair["height"]
    ratio_name = pair["ratio_name"]
    qw = pair.get("quoted_word")
    image_prompt = pair["image_prompt"]
    video_prompt = pair["video_prompt"]

    cfg = {
        "base_model": model,
        "video_model": VIDEO_MODEL if want_video else "none",
        "audio_model": "none",
        "tts_model": "none",
        "upscale_model": "none",
        "image_width": w,
        "image_height": h,
        "frames": 241 if want_video else 49,
        "fps": 24,
        "chains": 1,
    }
    req("POST", "/config", data=cfg)
    word_bit = f' · "{qw}"' if qw else ""
    tag = "i2v" if want_video else "img"
    # Shared salt + prompt_idx so same prompt is identifiable across models.
    title = (
        f"set{set_id:03d} p{prompt_idx:02d} · {model} · {pair['lane']} · "
        f"{ratio_name} · {salt} · {tag}{word_bit}"
    )
    form = {
        "prompt": title,
        "priority": "high",
        "stage_prompts": json.dumps(
            {"image": image_prompt, "video": video_prompt}
            if want_video
            else {"image": image_prompt}
        ),
    }
    if not want_video:
        form["image_only"] = "1"
    code_i, _body = req("POST", "/inject", form=form)
    print(
        f"  [{model}] p{prompt_idx:02d}/{PROMPT_BATCH} {w}x{h} "
        f"{'i2v' if want_video else 'img'} http={code_i}",
        flush=True,
    )
    log({
        "event": "inject",
        "set_id": set_id,
        "prompt_idx": prompt_idx,
        "lane": pair["lane"],
        "model": model,
        "image_only": not want_video,
        "ratio": ratio_name,
        "width": w,
        "height": h,
        "quoted_word": qw,
        "title": title,
        "http": code_i,
        "salt": salt,
        "subject_head": pair["subject"][:100],
    })
    wait_to = JOB_TIMEOUT_S if not want_video else max(JOB_TIMEOUT_S, 2400.0)
    it = wait_for_prompt(title, timeout_s=wait_to, image_only=(not want_video))
    by_model = counters["by_model"]
    if not it:
        counters["fail"] += 1
        by_model.setdefault(model, {"ok": 0, "fail": 0})
        by_model[model]["fail"] = by_model[model].get("fail", 0) + 1
        print(f"  TIMEOUT {model} p{prompt_idx:02d}", flush=True)
        log({"event": "timeout", "set_id": set_id, "prompt_idx": prompt_idx,
             "model": model, "title": title})
        return None

    stages = it.get("stages") or {}
    img = stages.get("image") or {}
    vid = stages.get("video") or {}
    img_st = img.get("status")
    vid_st = vid.get("status")
    sm = _stage_map(it)
    asset = img.get("asset")
    ok_img = img_st == "done" and bool(asset)
    if ok_img:
        counters["ok"] += 1
        counters["img_ok_total"] += 1
        by_model.setdefault(model, {"ok": 0, "fail": 0})
        by_model[model]["ok"] = by_model[model].get("ok", 0) + 1
        if want_video and vid_st != "done":
            by_model[model]["video_fail"] = by_model[model].get("video_fail", 0) + 1
    else:
        counters["fail"] += 1
        by_model.setdefault(model, {"ok": 0, "fail": 0})
        by_model[model]["fail"] = by_model[model].get("fail", 0) + 1
    print(
        f"  TERM {model} p{prompt_idx:02d} img={img_st} "
        f"{asset or img.get('error') or ''}",
        flush=True,
    )
    result = {
        "ok": ok_img,
        "set_id": set_id,
        "prompt_idx": prompt_idx,
        "lane": pair["lane"],
        "model": model,
        "asset": asset,
        "bytes": _file_score(asset),
        "pair": pair,
        "salt": salt,
        "title": title,
        "image_status": img_st,
        "image_error": img.get("error"),
    }
    log({
        "event": "terminal",
        **{k: result[k] for k in (
            "set_id", "prompt_idx", "lane", "model", "salt", "title",
            "image_status", "image_error",
        )},
        "image_only": not want_video,
        "ratio": ratio_name,
        "width": w,
        "height": h,
        "quoted_word": qw,
        "stages": sm,
        "image_asset": asset,
        "bytes": result["bytes"],
        "elapsed_s": round(time.time() - t0, 1),
        "ok_images": counters["img_ok_total"],
    })
    rate = counters["img_ok_total"] / max(1.0, (time.time() - t0) / 3600.0)
    save_state({
        "pid": os.getpid(),
        "mode": counters["mode_label"],
        "schedule": f"{PROMPT_BATCH}_prompts_then_next_model_then_animate_best",
        "elapsed_s": round(time.time() - t0, 1),
        "remaining_s": round(counters["deadline"] - time.time(), 1),
        "set_id": set_id,
        "ok": counters["ok"],
        "fail": counters["fail"],
        "ok_images": counters["img_ok_total"],
        "ok_anims": counters.get("ok_anims", 0),
        "images_per_hour": round(rate, 1),
        "eta_images_6h": int(rate * 6),
        "by_model": by_model,
        "by_lane": counters["by_lane"],
        "available_gb": _available_gb(),
        "last": {
            "model": model,
            "prompt_idx": prompt_idx,
            "lane": pair["lane"],
            "ratio": ratio_name,
            "stages": sm,
            "quoted_word": qw,
        },
    })
    time.sleep(0.5)
    return result


def main() -> int:
    PID_FILE.write_text(str(os.getpid()))
    t0 = time.time()
    deadline = t0 + DURATION_S
    rng = random.Random(0xC0A1E + int(t0) % 5003)
    ok = 0
    fail = 0
    by_model: dict[str, dict] = {m: {"ok": 0, "fail": 0} for m in MODELS}
    by_lane: dict[str, int] = {}
    lane_i = 0
    set_id = 0
    img_ok_total = 0

    mode_label = "IMAGE_ONLY" if IMAGE_ONLY else f"image→{VIDEO_MODEL}"
    anim_note = (
        f" then animate best×{PROMPT_BATCH} @ {VIDEO_FRAMES}f/{VIDEO_FPS}fps "
        f"(~{VIDEO_FRAMES/VIDEO_FPS:.1f}s)"
        if ANIMATE_BEST else " (no animate phase)"
    )
    print(
        f"START schedule={PROMPT_BATCH}_prompts×model models={MODELS} "
        f"mode={mode_label}{anim_note} duration_h={DURATION_S/3600:.1f} "
        f"ratios={len(RATIOS)} pure_categories=1",
        flush=True,
    )
    log({
        "event": "start",
        "models": MODELS,
        "prompt_batch": PROMPT_BATCH,
        "schedule": "N prompts on model A, same N on model B, then animate best per prompt",
        "image_only": IMAGE_ONLY,
        "animate_best": ANIMATE_BEST,
        "animate_prefer": ANIMATE_PREFER,
        "video_frames": VIDEO_FRAMES,
        "video_fps": VIDEO_FPS,
        "video_model": VIDEO_MODEL if ANIMATE_BEST else None,
        "ratios": RATIOS,
        "duration_s": DURATION_S,
        "rules": [
            f"warm one model, run {PROMPT_BATCH} stills, then next model on SAME prompts",
            "pick best still per prompt (size + prefer model), then ~10s LTX i2v",
            "NEVER mix gore+nudity in same generation",
            "lane-private subjects + aesthetics + fences",
        ],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    counters = {
        "ok": 0,
        "fail": 0,
        "img_ok_total": 0,
        "ok_anims": 0,
        "fail_anims": 0,
        "by_model": by_model,
        "by_lane": by_lane,
        "mode_label": mode_label,
        "deadline": deadline,
    }

    while time.time() < deadline:
        set_id += 1
        # Build fixed prompt set for this comparison round.
        prompts: list[dict] = []
        for i in range(PROMPT_BATCH):
            lane = LANES[lane_i % len(LANES)]
            lane_i += 1
            with_word = (i % 2 == 1)  # every other in the set
            pair = build_pair(rng, lane, with_quoted_word=with_word)
            salt = "".join(rng.choice("abcdefghijkmnpqrstuvwxyz") for _ in range(5))
            pair["_salt"] = salt
            pair["_prompt_idx"] = i + 1
            prompts.append(pair)
            by_lane[pair["lane"]] = by_lane.get(pair["lane"], 0) + 1

        print(
            f"\n######## SET {set_id} · {PROMPT_BATCH} prompts · "
            f"models={MODELS} · {mode_label} ########",
            flush=True,
        )
        log({
            "event": "prompt_set",
            "set_id": set_id,
            "n": PROMPT_BATCH,
            "lanes": [p["lane"] for p in prompts],
            "ratios": [p["ratio_name"] for p in prompts],
            "salts": [p["_salt"] for p in prompts],
        })
        for pi, p in enumerate(prompts, 1):
            print(
                f"  prompt {pi:02d}/{PROMPT_BATCH} lane={p['lane']} "
                f"{p['ratio_name']} {p['width']}x{p['height']} salt={p['_salt']}"
                + (f' word="{p.get("quoted_word")}"' if p.get("quoted_word") else ""),
                flush=True,
            )

        # Collect stills per prompt_idx across models for best-of picking.
        results_by_prompt: dict[int, list[dict]] = {
            p["_prompt_idx"]: [] for p in prompts
        }

        # For each model: warm once, run ALL prompts, then switch.
        for model in MODELS:
            if time.time() >= deadline:
                break
            want_video = not IMAGE_ONLY
            print(
                f"\n--- SET {set_id} MODEL {model} "
                f"({PROMPT_BATCH} stills, warm once) ---",
                flush=True,
            )
            if not ensure_model(model, for_video=want_video):
                fail += PROMPT_BATCH
                counters["fail"] = fail
                by_model.setdefault(model, {"ok": 0, "fail": 0})
                by_model[model]["fail"] = by_model[model].get("fail", 0) + PROMPT_BATCH
                print(f"  SKIP entire model {model} — headroom/warm failed", flush=True)
                log({
                    "event": "skip_model",
                    "set_id": set_id,
                    "model": model,
                    "available_gb": _available_gb(),
                })
                continue

            for pair in prompts:
                if time.time() >= deadline:
                    break
                rec = _run_one_still(
                    model=model,
                    pair=pair,
                    salt=pair["_salt"],
                    prompt_idx=pair["_prompt_idx"],
                    set_id=set_id,
                    want_video=want_video,
                    t0=t0,
                    counters=counters,
                )
                if rec is not None:
                    results_by_prompt[pair["_prompt_idx"]].append(rec)
                ok = counters["ok"]
                fail = counters["fail"]
                img_ok_total = counters["img_ok_total"]

            # Park before next model so UMA is free (safe mode).
            print(f"  park after {model} (set {set_id} done for this model)", flush=True)
            for sid in ("mage-image", "qwen-image", "comfyui"):
                req("POST", f"/services/{sid}/park", timeout=120)
            req("POST", "/free", timeout=60)
            time.sleep(2)

        # Phase 2: ~10s animations for the best still of each prompt.
        if ANIMATE_BEST and time.time() < deadline:
            winners = pick_best_stills(results_by_prompt, prefer=ANIMATE_PREFER)
            print(
                f"\n=== SET {set_id} ANIMATE BEST "
                f"({len(winners)}/{PROMPT_BATCH} winners, "
                f"{VIDEO_FRAMES}f @ {VIDEO_FPS}fps ≈ {VIDEO_FRAMES/VIDEO_FPS:.1f}s) ===",
                flush=True,
            )
            log({
                "event": "animate_phase",
                "set_id": set_id,
                "n_winners": len(winners),
                "winners": [
                    {
                        "prompt_idx": w["prompt_idx"],
                        "model": w["model"],
                        "asset": w.get("asset"),
                        "bytes": w.get("bytes"),
                        "lane": w.get("lane"),
                    }
                    for w in winners
                ],
            })
            for w in winners:
                if time.time() >= deadline:
                    break
                pair = w["pair"]
                anim = animate_still(
                    still_path=w["asset"],
                    video_prompt=pair.get("video_prompt") or "",
                    pair=pair,
                    set_id=set_id,
                    prompt_idx=w["prompt_idx"],
                    model_src=w["model"],
                    salt=w.get("salt") or pair.get("_salt") or "x",
                )
                if anim.get("ok"):
                    counters["ok_anims"] = counters.get("ok_anims", 0) + 1
                else:
                    counters["fail_anims"] = counters.get("fail_anims", 0) + 1
            # Free Comfy after animate burst.
            for sid in ("mage-image", "qwen-image", "comfyui"):
                req("POST", f"/services/{sid}/park", timeout=120)
            req("POST", "/free", timeout=60)

    summary = {
        "event": "summary",
        "elapsed_s": round(time.time() - t0, 1),
        "sets": set_id,
        "prompt_batch": PROMPT_BATCH,
        "ok": counters["ok"],
        "fail": counters["fail"],
        "ok_images": counters["img_ok_total"],
        "ok_anims": counters.get("ok_anims", 0),
        "fail_anims": counters.get("fail_anims", 0),
        "by_model": by_model,
        "by_lane": by_lane,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(summary)
    save_state(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
