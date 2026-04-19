#!/usr/bin/env python3
"""
All-permutations showcase orchestrator.

Runs 8 jobs sequentially in risk-ascending order:
  canary  — 1x T2V 49f+upscale (hero shot, fast fail if upscale hangs)
  path2_lost_letter      — chained i2v x5 @ 1280x720/193f (narrative)
  path2_midnight_train   — same
  path2_brains_cosmic    — same
  path2_anthology        — 4 standalone i2v @ 1280x720/193f (one per narrative)
  path1_subject_parade   — 6 T2V+upscale hero shots
  path1_ironic_wave      — 10 T2V+upscale satirical vignettes
  path1_tone_wave        — 12 T2V+upscale same base scene x 12 tones

Checkpoint file: $COMFY_OUTPUTS/all_perms_state.json tracks job status
so a mid-run crash is fully resumable. Pass --fresh to wipe state.

Container lifecycle:
  Path 2 = run_chained_wave.py as subprocess (it manages its own container)
  Path 1 = orchestrator starts container once, runs all T2V jobs, exits
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from pipelines import comfy_container, config  # noqa: E402
import upscale_t2v  # noqa: E402

STATE_FILE = os.path.join(config.OUTPUT_DIR, "all_perms_state.json")
EXAMPLE_DIR = os.path.join(ROOT, "tests", "example_prompts")


# --- Path 1 content inventories ---------------------------------------------

HERO_SHOT_PROMPT = (
    "a young woman in vintage clothes steps into a stone lighthouse interior "
    "and stops in stunned silence, the camera pulls back to reveal the "
    "circular room filled wall-to-wall with thousands of identical yellowed "
    "envelopes stacked to the ceiling, each addressed to her in the same "
    "handwriting, golden afternoon light pouring through high windows, "
    "audio is her sharp intake of breath and a swelling orchestral crescendo"
)

SUBJECT_PARADE = [
    ("parade_1_lighthouse", HERO_SHOT_PROMPT),
    ("parade_2_midnight_train",
     "cinematic wide shot of an empty vintage train carriage rushing through "
     "fog at midnight, warm amber sconces flickering, a lone passenger in a "
     "long coat staring out the rain-streaked window, reflection of a second "
     "figure briefly visible in the glass beside hers though the seat is "
     "empty, audio is rhythmic wheel clatter and distant mournful horn"),
    ("parade_3_cosmic_brain",
     "a wide cinematic shot of a glistening pink brain hovering serenely in "
     "a star-filled cosmic void, gentle golden pulse emanating outward, "
     "soft nebulae swirling in the background, camera orbiting slowly, "
     "audio is a deep ambient drone with subtle chimes and a single held "
     "choral note"),
    ("parade_4_dragon_office",
     "mid-air cinematic shot of an office worker clinging to the back of a "
     "fully grown red-scaled dragon flying down a corporate hallway, papers "
     "and cubicle walls exploding outward in their wake, fluorescent lights "
     "raining sparks, terrified coworkers diving for cover, audio is dragon "
     "roar, building alarms and swelling orchestral strings"),
    ("parade_5_neon_gladiator",
     "a gladiator in full bronze armor stands alone in the center of a 1980s "
     "neon-lit arcade, slowly raising a glowing pixelated sword above his "
     "head, CRT monitors flickering around him, joystick buttons illuminated "
     "in pink and cyan, audio is a deep synthwave bass swell and distant "
     "coin-drop chimes"),
    ("parade_6_marshmallow_moon",
     "an astronaut in a pristine white suit plants a tattered flag on the "
     "surface of a moon made entirely of pastel marshmallow, their boot "
     "sinking gently into the spongy ground, earthrise visible in the black "
     "sky, audio is their amplified breathing and the soft crunch of "
     "compressed sugar"),
]

IRONIC_WAVE = [
    ("ironic_1_gym_sloth",
     "a three-toed sloth in pristine athletic wear hangs motionless from "
     "a pull-up bar in a modern fluorescent-lit gym, a personal trainer "
     "next to it shouts 'GIVE ME ONE MORE' while looking very serious, "
     "audio is gym ambience, loud motivational music, and the sloth's "
     "extremely slow contented blink"),
    ("ironic_2_corporate_applause",
     "a packed corporate all-hands meeting where everyone stands and "
     "applauds enthusiastically at a blank whiteboard, the CEO at the front "
     "smiling proudly, a junior employee in the third row glancing nervously "
     "around, audio is thunderous applause, corporate motivational music "
     "swelling"),
    ("ironic_3_meditation_chaos",
     "a serene yoga instructor sits cross-legged on a beach at sunrise "
     "telling viewers to find their inner peace, meanwhile in the "
     "background a crab war breaks out, seagulls dive-bomb a picnic, and a "
     "jet ski explodes, audio is calm flute music over distant chaos"),
    ("ironic_4_self_driving_meeting",
     "four self-driving cars parked in a circle in an empty parking lot, "
     "headlights aimed inward as if holding a serious meeting, one car "
     "flashes its hazards slowly while the others appear to nod by dipping "
     "their hoods slightly, audio is mechanical servo sounds and tense "
     "corporate drum rhythm"),
    ("ironic_5_linkedin_influencer",
     "a man in an aggressively pressed blue suit stands in front of a "
     "whiteboard covered in meaningless arrows, pointing confidently at a "
     "single word 'SYNERGY' while flames silently rise behind him, audio is "
     "TED-talk ambient music with crackling fire gradually overwhelming"),
    ("ironic_6_kids_podcast",
     "two six-year-olds sit at a professional podcasting setup with foam "
     "microphones and soundproofed walls, one nodding thoughtfully while "
     "the other explains something with an extremely serious expression "
     "using a juice box as a prop, audio is adult podcast intro music and "
     "an impressively deep thoughtful hmmmmm"),
    ("ironic_7_billionaire_camping",
     "a billionaire in designer flannel roasts a single marshmallow over a "
     "twenty-foot bonfire maintained by six staff members, a helicopter "
     "idles nearby, he looks into the camera and says 'this is what it's "
     "all about', audio is crackling fire, helicopter rotor thrum, and "
     "crickets"),
    ("ironic_8_ai_therapist",
     "a worried man sits on a leather couch pouring his heart out to a "
     "desktop computer with a cartoon smiley face on the screen, the "
     "smiley face blinks once very slowly, the man nods and wipes a tear, "
     "audio is gentle therapy-office piano and the soft hum of a cooling "
     "fan"),
    ("ironic_9_motivational_rock",
     "a motivational speaker on stage points dramatically at a large rock "
     "sitting on a pedestal and shouts 'BE THIS ROCK' to a stadium crowd "
     "who rise to their feet weeping and nodding, stadium lights strobe, "
     "audio is roaring crowd and swelling inspirational power chords"),
    ("ironic_10_startup_pivot",
     "five startup founders in hoodies stand around a whiteboard covered "
     "in scribbles, one dramatically erases everything with a single swipe "
     "and writes the word 'TOAST', the others nod gravely and clap, audio "
     "is tense strings building to a triumphant cymbal crash"),
]

TONES = [
    "awestruck", "dread-laden", "tender", "absurd", "melancholy", "euphoric",
    "menacing", "whimsical", "grieving", "serene", "tense", "wonder-filled",
]

TONE_BASE = (
    "a young woman in vintage clothes kneels in a dusty sunlit attic among "
    "old steamer trunks, holding up a single yellowed envelope she has "
    "just found, turning it slowly in her hands, dust motes swirling in "
    "the light, the scene rendered in a {tone} mood"
)


def tone_wave_prompts():
    return [(f"tone_{i+1:02d}_{t.split('-')[0]}",
             TONE_BASE.format(tone=t)) for i, t in enumerate(TONES)]


# --- state ------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# --- Path 2 (chained i2v) ---------------------------------------------------

# Ordered best → most-conservative. First success wins; successful config is
# cached as "preferred" for subsequent chain jobs to skip re-probing.
# timeout_s is a per-subprocess wall-clock for a 5-scene chain.
# Empirical observations:
#   - 193f @ 1280x720: ~7 min/step * 8 steps = ~56 min/scene (HW hangs — blacklisted)
#   - 145f @ 1280x720: ~5 min/step * 8 = ~40 min/scene (untested; same-shape risk)
#   -  97f @ 1280x720: ~3 min/step * 8 = ~24 min/scene
#   - 145f @ 1024x576: ~2.5 min/step * 8 = ~20 min/scene (wave 1 baseline)
#   -  97f @  768x432: ~1.5 min/step * 8 = ~12 min/scene
# Plus Qwen phase (~2 min) + model load (~2 min). Per-scene estimates above,
# 5-scene chain: ~5*per_scene + 5 min overhead. Timeout = 2x expected + slack.
CHAIN_CONFIGS = [
    {"label": "hd_full",  "w": 1280, "h": 720, "f": 193, "fps": 24, "timeout_s": 36000},  # 2 hr/scene for retry (base for 5 scenes)
    {"label": "hd_long",  "w": 1280, "h": 720, "f": 145, "fps": 24, "timeout_s": 14400},  # 4 hr
    {"label": "hd_med",   "w": 1280, "h": 720, "f": 97,  "fps": 24, "timeout_s":  9000},  # 2.5 hr
    {"label": "proven",   "w": 1024, "h": 576, "f": 145, "fps": 24, "timeout_s":  7200},  # 2 hr
    {"label": "fallback", "w": 768,  "h": 432, "f": 97,  "fps": 24, "timeout_s":  4800},  # 80 min
]


def _invoke_chain(prompts_file, count, cfg, no_join=False):
    # Scale timeout with chain length (chain_configs timeout_s is for count=5).
    base_timeout = cfg.get("timeout_s", 3000)
    timeout = int(base_timeout * max(count, 1) / 5)
    cmd = [
        sys.executable, os.path.join(ROOT, "tests", "run_chained_wave.py"),
        "--prompts-file", prompts_file,
        "--count", str(count),
        "--width", str(cfg["w"]), "--height", str(cfg["h"]),
        "--frames", str(cfg["f"]), "--fps", str(cfg["fps"]),
    ]
    if no_join:
        cmd.append("--no-join")
    print(f"  config: {cfg['label']}  {cfg['w']}x{cfg['h']}/{cfg['f']}f"
          f"  timeout={timeout}s")
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s — killing subprocess + container")
        # container must be forcibly killed so next config's subprocess
        # gets a clean start; run_chained_wave's container-start is idempotent.
        subprocess.run(["docker", "kill", config.CONTAINER],
                       capture_output=True)
        subprocess.run(["docker", "rm", config.CONTAINER],
                       capture_output=True)
        return False, int(time.time() - t0)
    return rc == 0, int(time.time() - t0)


def _preferred_config(state):
    label = state.get("_preferred_chain_config")
    if not label:
        return None
    for c in CHAIN_CONFIGS:
        if c["label"] == label:
            return c
    return None


def _remember_preferred(state, cfg):
    state["_preferred_chain_config"] = cfg["label"]
    save_state(state)


def _blacklist_config(state, cfg_label):
    """Mark a config as known-broken for the rest of this run."""
    bl = set(state.get("_blacklisted_configs", []))
    bl.add(cfg_label)
    state["_blacklisted_configs"] = sorted(bl)
    save_state(state)


def _is_blacklisted(state, cfg_label):
    return cfg_label in state.get("_blacklisted_configs", [])


def _ollama_unload_all():
    """Free GPU from ollama before video gen; keeps models on disk."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/ps", timeout=2) as r:
            running = json.loads(r.read()).get("models", [])
    except Exception:
        return
    for m in running:
        name = m.get("name")
        if not name:
            continue
        subprocess.run(["ollama", "stop", name], capture_output=True)
    if running:
        print(f"  [ollama] unloaded: {', '.join(m.get('name','?') for m in running)}")


def run_chain_variety(name, prompts_file, state, start_label, count=5):
    """Variety-test mode: try `start_label` first; on failure, fall back through
    remaining CHAIN_CONFIGS in declared order, skipping blacklisted. Does NOT
    cache preferred (so the next chain job is free to test its own assigned
    config). Blacklists a config on failure so it won't be retried across jobs.
    Returns (ok, cfg_that_worked, attempts)."""
    _ollama_unload_all()
    tried = []
    ordered = []
    start_cfg = next((c for c in CHAIN_CONFIGS if c["label"] == start_label), None)
    if start_cfg and not _is_blacklisted(state, start_label):
        ordered.append(start_cfg)
    for c in CHAIN_CONFIGS:
        if c not in ordered and not _is_blacklisted(state, c["label"]):
            ordered.append(c)
    if not ordered:
        print(f"  [{name}] ALL configs blacklisted — giving up")
        return False, None, tried
    for cfg in ordered:
        print(f"\n{'-' * 72}\n  [{name}] trying {cfg['label']} "
              f"({cfg['w']}x{cfg['h']}/{cfg['f']}f)\n{'-' * 72}")
        ok, secs = _invoke_chain(prompts_file, count, cfg)
        tried.append({"label": cfg["label"], "ok": ok, "seconds": secs})
        if ok:
            return True, cfg, tried
        _blacklist_config(state, cfg["label"])
        print(f"  [{name}] {cfg['label']} FAILED in {secs}s; BLACKLISTED, falling back")
    return False, None, tried


def make_chain_job(name, prompts_file, start_label="proven", count=5):
    """Returns a callable for job_plan that pins the chain's starting config
    (variety-test mode). Adaptive fallback still engages on failure."""
    def _job(state):
        ok, cfg, tried = run_chain_variety(
            name, prompts_file, state, start_label=start_label, count=count)
        state[name + "_configs_tried"] = tried
        if cfg:
            state[name + "_won_with"] = cfg["label"]
        save_state(state)
        return ok
    return _job


def make_anthology_job():
    """Four standalone scenes (scene 1 of each narrative), each assigned a
    DIFFERENT starting config so we get variety data across narratives and
    configs in a single compact job. Succeeds if all 4 succeed."""
    assignments = [
        ("lost_letter",   "fallback"),   # quickest baseline
        ("midnight_train","proven"),     # wave-1 baseline
        ("brains_cosmic", "hd_med"),     # variety: 1280x720/97f
        ("coffee_break",  "hd_long"),    # variety: 1280x720/145f
    ]
    def _job(state):
        _ollama_unload_all()
        all_ok = True
        for narrative, start_label in assignments:
            path = os.path.join(EXAMPLE_DIR, f"{narrative}.json")
            if not os.path.exists(path):
                print(f"  SKIP anthology/{narrative}: {path} missing")
                continue
            ordered = []
            start_cfg = next(
                (c for c in CHAIN_CONFIGS if c["label"] == start_label), None)
            if start_cfg and not _is_blacklisted(state, start_label):
                ordered.append(start_cfg)
            for c in CHAIN_CONFIGS:
                if c not in ordered and not _is_blacklisted(state, c["label"]):
                    ordered.append(c)
            scene_ok = False
            for cfg in ordered:
                print(f"\n  anthology/{narrative} trying {cfg['label']}")
                ok, _ = _invoke_chain(path, 1, cfg, no_join=True)
                if ok:
                    scene_ok = True
                    break
                _blacklist_config(state, cfg["label"])
            all_ok = all_ok and scene_ok
        return all_ok
    return _job


# --- Path 1 (T2V + upscale) -------------------------------------------------

def ensure_container():
    """Start the always-on container for T2V+upscale jobs."""
    comfy_container.start()
    if not comfy_container.wait_ready():
        raise RuntimeError("ComfyUI failed to come up")


def run_upscale_batch(name, items, state):
    """Run a list of (prefix, prompt) tuples through upscale_t2v sequentially.

    Path 1 is locked to 49f@1280x720+2x upscale — that's the PROVEN-safe
    shape. No adaptive fallback (unlike chains) because upscale at larger
    frame counts GPU-hangs on gfx1151 (ROCm #5665).
    """
    ensure_container()
    results = []
    for prefix, prompt in items:
        print(f"\n  [{name}] {prefix}")
        status, err = upscale_t2v.submit_and_wait(prompt, prefix)
        results.append((prefix, status, err))
        if status != "success":
            print(f"  FAIL {prefix}: {err}")
    ok = sum(1 for _, s, _ in results if s == "success")
    print(f"\n  [{name}] {ok}/{len(results)} succeeded")
    return ok == len(results)


# --- job plan ---------------------------------------------------------------

def job_plan():
    """Variety-test assignment: each chain pins a DIFFERENT starting config.
    Ordering: cheapest failure first (hd_full probe as single-scene) so if
    the HW hang is real we learn in ≤ 1 hr; then proven baseline, then
    untested mid-HD configs, then anthology mix, then path 1 batches."""
    return [
        ("canary_hero",
         lambda state: run_upscale_batch(
             "canary_hero", [("canary_hero", HERO_SHOT_PROMPT)], state)),
        ("hd_full_probe", make_chain_job(
            "hd_full_probe",
            os.path.join(EXAMPLE_DIR, "lost_letter.json"),
            start_label="hd_full", count=1)),  # 1-scene test under clean VRAM
        ("path2_lost_letter", make_chain_job(
            "path2_lost_letter",
            os.path.join(EXAMPLE_DIR, "lost_letter.json"),
            start_label="proven")),            # baseline i2v (1024x576/145f)
        ("path2_midnight_train", make_chain_job(
            "path2_midnight_train",
            os.path.join(EXAMPLE_DIR, "midnight_train.json"),
            start_label="hd_med")),            # test 1280x720/97f
        ("path2_brains_cosmic", make_chain_job(
            "path2_brains_cosmic",
            os.path.join(EXAMPLE_DIR, "brains_cosmic.json"),
            start_label="hd_long")),           # test 1280x720/145f
        ("path2_anthology", make_anthology_job()),  # fallback/proven/hd_med/hd_long
        ("path1_subject_parade",
         lambda state: run_upscale_batch("subject_parade", SUBJECT_PARADE, state)),
        ("path1_ironic_wave",
         lambda state: run_upscale_batch("ironic_wave", IRONIC_WAVE, state)),
        ("path1_tone_wave",
         lambda state: run_upscale_batch("tone_wave", tone_wave_prompts(), state)),
        # === Second-pass additions (2026-04-19) ===
        # Extended hd_full retry (2 hr per-scene budget) after 1st probe failed
        # at 1 hr timeout; plus 3 new llama3.2-authored narratives across the
        # proven config spectrum.
        ("hd_full_retry", make_chain_job(
            "hd_full_retry",
            os.path.join(EXAMPLE_DIR, "lost_letter.json"),
            start_label="hd_full", count=1)),    # 1 scene, 2 hr timeout
        ("wave_underwater", make_chain_job(
            "wave_underwater",
            os.path.join(EXAMPLE_DIR, "underwater_archive.json"),
            start_label="hd_long")),             # repeat hd_long on new content
        ("wave_train", make_chain_job(
            "wave_train",
            os.path.join(EXAMPLE_DIR, "last_train.json"),
            start_label="hd_med")),              # repeat hd_med on new content
        ("wave_firefly", make_chain_job(
            "wave_firefly",
            os.path.join(EXAMPLE_DIR, "firefly_migration.json"),
            start_label="proven")),              # repeat proven on new content
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true",
                   help="Wipe checkpoint state and run all jobs from scratch")
    p.add_argument("--only", nargs="+",
                   help="Only run these job names (space-separated)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.fresh and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    state = load_state()
    plan = job_plan()
    if args.only:
        plan = [(n, fn) for n, fn in plan if n in args.only]

    print("=" * 72)
    print("  All-Permutations Showcase")
    print(f"  state: {STATE_FILE}")
    print(f"  jobs : {len(plan)}  (completed already: "
          f"{sum(1 for n, _ in plan if state.get(n, {}).get('status') == 'success')})")
    print("=" * 72)

    for name, fn in plan:
        existing = state.get(name, {})
        if existing.get("status") == "success":
            print(f"\n[SKIP] {name} (already completed "
                  f"{existing.get('finished_at', '?')})")
            continue
        if args.dry_run:
            print(f"\n[DRY]  {name}")
            continue

        print(f"\n[RUN]  {name}")
        state[name] = {"status": "running",
                       "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_state(state)

        t0 = time.time()
        try:
            ok = fn(state)
        except KeyboardInterrupt:
            state[name] = {"status": "interrupted",
                           "interrupted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_state(state)
            raise
        except Exception as e:
            state[name] = {"status": "error", "error": str(e),
                           "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_state(state)
            print(f"[FAIL] {name}: {e}")
            continue

        elapsed = int(time.time() - t0)
        state[name] = {
            "status": "success" if ok else "error",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
        }
        save_state(state)
        print(f"[{'OK' if ok else 'FAIL'}] {name} ({elapsed}s)")

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    for name, _ in plan:
        s = state.get(name, {})
        print(f"  {s.get('status', 'pending'):12s}  {name}")


if __name__ == "__main__":
    main()
