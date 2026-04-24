#!/usr/bin/env python3
"""
AI-slop caption overlay — chain mp4s + per-clip subsets.

Two modes:
  1. Chain overlay: multi-caption drawtext with per-scene time windows
  2. Single-clip overlay: one static caption spanning the whole clip

Output: <stem>_slop.mp4 next to each original.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipelines import comfy_container, config  # noqa: E402

# Override via env for a non-default font path; the default is the system
# DejaVu Sans Bold which ships with most Debian/Ubuntu installs.
FONT = os.environ.get(
    "SLOP_OVERLAY_FONT",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

# Scene duration per chain, in seconds (frames / fps)
CHAINS = {
    "chain_lost_letter_proven.mp4": {
        "scene_dur": 145 / 24,   # 6.04s
        "captions": [
            "AI-slop score: 8.7 / 10",
            "prompt: 'she reads mysteriously'",
            "43% of AI videos end at a lighthouse",
            "texture pass: peeling paint (licensed)",
            "CEO says the future is envelopes.",
        ],
    },
    "chain_midnight_train_hd_med.mp4": {
        "scene_dur": 97 / 24,    # 4.04s
        "captions": [
            "corporate nostalgia pack v2.1",
            "moodboard: 'quiet dignity'",
            "character archetype #47 loaded",
            "surreal filter: full tilt",
            "the algorithm wants awe. we delivered.",
        ],
    },
    "chain_brains_cosmic_hd_long.mp4": {
        "scene_dur": 145 / 24,
        "captions": [
            "scene 14,392: 'brain in rocket'",
            "vibes: dark occult (PG-13 palette)",
            "seed 0xBEEF. cello: inevitable.",
            "sports drama unlocked (LoRA: NBA 2013)",
            "frame 127 is the one people share",
        ],
    },
}


def _ffescape(s: str) -> str:
    """Escape a string for ffmpeg drawtext: single quote & colon are special."""
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")


def build_drawtext_chain(captions, scene_dur):
    """Build a chained drawtext filter. One drawtext per caption, enabled
    for its scene's time window (10% pad at start/end)."""
    filters = []
    for i, text in enumerate(captions):
        start = i * scene_dur + scene_dur * 0.05
        end = (i + 1) * scene_dur - scene_dur * 0.05
        txt = _ffescape(text)
        filters.append(
            f"drawtext=fontfile={FONT}:text='{txt}'"
            f":x=(w-text_w)/2:y=h-80"
            f":fontsize=32:fontcolor=white"
            f":box=1:boxcolor=black@0.55:boxborderw=10"
            f":enable='between(t,{start:.2f},{end:.2f})'"
        )
    return ",".join(filters)


def overlay_chain(filename, cfg):
    src = os.path.join(config.OUTPUT_DIR, filename)
    stem, ext = os.path.splitext(filename)
    dst = os.path.join(config.OUTPUT_DIR, f"{stem}_slop{ext}")
    if not os.path.exists(src):
        print(f"  SKIP: {src} missing")
        return False
    filter_chain = build_drawtext_chain(cfg["captions"], cfg["scene_dur"])
    rel_in = src.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
    rel_out = dst.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
    print(f"  {filename}")
    print(f"    scene_dur = {cfg['scene_dur']:.2f}s  captions = {len(cfg['captions'])}")
    # container's ffmpeg ships libopenh264 (not libx264); use that instead
    rc = comfy_container.exec_ffmpeg(
        ["-i", rel_in, "-vf", filter_chain,
         "-c:a", "copy", "-c:v", "libopenh264",
         "-b:v", "4M", rel_out]
    )
    if rc.returncode == 0 and os.path.exists(dst):
        size_mb = os.path.getsize(dst) / 1e6
        print(f"    OK: {os.path.basename(dst)} ({size_mb:.1f} MB)")
        return True
    print(f"    FAIL rc={rc.returncode}\n      stderr={rc.stderr[-500:] if rc.stderr else ''}")
    return False


# --- single-clip subsets (each gets one static caption) -------------------

SINGLE_CLIPS = {
    # ironic_wave (10) — meta-layer on top of existing satire
    "ironic_1_gym_sloth_00001_.mp4":           "procedurally generated motivation",
    "ironic_2_corporate_applause_00001_.mp4":  "engagement score: 9.4 (above target)",
    "ironic_3_meditation_chaos_00001_.mp4":    "calm overlay (premium tier)",
    "ironic_4_self_driving_meeting_00001_.mp4":"autonomy quarterly review",
    "ironic_5_linkedin_influencer_00001_.mp4": "thought leader detected: True",
    "ironic_6_kids_podcast_00001_.mp4":        "kid-safe brand experience",
    "ironic_7_billionaire_camping_00001_.mp4": "authenticity: sponsored",
    "ironic_8_ai_therapist_00001_.mp4":        "support simulation v3",
    "ironic_9_motivational_rock_00001_.mp4":   "growth mindset (procedurally generated)",
    "ironic_10_startup_pivot_00001_.mp4":      "toast: Series A ready",
    # subject_parade (6) — mock the "AI cinematic trailer" genre
    "parade_1_lighthouse_00001_.mp4":       "lighthouse asset #8 (overused)",
    "parade_2_midnight_train_00001_.mp4":   "moody transit pack",
    "parade_3_cosmic_brain_00001_.mp4":     "stock cosmic background — premium tier",
    "parade_4_dragon_office_00001_.mp4":    "genre: licensed chaos",
    "parade_5_neon_gladiator_00001_.mp4":   "retrowave × classical crossover",
    "parade_6_marshmallow_moon_00001_.mp4": "children's book aesthetic v2.0",
    # tone_wave (12) — each emotion as commoditized asset
    "tone_01_awestruck_00001_.mp4":   "licensable wonder",
    "tone_02_dread_00001_.mp4":       "monetizable horror (PG)",
    "tone_03_tender_00001_.mp4":      "stock nostalgia",
    "tone_04_absurd_00001_.mp4":      "brandable whimsy",
    "tone_05_melancholy_00001_.mp4":  "rain-track compatible",
    "tone_06_euphoric_00001_.mp4":    "ad-safe joy",
    "tone_07_menacing_00001_.mp4":    "thriller-trailer grade",
    "tone_08_whimsical_00001_.mp4":   "greeting-card ready",
    "tone_09_grieving_00001_.mp4":    "oscar bait v2",
    "tone_10_serene_00001_.mp4":      "wellness asset",
    "tone_11_tense_00001_.mp4":       "30-second ad suspense",
    "tone_12_wonder_00001_.mp4":      "family-film approved",
}


def overlay_single(filename, caption):
    src = os.path.join(config.OUTPUT_DIR, filename)
    stem, ext = os.path.splitext(filename)
    dst = os.path.join(config.OUTPUT_DIR, f"{stem}_slop{ext}")
    if not os.path.exists(src):
        print(f"  SKIP: {filename} missing")
        return False
    txt = _ffescape(caption)
    vf = (f"drawtext=fontfile={FONT}:text='{txt}'"
          f":x=(w-text_w)/2:y=h-80:fontsize=32:fontcolor=white"
          f":box=1:boxcolor=black@0.55:boxborderw=10")
    rel_in = src.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
    rel_out = dst.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")
    rc = comfy_container.exec_ffmpeg(
        ["-i", rel_in, "-vf", vf,
         "-c:a", "copy", "-c:v", "libopenh264", "-b:v", "4M", rel_out]
    )
    if rc.returncode == 0 and os.path.exists(dst):
        size_mb = os.path.getsize(dst) / 1e6
        print(f"  OK {filename}  ({size_mb:.1f} MB)  \"{caption}\"")
        return True
    print(f"  FAIL {filename} rc={rc.returncode}")
    return False


def main():
    ok_chains = 0
    print("=== chain overlays ===")
    for fn, cfg in CHAINS.items():
        if overlay_chain(fn, cfg):
            ok_chains += 1
    print(f"\n  chains: {ok_chains}/{len(CHAINS)}")

    print("\n=== single-clip overlays ===")
    ok_singles = 0
    for fn, cap in SINGLE_CLIPS.items():
        if overlay_single(fn, cap):
            ok_singles += 1
    print(f"\n  singles: {ok_singles}/{len(SINGLE_CLIPS)}")

    total_ok = ok_chains + ok_singles
    total = len(CHAINS) + len(SINGLE_CLIPS)
    print(f"\n{total_ok}/{total} slop overlays done")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
