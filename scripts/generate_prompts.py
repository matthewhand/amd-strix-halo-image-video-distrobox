#!/usr/bin/env python3
"""
LLM-driven prompt generator for the LTX-2.3 wave runners.

Takes a high-level theme and produces a JSON file with N scenes, each
containing a Qwen still-composition prompt and an LTX action+sound video
prompt. Optionally generates companion song lyrics for the chain.

Output format matches what tests/run_chained_wave.py and
tests/run_matrix.py consume via --prompts-file.

Provider auto-detection:
  1. ollama (http://localhost:11434) — local, free, ~5 GB CPU RAM at int4
  2. Anthropic API (env: ANTHROPIC_API_KEY)        — best quality
  3. OpenAI-compatible (env: OPENAI_API_KEY + optional OPENAI_BASE_URL)

Usage:
    # Auto-detect provider, save to public examples
    python scripts/generate_prompts.py \\
        --theme "underwater chess match in a sunken cathedral" \\
        --num-scenes 5 \\
        --output tests/example_prompts/underwater_chess.json

    # Force a specific tone + audio kind
    python scripts/generate_prompts.py \\
        --theme "abandoned amusement park at twilight" \\
        --tone sardonic --audio-kind music \\
        --output tests/example_prompts/amusement_park.json

    # Generate sensitive moderation-category content (lands in private dir)
    python scripts/generate_prompts.py \\
        --theme "noir detective unwinding" \\
        --moderation-category regulated_substances \\
        --output tests/private_prompts/detective_off_duty.json

    # Include companion song lyrics
    python scripts/generate_prompts.py \\
        --theme "midnight rooftop chase" \\
        --include-lyrics --genre "synthwave action" \\
        --output tests/example_prompts/rooftop_chase.json
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# Schema documentation for the LLM
SYSTEM_PROMPT = """\
You are a prompt-design assistant for a video-generation pipeline that uses
two models in sequence:

1. Qwen-Image: generates a still photograph of a static composition.
   Prompts should describe a SCENE (place, subject, props, lighting,
   framing) — like a photographer's setup notes. ~50-80 words.

2. LTX-2.3: takes that still as the first frame and animates it with
   sound. Prompts should describe MOTION + AUDIO over ~4-8 seconds.
   Include: what visibly happens, what is heard. Keep visual description
   tightly tied to the still. ~60-100 words.

Your job: given a THEME, produce N scenes that work together as a chain
where each scene's last frame plausibly leads into the next. Output STRICT
JSON only — no markdown, no commentary.

Schema:
{
  "scenes": [
    {
      "label": "snake_case_short_name",
      "qwen": "Qwen still prompt (only set on scene 1; null on chained scenes)",
      "video": "LTX video+audio prompt"
    },
    ...
  ]
}

Rules:
- Scene 1: include both qwen and video.
- Scenes 2..N: qwen MUST be null (last frame of scene N-1 seeds them).
- Video prompts must clearly motion-describe what changes from the previous
  scene, plus audio direction.
- Keep PG-13 unless the user explicitly asks for a moderation category.
- No real-person depictions, no minors in any sensitive context.
"""

LYRICS_SYSTEM_PROMPT = """\
You are a songwriter. Given a theme + genre + scene-by-scene narrative
description, write song lyrics where each verse maps to one scene of the
chain. Add a chorus that captures the emotional spine. Output JSON only.

Schema:
{
  "lyrics": "verse 1\\nverse 2\\nchorus\\nverse 3\\nverse 4\\nverse 5"
}
"""


def build_user_prompt(args):
    """Build the user-facing prompt for the LLM."""
    parts = [f"Theme: {args.theme}"]
    parts.append(f"Number of scenes: {args.num_scenes}")
    if args.tone:
        parts.append(f"Tonal register: {args.tone} "
                     "(ironic=visual/audio clash; sardonic=corporate cynicism; "
                     "comedic=lighthearted slapstick; absurd=dream-logic)")
    if args.audio_kind:
        parts.append(f"Audio style: {args.audio_kind} "
                     "(prescribed_speech=specific dialogue; "
                     "general_speech=vague voices; sound_effects=ambient/sfx; "
                     "music=musical performance)")
    if args.moderation_category and args.moderation_category != "safe_baseline":
        parts.append(f"Content category: {args.moderation_category} "
                     "(stay within PG-13/Hollywood broadcast bounds)")
    if args.subject:
        parts.append(f"All scenes feature: {args.subject}")
    return "\n".join(parts)


# ---------- providers ----------

# Preference order when --model is not set. First match wins.
# Note: gemma4:26b/31b excluded — confirmed broken on gfx1151 / this ollama
# build (HTTP 500 "model runner unexpectedly stopped" on first inference).
# Re-add them when ollama ships a fix.
OLLAMA_MODEL_PREFERENCE = [
    "gemma3:12b-it-qat", "gemma3:12b",
    "llama3.2:latest",
]


def _ollama_list_models():
    """Return set of installed ollama model tags, or empty set on error."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        return {m.get("name") for m in data.get("models", [])}
    except Exception:
        return set()


def _pick_ollama_model():
    """First installed model from OLLAMA_MODEL_PREFERENCE, else llama3.2."""
    installed = _ollama_list_models()
    for m in OLLAMA_MODEL_PREFERENCE:
        if m in installed:
            return m
    return "llama3.2:latest"


def call_ollama(system, user, model=None, timeout=300):
    """POST to local ollama. Returns the assistant's content string.

    If `model` is None, picks the highest-ranked installed model per
    OLLAMA_MODEL_PREFERENCE (gemma4 > gemma3 > llama3.2).

    Note: smaller models (llama3.2 3B) error 500 with format=json on long
    prompts. We omit it and parse JSON from the response with markdown
    fence stripping (see main()).
    """
    if model is None:
        model = _pick_ollama_model()
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.8, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def call_anthropic(system, user, model="claude-haiku-4-5-20251001", timeout=60):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body["content"][0]["text"]


def call_openai(system, user, model=None, timeout=60):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": 0.8,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def detect_provider():
    """Auto-detect which provider to use. Returns (name, callable)."""
    # Prefer local (no API cost, no network dependency)
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2):
            return "ollama", call_ollama
    except Exception:
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", call_anthropic
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", call_openai
    return None, None


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--theme", required=True,
                   help="High-level theme/concept the chain expresses")
    p.add_argument("--num-scenes", type=int, default=5)
    p.add_argument("--subject", help="Optional shared subject (e.g. 'floating brains') threaded through every scene")
    p.add_argument("--tone", choices=["ironic", "sardonic", "comedic", "absurd"])
    p.add_argument("--audio-kind", choices=["prescribed_speech", "general_speech",
                                             "sound_effects", "music"])
    p.add_argument("--moderation-category", default="safe_baseline",
                   choices=["safe_baseline", "violence_action", "weapons",
                            "suggestive_intimate", "regulated_substances",
                            "medical_distress", "frightening_imagery"])
    p.add_argument("--include-lyrics", action="store_true",
                   help="Also generate companion song lyrics matching the chain")
    p.add_argument("--genre", default="cinematic orchestral",
                   help="Genre hint for lyrics (only used with --include-lyrics)")
    p.add_argument("--output", required=True,
                   help="JSON output path (will be overwritten)")
    p.add_argument("--provider", choices=["ollama", "anthropic", "openai", "auto"],
                   default="auto")
    p.add_argument("--model", help="Override model (provider-specific)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the user prompt that would be sent; do not call LLM")
    args = p.parse_args()

    user_prompt = build_user_prompt(args)
    print("=" * 70)
    print(f"Theme: {args.theme}")
    print(f"Scenes: {args.num_scenes}, tone: {args.tone or 'any'}, "
          f"audio: {args.audio_kind or 'any'}, "
          f"moderation: {args.moderation_category}")
    print(f"Output: {args.output}")
    print("=" * 70)

    if args.dry_run:
        print("\n--- prompt that would be sent ---")
        print(user_prompt)
        return 0

    # Provider selection
    if args.provider == "auto":
        name, fn = detect_provider()
        if not name:
            sys.exit("No LLM provider available. Set ANTHROPIC_API_KEY or "
                     "OPENAI_API_KEY, or run ollama locally on :11434.")
    else:
        fn = {"ollama": call_ollama, "anthropic": call_anthropic,
              "openai": call_openai}[args.provider]
        name = args.provider
    print(f"Using provider: {name}")

    # Generate scenes
    print("\n--- generating scenes ---")
    t0 = time.time()
    kwargs = {"model": args.model} if args.model else {}
    raw = fn(SYSTEM_PROMPT, user_prompt, **kwargs)
    print(f"  ({time.time() - t0:.1f}s)")

    # Strip markdown fences and any preamble/coda text the model added.
    # Find the first { and the matching last } — robust to chatty models.
    raw_stripped = raw.strip()
    # Drop ``` ... ``` fences entirely
    if raw_stripped.startswith("```"):
        # remove leading fence and optional language tag
        lines = raw_stripped.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_stripped = "\n".join(lines)
    # Find outermost { ... } if there's surrounding prose
    first_brace = raw_stripped.find("{")
    last_brace = raw_stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        json_str = raw_stripped[first_brace:last_brace + 1]
    else:
        json_str = raw_stripped
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        sys.exit(f"Failed to parse LLM JSON: {e}\nRaw output:\n{raw[:600]}")

    scenes = data.get("scenes") or data.get("data", {}).get("scenes")
    if not scenes:
        sys.exit(f"No 'scenes' in LLM output:\n{json.dumps(data, indent=2)[:500]}")

    # Tag with metadata so downstream consumers know what they got
    for s in scenes:
        s["audio_kind"] = args.audio_kind or "general_speech"
        s["moderation_category"] = args.moderation_category
        if args.tone:
            s["tone"] = args.tone

    # Optional lyrics
    if args.include_lyrics:
        print("\n--- generating lyrics ---")
        t0 = time.time()
        narrative = "\n".join(f"Scene {i+1}: {s.get('video', '')[:120]}"
                              for i, s in enumerate(scenes))
        lyrics_prompt = (f"Theme: {args.theme}\nGenre: {args.genre}\n\n"
                         f"Scene-by-scene narrative:\n{narrative}")
        raw_lyr = fn(LYRICS_SYSTEM_PROMPT, lyrics_prompt, **kwargs)
        raw_lyr = raw_lyr.strip().lstrip("`").rstrip("`")
        if raw_lyr.startswith("json\n"):
            raw_lyr = raw_lyr[5:]
        try:
            lyr_data = json.loads(raw_lyr)
            data["_lyrics"] = lyr_data.get("lyrics", "")
            data["_genre"] = args.genre
            print(f"  ({time.time() - t0:.1f}s, {len(data['_lyrics'])} chars)")
        except json.JSONDecodeError:
            print("  (failed to parse lyrics, skipping)")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # The chain runners expect a top-level array, not the wrapper
    with open(args.output, "w") as f:
        json.dump(scenes, f, indent=2)

    if args.include_lyrics and data.get("_lyrics"):
        # Print the lyrics to stdout — user can append to .env.local manually
        print("\n--- LYRICS ---")
        print(data["_lyrics"])
        print("\n(append to .env.local manually as SONG_NN_* if you want)")

    print(f"\n✓ wrote {len(scenes)} scenes to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
