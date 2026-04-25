"""AI Magic fan-out: single-idea-to-multi-stage expansion with preserve-tokens.

Used by POST /enhance/distribute. Keeps the existing /enhance endpoint unchanged.

No third-party deps — stdlib only (urllib, re, json).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Iterable

STAGES = ("image", "video", "music", "tts")

# Common English function words we should NOT treat as proper nouns even when
# they happen to start a sentence (so they don't leak into preserve_tokens).
_STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Then", "Thus",
    "And", "But", "Or", "Nor", "For", "Yet", "So",
    "If", "When", "While", "Where", "Why", "How", "What", "Who", "Which",
    "A", "An", "Is", "It", "In", "On", "At", "Of", "To", "By", "As",
    "He", "She", "They", "We", "You", "His", "Her", "Their", "Our", "Your",
    "Him", "Them", "Us", "Me", "My", "Mine",
    "Was", "Were", "Been", "Being", "Have", "Has", "Had",
    "Will", "Would", "Could", "Should", "May", "Might", "Must",
    "Also", "After", "Before", "During", "Over", "Under",
}

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_QUOTED_RE = re.compile(r'"([^"]+)"|\u201c([^\u201d]+)\u201d|\'([^\']+)\'')
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def extract_preserve_tokens(core: str, stages: dict) -> list[str]:
    """Auto-detect tokens the LLM must keep intact.

    Rules:
      - Everything inside quotes (straight or curly, single or double) in `core`
        or any seeded stage becomes a preserve phrase (verbatim).
      - Capitalised proper-noun-like tokens (\\b[A-Z][a-z]{2,}\\b) that are NOT
        in the stopword set become preserve tokens.
    """
    sources = [core or ""]
    for name in STAGES:
        v = (stages or {}).get(name) or ""
        if v:
            sources.append(v)

    out: list[str] = []
    seen: set[str] = set()

    for text in sources:
        # Quoted phrases first (they take priority and may contain stopwords).
        for m in _QUOTED_RE.finditer(text):
            phrase = next((g for g in m.groups() if g), "").strip()
            if phrase and phrase.lower() not in seen:
                seen.add(phrase.lower())
                out.append(phrase)
        # Proper-noun-ish capitalised words.
        for m in _PROPER_NOUN_RE.finditer(text):
            w = m.group(0)
            if w in _STOPWORDS:
                continue
            if w.lower() in seen:
                continue
            seen.add(w.lower())
            out.append(w)
    return out


def _words(s: str) -> list[str]:
    return _WORD_RE.findall(s or "")


def verify_locked(
    returned: dict, stages: dict, locked: Iterable[str]
) -> tuple[dict, list[str]]:
    """For each locked stage, ensure every word in the user's seed survives.

    If any seed-word is missing (case-insensitive) from the returned text, append
    the missing words verbatim at the end. Returns (patched_returned, dropped).
    """
    locked_set = {s for s in (locked or []) if s in STAGES}
    dropped_overall: list[str] = []
    patched = dict(returned)

    for stage in locked_set:
        seed = (stages or {}).get(stage) or ""
        if not seed.strip():
            continue
        new_text = patched.get(stage) or ""
        lower_haystack = new_text.lower()
        missing: list[str] = []
        for w in _words(seed):
            if w.lower() not in lower_haystack:
                missing.append(w)
        if missing:
            dropped_overall.extend(missing)
            suffix = " " + " ".join(missing)
            patched[stage] = (new_text.rstrip() + suffix).strip()
    return patched, dropped_overall


def _try_parse_json(raw: str):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def build_system_prompt() -> str:
    # Honour the user's Settings \u2192 Prompts override; falls back to the
    # built-in default (see slopfinity.config.DEFAULT_FANOUT_SYSTEM_PROMPT).
    # Imported lazily so this module stays importable in test contexts that
    # don't construct a real config.
    try:
        from slopfinity import config as _config
        return _config.get_fanout_system_prompt()
    except Exception:
        return (
            "You are a master multi-stage cinematic director. Produce STRICT JSON "
            "with keys image, video, music, tts. If a stage's seed text is "
            "non-empty, you MUST extend it \u2014 keep every subject, name, and "
            "quoted token intact; only add sensory detail, lighting, motion, mood, "
            "or delivery. Under 40 words per stage. Return ONLY JSON."
        )


def build_user_payload(core: str, stages: dict, preserve_tokens: list[str]) -> str:
    payload = {
        "core_idea": core or "",
        "seeded_stages": {s: (stages or {}).get(s) or "" for s in STAGES},
        "preserve_tokens": preserve_tokens or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def fanout(
    core: str,
    stages: dict,
    locked: list[str],
    preserve_tokens: list[str] | None,
    llm_call: Callable[[str, str], str],
    max_retries: int = 2,
) -> dict:
    """Run the fan-out. `llm_call(sys_prompt, user_msg) -> raw_text`.

    Returns: {"ok": bool, "stages": {...}, "preserved_ok": bool,
              "preserved_dropped": [...], "raw": "..."}.
    """
    stages = stages or {}
    preserve_tokens = list(preserve_tokens or [])
    auto = extract_preserve_tokens(core or "", stages)
    for t in auto:
        if t not in preserve_tokens:
            preserve_tokens.append(t)

    sys_p = build_system_prompt()
    user_msg = build_user_payload(core or "", stages, preserve_tokens)

    parsed = None
    raw = ""
    nudge = ""
    for attempt in range(max_retries + 1):
        raw = llm_call(sys_p + nudge, user_msg) or ""
        parsed = _try_parse_json(raw)
        if isinstance(parsed, dict):
            break
        nudge = "\nReturn ONLY JSON, no prose."

    if not isinstance(parsed, dict):
        # Fall back to seed text (don't erase).
        parsed = {s: (stages.get(s) or "") for s in STAGES}

    returned = {s: (parsed.get(s) or (stages.get(s) or "")) for s in STAGES}
    patched, dropped = verify_locked(returned, stages, locked or [])
    return {
        "ok": True,
        "stages": patched,
        "preserved_ok": not dropped,
        "preserved_dropped": dropped,
        "raw": raw,
    }
