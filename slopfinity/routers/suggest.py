import json
import asyncio
from fastapi import APIRouter, Form, Body
from fastapi.responses import JSONResponse
import slopfinity.config as cfg
from slopfinity.llm import _LLM_LOCK
from slopfinity.llm import lmstudio_call
import slopfinity.fanout as _fanout


router = APIRouter()

@router.post("/enhance")
async def enhance(data: dict = Body(...)):
    config = cfg.load_config()
    prompt = data.get("prompt", "")
    distribute = bool(data.get("distribute"))
    if distribute:
        sys_p = (
            "You are a master multi-stage cinematic director. Given a single user idea, "
            "produce STRICT JSON with keys 'image', 'video', 'music', 'tts'. "
            "'image' = a vivid still-frame prompt, 'video' = a motion/camera prompt, "
            "'music' = a short mood/genre description for a music generator, "
            "'tts' = a one or two sentence voiceover line. Return ONLY JSON, no prose."
        )
        # JSON-schema constraint — same defense the /subjects/suggest path
        # uses (server.py:545-568). Without this, reasoning models leak their
        # planning preamble into the response and the client treats the
        # whole prose blob as the rewrite. With strict:true, compliant
        # providers (LM Studio / llama.cpp / vLLM) physically cannot emit
        # anything outside the {image, video, music, tts} envelope.
        distribute_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "stage_distribute",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "minLength": 1, "maxLength": 600},
                        "video": {"type": "string", "minLength": 1, "maxLength": 600},
                        "music": {"type": "string", "maxLength": 300},
                        "tts": {"type": "string", "maxLength": 400},
                    },
                    "required": ["image", "video", "music", "tts"],
                    "additionalProperties": False,
                },
            },
        }
        async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4), _LLM_LOCK:
            raw = await asyncio.to_thread(lmstudio_call, sys_p, prompt, distribute_schema)
        # Best-effort JSON extraction (schema-compliant providers will
        # already give us pure JSON; the loose-find paths handle providers
        # that ignore response_format).
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            parsed = {"image": prompt, "video": prompt, "music": "", "tts": ""}
        return {
            "suggestion": raw,
            "distribute": True,
            "stages": {
                "image": parsed.get("image", ""),
                "video": parsed.get("video", ""),
                "music": parsed.get("music", ""),
                "tts": parsed.get("tts", ""),
            },
        }
    # Single-stage rewrite — same json-schema treatment so the LLM cannot
    # leak "Sure, here's the rewritten prompt:" or any planning preamble
    # into the response. Returns just `{rewrite: "..."}`. Client extracts
    # `r.suggestion` (kept as the wire field name for back-compat) which
    # we now route through the schema-constrained extract.
    rewrite_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "stage_rewrite",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "rewrite": {"type": "string", "minLength": 1, "maxLength": 800},
                },
                "required": ["rewrite"],
                "additionalProperties": False,
            },
        },
    }
    # Append a one-line schema reminder for non-compliant providers.
    user_msg = prompt + '\n\nReturn ONLY a JSON object: {"rewrite": "..."}'
    async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4), _LLM_LOCK:
        raw = await asyncio.to_thread(lmstudio_call, config["enhancer_prompt"], user_msg, rewrite_schema)
    # Schema-compliant path: parse {"rewrite": "..."}.
    rewrite_text = ""
    try:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("rewrite"), str):
            rewrite_text = obj["rewrite"].strip()
    except Exception:
        pass
    # Fallback: model ignored the schema (raw prose response). Strip
    # common preamble patterns that reasoning models love. The QA agent
    # found local LM Studio models leaking "**Cinematic Director
    # Response:**\n\n```\n<actual rewrite>" — so we now also strip
    # markdown headers (**bold:** / ## H2 / etc.) AND surrounding
    # code-fence blocks. Last-resort we just hand the raw text back.
    if not rewrite_text:
        rewrite_text = (raw or "").strip()
        # Strip surrounding code fences (```lang\n...\n```).
        if rewrite_text.startswith("```"):
            rewrite_text = rewrite_text.split("\n", 1)[1] if "\n" in rewrite_text else rewrite_text
            if rewrite_text.endswith("```"):
                rewrite_text = rewrite_text[:-3]
            rewrite_text = rewrite_text.strip()
        # Strip leading **bold heading:** or ## H1/H2 markdown headers
        # that reasoning models put at the top of their response.
        rewrite_text = re.sub(r'^(?:\*\*[^*\n]+\*\*[:.]?\s*\n+|#{1,6}\s+[^\n]+\n+)+',
                              '', rewrite_text, flags=re.MULTILINE).strip()
        # Drop "Sure, here's…" / "Rewritten:" / "Here is the…" preludes.
        rewrite_text = re.sub(r'^(?:sure[,!.]?\s*|here\s*(?:is|are)\s*(?:the\s*)?(?:rewritten|revised|new)?\s*(?:prompt|version)?[:.]?\s*|rewritten[:.]?\s*|revised[:.]?\s*|cinematic\s+director\s+response[:.]?\s*)',
                              '', rewrite_text, flags=re.IGNORECASE).strip()
        # Re-strip code fences in case a header preceded them.
        if rewrite_text.startswith("```"):
            rewrite_text = rewrite_text.split("\n", 1)[1] if "\n" in rewrite_text else rewrite_text
            if rewrite_text.endswith("```"):
                rewrite_text = rewrite_text[:-3]
            rewrite_text = rewrite_text.strip()
    return {"suggestion": rewrite_text}

@router.post("/enhance/distribute")
async def enhance_distribute(data: dict = Body(...)):
    """Single-idea fan-out with preserve-tokens and lock support.

    Accepts: {core, stages: {image, video, music, tts}, locked: [...],
              preserve_tokens: [...], persist: bool=True}
    Returns: {ok, stages, preserved_ok, preserved_dropped, persisted}

    When `persist` is True (the default), the resulting per-stage prompts
    are written back to `config.{image,video,music,tts}_prompt` so they
    show up — and are editable — in the "prompts →" modal. The fleet
    runner re-reads config.json on each stage entry, so a fan-out followed
    by an immediate run will pick up the freshly persisted overrides.
    """
    core = (data.get("core") or "").strip()
    stages_in = data.get("stages") or {}
    locked = data.get("locked") or []
    preserve_tokens = data.get("preserve_tokens") or []
    persist = data.get("persist")
    if persist is None:
        persist = True
    # Fan-out makes multiple LLM calls. Hold acquire_gpu across the whole
    # batch so we suspend/resume LM Studio just once, not per call.
    # safety_gb=4 since this is just LLM rewrites, not a 60 GB diffusion stage.
    async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4), _LLM_LOCK:
        result = await asyncio.to_thread(
            _fanout.fanout,
            core,
            stages_in,
            locked,
            preserve_tokens,
            lmstudio_call,
        )
    persisted = False
    if persist:
        out_stages = result.get("stages") or {}
        try:
            config = cfg.load_config()
            if out_stages.get("image"):
                config["image_prompt"] = out_stages["image"]
            if out_stages.get("video"):
                config["video_prompt"] = out_stages["video"]
            if out_stages.get("music"):
                config["music_prompt"] = out_stages["music"]
            if out_stages.get("tts"):
                config["tts_prompt"] = out_stages["tts"]
            cfg.save_config(config)
            persisted = True
        except Exception as e:
            # Persistence is best-effort — never fail the fan-out itself
            # because of a transient config write hiccup.
            print(f"[enhance/distribute] config persist failed: {e}")
    return {
        "ok": result["ok"],
        "stages": result["stages"],
        "preserved_ok": result["preserved_ok"],
        "preserved_dropped": result["preserved_dropped"],
        "persisted": persisted,
    }

@router.get("/subjects/suggest")
async def subjects_suggest(n: int = 6, subjects: str = "", endless: int = 0, opener: int = 0,
                            fresh: int = 0, chat: int = 0, prompt_id: str = ""):
    """Generate N short visual subject ideas via the configured local LLM.

    Cache key includes both N and (subjects, settings flags) so toggling the
    "derive from subjects" switch or editing Subjects invalidates the cache.

    Settings honoured:
      * `suggest_use_subjects` (default True): when True, the user message
        seeds the LLM with the current Subjects textarea content for
        style/theme matching. The `subjects` query parameter carries that
        content from the client.
      * `suggest_custom_prompt` (default ""): when non-empty, replaces the
        built-in suggestion system prompt verbatim.

    `fresh=1` bypasses the cache AND injects a random salt into the user
    message so the LLM produces a different batch — used by the marquee
    drip-feed loop that fills rows 2..N (otherwise rows 2/3/4 hit the
    same cache_key as row 1 and all show identical chips).
    """
    import time
    config = cfg.load_config()
    use_subjects = bool(config.get("suggest_use_subjects", cfg.DEFAULT_SUGGEST_USE_SUBJECTS))
    # Env override wins over Settings → Prompts. Lets a user pin their
    # personal tone (e.g. "cynical philosophical") in `.env` so a fresh
    # install/checkout doesn't lose it. Empty/unset → fall back to the
    # Settings field, then the built-in default.
    env_override = (os.environ.get("SLOPFINITY_SUGGEST_CUSTOM_PROMPT") or "").strip()
    custom_prompt = env_override or (config.get("suggest_custom_prompt") or "").strip()
    subjects_in = (subjects or "").strip() if use_subjects else ""
    cache_key = (n, use_subjects, custom_prompt, subjects_in, bool(endless), bool(opener), (prompt_id or ""))
    cache = getattr(subjects_suggest, "_cache", None)
    now = time.time()
    # Cache persists indefinitely while the cache_key is unchanged — page
    # reloads and re-renders should NEVER re-fire the LLM unless the user
    # actually changed something (Subjects text, custom prompt, n, the
    # use_subjects toggle). The previous 30-second TTL caused every reload
    # past ~30 s to burn an unnecessary LLM call.
    # Exceptions that always re-fire fresh:
    #   * endless mode — every tick is a NEW story beat
    #   * opener — single-shot lucky-dip
    #   * fresh=1 — caller explicitly asked for variation (marquee drip-feed)
    if cache and cache[1] == cache_key and not endless and not opener and not fresh:
        return {"suggestions": cache[2], "cached": True}
    # prompt_id wins over custom_prompt: it points at a NAMED entry in
    # config.suggest_prompts (e.g. "yes-and", "plot-twist"). The named
    # prompt's .system text is rendered with {n} substituted. When the id
    # doesn't resolve, fall back to custom_prompt → built-in default.
    named_sys = ""
    if prompt_id:
        prompts_list = config.get("suggest_prompts") or []
        match = next((p for p in prompts_list if isinstance(p, dict) and p.get("id") == prompt_id), None)
        if match and match.get("system"):
            try:
                named_sys = match["system"].format(n=n)
            except Exception:
                named_sys = match["system"]
    sys_p = named_sys or custom_prompt or _default_suggest_system_prompt(n)
    # "I'm Feeling Lucky" — a single random story-opener used to seed an
    # Endless run when the textarea is empty. We override the system
    # prompt with one that asks for ONE evocative opening scene; the
    # rest of the cycle then asks for continuations off that seed.
    if opener:
        sys_p = (
            "You are a concept artist for an AI video fleet. "
            "Output exactly ONE short visual subject — 3-8 words, plain text, "
            "no numbering, no bullets, no quotes, no JSON, no markdown — "
            "an evocative opening scene that could anchor a longer story."
        )
        user_msg = "Give me one story-opening scene."
    # We're now fetching THREE modes simultaneously in one payload to save LLM roundtrips.
    user_msg_base = ""
    if subjects_in:
        user_msg_base = f"Current context / story beats: {subjects_in}\n\n"
        
    user_msg = user_msg_base + (
        f"Generate {n} suggestions for THREE distinct contexts:\n"
        f"1. 'story': short next-scene continuations building chronologically on the context above. Do NOT repeat beats.\n"
        f"2. 'simple': tangential visual subjects matching the tone/theme of the context.\n"
        f"3. 'chat': longer conversational replies or starters relating to the ongoing process."
    )

    # When fresh=1 we want EACH call to give different ideas (the marquee
    # drip-feed asks for rows 2..N and they'd otherwise be identical to
    # row 1 since the system+user prompts are the same). Append a small
    # randomized salt that the LLM will treat as a free-association nudge.
    if fresh:
        salt_themes = [
            "atmospheric weather", "industrial decay", "biological mutation",
            "celestial phenomena", "domestic surrealism", "geometric impossibility",
            "kinetic machinery", "underwater architecture", "nocturnal wildlife",
            "bureaucratic absurdity", "fungal growth", "crystalline structures",
            "deep-sea bioluminescence", "post-industrial landscapes", "neon arcades",
        ]
        chosen = random.choice(salt_themes)
        user_msg += f"\n\nNudge: lean toward {chosen}. Avoid repeating any earlier batch."
    # JSON-schema constraint: fetch all 3 KVP sets at once
    max_len_endless = config.get("suggest_max_len_endless") or 40
    max_len_simple = config.get("suggest_max_len_simple") or 80
    max_len_chat = config.get("suggest_max_len_chat") or 160

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "suggestions_multi",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "story": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": max_len_endless}
                    },
                    "simple": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": max_len_simple}
                    },
                    "chat": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": max_len_chat}
                    }
                },
                "required": ["story", "simple", "chat"],
                "additionalProperties": False,
            },
        },
    }
    # Append a one-line schema reminder to the user message
    user_msg += '\n\nReturn ONLY a JSON object: {"story": ["..."], "simple": ["..."], "chat": ["..."]}'
    
    async with sched.acquire_gpu("Concept", "lmstudio", safety_gb=4), _LLM_LOCK:
        raw = await asyncio.to_thread(lmstudio_call, sys_p, user_msg, response_format)

    suggestions_dict = {"story": [], "simple": [], "chat": []}
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if isinstance(obj.get("story"), list): suggestions_dict["story"] = list(dict.fromkeys([str(s).strip() for s in obj["story"] if s]))
            if isinstance(obj.get("simple"), list): suggestions_dict["simple"] = list(dict.fromkeys([str(s).strip() for s in obj["simple"] if s]))
            if isinstance(obj.get("chat"), list): suggestions_dict["chat"] = list(dict.fromkeys([str(s).strip() for s in obj["chat"] if s]))
    except Exception:
        pass
    # Don't store endless / opener results in the cache — we want each
    # tick fresh, and openers are one-shot by design.
    if any(suggestions_dict.values()) and not endless and not opener:
        subjects_suggest._cache = (now, cache_key, suggestions_dict)
    return {"suggestions": suggestions_dict, "cached": False}
