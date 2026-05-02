# Hardcoded Prompt Survey

A one-shot audit of every hardcoded prompt / prompt-template string the
local-only LLM, image, video and audio pipeline ships with. The goal is to
expose the prompts that meaningfully change output quality through the
slopfinity Settings modal so users can tune them without editing source.

## Categories

- **(a) Tunable** — meaningfully shapes user-visible output. Surface in Settings.
- **(b) Internal scaffolding** — JSON-extraction nudge, mock-server boilerplate,
  fan-out machinery. Leave in code.
- **(c) Constant** — model-name strings, schema docstrings for CLI tools that
  are not part of the slopfinity dashboard. Skip.

## Inventory

| File:line | Role | Current text (truncated to 80) | Cat. |
| --------- | ---- | ------------------------------ | ---- |
| `slopfinity/config.py:21` (`DEFAULT_PHILOSOPHICAL_PROMPT`) | Fleet system prompt — used by `run_fleet.py` to rewrite each seed prompt before image gen | `You are a master cinematic concept artist.` | a (already plumbed via `philosophical_prompt`) |
| `slopfinity/config.py:64` (`DEFAULT_CONFIG.enhancer_prompt`) | Concept-stage rewriter system prompt — `slopfinity/workers/concept.py` uses it to rewrite the user's prompt | `You are a master cinematic director. Rewrite the user's prompt into a hi…` | a |
| `slopfinity/server.py:288` (inside `/enhance` distribute branch) | Multi-stage fan-out system prompt — splits one idea into image/video/music/tts via `/enhance?distribute=1` | `You are a master multi-stage cinematic director. Given a single user ide…` | a |
| `slopfinity/fanout.py:121` (`build_system_prompt`) | Multi-stage fan-out system prompt — called by `/enhance/distribute` | `You are a master multi-stage cinematic director. Produce STRICT JSON wit…` | a |
| `slopfinity/server.py:390` (`_default_suggest_system_prompt`) | Subjects auto-suggest system prompt | `You are a concept artist for an AI video fleet. Output ONLY a JSON array…` | a (already overridable via `suggest_custom_prompt`) |
| `slopfinity/server.py:1690` (chaos-mode background loop) | Tangential subject suggestion (chaos-mode background re-suggest) | `You are a concept artist for an AI video fleet. The user is currently wo…` | a |
| `run_fleet.py:232` (`generate_prompt` queue branch — `sys_p`) | Fleet runner: queue-item rewriter system prompt | `You are a master cinematic concept artist. Output cinematic prose in con…` | a (covered by `philosophical_prompt` + new override) |
| `run_fleet.py:233-237` (`generate_prompt` queue branch — `user_p`) | Fleet runner: rewriter user-message template (the "Subject: {seed}. Under 40 words…" instruction) | `Rewrite as a detailed, visually evocative scene description for an AI im…` | a |
| `run_fleet.py:267-268` (`generate_prompt` infinity branch) | Infinity-mode runner: theme → cynical philosophical scene | `Describe a detailed, visually evocative cynical philosophical scene about: {theme}…` | a |
| `run_fleet.py:279` (final fallback) | VOID-style fallback when no LLM is reachable | `A cynical {style} scene. Text: 'VOID'.` | a (low priority — fires only on LLM outage) |
| `slopfinity/fanout.py:171` (retry nudge) | JSON-only retry nudge appended to fan-out prompt on parse failure | `Return ONLY JSON, no prose.` | b |
| `slopfinity/server.py:430-434` (subjects user-message) | "Match the style/theme of these existing subjects:" composer | `Match the style/theme of these existing subjects:\n{subjects}\n\nNow generate {n} more in the same vein.` | b (template scaffolding) |
| `tests/mock_llm_server.py:47` (`user_p = ""`) | Test mock server | n/a | b |
| `scripts/generate_prompts.py:51` (`SYSTEM_PROMPT`) | Standalone CLI prompt-design tool — not part of slopfinity runtime | (large schema doc) | c (out-of-scope; CLI-only utility) |
| `scripts/generate_prompts.py:89` (`LYRICS_SYSTEM_PROMPT`) | Same CLI utility | (lyrics doc) | c |
| `pipelines/qwen_runner.py:64`, `tests/legacy/*` (`negative_prompt = 'blurry, low quality…'`) | Per-pipeline negative prompts | `blurry, low quality, distorted, watermark` | c (model-tuning, not user-facing prompt rewriting) |
| `run_fleet.py:350,502,525` (LTX `CLIPTextEncode` negative `"blurry"` etc.) | Per-stage negative prompts in workflow JSON | `blurry, low quality` | c |

## What was already plumbed

- **`philosophical_prompt`** — fleet system prompt — backend already exposes
  it via `/settings` GET/POST and the LLM tab's `set-fleet-prompt` textarea
  with a "Reset to default" button.
- **`suggest_custom_prompt`** — subjects-suggest system-prompt override —
  backend already exposes it in the LLM tab as `set-suggest-custom-prompt`.

## What this PR exposes (new)

A new "Prompts" tab in the Settings modal (between Triggers and Scheduler)
collects every category-(a) prompt that was *not* already plumbed:

| Settings field | Config key | Default constant |
| -------------- | ---------- | ---------------- |
| Concept-stage rewriter | `enhancer_prompt` | `DEFAULT_CONFIG["enhancer_prompt"]` |
| Fleet rewriter user-message template | `fleet_user_prompt_template` | `DEFAULT_FLEET_USER_PROMPT_TEMPLATE` |
| Multi-stage fan-out system prompt | `fanout_system_prompt` | `DEFAULT_FANOUT_SYSTEM_PROMPT` |
| Infinity-mode user-message template | `infinity_user_prompt_template` | `DEFAULT_INFINITY_USER_PROMPT_TEMPLATE` |
| Chaos-mode tangential-suggest system prompt | `chaos_suggest_system_prompt` | `DEFAULT_CHAOS_SUGGEST_SYSTEM_PROMPT` |
| VOID-style fallback template | `void_fallback_template` | `DEFAULT_VOID_FALLBACK_TEMPLATE` |

The existing `philosophical_prompt` and `suggest_custom_prompt` fields are
linked from the new tab too (cross-reference helpers; the canonical editors
remain in the LLM tab to preserve every existing handler).

## What was left as code-only

- `fanout.py:171` retry nudge (`Return ONLY JSON, no prose.`) — internal
  scaffolding; tweaking it changes nothing user-visible and changing it
  badly (e.g. translating to non-English) breaks the JSON parser.
- `server.py:430-434` subjects user-message composer — template glue, not a
  prompt content knob.
- `scripts/generate_prompts.py` — separate CLI utility; out-of-scope for the
  slopfinity dashboard.
- ComfyUI-workflow negative prompts (`"blurry"`, `"blurry, low quality"`,
  etc.) — per-stage model knobs, not LLM prompts.
- Pipeline-script defaults under `tests/legacy/`, `scripts/upscale_smoke.py`
  etc. — example prompts in test fixtures, not runtime defaults.
