# Auto-suspend framework

## Why generalize

PR #40 added a single hard-coded toggle: "Auto-suspend LLM during GPU
inference". When ON, the scheduler `pgrep`s for `LM Studio` / `lm-studio` /
`ollama serve` on the host and sends `SIGSTOP` before each heavy GPU stage,
then `SIGCONT` after. This was a pragmatic fix for one specific memory
pressure case — LM Studio sitting on ~8 GB of unified RAM while a Wan video
render needs every GB it can get.

But the same idea applies to lots of other co-resident services:

- **ComfyUI** — when it's not actively the inference target, its loaded
  models are dead weight. There's already a `/free` endpoint that unloads
  them; we should be able to fire it from the scheduler too, not just from
  `free_between` mid-stage.
- **Qwen-TTS worker container** — sits idle 99% of the time but eats GPU
  context. `docker stop` it during long video renders and `docker start`
  on stage exit; the next `/tts` request takes a cold-start hit but that's
  acceptable.
- **Ollama, vLLM, llama.cpp, etc.** — different LLM runtimes, same
  pattern: pause the daemon, free the RAM.

PR #40's one-toggle, one-method design doesn't scale. Adding each new
service as its own toggle would clutter Settings and duplicate scheduler
plumbing. Instead, this PR generalizes the lifecycle into a list of
services, each with its own suspension method, dispatched centrally.

## Existing pieces to reuse

- **`slopfinity/scheduler.py::acquire_gpu`** — the per-stage entry/exit
  context manager. Already the right hook point: it fires after the GPU
  lock is held and before the stage yields, then again on `finally`.
- **`slopfinity/scheduler.py::free_between`** — the existing REST
  `unload_models` POST to ComfyUI's `/free`. The new `rest_unload`
  method generalizes this pattern to any endpoint.
- **PR #40's `suspend_llm()` / `resume_llm()`** (lines 304–331 of
  `scheduler.py`) — the SIGSTOP/SIGCONT mechanic. Becomes the `sigstop`
  method in the new framework. The `LLM_PROCESS_HINTS` list + `pgrep`
  matching logic moves into the dispatcher's `_sigstop` helper.

## Methods

| Method        | Suspend op                 | Resume op           | Latency on suspend | Latency on resume | Memory freed | Notes |
|---------------|----------------------------|---------------------|--------------------|-------------------|--------------|-------|
| `sigstop`     | `kill -SIGSTOP <pid>`      | `kill -SIGCONT`     | ~10 ms             | ~10 ms            | ~0 (RSS stays resident — process is descheduled, not unloaded) | Truly reversible. Frees CPU and reduces memory bus contention; does NOT actually return RAM to the kernel. Best for "get out of the way" semantics. |
| `rest_unload` | POST configured endpoint   | (no-op — lazy)      | ~50–500 ms         | next request pays | Up to all model RAM | Keeps server responsive; user pays a model-reload tax on next request. ComfyUI `/free`, vLLM `/v1/load_lora_adapters?unload`, etc. |
| `docker_stop` | `docker stop <name>`       | `docker start`      | ~5 s               | ~30 s cold-start  | All container RAM | Heavyweight. Use for services with their own process tree (Qwen-TTS, dedicated llama.cpp container) and only for long stages. |
| `sigterm`     | `kill -SIGTERM <pid>`      | (no-op)             | ~100 ms            | manual restart    | All process RAM | One-shot graceful shutdown. Use sparingly — there's no automatic resume. |

Resume semantics are deliberately asymmetric: `rest_unload` and `sigterm`
have no automatic resume because the cost of "bringing it back" doesn't
belong on the GPU stage exit's critical path. The next request from the
user (or the background watchdog) will re-warm the service organically.

## Config schema

```python
DEFAULT_AUTO_SUSPEND = [
    {"id": "lmstudio", "label": "LLM (LM Studio)", "enabled": True,
     "method": "sigstop", "process_name": "LM Studio"},
    {"id": "comfyui", "label": "ComfyUI", "enabled": False,
     "method": "rest_unload", "endpoint": "http://localhost:8188/free"},
    {"id": "qwen-tts", "label": "Qwen-TTS worker", "enabled": False,
     "method": "docker_stop", "container": "strix-halo-qwen-tts"},
    {"id": "ollama", "label": "Ollama LLM", "enabled": False,
     "method": "sigstop", "process_name": "ollama"},
]
```

Persisted under `auto_suspend` at the top level of `config.json`.
`load_config()` merges defaults so an existing config file gets the
canonical list on first read after upgrade.

## Scheduler wiring

`acquire_gpu` calls `auto_suspend.suspend_all(...)` after the GPU lock is
acquired and `stage_start` is emitted, and `auto_suspend.resume_all(...)`
in the `finally` block, before `stage_end`. Each call emits an
`auto_suspend_start` / `auto_suspend_end` event with per-entry results so
the dashboard can show what happened (e.g. `lmstudio: ok`,
`comfyui: ok freed 12.4 GB`, `qwen-tts: error container not found`).

## Backwards compatibility

PR #40's `/llm/suspend` and `/llm/resume` REST endpoints stay as thin
wrappers that dispatch a synthetic `lmstudio`-only entry through the new
framework. The fleet runner uses these endpoints; we don't break it.

The PR #40 Settings toggle ("Auto-suspend LLM during GPU inference") is
removed. Its semantic equivalent is the `lmstudio` entry's `enabled`
checkbox in the new list UI, which defaults to ON — matching the value
most users would have set the old toggle to anyway.

The legacy `config.llm.auto_suspend` field is not migrated. Users who had
it OFF will see the new default ON for `lmstudio`; they can flip it back
in one click. This is the simplest path and avoids dragging migration
code along forever.

## Out of scope

- Wiring `acquire_gpu` into the fleet runner. The runner currently
  bypasses the scheduler; this PR doesn't change that. Auto-suspend will
  only fire for dashboard-spawned work, which mirrors the current state.
- A "test" button per entry. Worth adding later but not needed for v1.
- Conditional methods (e.g. "only suspend if stage budget > N GB").
  Add if real usage motivates it.
