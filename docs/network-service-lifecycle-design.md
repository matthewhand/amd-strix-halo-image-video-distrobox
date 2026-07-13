# Network service lifecycle — Design

> **Status:** Implemented as `slopfinity/service_registry.py` +
> `config.network_services` + worker ensure hooks + `GET /services`.
> Complements (does not replace) [auto-suspend](auto-suspend-design.md) and the
> [memory-stage planner](memory-stage-planner-design.md).

## Problem

Toolbox workers are available as **network HTTP endpoints** (Qwen Image Studio
`:8180`, Qwen/Kokoro TTS `:8010`, HeartMuLa `:8011`, ComfyUI `:8188`). Slopfinity
can call them by URL instead of per-job `docker run` scripts.

Running the full `docker compose --profile slop` stack **always warm** keeps
every model-capable process resident on unified memory. On gfx1151 / 128 GB UMA
that routinely OOMs the host (idle Qwen Image alone can pin tens of GB).

We need:

1. **Awareness** — is the service listening and healthy?
2. **Ensure** — if a stage needs it, start it via a configured compose/script.
3. **Park** — when the planner evicts a model (or exclusive group conflicts),
   stop the container so RAM returns to the kernel.
4. **Generation stays on the wire** — scripts/compose are **lifecycle only**.

## Boundary vs existing subsystems

| Module | Owns | Does not |
|--------|------|----------|
| **`service_registry`** | Pipeline HTTP workers: probe / start / stop | Host LLMs, SIGSTOP of LM Studio |
| **`auto_suspend`** | Park **competitors** during a GPU stage (LLM, optional extras) | Bring pipeline workers up |
| **`memory_planner`** | Belady load/keep/evict of **model names** under GB budget | Docker I/O (registry maps models → services) |

**Rule:** default `auto_suspend` entries for `qwen-tts` / pipeline containers stay
**disabled**. The registry owns those containers so we do not double-stop or
fight ensure_up.

## Flow

```
memory_planner (optional use_planner)
   load model X  ──►  ensure_up(service_for(X))
   evict model Y ──►  ensure_down(service_for(Y))

worker.run_stage
   ensure_for_stage(stage, model)
   POST/GET to base URL from env / registry
```

```
ensure_up(id):
  if probe(id).ok → return healthy
  run start_cmd
  poll probe until ok or timeout → fail with detail
```

## Schema (`config.network_services`)

List of dicts (merged like `auto_suspend` on load):

| Field | Meaning |
|-------|---------|
| `id` | Stable key (`qwen-image`, `qwen-tts`, `heartmula`, `comfyui`) |
| `label` | UI label |
| `enabled` | If false, ensure is no-op (probe still works) |
| `health_url` | GET probe target (optional if derivable from base) |
| `health_path` | Path joined to base when deriving health (`/health`, `/docs`, …) |
| `base_url` | Default base for workers (overridden by env if set) |
| `base_url_env` | Env var name workers already use (`TTS_WORKER_URL`, …) |
| `container_name` | Fixed compose `container_name` for start/stop |
| `compose_service` / `compose_profile` | For `lifecycle_mode=compose` |
| `lifecycle_mode` | `compose` \| `container` \| `cmd` \| `none` |
| `start_cmd` / `stop_cmd` | Override or compose-mode start; stop usually synthesized |
| `stage_keys` | List of `"stage:model"` or `"stage:*"` matchers |
| `budget_gb` | Idle / peak hint for UI (planner still uses `STAGE_BUDGETS`) |
| `exclusive_group` | Optional string; ensure_up stops other members first |
| `ensure_timeout_s` | Max wait after start (default 120) |
| `docker_host` / `docker_context` | Optional per-service Docker targeting |

### Defaults (toolbox)

| id | container | stages | exclusive |
|----|-----------|--------|-----------|
| `qwen-image` | `strix-halo-qwen-image` | `image:qwen` only | uma-heavy |
| `qwen-tts` | `strix-halo-qwen-tts` | `tts:*` | — |
| `heartmula` | `strix-halo-heartmula` | `audio:*` | uma-heavy |
| `comfyui` | `strix-halo-comfyui` | LTX image/video/upscale | uma-heavy |

Default start remains **compose up**; stop is **docker stop**. Set
`lifecycle_mode: container` for pure `docker start` after first create.

`slop` profile means **defined for ensure**, not **always running**.

### Control plane env

`DOCKER_HOST` / `SLOPFINITY_DOCKER_HOST`, `DOCKER_CONTEXT` /
`SLOPFINITY_DOCKER_CONTEXT`, `SLOPFINITY_COMPOSE_DIR`, `SLOPFINITY_DOCKER_BIN`.

Remote: SSH Docker context + LAN HTTP URLs (not multi-host k8s).

## API

| Function | Behaviour |
|----------|-----------|
| `probe(id)` | GET derived/explicit health → `{ok, status, latency_ms, detail}` |
| `ensure_up(id)` | probe → resolve start → poll |
| `ensure_down(id)` | stop (idempotent) |
| `ensure_down_group(group)` | stop all members of exclusive group |
| `ensure_for_stage(stage, model)` | match service → ensure_up; or clear uma-heavy for one-shots |
| `normalize_stage(stage)` | map `Base Image` / `TTS` → `image` / `tts` |
| `status_all()` | probe every enabled service |
| `service_for_stage(stage, model)` | resolve id or None |

HTTP: `GET /services` → `status_all()` snapshot for the dashboard.

## Worker contract

Before network generation:

1. `ensure_for_stage(...)` (or explicit `ensure_up("qwen-tts")`).
2. On failure → stage error string includes start/probe detail (no silent skip).
3. POST using URL from env (`TTS_WORKER_URL`, `HEARTMULA_URL`, `IMAGE_API_URL`,
   `SLOPFINITY_COMFY_URL`) with registry `base_url` as fallback.

Image/Audio keep a **docker-run fallback** only when
`SLOPFINITY_IMAGE_MODE=docker` / `SLOPFINITY_AUDIO_MODE=docker` (legacy).

## Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| Container stop on eviction | True UMA free | Cold-start 30–120 s next use |
| Comfy `rest_unload` for weights | Faster than container bounce | Idle Comfy process still holds some RAM |
| Ensure in worker vs only scheduler | Works even without coordinator | Slightly more call sites |
| Probe via HTTP not docker inspect | Matches real client path | Needs a reachable health URL |

## Non-goals

- Multi-cluster k8s scheduling (single Docker context + LAN HTTP is in scope)
- Replacing Comfy graph unload with stop/start every step
- Changing toolbox serve contracts (`/health`, `/tts`, `/music`)

## Related

- Toolbox: `docker-compose.yaml` profiles, `scripts/*_serve.py`,
  `scripts/slopfinity_http.py`
- [Auto-suspend](auto-suspend-design.md) — park competitors
- [Memory-stage planner](memory-stage-planner-design.md) — Belady resident set
