# Toolbox vs Slopfinity boundary

Honest ownership map for this monorepo as of 2026-07-13 (branch `wip/from-toolbox-20260713`).
Not aspirational design — what the tree actually contains today.

Related:
- Private-repo / submodule status: [`docs/slopfinity-private-repo.md`](slopfinity-private-repo.md)
- Product / operator docs: [`README.slopfinity.md`](../README.slopfinity.md)
- Toolbox stack: [`README.md`](../README.md)

## Two products, one checkout

| Layer | What it is | Canonical paths |
|-------|------------|-----------------|
| **Toolbox** | AMD Strix Halo + ROCm image/video **backends**: Docker image, compose profiles, model download/patch scripts, ComfyUI wiring, per-service launchers | `Dockerfile`, `docker-compose.yaml`, `scripts/*_launcher.py`, `scripts/*_serve.py`, `scripts/comfyui_api.py`, `generate_ltx_workflow.py`, model/cache dirs |
| **Slopfinity** | FastAPI + Jinja + vanilla-JS **dashboard** that queues work and talks to backends over **HTTP URLs + env** | In-tree `slopfinity/` package, `dark_server.py`, `bin/slopfinity`, `requirements-slopfinity.txt`, `README.slopfinity.md` |

They are **not** a clean submodule split yet. Dashboard code lives **in-tree** under `slopfinity/`. A private mirror (`matthewhand/slopfinity`) exists for extraction WIP; see the private-repo doc.

## Who owns what

### Toolbox owns

- Host/container image and GPU runtime assumptions (ROCm, Strix Halo).
- Docker Compose services/profiles: ComfyUI, Qwen-Image, Qwen3-TTS, HeartMuLa, etc.
- Launcher / serve wrappers under `scripts/` (e.g. `qwen_launcher.py`, `qwen_tts_serve.py`, `heartmula_serve.py`, `ltx_launcher.py`, `dramabox_launcher.py`).
- Workflow generators and Comfy patches (`generate_ltx_workflow.py`, `scripts/apply_qwen_patches.py`, `scripts/patch_comfyui.py`, …).
- Model download helpers and large artifact dirs (not documented as product surface).

### Slopfinity owns

- Dashboard HTTP API, WebSocket, queue/state/config persistence (`slopfinity/server.py`, `slopfinity/routers/`, `slopfinity/config.py`).
- UI: `slopfinity/templates/`, `slopfinity/static/` (no React build step; Tailwind/DaisyUI from CDN + hand CSS).
- Orchestration helpers: `scheduler.py`, `stage_gate.py`, `memory_planner.py`, `service_registry.py`, `auto_suspend.py`, `coordinator.py`, `workers/`, `llm/`.
- Operator contract: bind host/port, CSRF, SSRF guards, `SLOPFINITY_*` env vars, Settings → Endpoints.

### Shared / grey area (honest)

| Item | Reality today |
|------|----------------|
| `run_fleet.py` (repo root) | Legacy **linear** fleet orchestrator still used as primary runner from the dashboard path. Lives at toolbox root, not under `slopfinity/`. |
| `slopfinity/workers.py` | Thin async wrappers that still call `docker run --rm` for some stages (fallback / WAN). Couples dashboard to compose image names. |
| `slopfinity/service_registry.py` | Dashboard-side lifecycle for **toolbox** compose profiles (`ensure_up` / `ensure_down`). Boundary is HTTP+compose, not a pure URL-only plug-in. |
| `slopfinity/ltx_comfy.py`, `wan_cli.py` | Backend-specific adapters inside the dashboard package. |
| `scripts/slopfinity_http.py` | Toolbox-side helper for dashboard HTTP, not the main entrypoint. |
| Design notes under `docs/*-design.md` | Plans / history for queueing, concurrency, auto-suspend, memory planner, network services. Prefer code + `README.slopfinity.md` for **current** behavior. |

## Runtime contract (stable)

Slopfinity should treat inference as **remote services**:

1. **LLM** — OpenAI-compatible HTTP (Settings → Endpoints or `SLOPFINITY_LLM_*`).
2. **ComfyUI** — `/prompt` (and related) HTTP; LTX-2 path prefers this when `SLOPFINITY_LTX_MODE` defaults to `http`.
3. **TTS** — HTTP worker (default `TTS_WORKER_URL` → `http://localhost:8010/tts`).
4. **Filesystem** — experiment/output dir via `SLOPFINITY_EXP_DIR` / `SLOPFINITY_STATE_DIR`, served at `/files/`.

Anything that still requires `docker run --rm` (notably **WAN 2.2** paths today) is a known leak past that contract — tracked as P0 in `README.slopfinity.md`.

## Entrypoints (verified present)

| Entry | Role |
|-------|------|
| `python3 dark_server.py` | Thin wrapper; primary documented standalone start |
| `bin/slopfinity` | CLI/wrapper around the dashboard |
| `docker compose --profile slop up -d` | Bundled stack (toolbox compose) |

Default bind: `127.0.0.1:9099` (loopback). See `README.slopfinity.md` for env and security.

## What is *not* the boundary

- **Private GitHub ACL** is packaging history, not runtime architecture. Canonical dashboard code for day-to-day work is this toolbox tree's `slopfinity/`.
- **PWA / service worker** docs in `slopfinity/PWA.md` are product details, not toolbox ownership.
- **USER_GUIDE.md** and screenshot-era UI notes may lag the live template/JS; treat them as operator aids, not source of truth for layout.

## When extracting a standalone Slopfinity repo

1. Keep the **URL + env** contract above.
2. Do not submodule-add private `main` as-is (nested `slopfinity/slopfinity/` break); reshape first — steps in `slopfinity-private-repo.md`.
3. Decide explicitly what happens to `run_fleet.py` and docker-coupled workers (move, replace with HTTP-only, or leave as optional toolbox plugin).


## What is NOT shipped (public product surface)

These exist in a developer checkout or private mirror but are **not** the public "run me" surface:

| Item | Notes |
|------|-------|
| Model weights / HF caches | `huggingface-cache/`, `comfy-models/`, `wan-models/`, etc. — large, machine-local |
| `dist/demo/` | Build artifact from `make demo`; **not** committed; gh-pages workflow **not** active |
| Private `matthewhand/slopfinity` history | Packaging/extract WIP; **canonical dashboard code is in-tree `slopfinity/`** |
| `.claude/worktrees/*` | Local agent worktrees if present — ignore |
| Host secrets / `config.json` runtime state | Under `SLOPFINITY_STATE_DIR` / experiment dirs — never commit |
| Root `LICENSE` | **Currently missing** — do not claim a license until one is added |

## Sync rules (private remote)

```text
origin     → matthewhand/amd-strix-halo-image-video-distrobox   (public toolbox)
slopfinity → matthewhand/slopfinity                            (private mirror)
```

1. Day-to-day development: commit in the **toolbox** tree (`origin`).
2. Optional mirror: `git push -u slopfinity HEAD:wip/from-toolbox-YYYYMMDD` (never force-push private `main` without review).
3. Do **not** `git submodule add` private `main` into `slopfinity/` until the private tree is reshaped (nested `slopfinity/slopfinity/` break).
4. Local clone directory may be named `amd-strix-halo-image-video-toolboxes` while GitHub remote is `…-distrobox` — same remote, different folder name.

## Grey packaging paths (also see Shared table)

| Path | Role |
|------|------|
| `branding/*.json` | Dashboard theme profiles (repo root) |
| `demo/` + `demo/skill-templates/` | Static demo **source** + local builder (not Claude Code / not deployed gh-pages) |
| `Makefile` targets `demo`, `demo-serve`, `demo-smoke`, `demo-clean` | Build/serve demo locally |

## Verification stamp

- **Date:** 2026-07-13
- **Branch:** `wip/from-toolbox-20260713`
- **Method:** filesystem presence of entrypoints and launchers; `git remote -v`; cross-read of README.slopfinity status/architecture; skeptic audit `/tmp/goal-achievement-skeptic-report.md`
- **Related index:** [`slopfinity-docs-index.md`](slopfinity-docs-index.md)

