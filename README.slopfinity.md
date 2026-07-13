# Slopfinity

> A self-hosted dashboard for orchestrating local generative-AI fleets — image, video, music, voice — into chained "stories" or one-shot drops. Comfy slop, made queueable.

[![Demo bundle](https://img.shields.io/badge/demo-source_only-lightgrey)](#demo)
[![License](https://img.shields.io/badge/license-not_in_tree-lightgrey)](#license)
[![Status](https://img.shields.io/badge/status-v0.x_preview-orange)](#status)

---

## Documentation map (read this first)

| Doc | Role |
|-----|------|
| **This file** (`README.slopfinity.md`) | Operator / product surface for the dashboard |
| [`README.md` §14](README.md#14-slopfinity-dashboard-ui) | Toolbox monorepo entry + UI layout (current mermaid) |
| [`docs/slopfinity-toolbox-boundary.md`](docs/slopfinity-toolbox-boundary.md) | What toolbox owns vs dashboard; runtime contract; NOT shipped |
| [`docs/slopfinity-private-repo.md`](docs/slopfinity-private-repo.md) | Private GitHub mirror / extract / submodule plan |
| [`docs/slopfinity-docs-index.md`](docs/slopfinity-docs-index.md) | Full index + last-verified method |

**Last verified:** 2026-07-13 — method: `wc -l` on package paths, `test -f` entrypoints, `git remote -v`, `ls` demo/Makefile workflows, greps for overclaims. See `/tmp/goal-achievement-skeptic-report.md` for the adversarial audit of the prior docs pass.

---

## TL;DR

You run the inference services (LLM, ComfyUI, TTS, music). Slopfinity is the dashboard that pipes a prompt through them and gives you slop with progress bars, queue management, and a story-stitcher.

```
┌─ Slopfinity (FastAPI + Jinja + DaisyUI on :9099) ─────────────────────┐
│                                                                       │
│   Subjects ──┐                ┌── Queue ──┐         ┌── Slop gallery  │
│   prompt /   │   /chat      │  pending   │         │  PNG / MP4 /    │
│   beats /    ├─→ /inject    ├── running ──┼─pubs──→ │  WAV thumbs     │
│   chat       │   /enhance   │  done      │         │  + lightbox     │
│              └──────────────┘            └─────────┘                  │
└──────┬────────────┬──────────────┬───────────────┬────────────────────┘
       │            │              │               │
       ↓            ↓              ↓               ↓
   LLM HTTP    ComfyUI HTTP    TTS HTTP      filesystem (EXP_DIR)
  (LM Studio,  (any backend     (Qwen3-TTS    (writes go here, dashboard
   Ollama,      with /prompt)    or Kokoro)    serves them at /files/)
   OpenAI…)
```

Three external services. Each addressed by a URL configurable in **Settings → Endpoints**. Defaults to loopback so the bundled stack works out of the box; point at remote services when you want.

---

## What's actually in here

| Surface | What it does |
|---|---|
| **Simple mode** | One prompt textbox → one queue submit → one slop output. The "I just want art" flow. |
| **Story mode** (Endless) | Multi-beat composer. Each beat → a clip. Optional auto-stitch concatenates `FINAL_*.mp4` clips into one continuous story. |
| **Raw mode** | Per-stage textareas (image / video / music / voice). Skips the LLM rewrite — you write the prompts the model sees. Power users only. |
| **Chat mode** | Conversational driver. The LLM has tool-calls for `queue_inject`, `pause_queue`, `requeue`, etc. — type natural language, watch the dashboard react. |
| **Suggestions toggle** | LLM-generated prompt chips. Counterintuitive-by-design: turning Suggestions **ON** makes the surrounding config controls *recede* (the chips are now the active surface) — on desktop the secondary controls dim to ~45% at rest and restore on hover/focus; on mobile (≤767px) the cluster collapses to just the toggle (shortened label) to hand the vertical space back to the prompt. |
| **Queue card** | All queued / running / completed / failed jobs with pause, cancel, edit, requeue. |
| **Slop gallery** | Thumbnail grid of every generated PNG/MP4/WAV with filter pills (image / video / music / speech / intermediates / frames) and lightbox. |
| **Settings → Endpoints** | Three URLs (LLM, TTS, ComfyUI) so you can point at your own services. SSRF-guarded; metadata services blocked unconditionally. |
| **Branding profiles** | Re-skin via `branding/*.json` profiles. Theme colors, app name, taglines, footer text — all overridable. |

---

## Status

**v0.x preview.** Usable for **single-user loopback** installs on the bundled toolbox stack; not a hardened product release. Multi-user / public-internet exposure is **not** supported — see Security. Several paths remain preview-tier (below).

What's stable:
- All 4 modes work end-to-end on the bundled backend (Strix Halo + ROCm + ComfyUI + Qwen-Image + LTX + Heartmula + Qwen3-TTS).
- **TTS is wired into the fleet loop.** When `tts_model` is set (≠ `none`) in the pipeline config, each iteration synthesizes narration through the TTS worker on `:8010` (Qwen3-TTS / Kokoro) and muxes the voice track onto the `FINAL_*.mp4`. If music (HeartMuLa) is also produced, the two tracks are mixed (`amix`) so the narration rides over the music; otherwise the single track is used directly. Driven by `tts_model` / `tts_voice` / `tts_text` config (see `run_fleet.py`).
- 0-auth single-user-loopback by default. CSRF + Origin checks block drive-by abuse.
- SSRF guard validates every backend URL written through Settings.
- Atomic queue writes with flock-backed persistence; residual multi-writer races are called out in `config.py` comments (known correctness gap under heavy concurrency).
- `/healthz` + `/readyz` probes for monitors / docker healthchecks.
- Service worker auto-versions cache from shell-asset hash.

What's preview-tier:
- Single-user only. No auth, no roles, no shared workspaces.
- Settings UX on phones is functional but cramped — 9 horizontal tabs need vertical-tabs refactor.
- **Coordinator (Phase 4)** exists as `slopfinity/coordinator.py` with 7 concurrent stage workers, but the legacy `run_fleet.py` linear orchestrator is still the primary runner. The coordinator is opt-in via its CLI or API endpoints.
- Demo bundle source exists (`demo/`) but the `dist/demo/` build and gh-pages deployment workflow are not yet active.
- Backend abstraction is leaky: LTX-2 stages prefer ComfyUI HTTP (controlled by `SLOPFINITY_LTX_MODE`), but WAN 2.2 still requires `docker run --rm` paths. Service registry (`service_registry.py`) manages Docker Compose lifecycle for network workers.

---

## Quick start (bundled toolbox image)

The fastest path: run the AMD Strix Halo Image+Video toolbox stack via docker-compose. Slopfinity is included.

```bash
git clone https://github.com/matthewhand/amd-strix-halo-image-video-distrobox.git
cd amd-strix-halo-image-video-distrobox
cp docker-compose.override.yaml.example docker-compose.override.yaml
docker compose --profile slop up -d
# Dashboard at http://localhost:9099
```

That gives you Slopfinity + ComfyUI + Qwen-Image + Qwen3-TTS + Heartmula all wired together.

---

## Quick start (standalone, against your own backends)

Slopfinity itself is a Python+FastAPI app that talks to three URLs. If you have your own LLM / ComfyUI / TTS already running, you can run Slopfinity standalone:

```bash
# Slopfinity ships in-tree in the toolbox repo today. The private slopfinity
# standalone repo (https://github.com/matthewhand/slopfinity) is not yet
# public — see docs/slopfinity-private-repo.md for the extraction plan.
git clone https://github.com/matthewhand/amd-strix-halo-image-video-distrobox.git
cd amd-strix-halo-image-video-distrobox
pip install -r requirements-slopfinity.txt
SLOPFINITY_BIND_HOST=127.0.0.1 \
  SLOPFINITY_EXP_DIR=$PWD/comfy-outputs/experiments \
  python3 dark_server.py
# Open http://localhost:9099
# Then: Settings → Endpoints → set LLM / TTS / ComfyUI URLs to your services
```

What URLs you point at:
- **LLM**: anything OpenAI-compatible. LM Studio (`:1234/v1`), Ollama (`:11434/v1`), vLLM, OpenRouter, OpenAI itself.
- **ComfyUI**: any ComfyUI install with `/prompt` HTTP API. Default port 8188.
- **TTS**: any HTTP service that takes `{ text, voice }` and returns a WAV. The bundled stack uses [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice); Kokoro-82M is a documented fallback. Roll your own with ~30 LOC of FastAPI if needed.

---

## Currently bundled with the toolbox image

Slopfinity ships in the [amd-strix-halo-image-video-distrobox](https://github.com/matthewhand/amd-strix-halo-image-video-distrobox) repo today. The bundled stack provides:

- **ComfyUI** with Strix Halo / ROCm support (image + video workflows)
- **Qwen-Image** (image generation backbone)
- **LTX-Video** (video generation)
- **Qwen3-TTS** (voice synthesis)
- **Heartmula** (text→music; HTTP wrapper shipped — see `scripts/heartmula_serve.py` and `compose` profile `heartmula`)

If you don't have AMD Strix Halo + ROCm hardware, you'll want to point Slopfinity at your own services instead. The dashboard doesn't care what runs the inference as long as the URLs respond to the expected schemas.

---

## Architecture

Slopfinity itself is small (~280 KB of Python + ~700 KB of hand-written JS, no React/Vue, no build step beyond Tailwind). Most of the code is the queue + worker orchestration; the HTTP integrations are thin wrappers.

| Subsystem | LOC (`wc -l`, 2026-07-13) | Purpose |
|---|---:|---|
| `slopfinity/server.py` | 1379 | FastAPI app, HTTP + WebSocket handlers |
| `slopfinity/static/app.js` | 11120 | Hand-authored vanilla JS (modes, queue, chat, story) |
| `slopfinity/static/app.css` | 4572 | Hand CSS + Tailwind/DaisyUI (CDN) |
| `slopfinity/templates/index.html` | 3732 | Single-page Jinja template |
| `slopfinity/workers/` | 1279 | Per-stage worker classes + base framework |
| `slopfinity/workers.py` | 186 | Thin async wrappers around `docker run --rm` fleet calls |
| `slopfinity/coordinator.py` | 332 | Phase-4 multi-worker dispatcher (**opt-in**; primary runner remains root `run_fleet.py`) |
| `slopfinity/scheduler.py` | 658 | GPU budget + auto-suspend + service lifecycle |
| `slopfinity/stage_gate.py` | 367 | Host-safe memory floor |
| `slopfinity/memory_planner.py` | 262 | Look-ahead planner for resident model sets |
| `slopfinity/service_registry.py` | 552 | Compose profile probe / ensure_up / ensure_down |
| `slopfinity/config.py` | 494 | Config + state + queue persistence (flock; residual races documented in-file) |
| `slopfinity/auto_suspend.py` | 302 | Suspend/resume co-resident services during GPU stages |
| `slopfinity/fanout.py` | 193 | Single-idea → multi-stage expansion |
| `slopfinity/llm/` | 723 | Provider registry + pool failover |
| `slopfinity/routers/` | 2833 | HTTP API surface by domain |
| `slopfinity/ltx_comfy.py` | ~0.5k | LTX ComfyUI HTTP adapter (backend-specific) |
| `slopfinity/wan_cli.py` | ~0.2k | WAN CLI/docker adapter (backend-specific) |
| `slopfinity/stats.py` | ~0.3k | Host/GPU stats helpers (ROCm-oriented today) |

The full integration tree:

1. **HTTP direct** — LLM, TTS, ComfyUI submit, health probes
2. **subprocess** — host ffmpeg (mux, concat)
3. **`docker run --rm`** — image, video, audio, post (fallback path when ComfyUI HTTP is unavailable; see `workers.py`)
4. **ComfyUI HTTP** — LTX-2 image/video/upscale via `/prompt` API (preferred path; controlled by `SLOPFINITY_LTX_MODE` env var, defaults to `http`)
5. **`docker exec`** — fallback ffmpeg into long-running ComfyUI container
6. **Service registry lifecycle** — `service_registry.py` manages Docker Compose profiles (qwen-image, qwen-tts, heartmula, comfyui) for network workers via `ensure_up`/`ensure_down`

See also **[`docs/slopfinity-toolbox-boundary.md`](docs/slopfinity-toolbox-boundary.md)** for what the toolbox owns vs what the dashboard owns, and [`docs/slopfinity-private-repo.md`](docs/slopfinity-private-repo.md) for the private-repo extract plan.

Items 3 + 5 are the parts that tie Slopfinity to a specific backend image. The `docker run --rm` paths remain as fallback but the primary path for LTX-2 stages is now ComfyUI HTTP. WAN 2.2 still requires docker run (no persistent HTTP service).

---

## Configuration

**Loopback bind by default** (since v330). The dashboard binds `127.0.0.1:9099` so it isn't exposed to the LAN by default. Operators who need LAN/proxy access:

```bash
export SLOPFINITY_BIND_HOST=0.0.0.0           # or specific interface
export SLOPFINITY_TRUSTED_ORIGINS=http://my-host:9099,https://kiosk.local
python3 dark_server.py
```

The CSRF middleware accepts only same-origin (the bind host) by default; `SLOPFINITY_TRUSTED_ORIGINS` extends the allowlist.

### Server / bind

- `SLOPFINITY_BIND_HOST` (default `127.0.0.1`) — interface to bind. Set `0.0.0.0` for LAN/proxy access.
- `SLOPFINITY_BIND_PORT` (default `9099`)
- `SLOPFINITY_TRUSTED_ORIGINS` — comma-separated full origins (e.g. `http://my-host:9099`) added to the CSRF allowlist.
- `SLOPFINITY_DISABLE_CSRF=1` — escape hatch for scripted automation that can't carry Origin/Referer headers.

### Paths / storage

- `SLOPFINITY_EXP_DIR` — where slop files land (writes go here, dashboard serves them at `/files/<name>`). Defaults to `/workspace` inside the container; if `/workspace` is missing **or not writable** (e.g. a read-only bind in CI/sandboxes), it falls back to `./comfy-outputs/experiments`. Set this when running locally so the dashboard isn't fighting a read-only `/workspace`.
- `SLOPFINITY_STATE_DIR` — where `queue.json` + `state.json` + `config.json` live (default `comfy-outputs/experiments`).
- `HF_HOME` — HuggingFace cache dir for any downloaded models.

### Backend endpoints (env overrides Settings)

The LLM pool reads these per request (`slopfinity/llm/pool.py::get_env_pool_config`). When `SLOPFINITY_LLM_PRIMARY_URL` is unset the primary falls back to the `llm` block saved in `config.json` (Settings → Endpoints), then to the legacy `localhost:1234` default.

- `SLOPFINITY_LLM_PRIMARY_URL` / `SLOPFINITY_LLM_PRIMARY_MODEL` — primary OpenAI-compatible LLM endpoint + model id.
- `SLOPFINITY_LLM_CPU_URL` / `SLOPFINITY_LLM_CPU_MODEL` — secondary "CPU" endpoint (default `http://localhost:11434/v1`, i.e. Ollama). Set to `""` to disable.
- `SLOPFINITY_LLM_FAILOVER_URLS` — comma-separated list of additional failover endpoints.
- `SLOPFINITY_LLM_FAILOVER_MODELS` — comma-separated model ids, positionally paired with the failover URLs (padded with blanks when shorter).
- `SLOPFINITY_COMFY_URL` — ComfyUI base URL used for the `/free` VRAM-unload calls in the scheduler (default `http://localhost:8188`).
- `TTS_WORKER_URL` — full URL of the TTS worker `/tts` endpoint (default `http://localhost:8010/tts`). Preferred over Settings if both set (env wins).

Settings → Endpoints persists the URLs to `config.json`; env vars override on each restart.

---

## Security model

- **Single-user, loopback-only by default.** Any deviation requires explicit env-var opt-in.
- **CSRF protection** on every mutating endpoint and the WebSocket handshake. Cross-origin requests rejected with 403.
- **SSRF guard** on every URL field in `/settings`. `file://`, RFC1918, and cloud metadata services blocked unless `allow_cloud_endpoints=true` is on AND the URL isn't a metadata endpoint.
- **No telemetry, no analytics, no third-party API calls.** Any third-party CDN dependency (DaisyUI from jsdelivr today) is documented and replaceable.
- **No credentials accepted as URL params.** API keys, when configured, live in `config.json` only.

What it doesn't have yet (v0.x preview limitations):
- No multi-user auth. **Don't expose to the public internet.** A future v1 may add OIDC.
- No rate limiting beyond the GPU's natural single-stream serialization.
- No input-length validation at the API layer — large prompts can OOM the LLM, gracefully but noisily.

---

## Demo

A static interactive demo can be built from canned fixtures with no backend — try every UI flow without installing anything.

→ **Planned public URL:** `https://matthewhand.github.io/amd-strix-halo-image-video-distrobox/` — **not deployed yet**. Demo source lives in `demo/`; build with `make demo` (serves via `make demo-serve` on port 8765).

The demo bundle is built by `make demo` using templates under `demo/skill-templates/` (local static-demo builder) — a self-contained `dist/demo/` you can drop on any static host. **`dist/demo/` is not committed; gh-pages is not deployed yet.** Embed via:

```html
<iframe src="…demo url…" width="100%" height="900" loading="lazy"
        sandbox="allow-scripts allow-same-origin allow-forms"></iframe>
```

The demo banner at the top says *"DEMO — AI outputs are samples"* so visitors know the variety they see is canned, not generated from their input.

---

## Roadmap (post-spinoff)

When this project moves to its own repo, the priorities are:

| Priority | Item | Why |
|---|---|---|
| **P0** | Drop `docker run --rm IMAGE=…` paths in workers/; route everything through ComfyUI HTTP | Removes the only non-URL backend coupling. LTX-2 stages already prefer HTTP (via `SLOPFINITY_LTX_MODE`). WAN 2.2 still docker-only. |
| **P0** | Pluggable stats provider (rocm-smi / nvidia-smi / fallback) | Currently ROCm-specific; non-AMD users see "—" everywhere |
| **P1** | ~~Heartmula HTTP wrapper~~ ✅ **DONE** | `scripts/heartmula_serve.py` + compose `heartmula` profile shipped |
| **P1** | Settings vertical-tabs refactor (DaisyUI menu component) | 9 horizontal tabs cramped on phones |
| **P2** | Multi-user auth (OIDC or simple session cookies) | Enables small-team / kiosk-with-login deployments |
| **P2** | Replay attack tests in CI for the CSRF/SSRF middleware | Lock in the security fixes |
| **P3** | rrweb-recorded interactive walkthrough on the docs page | Better than MP4 for engagement |
| **P3** | Browser-side ffmpeg.wasm story stitching | Removes the last host-binary dependency |

---

## Testing

CI runs three lanes on every push / PR to `main`:

- **Python suite** (`.github/workflows/python-tests.yml`) — the full `tests/` suite (excluding `tests/e2e_qwen_web_test.py`, which needs a real AMD GPU / ROCm). This job *gates* (no `continue-on-error`).
- **Playwright e2e** (`.github/workflows/playwright.yml`) — boots the dashboard against a stdlib mock LLM (`tests/mock_llm_server.py`) and runs the specs in `e2e/`. Currently non-blocking (`continue-on-error`) while the suite stabilises; failure-packet + report artifacts are always uploaded.
- **Lint** (`.github/workflows/lint.yml`) — stylelint + `node --check` on the hand-authored CSS/JS.

Running locally:

```bash
# Python suite. SLOPFINITY_EXP_DIR is required so tests don't try to
# write to a read-only /workspace (paths.py falls back otherwise, but
# pinning it keeps the state + outputs dirs writable and predictable).
export SLOPFINITY_EXP_DIR="$PWD/.pytest-exp"
export SLOPFINITY_STATE_DIR="$PWD/.pytest-exp"
mkdir -p "$SLOPFINITY_EXP_DIR"
python -m pytest tests/ --ignore=tests/e2e_qwen_web_test.py -q

# Frontend unit tests + lint (no build step).
npm ci
npm test          # vitest run js-tests/
npm run lint      # stylelint + node --check

# Playwright e2e (boots dark_server.py against the mock LLM).
npx playwright install --with-deps chromium
npx playwright test
```

`pytest.ini` sets `asyncio_mode = auto` so async tests need no per-test marker.

---

## Contributing

This README is the spin-off draft. Until the project is forked into its own repo, contributions land in the toolbox repo: [`matthewhand/amd-strix-halo-image-video-distrobox`](https://github.com/matthewhand/amd-strix-halo-image-video-distrobox).

The dashboard is hand-written vanilla JS by deliberate choice. PRs that introduce a build pipeline (Vite, Webpack, etc.) or framework (React, Vue) will get a hard "no" — the no-build-step constraint is the project's identity. Tailwind CLI for CSS is the one exception.

---

## License

**No root `LICENSE` file is present in this checkout** (verified 2026-07-13). The previous docs claimed Apache-2.0; that claim is **withdrawn until a root license is added**.

- Vendored/third-party trees carry their own licenses (e.g. `LTX-2/LICENSE`, `ComfyUI-LTXVideo/LICENSE`, `.heartlib/LICENSE`).
- Do not assume the dashboard package is Apache-2.0-licensed solely from older badges or spin-off drafts.
- When a root license is chosen, add `LICENSE` at the repo root and restore an accurate badge here.

---

## Acknowledgements

- **DaisyUI** for the component layer
- **ComfyUI** for the workflow engine the gen-side leans on
- **Qwen** team for Qwen-Image, Qwen3-TTS, and the LLMs that drive the pipeline
- **HeartMuLa** team for the music model
- **kyuz0** for the upstream toolbox repo this dashboard was built against
