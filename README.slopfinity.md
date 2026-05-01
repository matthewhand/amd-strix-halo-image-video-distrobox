# Slopfinity

> A self-hosted dashboard for orchestrating local generative-AI fleets — image, video, music, voice — into chained "stories" or one-shot drops. Comfy slop, made queueable.

[![Demo bundle](https://img.shields.io/badge/demo-online-ff79c6)](https://matthewhand.github.io/amd-strix-halo-image-video-distrobox/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.x_preview-orange)](#status)

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
| **Queue card** | All queued / running / completed / failed jobs with pause, cancel, edit, requeue. |
| **Slop gallery** | Thumbnail grid of every generated PNG/MP4/WAV with filter pills (image / video / music / speech / intermediates / frames) and lightbox. |
| **Settings → Endpoints** | Three URLs (LLM, TTS, ComfyUI) so you can point at your own services. SSRF-guarded; metadata services blocked unconditionally. |
| **Branding profiles** | Re-skin via `branding/*.json` profiles. Theme colors, app name, taglines, footer text — all overridable. |

---

## Status

**v0.x preview.** Production-ready for single-user local installs. Multi-user / public-internet exposure is **not** supported yet — see the security section below for why.

What's stable:
- All 4 modes work end-to-end on the bundled backend (Strix Halo + ROCm + ComfyUI + Qwen-Image + LTX + Heartmula + Qwen3-TTS).
- 0-auth single-user-loopback by default. CSRF + Origin checks block drive-by abuse.
- SSRF guard validates every backend URL written through Settings.
- Atomic queue writes; multi-worker race-free.
- `/healthz` + `/readyz` probes for monitors / docker healthchecks.
- Service worker auto-versions cache from shell-asset hash.

What's preview-tier:
- Single-user only. No auth, no roles, no shared workspaces.
- Settings UX on phones is functional but cramped — vertical-tabs refactor planned.
- Demo bundle works but doesn't prove every flow end-to-end (image+video preview yes; chat yes; story stitching yes; music yes).
- Backend abstraction is leaky (see "Currently bundled with the toolbox image" section below) — most things go through HTTP cleanly, but a few `docker run --rm` codepaths still exist in workers/.

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
git clone https://github.com/matthewhand/slopfinity.git    # (post-spinoff)
cd slopfinity
pip install -r requirements.txt
SLOPFINITY_BIND_HOST=127.0.0.1 \
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
- **Heartmula** (text→music; CLI launcher today, HTTP wrapper planned)

If you don't have AMD Strix Halo + ROCm hardware, you'll want to point Slopfinity at your own services instead. The dashboard doesn't care what runs the inference as long as the URLs respond to the expected schemas.

---

## Architecture

Slopfinity itself is small (~280 KB of Python + ~700 KB of hand-written JS, no React/Vue, no build step beyond Tailwind). Most of the code is the queue + worker orchestration; the HTTP integrations are thin wrappers.

| Subsystem | LOC | Purpose |
|---|---|---|
| `slopfinity/server.py` | ~3.2k | FastAPI app, all HTTP + WebSocket handlers |
| `slopfinity/static/app.js` | ~10.6k | Hand-authored vanilla JS, mode switching, queue rendering, chat, story log |
| `slopfinity/static/app.css` | ~4.3k | Hand + Tailwind CSS (DaisyUI 4.10 from CDN) |
| `slopfinity/templates/index.html` | ~3.7k | Single-page Jinja template |
| `slopfinity/workers/` | ~1.2k | Per-stage worker classes (Concept / Image / Video / Audio / TTS / Post / Merge) |
| `slopfinity/coordinator.py` | ~0.4k | Phase-4 multi-worker dispatcher |
| `slopfinity/scheduler.py` | ~0.7k | GPU lock + budget gating |
| `slopfinity/llm/` | ~0.5k | Provider registry (LM Studio / Ollama / vLLM / OpenAI) |

The full integration tree:

1. **HTTP direct** — LLM, TTS, ComfyUI submit, health probes
2. **subprocess** — host ffmpeg (mux, concat)
3. **`docker run --rm`** — image, video, audio, post (currently routes through the bundled toolbox image; abstraction in progress)
4. **`docker exec`** — fallback ffmpeg into long-running ComfyUI container

Items 3 + 4 are the parts that tie Slopfinity to a specific backend image. The plan is to route everything through ComfyUI HTTP workflows and drop the `docker run --rm` paths.

---

## Configuration

**Loopback bind by default** (since v330). The dashboard binds `127.0.0.1:9099` so it isn't exposed to the LAN by default. Operators who need LAN/proxy access:

```bash
export SLOPFINITY_BIND_HOST=0.0.0.0           # or specific interface
export SLOPFINITY_TRUSTED_ORIGINS=http://my-host:9099,https://kiosk.local
python3 dark_server.py
```

The CSRF middleware accepts only same-origin (the bind host) by default; `SLOPFINITY_TRUSTED_ORIGINS` extends the allowlist.

Other env vars:
- `SLOPFINITY_BIND_PORT` (default `9099`)
- `SLOPFINITY_DISABLE_CSRF=1` — escape hatch for scripted automation that can't carry Origin/Referer headers
- `SLOPFINITY_STATE_DIR` — where queue.json + state.json + config.json live
- `EXP_DIR` — where slop files land (writes go here, dashboard serves at `/files/<name>`)
- `TTS_WORKER_URL` — preferred over Settings if both set (env wins)
- `HF_HOME` — HuggingFace cache dir for any downloaded models

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

A static interactive demo runs entirely from canned fixtures with no backend — try every UI flow without installing anything.

→ **[matthewhand.github.io/amd-strix-halo-image-video-distrobox/](https://matthewhand.github.io/amd-strix-halo-image-video-distrobox/)** (after gh-pages workflow lands)

The demo bundle is built by `make demo` from the [static-demo-builder](https://github.com/anthropics/claude-code) skill — a self-contained `dist/demo/` you can drop on any static host. Embed via:

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
| **P0** | Drop `docker run --rm IMAGE=…` paths in workers/; route everything through ComfyUI HTTP | Removes the only non-URL backend coupling |
| **P0** | Pluggable stats provider (rocm-smi / nvidia-smi / fallback) | Currently ROCm-specific; non-AMD users see "—" everywhere |
| **P1** | Heartmula HTTP wrapper (~50 LOC FastAPI) | Music gen becomes URL-configurable like everything else |
| **P1** | Settings vertical-tabs refactor (DaisyUI menu component) | 10 horizontal tabs cramped on phones |
| **P2** | Multi-user auth (OIDC or simple session cookies) | Enables small-team / kiosk-with-login deployments |
| **P2** | Replay attack tests in CI for the CSRF/SSRF middleware | Lock in the security fixes |
| **P3** | rrweb-recorded interactive walkthrough on the docs page | Better than MP4 for engagement |
| **P3** | Browser-side ffmpeg.wasm story stitching | Removes the last host-binary dependency |

---

## Contributing

This README is the spin-off draft. Until the project is forked into its own repo, contributions land in the toolbox repo: [`matthewhand/amd-strix-halo-image-video-distrobox`](https://github.com/matthewhand/amd-strix-halo-image-video-distrobox).

The dashboard is hand-written vanilla JS by deliberate choice. PRs that introduce a build pipeline (Vite, Webpack, etc.) or framework (React, Vue) will get a hard "no" — the no-build-step constraint is the project's identity. Tailwind CLI for CSS is the one exception.

---

## License

Apache-2.0. See `LICENSE`.

---

## Acknowledgements

- **DaisyUI** for the component layer
- **ComfyUI** for the workflow engine the gen-side leans on
- **Qwen** team for Qwen-Image, Qwen3-TTS, and the LLMs that drive the pipeline
- **HeartMuLa** team for the music model
- **kyuz0** for the upstream toolbox repo this dashboard was built against
