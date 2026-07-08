# Slopfinity network services (lean lifecycle)

Slopfinity talks to toolbox workers over HTTP. Containers need not stay warm.

## Env URLs

| Env | Default | Role |
|-----|---------|------|
| `TTS_WORKER_URL` | `http://localhost:8010/tts` | TTS POST |
| `HEARTMULA_URL` | `http://127.0.0.1:8011` | Music base (`/music`, `/health`) |
| `IMAGE_API_URL` | `http://127.0.0.1:8180` | Qwen Image Studio base |
| `SLOPFINITY_COMFY_URL` | `http://localhost:8188` | ComfyUI |

## Ensure-on-demand

`slopfinity/service_registry.py` probes `health_url` and runs configured
`start_cmd` / `stop_cmd` (usually `docker compose … up -d` / `docker stop`).

- Dashboard-only: leave heavy workers stopped (saves UMA).
- First stage that needs a worker: `ensure_up` cold-starts it (can take 30–120s).
- Planner eviction / exclusive groups: `ensure_down` frees RAM.

Status: `GET /services` on the dashboard port.

## Compose profiles (toolbox)

```bash
# Define services; do not require all up forever
docker compose --profile slop config

# Manual stop (same as ensure_down)
docker stop strix-halo-qwen-image

# Manual start (same as ensure_up)
docker compose --profile qwen-image up -d qwen-image-service
```

Legacy per-job docker generation: set `SLOPFINITY_IMAGE_MODE=docker` or
`SLOPFINITY_AUDIO_MODE=docker`.

See [network-service-lifecycle-design.md](network-service-lifecycle-design.md).
