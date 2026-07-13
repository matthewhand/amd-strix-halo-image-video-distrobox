# Slopfinity network services (lean lifecycle)

Slopfinity talks to toolbox workers over HTTP. Containers need not stay warm.

## Env URLs (data plane)

| Env | Default | Role |
|-----|---------|------|
| `TTS_WORKER_URL` | `http://localhost:8010/tts` | TTS POST |
| `HEARTMULA_URL` | `http://127.0.0.1:8011` | Music base (`/music`, `/health`) |
| `IMAGE_API_URL` | `http://127.0.0.1:8180` | Qwen Image Studio base |
| `SLOPFINITY_COMFY_URL` | `http://localhost:8188` | ComfyUI |

When these point at a remote GPU host, probes derive health from the same base
(`{base}/health`, `/docs`, or `/system_stats`) so you do not leave
`health_url` stuck on `127.0.0.1`.

## Ensure-on-demand (control plane)

`slopfinity/service_registry.py` probes health and runs lifecycle cmds:

| Mode | Start | Stop |
|------|-------|------|
| `compose` (default) | `docker compose --profile … up -d …` | `docker stop <container>` |
| `container` | `docker start <container>` | `docker stop <container>` |
| `none` | no-op | no-op |

- Dashboard-only: leave heavy workers stopped (saves UMA).
- First stage that needs a worker: `ensure_up` cold-starts it (can take 30–120s).
- Exclusive group **`uma-heavy`**: `qwen-image` / `heartmula` / `comfyui` — ensuring one stops the others.
- One-shot models (ernie, wan): no service; registry still **stops uma-heavy** peers first.
- Status: `GET /services` on the dashboard port.

### Docker targeting (local sock / remote)

Lifecycle subprocesses inherit Docker CLI env. Optional overrides:

| Env | Role |
|-----|------|
| `DOCKER_HOST` / `SLOPFINITY_DOCKER_HOST` | Engine endpoint (`unix:///var/run/docker.sock`, `ssh://user@gpu`, …) |
| `DOCKER_CONTEXT` / `SLOPFINITY_DOCKER_CONTEXT` | Named context (preferred for SSH) |
| `SLOPFINITY_COMPOSE_DIR` | Directory with `docker-compose.yaml` |
| `SLOPFINITY_DOCKER_BIN` | Path to `docker` binary |

Prefer **SSH Docker context** over open TCP `:2375`.

### Pure start/stop day-to-day

```bash
# One-time create on the GPU host
docker compose --profile slop up -d
docker compose --profile slop stop

# Park / unpark (same as ensure_down / ensure_up in container mode)
docker stop strix-halo-qwen-image strix-halo-qwen-tts strix-halo-heartmula strix-halo-comfyui
docker start strix-halo-qwen-tts   # etc.
```

Config example (remote host, container mode) — partial `network_services` in `config.json`:

```json
{
  "network_services": [
    {
      "id": "qwen-tts",
      "lifecycle_mode": "container",
      "container_name": "strix-halo-qwen-tts",
      "base_url": "http://192.168.1.50:8010",
      "health_url": "http://192.168.1.50:8010/health"
    }
  ]
}
```

Plus process env:

```bash
export DOCKER_CONTEXT=gpu
export TTS_WORKER_URL=http://192.168.1.50:8010/tts
export IMAGE_API_URL=http://192.168.1.50:8180
export HEARTMULA_URL=http://192.168.1.50:8011
export SLOPFINITY_COMFY_URL=http://192.168.1.50:8188
```

### Container map

| id | container_name | port | stages |
|----|----------------|------|--------|
| qwen-image | `strix-halo-qwen-image` | 8180 | `image:qwen` |
| qwen-tts | `strix-halo-qwen-tts` | 8010 | `tts:*` |
| heartmula | `strix-halo-heartmula` | 8011 | `audio:*` |
| comfyui | `strix-halo-comfyui` | 8188 | LTX image/video/upscale |

Do **not** enable auto-suspend `docker_stop` on these four containers — the registry owns them.

Legacy per-job docker generation: `SLOPFINITY_IMAGE_MODE=docker` or
`SLOPFINITY_AUDIO_MODE=docker`.

See [network-service-lifecycle-design.md](network-service-lifecycle-design.md).
